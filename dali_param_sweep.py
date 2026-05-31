"""Parameter sweep: data-loading throughput and end-to-end training throughput.

Three sections:
  1. Data-only sweep — batch_size × worker_count grid for each pipeline.
     Finds the optimal configuration for raw loading/augmentation.

  2. Forward sweep — data + model forward, fixed workers, sweep batch_size.
     Shows how the data bottleneck changes as the model gets heavier.

  3. Forward+backward sweep — realistic training proxy (no optimizer step,
     but gradient computation included).

Models: ResNet-50 and ViT-B/16 (both 224×224 inputs).

Usage:
    python dali_param_sweep.py

Adjust the config block to match your system.
"""

import csv
import gc
import time
from pathlib import Path
from typing import assert_never

import torch
from torchvision.models import resnet50, vit_b_16
from tqdm import tqdm

from lightly.data import LightlyDataset
from lightly.transforms import BYOLTransform
from lightly.transforms.dali_byol_transform import DaliBYOLDataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MATMUL_PRECISION = "medium"
DATA_ROOT = "/data/datasets/imagenet/val/"
torch.set_float32_matmul_precision(MATMUL_PRECISION)

# Section 1: data-only sweep
BATCH_SIZES = [64, 128, 256, 512]
WORKER_COUNTS = [2, 4, 8, 16]

# Sections 2 & 3: model benchmarks
MODEL_BATCH_SIZES = [64, 128, 256, 512]
MODEL_WORKERS = 16

NUM_WARMUP = 5
NUM_BATCHES = 50


# ---------------------------------------------------------------------------
# Batch iterators
# ---------------------------------------------------------------------------
def dali_batches(loader):
    for v1, v2 in loader:
        yield v1, v2  # already on GPU


def tv_batches(loader):
    for (v1, v2), _, _ in loader:
        yield (
            v1.to("cuda", non_blocking=True),
            v2.to("cuda", non_blocking=True),
        )


# ---------------------------------------------------------------------------
# Model-usage functions with/without accelerators
# ---------------------------------------------------------------------------
def make_forward_fn(model: str, compile: bool = False, amp: bool = False):
    match model:
        case "resnet50":
            model = resnet50().cuda().eval()
        case "vit_b_16":
            model = vit_b_16().cuda().eval()
        case _:
            assert_never(model)

    if compile:
        model = torch.compile(model)

    def forward(v1, v2):
        out1 = model(v1)
        out2 = model(v2)
        _ = (out1 - out2).pow(2).sum()

    if amp:
        forward = torch.autocast("cuda", torch.float16)(forward)

    return forward


def make_forward_backward_fn(model: str, compile: bool = False, amp: bool = False):
    match model:
        case "resnet50":
            model = resnet50().cuda().train()
        case "vit_b_16":
            model = vit_b_16().cuda().train()
        case _:
            assert_never(model)

    if compile:
        model = torch.compile(model)

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    if amp:
        scaler = torch.amp.GradScaler("cuda")

        def forward_backward(v1, v2):
            with torch.autocast("cuda", torch.float16):
                out1 = model(v1)
                out2 = model(v2)
                loss = (out1 - out2).pow(2).sum()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

    else:

        def forward_backward(v1, v2):
            out1 = model(v1)
            out2 = model(v2)
            loss = (out1 - out2).pow(2).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

    return forward_backward


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------
def measure_throughput(batches, batch_size: int, process_fn=None) -> float:
    """Returns mean img/s over NUM_BATCHES after NUM_WARMUP warmup batches.

    process_fn(v1, v2) is called after each batch arrives on GPU; use it for
    model forward / forward+backward.
    """
    # Prepare timer following https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Warmup batches
    it = iter(batches)
    for _ in range(NUM_WARMUP):
        v1, v2 = next(it)
        if process_fn:
            process_fn(v1, v2)
        torch.cuda.synchronize()

    # Do actual timing
    intervals = []
    t = time.perf_counter()
    for _ in range(NUM_BATCHES):
        v1, v2 = next(it)
        if process_fn:
            process_fn(v1, v2)
        torch.cuda.synchronize()
        now = time.perf_counter()
        intervals.append(now - t)
        t = now

    return batch_size / (sum(intervals) / len(intervals))


def make_dali(batch_size, num_threads):
    return DaliBYOLDataLoader(DATA_ROOT, batch_size=batch_size, num_threads=num_threads)


def make_tv(batch_size, num_workers):
    ds = LightlyDataset(DATA_ROOT, transform=BYOLTransform())
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )


def run_config(backend, batch_size, workers, mode, model=None, compile=False, amp=False):
    match backend:
        case "DALI":
            iterator = dali_batches(make_dali(batch_size, workers))
        case "TV":
            iterator = tv_batches(make_tv(batch_size, workers))
        case _:
            assert_never(backend)

    match mode:
        case "data-only":
            process_fn = None
        case "fwd":
            process_fn = make_forward_fn(model, compile=compile, amp=amp)
        case "fwd+bwd":
            process_fn = make_forward_backward_fn(model, compile=compile, amp=amp)
        case _:
            assert_never(mode)

    try:
        throughput = measure_throughput(iterator, batch_size, process_fn)
    except RuntimeError:
        throughput = float("nan")
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "backend": backend,
        "model": model,
        "batch_size": batch_size,
        "workers": workers,
        "mode": mode,
        "compile": compile,
        "amp": amp,
        "throughput": throughput,
    }


if __name__ == "__main__":
    # — Part 1: Setup jobs to be run —————————————————————————

    # Start with 'data-only' results (no model, no model accelerators)
    jobs = set()
    for backend in ["DALI", "TV"]:
        for bs in BATCH_SIZES:
            for wrk in WORKER_COUNTS:
                jobs.add(
                    frozenset(
                        [
                            ("backend", backend),
                            ("batch_size", bs),
                            ("workers", wrk),
                            ("mode", "data-only"),
                        ]
                    )
                )

    # Further iterate over models and accelerators; use fixed batch size though
    for backend in ["DALI", "TV"]:
        for bs in MODEL_BATCH_SIZES:
            for model in ["resnet50", "vit_b_16"]:
                for compile in [True, False]:
                    for amp in [True, False]:
                        for mode in ["fwd", "fwd+bwd"]:
                            jobs.add(
                                frozenset(
                                    [
                                        ("backend", backend),
                                        ("batch_size", bs),
                                        ("workers", MODEL_WORKERS),
                                        ("mode", mode),
                                        ("model", model),
                                        ("compile", compile),
                                        ("amp", amp),
                                    ]
                                )
                            )

    # — Part 2: Load existing outputs and run those that need running ——————————
    results_file = Path("dali_param_sweep.csv")
    if not results_file.exists():
        results_file.touch()

    finished_jobs = set()
    with results_file.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            finished_jobs.add(frozenset((k, v) for (k, v) in row.items() if k != "throughput"))

    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "model",
                "batch_size",
                "workers",
                "mode",
                "compile",
                "amp",
                "throughput",
            ],
        )
        if f.tell() == 0:
            writer.writeheader()

        for job in tqdm(jobs - finished_jobs, initial=len(finished_jobs), total=len(jobs)):
            result = run_config(**dict(job))
            writer.writerow(result)
            f.flush()
