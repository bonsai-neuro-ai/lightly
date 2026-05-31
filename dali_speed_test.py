"""BYOL data-loading speed benchmark: DALI vs torchvision.

Two scenarios:
  1. Data only  — measures raw data loading + augmentation throughput.
  2. Data + model — measures end-to-end per-iteration wall time (the number
     that determines actual training speed).

Fairness measures applied vs the original benchmark:
  - Warm-up: NUM_WARMUP batches are consumed and synced before timing starts,
    so DALI's prefetch pipeline is fully primed and torchvision's worker pool
    is hot before any measurement is recorded.
  - torch.cuda.synchronize() before every timer tick: GPU work (NVJPEG decode,
    augmentation, H→D transfer, model forward) is async; without a sync we
    measure CPU enqueue latency, not actual throughput.
  - torchvision DataLoader uses persistent_workers=True and prefetch_factor=4
    so workers stay alive across epochs and always have data ready.
  - Both pipelines use the same NUM_WORKERS and BATCH_SIZE.
  - A fresh DALI loader is created for each benchmark to avoid the mid-epoch
    reset warning that appeared when reusing the same iterator.
  - torch.no_grad() in model benchmarks (we're measuring throughput, not
    training; gradients would add overhead unrelated to data loading).
"""

import time

import torch
from torchvision.models import resnet50

from lightly.data import LightlyDataset
from lightly.transforms import BYOLTransform
from lightly.transforms.dali_byol_transform import DaliBYOLDataLoader

# ---------------------------------------------------------------------------
# Config — adjust to your system
# ---------------------------------------------------------------------------
DATA_ROOT = "/data/datasets/imagenet/val/"
BATCH_SIZE = 256
NUM_WORKERS = 8  # CPU threads/workers; matched across both pipelines
NUM_WARMUP = 10  # batches to discard before timing
NUM_BATCHES = 100  # batches to time per benchmark


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------
class Timer:
    def __init__(self, name: str) -> None:
        self.name = name
        self._start: float | None = None
        self.intervals: list[float] = []

    def tick(self) -> None:
        """Record one interval: time since the previous tick (or first call)."""
        now = time.perf_counter()
        if self._start is not None:
            self.intervals.append(now - self._start)
        self._start = now

    def report(self) -> None:
        if not self.intervals:
            print(f"[{self.name}] no data")
            return
        t = self.intervals
        avg = sum(t) / len(t)
        throughput = BATCH_SIZE / avg
        print(
            f"[{self.name}]  "
            f"avg {avg * 1e3:.1f} ms/batch  "
            f"({throughput:.0f} img/s)  "
            f"min {min(t) * 1e3:.1f}  max {max(t) * 1e3:.1f}  "
            f"n={len(t)}"
        )


# ---------------------------------------------------------------------------
# Loader factories
# ---------------------------------------------------------------------------
def make_dali() -> DaliBYOLDataLoader:
    return DaliBYOLDataLoader(DATA_ROOT, batch_size=BATCH_SIZE, num_threads=NUM_WORKERS)


def make_tv() -> torch.utils.data.DataLoader:
    ds = LightlyDataset(DATA_ROOT, transform=BYOLTransform())
    return torch.utils.data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )


# ---------------------------------------------------------------------------
# Batch iterators — unified interface: yield (v1, v2) as CUDA tensors
# ---------------------------------------------------------------------------
def dali_batches(loader: DaliBYOLDataLoader):
    for v1, v2 in loader:
        yield v1, v2  # already on GPU


def tv_batches(loader: torch.utils.data.DataLoader):
    for (v1, v2), _, _ in loader:
        # pin_memory=True + non_blocking=True: async H→D transfer;
        # torch.cuda.synchronize() in the benchmark loop ensures completion.
        yield (
            v1.to("cuda", non_blocking=True),
            v2.to("cuda", non_blocking=True),
        )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run(name: str, batches, extra_fn=None) -> Timer:
    """
    Warm up for NUM_WARMUP batches (synced but not timed), then time
    NUM_BATCHES batches.  extra_fn(v1, v2) is called after each batch
    is on GPU (used for model-forward benchmarks).
    """
    print(f"\n  [{name}] warming up ({NUM_WARMUP} batches)…", flush=True)
    it = iter(batches)
    for _ in range(NUM_WARMUP):
        v1, v2 = next(it)
        if extra_fn:
            extra_fn(v1, v2)
        torch.cuda.synchronize()

    print(f"  [{name}] timing ({NUM_BATCHES} batches)…", flush=True)
    timer = Timer(name)
    timer.tick()  # mark start; first interval measured from here
    for _ in range(NUM_BATCHES):
        v1, v2 = next(it)
        if extra_fn:
            extra_fn(v1, v2)
        torch.cuda.synchronize()
        timer.tick()

    timer.report()
    return timer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
model = resnet50().eval().cuda()


def forward(v1: torch.Tensor, v2: torch.Tensor) -> None:
    out1 = model(v1)
    out2 = model(v2)
    _ = (out1 - out2).pow(2).sum()


print("=" * 60)
print("Benchmark: data loading only")
print("=" * 60)
run("DALI ", dali_batches(make_dali()))
run("TV   ", tv_batches(make_tv()))

print()
print("=" * 60)
print("Benchmark: data loading + ResNet-50 forward")
print("=" * 60)
with torch.no_grad():
    run("DALI ", dali_batches(make_dali()), forward)
    run("TV   ", tv_batches(make_tv()), forward)


# RESULTS:
# ============================================================
# Benchmark: data loading only
# ============================================================
# [DALI ]  avg 15.6 ms/batch  (16374 img/s)  min 12.3  max 19.7  n=100
# [TV   ]  avg 294.8 ms/batch  (868 img/s)  min 12.1  max 2330.8  n=100
#
# ============================================================
# Benchmark: data loading + ResNet-50 forward
# ============================================================
# [DALI ]  avg 314.5 ms/batch  (814 img/s)  min 309.1  max 318.5  n=100
# [TV   ]  avg 330.3 ms/batch  (775 img/s)  min 319.9  max 418.4  n=100


# RESULTS (no TV prefetch):
# ============================================================
# Benchmark: data loading only
# ============================================================
# [DALI ]  avg 15.5 ms/batch  (16473 img/s)  min 10.3  max 19.9  n=100
# [TV   ]  avg 303.3 ms/batch  (844 img/s)  min 12.1  max 2017.2  n=100
#
# ============================================================
# Benchmark: data loading + ResNet-50 forward
# ============================================================
# [DALI ]  avg 315.0 ms/batch  (813 img/s)  min 309.3  max 319.4  n=100
# [TV   ]  avg 326.4 ms/batch  (784 img/s)  min 320.4  max 423.1  n=100
