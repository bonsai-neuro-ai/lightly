"""Correctness check: DALI BYOL pipeline vs torchvision BYOL pipeline.

Runs both pipelines over the same image directory many times and compares
the distributions of key statistics.  We cannot compare individual outputs
(both are stochastic) so we compare distributional summaries:

  1. Per-channel mean / std of the *unnormalized* pixel values.
     Both pipelines apply the same ImageNet normalization, so unnormalizing
     and checking that the raw distributions match is the cleanest signal.

  2. Grayscale rate: fraction of samples where |R_mean - G_mean| < tol and
     |R_mean - B_mean| < tol (per-image, over the channel dimension).
     Should match random_gray_scale probability (~0.2).

  3. Blur detection rate: fraction of samples with low high-frequency energy
     (Laplacian variance below a threshold), a proxy for Gaussian blur being
     applied.  View 1 has blur_prob=1.0, view 2 has blur_prob=0.1.

  4. Value range: after normalization outputs should sit in roughly [-3, 3].

Tolerances are intentionally loose because the pipelines are not identical:
- fn.color_twist applies brightness/contrast/saturation/hue in a fixed order;
  torchvision ColorJitter randomizes the order.
- DALI's grayscale uses ITU-R BT.601 weights; torchvision uses PIL/BT.601 too,
  but rounding may differ.

Usage:
    python dali_correctness_test.py /data/datasets/imagenet/val

The directory must be ImageFolder-style (class subdirectories).
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tv

from lightly.data import LightlyDataset
from lightly.transforms import BYOLTransform
from lightly.transforms.dali_byol_transform import DaliBYOLDataLoader
from lightly.transforms.utils import IMAGENET_NORMALIZE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
NUM_BATCHES = 50  # batches to collect stats over (~3200 images)
NUM_THREADS = 8
DEVICE = "cuda"

IMAGENET_MEAN = torch.tensor(IMAGENET_NORMALIZE["mean"], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor(IMAGENET_NORMALIZE["std"], device=DEVICE).view(1, 3, 1, 1)


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization → values in [0, 1]."""
    return x * IMAGENET_STD + IMAGENET_MEAN


def laplacian_variance(x: torch.Tensor) -> torch.Tensor:
    """Mean per-image variance of the Laplacian (proxy for sharpness). [B]"""
    # Grayscale for simplicity
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # [B, H, W]
    gray = gray.unsqueeze(1)  # [B, 1, H, W]
    lap_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=x.device
    ).view(1, 1, 3, 3)
    lap = F.conv2d(gray, lap_kernel, padding=1)
    return lap.var(dim=[1, 2, 3])


def is_grayscale(x: torch.Tensor, tol: float = 0.01) -> torch.Tensor:
    """Return bool tensor [B]: True if the image appears grayscale (R=G=B).

    After a grayscale round-trip (RGB→Gray→RGB) every pixel has R=G=B exactly,
    so per-pixel std across channels is 0.  Natural images with any color have
    per-pixel std well above 0.01, so this threshold cleanly separates the two
    cases.  Checking only R-G fails because many natural scenes (blue sky,
    concrete) have R≈G while B differs, producing false positives.

    Must be called on normalized tensors; we unnormalize internally so that
    different per-channel mean/std don't break the R=G=B equality.
    """
    raw = unnormalize(x).clamp(0.0, 1.0)
    per_pixel_std = raw.std(dim=1)  # [B, H, W] — 0 for perfect gray
    return per_pixel_std.mean(dim=[1, 2]) < tol


class Stats:
    """Accumulates per-batch statistics and reports a summary."""

    def __init__(self, label: str):
        self.label = label
        self.channel_means: list[torch.Tensor] = []
        self.channel_stds: list[torch.Tensor] = []
        self.gray_flags: list[torch.Tensor] = []
        self.lap_vars: list[torch.Tensor] = []
        self.min_vals: list[float] = []
        self.max_vals: list[float] = []

    def update(self, x: torch.Tensor) -> None:
        """x: [B, 3, H, W] normalized tensor on GPU."""
        raw = unnormalize(x)  # [0, 1]
        # Per-channel mean/std across spatial dims and batch
        self.channel_means.append(raw.mean(dim=[0, 2, 3]).cpu())
        self.channel_stds.append(raw.std(dim=[0, 2, 3]).cpu())
        self.gray_flags.append(is_grayscale(x).cpu())
        self.lap_vars.append(laplacian_variance(raw).cpu())
        self.min_vals.append(x.min().item())
        self.max_vals.append(x.max().item())

    def summarize(self) -> dict:
        means = torch.stack(self.channel_means).mean(0)  # [3]
        stds = torch.stack(self.channel_stds).mean(0)  # [3]
        gray_rate = torch.cat(self.gray_flags).float().mean().item()
        lap_mean = torch.cat(self.lap_vars).mean().item()
        return {
            "channel_mean": means.tolist(),
            "channel_std": stds.tolist(),
            "gray_rate": gray_rate,
            "lap_mean": lap_mean,
            "min_val": min(self.min_vals),
            "max_val": max(self.max_vals),
        }

    def report(self, other: "Stats | None" = None) -> None:
        s = self.summarize()
        print(f"\n=== {self.label} ===")
        print(f"  channel mean (R,G,B):  {[f'{v:.4f}' for v in s['channel_mean']]}")
        print(f"  channel std  (R,G,B):  {[f'{v:.4f}' for v in s['channel_std']]}")
        print(f"  grayscale rate:        {s['gray_rate']:.4f}  (expected ~0.20)")
        print(f"  Laplacian variance:    {s['lap_mean']:.6f}  (higher = sharper)")
        print(f"  value range:           [{s['min_val']:.3f}, {s['max_val']:.3f}]")
        if other is not None:
            o = other.summarize()
            delta_mean = [
                abs(a - b) for a, b in zip(s["channel_mean"], o["channel_mean"])
            ]
            delta_std = [abs(a - b) for a, b in zip(s["channel_std"], o["channel_std"])]
            delta_gray = abs(s["gray_rate"] - o["gray_rate"])
            delta_lap = abs(s["lap_mean"] - o["lap_mean"]) / max(
                s["lap_mean"], o["lap_mean"]
            )
            print(f"\n  vs {other.label}:")
            print(f"    |Δ channel mean|:   {[f'{v:.4f}' for v in delta_mean]}")
            print(f"    |Δ channel std|:    {[f'{v:.4f}' for v in delta_std]}")
            print(f"    |Δ gray rate|:      {delta_gray:.4f}  (tolerance 0.05)")
            print(f"    |Δ Laplacian| (%):  {delta_lap * 100:.1f}%  (tolerance 20%)")


def check(condition: bool, msg: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(file_root: str) -> None:
    print(f"Collecting {NUM_BATCHES} batches × {BATCH_SIZE} images from {file_root}")

    # --- DALI ----------------------------------------------------------------
    dali_v1 = Stats("DALI view1")
    dali_v2 = Stats("DALI view2")
    dali_dl = DaliBYOLDataLoader(
        file_root, batch_size=BATCH_SIZE, num_threads=NUM_THREADS
    )
    for i, (v1, v2) in enumerate(dali_dl):
        if i >= NUM_BATCHES:
            break
        dali_v1.update(v1)
        dali_v2.update(v2)

    # --- Torchvision ---------------------------------------------------------
    tv_v1 = Stats("Torchvision view1")
    tv_v2 = Stats("Torchvision view2")
    lightly_ds = LightlyDataset(file_root, transform=BYOLTransform())
    tv_dl = torch.utils.data.DataLoader(
        lightly_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_THREADS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    for i, (batch, _, _) in enumerate(tv_dl):
        if i >= NUM_BATCHES:
            break
        v1 = batch[0].to(DEVICE, non_blocking=True)
        v2 = batch[1].to(DEVICE, non_blocking=True)
        tv_v1.update(v1)
        tv_v2.update(v2)

    # --- Report & checks -----------------------------------------------------
    dali_v1.report(tv_v1)
    dali_v2.report(tv_v2)

    print("\n=== Assertion checks (view1) ===")
    ds = dali_v1.summarize()
    ts = tv_v1.summarize()
    check(ds["min_val"] > -4.0 and ds["max_val"] < 4.0, "DALI value range in [-4, 4]")
    check(ts["min_val"] > -4.0 and ts["max_val"] < 4.0, "TV   value range in [-4, 4]")
    check(
        abs(ds["gray_rate"] - 0.20) < 0.08,
        f"DALI gray rate near 0.20 (got {ds['gray_rate']:.3f})",
    )
    check(
        abs(ts["gray_rate"] - 0.20) < 0.08,
        f"TV   gray rate near 0.20 (got {ts['gray_rate']:.3f})",
    )
    delta_means = [abs(a - b) for a, b in zip(ds["channel_mean"], ts["channel_mean"])]
    delta_stds = [abs(a - b) for a, b in zip(ds["channel_std"], ts["channel_std"])]
    # fn.color_twist applies brightness/contrast/saturation/hue in a fixed order;
    # torchvision ColorJitter randomizes the order per sample.  This produces a
    # systematic ~0.04 shift in raw channel means that cannot be eliminated
    # without reimplementing randomized-order jitter in DALI.  Tolerance is set
    # to 0.06 to flag gross errors while accepting this known approximation.
    check(
        all(d < 0.06 for d in delta_means),
        f"Channel means agree within 0.06: {[f'{v:.4f}' for v in delta_means]}",
    )
    check(
        all(d < 0.04 for d in delta_stds),
        f"Channel stds agree within 0.04:  {[f'{v:.4f}' for v in delta_stds]}",
    )

    print("\n=== Assertion checks (view2) ===")
    ds2 = dali_v2.summarize()
    ts2 = tv_v2.summarize()
    check(
        abs(ds2["gray_rate"] - 0.20) < 0.08,
        f"DALI gray rate near 0.20 (got {ds2['gray_rate']:.3f})",
    )
    check(
        abs(ts2["gray_rate"] - 0.20) < 0.08,
        f"TV   gray rate near 0.20 (got {ts2['gray_rate']:.3f})",
    )
    # View 2: blur_prob=0.1 so Laplacian should be higher (sharper) than view 1
    lap_ratio = ds2["lap_mean"] / max(ds["lap_mean"], 1e-9)
    check(
        lap_ratio > 1.5,
        f"DALI view2 sharper than view1 (ratio {lap_ratio:.2f}, expected > 1.5)",
    )
    lap_ratio_tv = ts2["lap_mean"] / max(ts["lap_mean"], 1e-9)
    check(
        lap_ratio_tv > 1.5,
        f"TV   view2 sharper than view1 (ratio {lap_ratio_tv:.2f}, expected > 1.5)",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <file_root>")
        sys.exit(1)
    main(sys.argv[1])
