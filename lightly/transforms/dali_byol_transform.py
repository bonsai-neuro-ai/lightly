"""DALI-based BYOL augmentation pipeline prototype for GPU-accelerated data loading.

Requires nvidia-dali-cuda* installed. Feed an ImageFolder-style directory and get
a DataLoader-like iterator that yields (view1, view2) pairs as GPU tensors.

Usage::

    loader = DaliBYOLDataLoader(
        file_root="/data/imagenet/train",
        batch_size=256,
        num_threads=8,
        device_id=0,
    )
    for view1, view2 in loader:
        # view1, view2: torch.Tensor [B, 3, 224, 224] on GPU
        ...
"""

from __future__ import annotations

from typing import Iterator, Tuple

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator

    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False

from lightly.transforms.utils import IMAGENET_NORMALIZE

# ImageNet mean/std scaled to [0,255] for fn.crop_mirror_normalize.
_IMAGENET_MEAN_255 = [m * 255 for m in IMAGENET_NORMALIZE["mean"]]
_IMAGENET_STD_255 = [s * 255 for s in IMAGENET_NORMALIZE["std"]]


def _augment_view(
    images,
    *,
    input_size: int,
    min_scale: float,
    hf_prob: float,
    cj_prob: float,
    cj_bright: float,
    cj_contrast: float,
    cj_sat: float,
    cj_hue: float,
    gray_prob: float,
    blur_prob: float,
    sigma_min: float,
    sigma_max: float,
    solar_prob: float,
):
    """One BYOL view augmentation chain inside a @pipeline_def context."""

    # --- Random resized crop --------------------------------------------------
    images = fn.random_resized_crop(
        images,
        device="gpu",
        size=input_size,
        random_area=[min_scale, 1.0],
    )

    # --- Horizontal flip ------------------------------------------------------
    images = fn.flip(
        images,
        device="gpu",
        horizontal=fn.random.coin_flip(probability=hf_prob),
    )

    # --- Color jitter (applied with probability cj_prob) ----------------------
    # Torchvision hue ∈ [-cj_hue, cj_hue] (fraction of 1.0) → DALI degrees.
    do_cj = fn.random.coin_flip(probability=cj_prob)
    jittered = fn.color_twist(
        images,
        device="gpu",
        brightness=fn.random.uniform(
            range=[max(0.0, 1.0 - cj_bright), 1.0 + cj_bright]
        ),
        contrast=fn.random.uniform(
            range=[max(0.0, 1.0 - cj_contrast), 1.0 + cj_contrast]
        ),
        saturation=fn.random.uniform(
            range=[max(0.0, 1.0 - cj_sat), 1.0 + cj_sat]
        ),
        hue=fn.random.uniform(range=[-cj_hue * 360.0, cj_hue * 360.0]),
    )
    if do_cj:
        images = jittered

    # --- Random grayscale -----------------------------------------------------
    do_gray = fn.random.coin_flip(probability=gray_prob)
    if do_gray:
        gray = fn.color_space_conversion(
            images, device="gpu", image_type=types.RGB, output_type=types.GRAY
        )
        images = fn.color_space_conversion(
            gray, device="gpu", image_type=types.GRAY, output_type=types.RGB
        )

    # --- Gaussian blur (conditional) -----------------------------------------
    do_blur = fn.random.coin_flip(probability=blur_prob)
    if do_blur:
        images = fn.gaussian_blur(
            images,
            device="gpu",
            sigma=fn.random.uniform(range=[sigma_min, sigma_max]),
        )

    # --- Solarization (conditional) via arithmetic ----------------------------
    # f(p) = 255 - p if p >= 128 else p.  DALI DataNodes support arithmetic and comparison
    # operators element-wise, so we avoid fn.lookup_table whose API requires the LUT as a
    # positional tensor input (not a kwarg).
    do_solar = fn.random.coin_flip(probability=solar_prob)
    if do_solar:
        img_f = fn.cast(images, dtype=types.FLOAT, device="gpu")
        mask_f = fn.cast(img_f >= 128.0, dtype=types.FLOAT, device="gpu")
        images = fn.cast(
            mask_f * (255.0 - img_f) + (1.0 - mask_f) * img_f,
            dtype=types.UINT8,
            device="gpu",
        )

    # --- Normalize + HWC→CHW -------------------------------------------------
    # fn.crop_mirror_normalize converts uint8 HWC → float32 CHW with ImageNet stats.
    images = fn.crop_mirror_normalize(
        images,
        device="gpu",
        dtype=types.FLOAT,
        mean=_IMAGENET_MEAN_255,
        std=_IMAGENET_STD_255,
        output_layout="CHW",
    )

    return images


def _build_pipeline(
    file_root: str,
    batch_size: int,
    num_threads: int,
    device_id: int,
    # View 1 defaults match BYOLView1Transform
    v1_input_size: int = 224,
    v1_min_scale: float = 0.08,
    v1_hf_prob: float = 0.5,
    v1_cj_prob: float = 0.8,
    v1_cj_bright: float = 0.4,
    v1_cj_contrast: float = 0.4,
    v1_cj_sat: float = 0.2,
    v1_cj_hue: float = 0.1,
    v1_gray_prob: float = 0.2,
    v1_blur_prob: float = 1.0,
    v1_sigma_min: float = 0.1,
    v1_sigma_max: float = 2.0,
    v1_solar_prob: float = 0.0,
    # View 2 defaults match BYOLView2Transform
    v2_input_size: int = 224,
    v2_min_scale: float = 0.08,
    v2_hf_prob: float = 0.5,
    v2_cj_prob: float = 0.8,
    v2_cj_bright: float = 0.4,
    v2_cj_contrast: float = 0.4,
    v2_cj_sat: float = 0.2,
    v2_cj_hue: float = 0.1,
    v2_gray_prob: float = 0.2,
    v2_blur_prob: float = 0.1,
    v2_sigma_min: float = 0.1,
    v2_sigma_max: float = 2.0,
    v2_solar_prob: float = 0.2,
):
    @pipeline_def(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        enable_conditionals=True,
    )
    def pipeline():
        # Read raw JPEG bytes from an ImageFolder-style directory tree.
        jpegs, _ = fn.readers.file(
            file_root=file_root,
            random_shuffle=True,
            name="Reader",
        )
        # GPU-decode: yields uint8 HWC RGB tensors on GPU.
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

        view1 = _augment_view(
            images,
            input_size=v1_input_size,
            min_scale=v1_min_scale,
            hf_prob=v1_hf_prob,
            cj_prob=v1_cj_prob,
            cj_bright=v1_cj_bright,
            cj_contrast=v1_cj_contrast,
            cj_sat=v1_cj_sat,
            cj_hue=v1_cj_hue,
            gray_prob=v1_gray_prob,
            blur_prob=v1_blur_prob,
            sigma_min=v1_sigma_min,
            sigma_max=v1_sigma_max,
            solar_prob=v1_solar_prob,
        )
        view2 = _augment_view(
            images,
            input_size=v2_input_size,
            min_scale=v2_min_scale,
            hf_prob=v2_hf_prob,
            cj_prob=v2_cj_prob,
            cj_bright=v2_cj_bright,
            cj_contrast=v2_cj_contrast,
            cj_sat=v2_cj_sat,
            cj_hue=v2_cj_hue,
            gray_prob=v2_gray_prob,
            blur_prob=v2_blur_prob,
            sigma_min=v2_sigma_min,
            sigma_max=v2_sigma_max,
            solar_prob=v2_solar_prob,
        )
        return view1, view2

    return pipeline()


class DaliBYOLDataLoader:
    """DataLoader-like wrapper around a DALI BYOL augmentation pipeline.

    Reads JPEG images directly from disk, decodes and augments on GPU, and
    yields (view1, view2) pairs as GPU torch.Tensors — bypassing PIL and CPU
    augmentation entirely.

    Args:
        file_root:
            Root directory of an ImageFolder-style dataset (subdirs = classes).
        batch_size:
            Number of samples per batch.
        num_threads:
            Number of CPU threads for prefetching / decoding.
        device_id:
            CUDA device index.
        **pipeline_kwargs:
            Override any v1_* / v2_* augmentation hyperparameters; see
            _build_pipeline() for the full list.
    """

    def __init__(
        self,
        file_root: str,
        batch_size: int = 256,
        num_threads: int = 8,
        device_id: int = 0,
        **pipeline_kwargs,
    ):
        if not DALI_AVAILABLE:
            raise ImportError(
                "nvidia-dali-cuda* is required. "
                "Install with: pip install nvidia-dali-cuda130"
            )

        self._pipe = _build_pipeline(
            file_root=file_root,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            **pipeline_kwargs,
        )
        self._pipe.build()

        self._iter = DALIGenericIterator(
            self._pipe,
            output_map=["view1", "view2"],
            reader_name="Reader",
            auto_reset=True,
        )

    def __iter__(self) -> Iterator[Tuple]:
        for batch in self._iter:
            yield batch[0]["view1"], batch[0]["view2"]

    def __len__(self) -> int:
        return len(self._iter)
