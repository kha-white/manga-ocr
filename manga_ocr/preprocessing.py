"""Bit-identical single-plane preprocessing for the MangaOCR model.

MangaOcr converts a crop to L and back to RGB, so the three channels fed to the stock
ViTImageProcessor are identical. Resizing the luma plane once with the arithmetic of the
installed processor (PIL on transformers 4.x, torchvision on 5.x, they differ) and expanding
it to three channels on the device saves two thirds of the resampling work and host-to-device
bytes. Unusual processor configs are detected and marked single_plane=False; the module does
not import transformers, so worker processes can use it without loading the OCR stack.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

_OCR_SIZE = [224, 224]
_DEFAULT_MEAN = 0.5
_DEFAULT_STD = 0.5
_DEFAULT_RESCALE = 1 / 255


def _stock_resize_backend() -> str:
    """Return "torchvision" (transformers >= 5) or "pil" (transformers 4.x) without importing it."""
    try:
        from importlib.metadata import version

        major = int(version("transformers").split(".")[0])
    except (ImportError, ValueError):  # transformers not installed / unparsable version
        return "torchvision"
    return "pil" if major < 5 else "torchvision"


OCR_RESIZE_BACKEND = _stock_resize_backend()


def _processor_backend(processor) -> str:
    """Resize backend of an actual processor instance (falls back to the version default)."""
    backend = getattr(processor, "backend", None)  # transformers >= 5: "torchvision" or "pil"
    if backend in ("pil", "torchvision"):
        return backend
    return _stock_resize_backend()  # transformers 4.x loads the PIL processor


def _size_hw(size) -> list[int] | None:
    """Height and width of a processor size field, or None if it is not a fixed size."""
    if size is None:
        return None
    if isinstance(size, int):
        return [size, size]
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return [int(size[0]), int(size[1])]
    if hasattr(size, "get"):
        height, width = size.get("height"), size.get("width")
        if height and width:
            return [int(height), int(width)]
    return None


def _is_bilinear(resample) -> bool:
    # PILImageResampling.BILINEAR == torchvision InterpolationMode.BILINEAR == 2
    try:
        return int(resample) == 2
    except (TypeError, ValueError):
        return False


def preprocess_config_from_processor(processor) -> dict:
    """Describe how to reproduce processor's output on a single plane.

    Returns a picklable dict with size, mean, std, rescale_factor, backend ("torchvision" or
    "pil") and single_plane. single_plane is False when the processor cannot be reproduced
    from one channel (unusual size/resample/mean/std, missing rescale or normalize), in which
    case callers must pass the stock processor instead.
    """
    mean = [float(v) for v in getattr(processor, "image_mean", (_DEFAULT_MEAN,))]
    std = [float(v) for v in getattr(processor, "image_std", (_DEFAULT_STD,))]
    rescale_factor = float(getattr(processor, "rescale_factor", _DEFAULT_RESCALE))
    size = _size_hw(getattr(processor, "size", None)) or list(_OCR_SIZE)

    single_plane = (
        len(mean) == len(std) == 3
        and all(v == mean[0] for v in mean)
        and all(v == std[0] for v in std)
        and bool(getattr(processor, "do_resize", False))
        and bool(getattr(processor, "do_rescale", False))
        and bool(getattr(processor, "do_normalize", False))
        and _is_bilinear(getattr(processor, "resample", None))
        and size == _OCR_SIZE
    )
    return {
        "size": size,
        "mean": mean,
        "std": std,
        "rescale_factor": rescale_factor,
        "backend": _processor_backend(processor),
        "single_plane": single_plane,
    }


def _to_gray(item) -> np.ndarray:
    """PIL image or uint8 array -> uint8 HxW luma, as MangaOcr.__call__ computes it."""
    if isinstance(item, np.ndarray):
        if item.ndim == 2:
            return item
        item = Image.fromarray(item)
    if item.mode != "RGB":
        item = item.convert("RGB")
    return np.asarray(item.convert("L"))


def _stock_preprocess(images, processor) -> torch.Tensor:
    crops = [im.convert("L").convert("RGB") for im in images]
    return processor(crops, return_tensors="pt").pixel_values


def preprocess_images(images, config, processor=None) -> torch.Tensor:
    """Preprocess PIL images into a float32 CPU tensor (N,1,H,W) or (N,3,H,W).

    config is a preprocess_config_from_processor() dict. With single_plane the result is the
    luma plane, bit-identical to the stock processor's three-channel output (the model expands
    it on the device). Otherwise the stock processor is used and must be passed. Empty input
    returns an empty (0,1,224,224) tensor.
    """
    if not images:
        return torch.empty((0, 1, _OCR_SIZE[0], _OCR_SIZE[1]), dtype=torch.float32)
    size = list(config.get("size") or _OCR_SIZE)
    if not config.get("single_plane", True):
        if processor is None:
            raise ValueError(
                "preprocess_images: config['single_plane'] is False, so the stock processor is needed; "
                "pass it as processor=..."
            )
        return _stock_preprocess(images, processor)

    backend = config.get("backend") or OCR_RESIZE_BACKEND
    rescale_factor = float(config.get("rescale_factor", _DEFAULT_RESCALE))
    mean = float(config.get("mean", [_DEFAULT_MEAN])[0])
    std = float(config.get("std", [_DEFAULT_STD])[0])

    planes = []
    if backend == "pil":
        # transformers 4.x: PIL resize, then rescale (float64 -> float32) and normalise (float32)
        for item in images:
            im = Image.fromarray(np.ascontiguousarray(_to_gray(item)))
            im = im.resize((size[1], size[0]), resample=Image.BILINEAR, reducing_gap=None)
            planes.append(torch.from_numpy(np.array(im)).unsqueeze(0))
        x = torch.stack(planes)
        x = x.to(torch.float64).mul_(rescale_factor).to(torch.float32)
        return x.sub_(mean).div_(std)
    # torchvision is only needed for a torchvision-backed processor (transformers >= 5), so it is
    # imported lazily like transformers does.
    from torchvision.transforms.v2 import functional as tvF

    for item in images:
        t = torch.from_numpy(np.array(_to_gray(item))).unsqueeze(0)  # 1xHxW uint8, writable copy
        t = tvF.resize(t, size, interpolation=tvF.InterpolationMode.BILINEAR, antialias=True)
        planes.append(t)
    x = torch.stack(planes)
    # transformers 5.x fuses rescale and normalize: (x - mean/rescale_factor) / (std/rescale_factor)
    mean = np.float32(mean * (1.0 / rescale_factor))
    std = np.float32(std * (1.0 / rescale_factor))
    return x.to(torch.float32).sub_(float(mean)).div_(float(std))
