"""The single-plane preprocessing must be bit-identical to the stock ViTImageProcessor path."""

import pickle

import numpy as np
import pytest
import torch
from PIL import Image

from manga_ocr.preprocessing import OCR_RESIZE_BACKEND, preprocess_config_from_processor, preprocess_images


@pytest.fixture(scope="module")
def processor():
    try:
        from transformers import ViTImageProcessor

        return ViTImageProcessor.from_pretrained("kha-white/manga-ocr-base")
    except Exception as e:  # noqa: BLE001 - no model cache or no network: skip
        pytest.skip(f"manga-ocr processor unavailable: {e}")


@pytest.fixture(scope="module")
def config(processor):
    return preprocess_config_from_processor(processor)


def _random_crops(rng, n):
    crops = []
    for i in range(n):
        h = int(rng.integers(20, 120))
        w = int(rng.integers(20, 900))
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        if i % 3 == 0:  # some crops with duplicate shapes (the stock path batches equal shapes)
            arr = rng.integers(0, 256, size=(64, 300, 3), dtype=np.uint8)
        crops.append(Image.fromarray(arr).convert("RGB"))
    return crops


def test_config_describes_the_processor(processor, config):
    assert config["size"] == [224, 224]
    assert config["mean"] == list(processor.image_mean)
    assert config["std"] == list(processor.image_std)
    assert config["rescale_factor"] == processor.rescale_factor
    assert config["single_plane"] is True
    assert pickle.loads(pickle.dumps(config)) == config  # worker processes get it pickled


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_single_plane_equals_stock(processor, config, seed):
    rng = np.random.default_rng(seed)
    crops = _random_crops(rng, 24)
    stock = preprocess_images(crops, {**config, "single_plane": False}, processor=processor)
    single = preprocess_images(crops, config)
    assert single.shape == (len(crops), 1, 224, 224)
    assert single.dtype == torch.float32 and single.device.type == "cpu"
    assert stock.shape == (len(crops), 3, 224, 224)
    expanded = single.expand(-1, 3, -1, -1).contiguous()
    assert torch.equal(stock, expanded)
    assert torch.equal(stock.half(), expanded.half())


def test_accepts_uint8_arrays(config):
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(70, 200, 3), dtype=np.uint8)
    from_pil = preprocess_images([Image.fromarray(arr).convert("RGB")], config)
    from_array = preprocess_images([arr], config)
    assert torch.equal(from_pil, from_array)


def test_backend_follows_installed_stack(processor, config):
    """The single-plane path must reproduce whichever backend the installed processor uses."""
    expected = getattr(processor, "backend", None)
    if expected is None:  # transformers 4.x has no backend attribute
        import transformers

        expected = "pil" if int(transformers.__version__.split(".")[0]) < 5 else "torchvision"
    assert expected in ("pil", "torchvision")
    assert config["backend"] == expected
    assert OCR_RESIZE_BACKEND in ("pil", "torchvision")


def test_backends_differ(processor, config):
    """Sanity check that the two backends are really distinct arithmetic (else the version switch is moot)."""
    pytest.importorskip("torchvision")
    rng = np.random.default_rng(3)
    crops = _random_crops(rng, 8)
    a = preprocess_images(crops, {**config, "backend": "pil"})
    b = preprocess_images(crops, {**config, "backend": "torchvision"})
    assert a.shape == b.shape == (len(crops), 1, 224, 224)
    assert not torch.equal(a, b)
    stock = preprocess_images(crops, {**config, "single_plane": False}, processor=processor)
    assert torch.equal(stock, preprocess_images(crops, config).expand(-1, 3, -1, -1).contiguous())


def test_unusual_processor_config_falls_back():
    """A processor that cannot be reproduced from one plane must be detected (and the stock path required)."""
    from transformers import ViTImageProcessor

    processor = ViTImageProcessor(size={"height": 512, "width": 512})
    config = preprocess_config_from_processor(processor)
    assert config["single_plane"] is False


def test_stock_path_needs_processor(config):
    with pytest.raises(ValueError):
        preprocess_images([Image.new("RGB", (10, 10))], {**config, "single_plane": False})


def test_empty(config):
    empty = preprocess_images([], config)
    assert empty.shape == (0, 1, 224, 224)
    assert empty.dtype == torch.float32
