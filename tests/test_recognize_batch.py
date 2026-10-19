"""The batched recognition API must give the same text as the single-image __call__ path."""

import subprocess
import sys
from pathlib import Path

import PIL.Image
import pytest
import torch

from manga_ocr import MangaOcr

TEST_DATA_ROOT = Path(__file__).parent / "data"
N_IMAGES = 3


@pytest.fixture(scope="module")
def mocr():
    try:
        return MangaOcr()
    except Exception as e:  # noqa: BLE001 - no model cache or no network: skip
        pytest.skip(f"manga-ocr model unavailable: {e}")


@pytest.fixture(scope="module")
def images():
    paths = sorted((TEST_DATA_ROOT / "images").iterdir())[:N_IMAGES]
    return [PIL.Image.open(path).copy() for path in paths]


def test_recognize_single_equals_call(mocr, images):
    assert mocr.recognize([images[0]])[0] == mocr(images[0])


def test_recognize_batch_is_per_image(mocr, images):
    batch = mocr.recognize(images, batch_size=N_IMAGES)
    one_by_one = mocr.recognize(images, batch_size=1)
    assert batch == one_by_one
    assert batch == [mocr(img) for img in images]


def test_preprocess_then_recognize_preprocessed(mocr, images):
    pixel_values = mocr.preprocess(images)
    assert pixel_values.shape == (len(images), 1, 224, 224)
    assert pixel_values.dtype == torch.float32 and pixel_values.device.type == "cpu"
    assert mocr.recognize_preprocessed(pixel_values, batch_size=2) == mocr.recognize(images)


def test_empty_inputs(mocr):
    assert mocr.preprocess([]).shape == (0, 1, 224, 224)
    assert mocr.recognize([]) == []


def test_custom_beam_matches_generate(mocr, images):
    """The default custom beam search must decode exactly like model.generate() with the model config."""
    beam = mocr._beam
    assert beam is not None
    pixel_values = mocr.preprocess(images)
    pixel_values = pixel_values.to(mocr.model.device).expand(-1, 3, -1, -1).contiguous()
    gen_args = mocr._generation_args()
    with torch.inference_mode():
        ids_beam = beam.generate(pixel_values)
        ids_generate = mocr.model.generate(pixel_values, **gen_args)
    assert mocr.tokenizer.batch_decode(ids_beam, skip_special_tokens=True) == mocr.tokenizer.batch_decode(
        ids_generate, skip_special_tokens=True
    )


def test_importing_preprocessing_does_not_load_transformers():
    """Worker processes import only manga_ocr.preprocessing, so it must not pull in the OCR stack."""
    code = (
        "import sys\n"
        "import manga_ocr.preprocessing\n"
        "assert 'transformers' not in sys.modules, sorted(m for m in sys.modules if 'transformers' in m)\n"
        "assert 'manga_ocr.ocr' not in sys.modules, 'manga_ocr.ocr imported eagerly'\n"
        "assert manga_ocr.__version__\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parent.parent)
