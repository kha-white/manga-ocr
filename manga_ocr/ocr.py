import re
from pathlib import Path

import jaconv
import torch
from loguru import logger
from PIL import Image
from transformers import AutoTokenizer, GenerationMixin, VisionEncoderDecoderModel, ViTImageProcessor

from manga_ocr.beam import DEFAULT_SYNC_LAG, BeamSearchOCR
from manga_ocr.hf_patches import install_skip_cross_attn_cache_reorder
from manga_ocr.preprocessing import preprocess_config_from_processor, preprocess_images


class MangaOcrModel(VisionEncoderDecoderModel, GenerationMixin):
    pass


def _default_batch_size(device_type: str) -> int:
    """Tuned number of crops per generate() call."""
    if device_type == "cuda":
        return 128
    if device_type == "mps":
        return 64
    return 32


class MangaOcr:
    def __init__(
        self,
        pretrained_model_name_or_path="kha-white/manga-ocr-base",
        force_cpu=False,
        fp16=False,
        use_custom_beam=True,
    ):
        logger.info(f"Loading OCR model from {pretrained_model_name_or_path}")
        self.processor = ViTImageProcessor.from_pretrained(pretrained_model_name_or_path)
        # explicit tokenizer_type works around transformers>=5.13 misdetecting the tokenizer class
        # for VisionEncoderDecoderModel configs and falling back to an incompatible fast-only backend
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, tokenizer_type="bert-japanese")
        self.model = MangaOcrModel.from_pretrained(pretrained_model_name_or_path)

        if not force_cpu and torch.cuda.is_available():
            logger.info("Using CUDA")
            self.model.cuda()
        elif not force_cpu and torch.backends.mps.is_available():
            logger.info("Using MPS")
            self.model.to("mps")
        else:
            logger.info("Using CPU")

        # Picklable description of the preprocessing path (see manga_ocr.preprocessing).
        self.preprocess_config = preprocess_config_from_processor(self.processor)
        if not self.preprocess_config["single_plane"]:
            logger.warning(
                "MangaOcr: this processor config cannot use single-plane preprocessing; using stock processor"
            )

        # fp16 only makes sense on an accelerator; on CPU the model stays fp32.
        self.fp16 = False
        if fp16:
            if self.model.device.type == "cpu":
                logger.warning("fp16 requested, but the model runs on CPU; staying in fp32")
            else:
                self.model.half()
                self.fp16 = True

        # Custom beam search for the model's own generation config; generate() is the fallback.
        self._beam = None
        if use_custom_beam:
            try:
                self._beam = BeamSearchOCR(self.model, sync_lag=DEFAULT_SYNC_LAG)
            except Exception as e:  # noqa: BLE001 - unexpected model layout: use the stock generate() path
                logger.warning(f"Custom beam search unavailable ({e}); falling back to generate()")
        install_skip_cross_attn_cache_reorder(self.model)

        example_path = Path(__file__).parent / "assets/example.jpg"
        if not example_path.is_file():
            raise FileNotFoundError(f"Missing example image {example_path}")
        self(example_path)

        logger.info("OCR ready")

    def __call__(self, img_or_path):
        img = self._load_image(img_or_path)
        img = img.convert("L").convert("RGB")
        return self.recognize([img], max_length=300)[0]

    def preprocess(self, images) -> torch.Tensor:
        """Preprocess PIL images or image paths into a float32 CPU tensor.

        Returns (N,1,224,224) with the single-plane path, or the stock (N,3,224,224) processor
        output for an unusual processor config.
        """
        images = self._load_images(images)
        if not images:
            return torch.empty((0, 1, 224, 224), dtype=torch.float32)
        return preprocess_images(images, self.preprocess_config, processor=self.processor)

    def recognize_preprocessed(self, pixel_values, *, batch_size=None, **generate_kwargs) -> list[str]:
        """Recognize a preprocessed (N,1,224,224) or (N,3,224,224) tensor; one text per crop.

        batch_size caps how many crops are decoded per generate() call; other keyword arguments go to
        the decoder (e.g. num_beams=1 uses model.generate instead of the custom beam search).
        """
        n = pixel_values.shape[0]
        if n == 0:
            return []
        if batch_size is None:
            batch_size = _default_batch_size(self.model.device.type)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        gen_args = self._generation_args(**generate_kwargs)
        texts = []
        for i in range(0, n, batch_size):
            texts.extend(self._generate_texts(pixel_values[i : i + batch_size], gen_args))
        return texts

    def recognize(self, images, *, batch_size=None, **generate_kwargs) -> list[str]:
        """Preprocess and recognize PIL images or image paths; one text per image.

        Same as preprocess() followed by recognize_preprocessed(), with preprocessing done in
        batch_size chunks to keep the memory bounded.
        """
        images = self._load_images(images)
        if not images:
            return []
        if batch_size is None:
            batch_size = _default_batch_size(self.model.device.type)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        texts = []
        for i in range(0, len(images), batch_size):
            pixel_values = self.preprocess(images[i : i + batch_size])
            texts.extend(self.recognize_preprocessed(pixel_values, batch_size=batch_size, **generate_kwargs))
        return texts

    def _generation_args(self, **generation_kwargs) -> dict:
        """Resolved generate() kwargs: the model generation config plus explicit overrides."""
        gen_config = self.model.generation_config
        gen_args = {
            "max_length": getattr(gen_config, "max_length", 300),
            "num_beams": getattr(gen_config, "num_beams", 1),
            "do_sample": getattr(gen_config, "do_sample", False),
            "use_cache": True,
        }
        gen_args.update({k: v for k, v in generation_kwargs.items() if v is not None})
        return gen_args

    def _generate_texts(self, pixel_values: torch.Tensor, gen_args: dict) -> list[str]:
        """Decode one preprocessed batch (CPU or device tensor) into post-processed texts."""
        pixel_values = pixel_values.to(self.model.device, non_blocking=True)
        if self.fp16:
            pixel_values = pixel_values.half()
        if pixel_values.shape[1] == 1:
            # single grayscale plane -> the model's 3 identical channels
            pixel_values = pixel_values.expand(-1, 3, -1, -1).contiguous()

        with torch.inference_mode():
            if self._beam is not None and self._beam.matches(gen_args):
                generated_ids = self._beam.generate(pixel_values)
            else:
                generated_ids = self.model.generate(pixel_values, **gen_args)

        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [post_process(text) for text in texts]

    @staticmethod
    def _load_image(img_or_path) -> Image.Image:
        if isinstance(img_or_path, (str, Path)):
            return Image.open(img_or_path)
        if isinstance(img_or_path, Image.Image):
            return img_or_path
        raise ValueError(f"img_or_path must be a path or PIL.Image, instead got: {img_or_path}")

    @classmethod
    def _load_images(cls, images) -> list[Image.Image]:
        if isinstance(images, (str, Path, Image.Image)):
            images = [images]
        return [cls._load_image(img) for img in images]


def post_process(text):
    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text
