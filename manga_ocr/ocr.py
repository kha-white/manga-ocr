import re
from pathlib import Path

import jaconv
import torch
from PIL import Image
from loguru import logger
from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel


class MangaOcr:
    def __init__(self, pretrained_model_name_or_path='kha-white/manga-ocr-base', force_cpu=False):
        logger.info(f'Loading OCR model from {pretrained_model_name_or_path}')
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)

        if not force_cpu and torch.cuda.is_available():
            logger.info('Using CUDA')
            self.model.cuda()
        else:
            logger.info('Using CPU')

        example_path = Path(__file__).parent / 'assets/example.jpg'
        if not example_path.is_file():
            example_path = Path(__file__).parent.parent / 'assets/example.jpg'
        self(example_path)

        logger.info('OCR ready')

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'Invalid value of img_or_path: {img_or_path}')

        img = img.convert('L').convert('RGB')

        x = self._preprocess(img)
        x = self.model.generate(x[None].to(self.model.device), max_length=300)[0].cpu()
        x = self.tokenizer.decode(x, skip_special_tokens=True)
        x = post_process(x)
        return x

    def _preprocess(self, img):
        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()


def post_process(text):
    text = ''.join(text.split())
    text = text.replace('…', '...')
    text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text
