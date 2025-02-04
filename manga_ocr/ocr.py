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
            print('Using CUDA')
            self.model.cuda()
        elif not force_cpu and torch.backends.mps.is_available():
            logger.info('Using MPS')
            print('Using MPS')
            self.model.to('mps')
        else:
            logger.info('Using CPU')
            print('Using CPU')

        example_path = Path(__file__).parent / 'assets/example.jpg'
        if not example_path.is_file():
            example_path = Path(__file__).parent.parent / 'assets/example.jpg'
        self(example_path)

        logger.info('OCR ready')

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = [Image.open(img_or_path)]
        elif isinstance(img_or_path, Image.Image):
            img = [img_or_path]
        elif type(img_or_path) in (tuple, list):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        img = [i.convert('L').convert('RGB') for i in img]

        x = self._preprocess(img)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.model.generate(x.to(self.model.device), max_length=300).cpu()
        x = [self.tokenizer.decode(x[i], skip_special_tokens=True) for i in range(x.size(dim=0))]
        x = [post_process(i) for i in x]
        return x

    def _preprocess(self, img):
        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()


def post_process(text):
    text = ''.join(text.split())
    text = text.replace('…', '...')
    text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
    text = jaconv.h2z(text, ascii=True, digit=True)

    if not bool(text.strip()):
        return "<no ocr>"
    return text
