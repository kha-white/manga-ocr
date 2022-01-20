import time
from pathlib import Path

import PIL.Image
import fire
import numpy as np
import pyperclip
from loguru import logger

from manga_ocr import MangaOcr


def are_images_identical(img1, img2):
    if None in (img1, img2):
        return img1 == img2

    img1 = np.array(img1)
    img2 = np.array(img2)

    return (img1.shape == img2.shape) and (img1 == img2).all()


def process_and_write_results(mocr, img_or_path, write_to):
    t0 = time.time()
    text = mocr(img_or_path)
    t1 = time.time()

    logger.info(f'Text recognized in {t1 - t0:0.03f} s: {text}')

    if write_to == 'clipboard':
        pyperclip.copy(text)
    else:
        write_to = Path(write_to)
        if write_to.suffix != '.txt':
            raise ValueError('write_to must be either "clipboard" or a path to a text file')

        with write_to.open('a') as f:
            f.write(text + '\n')


def run(read_from='clipboard',
        write_to='clipboard',
        pretrained_model_name_or_path='kha-white/manga-ocr-base',
        force_cpu=False,
        delay_secs=0.1
        ):
    """
    Run OCR in the background, waiting for new images to appear either in system clipboard, or a directory.
    Recognized texts can be either saved to system clipboard, or appended to a text file.

    :param read_from: Specifies where to read input images from. Can be either "clipboard", or a path to a directory.
    :param write_to: Specifies where to save recognized texts to. Can be either "clipboard", or a path to a text file.
    :param pretrained_model_name_or_path: Path to a trained model, either local or from Transformers' model hub.
    :param force_cpu: If True, OCR will use CPU even if GPU is available.
    :param delay_secs: How often to check for new images, in seconds.
    """

    mocr = MangaOcr(pretrained_model_name_or_path, force_cpu)

    if read_from == 'clipboard':
        import PIL.ImageGrab
        logger.info('Reading from clipboard')

        img = None
        while True:
            old_img = img

            try:
                img = PIL.ImageGrab.grabclipboard()
            except OSError:
                logger.warning('Error while reading from clipboard')
            else:
                if isinstance(img, PIL.Image.Image) and not are_images_identical(img, old_img):
                    process_and_write_results(mocr, img, write_to)

            time.sleep(delay_secs)


    else:
        read_from = Path(read_from)
        if not read_from.is_dir():
            raise ValueError('read_from must be either "clipboard" or a path to a directory')

        logger.info(f'Reading from directory {read_from}')

        old_paths = set()
        for path in read_from.iterdir():
            old_paths.add(path)

        while True:
            for path in read_from.iterdir():
                if path not in old_paths:
                    old_paths.add(path)

                    try:
                        img = PIL.Image.open(path)
                    except PIL.UnidentifiedImageError:
                        logger.warning(f'Error while reading file {path}')
                    else:
                        process_and_write_results(mocr, img, write_to)

            time.sleep(delay_secs)


if __name__ == '__main__':
    fire.Fire(run)
