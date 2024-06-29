import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from manga_ocr_dev.env import MANGA109_ROOT, DATA_SYNTHETIC_ROOT


class MangaDataset(Dataset):
    def __init__(
        self,
        processor,
        split,
        max_target_length,
        limit_size=None,
        augment=False,
        skip_packages=None,
    ):
        self.processor = processor
        self.max_target_length = max_target_length

        data = []

        print(f"Initializing dataset {split}...")

        if skip_packages is None:
            skip_packages = set()
        else:
            skip_packages = {f"{x:04d}" for x in skip_packages}

        for path in sorted((DATA_SYNTHETIC_ROOT / "meta").glob("*.csv")):
            if path.stem in skip_packages:
                print(f"Skipping package {path}")
                continue
            if not (DATA_SYNTHETIC_ROOT / "img" / path.stem).is_dir():
                print(f"Missing image data for package {path}, skipping")
                continue
            df = pd.read_csv(path)
            df = df.dropna()
            df["path"] = df.id.apply(lambda x: str(DATA_SYNTHETIC_ROOT / "img" / path.stem / f"{x}.jpg"))
            df = df[["path", "text"]]
            df["synthetic"] = True
            data.append(df)

        df = pd.read_csv(MANGA109_ROOT / "data.csv")
        df = df[df.split == split].reset_index(drop=True)
        df["path"] = df.crop_path.apply(lambda x: str(MANGA109_ROOT / x))
        df = df[["path", "text"]]
        df["synthetic"] = False
        data.append(df)

        data = pd.concat(data, ignore_index=True)

        if limit_size:
            data = data.iloc[:limit_size]
        self.data = data

        print(f"Dataset {split}: {len(self.data)}")

        self.augment = augment
        self.transform_medium, self.transform_heavy = self.get_transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.loc[idx]
        text = sample.text

        if self.augment:
            medium_p = 0.8
            heavy_p = 0.02
            transform_variant = np.random.choice(
                ["none", "medium", "heavy"],
                p=[1 - medium_p - heavy_p, medium_p, heavy_p],
            )
            transform = {
                "none": None,
                "medium": self.transform_medium,
                "heavy": self.transform_heavy,
            }[transform_variant]
        else:
            transform = None

        pixel_values = self.read_image(self.processor, sample.path, transform)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        labels = np.array(labels)
        # important: make sure that PAD tokens are ignored by the loss function
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        encoding = {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels),
        }
        return encoding

    @staticmethod
    def read_image(processor, path, transform=None):
        img = cv2.imread(str(path))

        if transform is None:
            transform = A.ToGray(always_apply=True)

        img = transform(image=img)["image"]

        pixel_values = processor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()

    @staticmethod
    def get_transforms():
        t_medium = A.Compose(
            [
                A.Rotate(5, border_mode=cv2.BORDER_REPLICATE, p=0.2),
                A.Perspective((0.01, 0.05), pad_mode=cv2.BORDER_REPLICATE, p=0.2),
                A.InvertImg(p=0.05),
                A.OneOf(
                    [
                        A.Downscale(0.25, 0.5, interpolation=cv2.INTER_LINEAR),
                        A.Downscale(0.25, 0.5, interpolation=cv2.INTER_NEAREST),
                    ],
                    p=0.1,
                ),
                A.Blur(p=0.2),
                A.Sharpen(p=0.2),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise((50, 200), p=0.3),
                A.ImageCompression(0, 30, p=0.1),
                A.ToGray(always_apply=True),
            ]
        )

        t_heavy = A.Compose(
            [
                A.Rotate(10, border_mode=cv2.BORDER_REPLICATE, p=0.2),
                A.Perspective((0.01, 0.05), pad_mode=cv2.BORDER_REPLICATE, p=0.2),
                A.InvertImg(p=0.05),
                A.OneOf(
                    [
                        A.Downscale(0.1, 0.2, interpolation=cv2.INTER_LINEAR),
                        A.Downscale(0.1, 0.2, interpolation=cv2.INTER_NEAREST),
                    ],
                    p=0.1,
                ),
                A.Blur((4, 9), p=0.5),
                A.Sharpen(p=0.5),
                A.RandomBrightnessContrast(0.8, 0.8, p=1),
                A.GaussNoise((1000, 10000), p=0.3),
                A.ImageCompression(0, 10, p=0.5),
                A.ToGray(always_apply=True),
            ]
        )

        return t_medium, t_heavy


if __name__ == "__main__":
    from manga_ocr_dev.training.get_model import get_processor
    from manga_ocr_dev.training.utils import tensor_to_image

    encoder_name = "facebook/deit-tiny-patch16-224"
    decoder_name = "cl-tohoku/bert-base-japanese-char-v2"

    max_length = 300

    processor = get_processor(encoder_name, decoder_name)
    ds = MangaDataset(processor, "train", max_length, augment=True)

    for i in range(20):
        sample = ds[0]
        img = tensor_to_image(sample["pixel_values"])
        tokens = sample["labels"]
        tokens[tokens == -100] = processor.tokenizer.pad_token_id
        text = "".join(processor.decode(tokens, skip_special_tokens=True).split())

        print(f"{i}:\n{text}\n")
        plt.imshow(img)
        plt.show()
