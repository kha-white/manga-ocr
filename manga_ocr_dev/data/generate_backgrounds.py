from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from manga_ocr_dev.env import MANGA109_ROOT, BACKGROUND_DIR


def find_rectangle(mask, y, x, aspect_ratio_range=(0.33, 3.0)):
    ymin_ = ymax_ = y
    xmin_ = xmax_ = x

    ymin = ymax = xmin = xmax = None

    while True:
        if ymin is None:
            ymin_ -= 1
            if ymin_ == 0 or mask[ymin_, xmin_:xmax_].any():
                ymin = ymin_

        if ymax is None:
            ymax_ += 1
            if ymax_ == mask.shape[0] - 1 or mask[ymax_, xmin_:xmax_].any():
                ymax = ymax_

        if xmin is None:
            xmin_ -= 1
            if xmin_ == 0 or mask[ymin_:ymax_, xmin_].any():
                xmin = xmin_

        if xmax is None:
            xmax_ += 1
            if xmax_ == mask.shape[1] - 1 or mask[ymin_:ymax_, xmax_].any():
                xmax = xmax_

        h = ymax_ - ymin_
        w = xmax_ - xmin_
        if h > 1 and w > 1:
            ratio = w / h
            if ratio < aspect_ratio_range[0] or ratio > aspect_ratio_range[1]:
                return ymin_, ymax_, xmin_, xmax_

        if None not in (ymin, ymax, xmin, xmax):
            return ymin, ymax, xmin, xmax


def generate_backgrounds(crops_per_page=5, min_size=40):
    data = pd.read_csv(MANGA109_ROOT / "data.csv")
    frames_df = pd.read_csv(MANGA109_ROOT / "frames.csv")

    BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

    page_paths = data.page_path.unique()
    for page_path in tqdm(page_paths):
        page = cv2.imread(str(MANGA109_ROOT / page_path))
        mask = np.zeros((page.shape[0], page.shape[1]), dtype=bool)
        for row in data[data.page_path == page_path].itertuples():
            mask[row.ymin : row.ymax, row.xmin : row.xmax] = True

        frames_mask = np.zeros((page.shape[0], page.shape[1]), dtype=bool)
        for row in frames_df[frames_df.page_path == page_path].itertuples():
            frames_mask[row.ymin : row.ymax, row.xmin : row.xmax] = True

        mask = mask | ~frames_mask

        if mask.all():
            continue

        unmasked_points = np.stack(np.where(~mask), axis=1)
        for i in range(crops_per_page):
            p = unmasked_points[np.random.randint(0, unmasked_points.shape[0])]
            y, x = p
            ymin, ymax, xmin, xmax = find_rectangle(mask, y, x)
            crop = page[ymin:ymax, xmin:xmax]

            if crop.shape[0] >= min_size and crop.shape[1] >= min_size:
                out_filename = (
                    "_".join(Path(page_path).with_suffix("").parts[-2:]) + f"_{ymin}_{ymax}_{xmin}_{xmax}.png"
                )
                cv2.imwrite(str(BACKGROUND_DIR / out_filename), crop)


if __name__ == "__main__":
    generate_backgrounds()
