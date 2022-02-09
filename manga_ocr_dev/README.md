# Project structure

```
assets/                       # assets (see description below)
manga_ocr/                    # release code (inference only)
manga_ocr_dev/                # development code
   env.py                     # global constants
   data/                      # data preprocessing
   synthetic_data_generator/  # generation of synthetic image-text pairs
   training/                  # model training
```

## assets

### fonts.csv
csv with columns:
- font_path: path to font file, relative to `FONTS_ROOT`
- supported_chars: string of characters supported by this font
- num_chars: number of supported characters
- label: common/regular/special (used to sample regular fonts more often than special)

List of fonts with metadata used by synthetic data generator.
Provided file is just an example, you have to generate similar file for your own set of fonts,
using `manga_ocr_dev/synthetic_data_generator/scan_fonts.py` script.
Note that `label` will be filled with `regular` by default. You have to label your special fonts manually.

### lines_example.csv
csv with columns:
- source: source of text
- id: unique id of the line
- line: line from language corpus

Example of csv used for synthetic data generation.

### len_to_p.csv
csv with columns:
- len: length of text
- p: probability of text of this length occurring in manga

Used by synthetic data generator to more-or-less match the natural distribution of text lengths.
Computed based on Manga109-s dataset.

### vocab.csv
List of all characters supported by tokenizer.

# Training OCR

`env.py` contains global constants used across the repo. Set your paths to data etc. there.

1. Download [Manga109-s](http://www.manga109.org/en/download_s.html) dataset.
2. Set `MANGA109_ROOT`, so that your directory structure looks like this: 
    ```
    <MANGA109_ROOT>/
        Manga109s_released_2021_02_28/
            annotations/
            annotations.v2018.05.31/
            images/
            books.txt
            readme.txt
    ```
3. Preprocess Manga109-s with `data/process_manga109s.py`
4. Optionally generate synthetic data (see below)
5. Train with `manga_ocr_dev/training/train.py`

# Synthetic data generation

Generated data is split into packages (named `0000`, `0001` etc.) for easier management of large dataset.
Each package is assumed to have similar data distribution, so that a properly balanced dataset
can be built from any subset of packages.

Data generation pipeline assumes following directory structure:

```
<DATA_SYNTHETIC_ROOT>/
   img/           # generated images (output from generation pipeline)
      0000/
      0001/
      ...
   lines/         # lines from corpus (input to generation pipeline)
      0000.csv
      0001.csv
      ...
   meta/          # metadata (output from generation pipeline)
      0000.csv
      0001.csv
      ...
```

To use a language corpus for data generation, `lines/*.csv` files must be provided.
For a small example of such file see `assets/lines_example.csv`.

To generate synthetic data:
1. Generate backgrounds with `data/generate_backgrounds.py`.
2. Put your fonts in `<FONTS_ROOT>`.
3. Generate fonts metadata with `synthetic_data_generator/scan_fonts.py`.
4. Optionally manually label your fonts with `common/regular/special` labels.
5. Provide `<DATA_SYNTHETIC_ROOT>/lines/*.csv`.
6. Run `synthetic_data_generator/run_generate.py` for each package.
