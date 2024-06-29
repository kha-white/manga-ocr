import pandas as pd
import unicodedata

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT


def get_background_df(background_dir):
    background_df = []
    for path in background_dir.iterdir():
        ymin, ymax, xmin, xmax = [int(v) for v in path.stem.split("_")[-4:]]
        h = ymax - ymin
        w = xmax - xmin
        ratio = w / h

        background_df.append(
            {
                "path": str(path),
                "h": h,
                "w": w,
                "ratio": ratio,
            }
        )
    background_df = pd.DataFrame(background_df)
    return background_df


def is_kanji(ch):
    return "CJK UNIFIED IDEOGRAPH" in unicodedata.name(ch)


def is_hiragana(ch):
    return "HIRAGANA" in unicodedata.name(ch)


def is_katakana(ch):
    return "KATAKANA" in unicodedata.name(ch)


def is_ascii(ch):
    return ord(ch) < 128


def get_charsets(vocab_path=None):
    if vocab_path is None:
        vocab_path = ASSETS_PATH / "vocab.csv"
    vocab = pd.read_csv(vocab_path).char.values
    hiragana = vocab[[is_hiragana(c) for c in vocab]][:-6]
    katakana = vocab[[is_katakana(c) for c in vocab]][3:]
    return vocab, hiragana, katakana


def get_font_meta():
    df = pd.read_csv(ASSETS_PATH / "fonts.csv")
    df.font_path = df.font_path.apply(lambda x: str(FONTS_ROOT / x))
    font_map = {row.font_path: set(row.supported_chars) for row in df.dropna().itertuples()}
    return df, font_map
