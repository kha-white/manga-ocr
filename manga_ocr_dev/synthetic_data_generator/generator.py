import budou
import numpy as np
import pandas as pd

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
from manga_ocr_dev.synthetic_data_generator.utils import (
    get_font_meta,
    get_charsets,
    is_ascii,
    is_kanji,
)


class SyntheticDataGenerator:
    def __init__(self):
        self.vocab, self.hiragana, self.katakana = get_charsets()
        self.len_to_p = pd.read_csv(ASSETS_PATH / "len_to_p.csv")
        self.parser = budou.get_parser("tinysegmenter")
        self.fonts_df, self.font_map = get_font_meta()
        self.font_labels, self.font_p = self.get_font_labels_prob()
        self.renderer = Renderer()

    def process(self, text=None, override_css_params=None):
        """
        Generate image, text pair. Use source text if provided, otherwise generate random text.
        """

        if override_css_params is None:
            override_css_params = {}

        if text is None:
            # if using random text, choose font first,
            # and then generate text using only characters supported by that font
            if "font_path" not in override_css_params:
                font_path = self.get_random_font()
                vocab = self.font_map[font_path]
                override_css_params["font_path"] = font_path
            else:
                font_path = override_css_params["font_path"]
                vocab = self.font_map[font_path]

            words = self.get_random_words(vocab)

        else:
            text = text.replace("　", " ")
            text = text.replace("…", "...")
            words = self.split_into_words(text)

        lines = self.words_to_lines(words)
        text_gt = "\n".join(lines)

        if "font_path" not in override_css_params:
            override_css_params["font_path"] = self.get_random_font(text_gt)

        font_path = override_css_params.get("font_path")
        if font_path:
            vocab = self.font_map.get(font_path)

            # remove unsupported characters
            lines = ["".join([c for c in line if c in vocab]) for line in lines]
            text_gt = "\n".join(lines)
        else:
            vocab = None

        if np.random.random() < 0.5:
            word_prob = np.random.choice([0.33, 1.0], p=[0.3, 0.7])

            lines = [self.add_random_furigana(line, word_prob, vocab) for line in lines]

        img, params = self.renderer.render(lines, override_css_params)
        return img, text_gt, params

    def get_random_words(self, vocab):
        vocab = list(vocab)
        max_text_len = np.random.choice(self.len_to_p.len, p=self.len_to_p.p)

        words = []
        text_len = 0
        while True:
            word = "".join(np.random.choice(vocab, np.random.randint(1, 4)))
            words.append(word)
            text_len += len(word)
            if text_len + len(word) >= max_text_len:
                break

        return words

    def split_into_words(self, text):
        max_text_len = np.random.choice(self.len_to_p.len, p=self.len_to_p.p)

        words = []
        text_len = 0
        for chunk in self.parser.parse(text)["chunks"]:
            words.append(chunk.word)
            text_len += len(chunk.word)
            if text_len + len(chunk.word) >= max_text_len:
                break

        return words

    def words_to_lines(self, words):
        text = "".join(words)

        max_num_lines = 10
        min_line_len = len(text) // max_num_lines
        max_line_len = 20
        max_line_len = np.clip(np.random.poisson(6), min_line_len, max_line_len)
        lines = []
        line = ""
        for word in words:
            line += word
            if len(line) >= max_line_len:
                lines.append(line)
                line = ""
        if line:
            lines.append(line)

        return lines

    def add_random_furigana(self, line, word_prob=1.0, vocab=None):
        if vocab is None:
            vocab = self.vocab
        else:
            vocab = list(vocab)

        processed = ""
        kanji_group = ""
        ascii_group = ""
        for i, c in enumerate(line):
            if is_kanji(c):
                c_type = "kanji"
                kanji_group += c
            elif is_ascii(c):
                c_type = "ascii"
                ascii_group += c
            else:
                c_type = "other"

            if c_type != "kanji" or i == len(line) - 1:
                if kanji_group:
                    if np.random.uniform() < word_prob:
                        furigana_len = int(np.clip(np.random.normal(1.5, 0.5), 1, 4) * len(kanji_group))
                        char_source = np.random.choice(["hiragana", "katakana", "all"], p=[0.8, 0.15, 0.05])
                        char_source = {
                            "hiragana": self.hiragana,
                            "katakana": self.katakana,
                            "all": vocab,
                        }[char_source]
                        furigana = "".join(np.random.choice(char_source, furigana_len))
                        processed += f"<ruby>{kanji_group}<rt>{furigana}</rt></ruby>"
                    else:
                        processed += kanji_group
                    kanji_group = ""

            if c_type != "ascii" or i == len(line) - 1:
                if ascii_group:
                    if len(ascii_group) <= 3 and np.random.uniform() < 0.7:
                        processed += f'<span style="text-combine-upright: all">{ascii_group}</span>'
                    else:
                        processed += ascii_group
                    ascii_group = ""

            if c_type == "other":
                processed += c

        return processed

    def is_font_supporting_text(self, font_path, text):
        chars = self.font_map[font_path]
        for c in text:
            if c.isspace():
                continue
            if c not in chars:
                return False
        return True

    def get_font_labels_prob(self):
        labels = {
            "common": 0.2,
            "regular": 0.75,
            "special": 0.05,
        }
        labels = {k: labels[k] for k in self.fonts_df.label.unique()}
        p = np.array(list(labels.values()))
        p = p / p.sum()
        labels = list(labels.keys())
        return labels, p

    def get_random_font(self, text=None):
        label = np.random.choice(self.font_labels, p=self.font_p)
        df = self.fonts_df[self.fonts_df.label == label]

        if text is None:
            return df.sample(1).iloc[0].font_path

        valid_mask = df.font_path.apply(lambda x: self.is_font_supporting_text(x, text))
        if not valid_mask.any():
            # if text contains characters not supported by any font, just pick some of the more capable fonts
            valid_mask = df.num_chars >= 4000

        return str(FONTS_ROOT / df[valid_mask].sample(1).iloc[0].font_path)
