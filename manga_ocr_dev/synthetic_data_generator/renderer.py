import os
import uuid

import albumentations as A
import cv2
import numpy as np
from html2image import Html2Image

from manga_ocr_dev.env import BACKGROUND_DIR
from manga_ocr_dev.synthetic_data_generator.utils import get_background_df


class Renderer:
    def __init__(self):
        self.hti = Html2Image()
        self.background_df = get_background_df(BACKGROUND_DIR)
        self.max_size = 600

    def render(self, lines, override_css_params=None):
        img, params = self.render_text(lines, override_css_params)
        img = self.render_background(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = A.LongestMaxSize(self.max_size)(image=img)["image"]
        return img, params

    def render_text(self, lines, override_css_params=None):
        """Render text on transparent background and return as BGRA image."""

        params = self.get_random_css_params()
        if override_css_params:
            params.update(override_css_params)

        css = get_css(**params)

        # this is just a rough estimate, image is cropped later anyway
        size = (
            int(max(len(line) for line in lines) * params["font_size"] * 1.5),
            int(len(lines) * params["font_size"] * (3 + params["line_height"])),
        )
        if params["vertical"]:
            size = size[::-1]
        html = self.lines_to_html(lines)

        filename = str(uuid.uuid4()) + ".png"
        self.hti.screenshot(html_str=html, css_str=css, save_as=filename, size=size)
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        os.remove(filename)
        return img, params

    @staticmethod
    def get_random_css_params():
        params = {
            "font_size": 48,
            "vertical": True if np.random.rand() < 0.7 else False,
            "line_height": 0.5,
            "background_color": "transparent",
            "text_color": "black",
        }

        if np.random.rand() < 0.7:
            params["text_orientation"] = "upright"

        stroke_variant = np.random.choice(["stroke", "shadow", "none"], p=[0.8, 0.15, 0.05])
        if stroke_variant == "stroke":
            params["stroke_size"] = np.random.choice([1, 2, 3, 4, 8])
            params["stroke_color"] = "white"
        elif stroke_variant == "shadow":
            params["shadow_size"] = np.random.choice([2, 5, 10])
            params["shadow_color"] = ("white" if np.random.rand() < 0.8 else "black",)
        elif stroke_variant == "none":
            pass

        return params

    def render_background(self, img):
        """Add background and/or text bubble to a BGRA image, crop and return as BGR image."""
        draw_bubble = np.random.random() < 0.7

        m0 = int(min(img.shape[:2]) * 0.3)
        img = crop_by_alpha(img, m0)

        background_path = self.background_df.sample(1).iloc[0].path
        background = cv2.imread(background_path)

        t = [
            A.HorizontalFlip(),
            A.RandomRotate90(),
            A.InvertImg(),
            A.RandomBrightnessContrast((-0.2, 0.4), (-0.8, -0.3), p=0.5 if draw_bubble else 1),
            A.Blur((3, 5), p=0.3),
            A.Resize(img.shape[0], img.shape[1]),
        ]

        background = A.Compose(t)(image=background)["image"]

        if not draw_bubble:
            if np.random.rand() < 0.5:
                img[:, :, :3] = 255 - img[:, :, :3]

        else:
            radius = np.random.uniform(0.7, 1.0)
            thickness = np.random.choice([1, 2, 3])
            alpha = np.random.randint(60, 100)
            sigma = np.random.randint(10, 15)

            ymin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))
            ymax = img.shape[0] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))
            xmin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))
            xmax = img.shape[1] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))

            bubble_fill_color = (255, 255, 255, 255)
            bubble_contour_color = (0, 0, 0, 255)
            bubble = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            bubble = rounded_rectangle(
                bubble,
                (xmin, ymin),
                (xmax, ymax),
                radius=radius,
                color=bubble_fill_color,
                thickness=-1,
            )
            bubble = rounded_rectangle(
                bubble,
                (xmin, ymin),
                (xmax, ymax),
                radius=radius,
                color=bubble_contour_color,
                thickness=thickness,
            )

            t = [
                A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=0, p=0.8),
            ]
            bubble = A.Compose(t)(image=bubble)["image"]

            background = blend(bubble, background)

        img = blend(img, background)

        ymin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        ymax = img.shape[0] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        xmin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        xmax = img.shape[1] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        img = img[ymin:ymax, xmin:xmax]
        return img

    def lines_to_html(self, lines):
        lines_str = "\n".join(["<p>" + line + "</p>" for line in lines])
        html = f"<html><body>\n{lines_str}\n</body></html>"
        return html


def crop_by_alpha(img, margin):
    y, x = np.where(img[:, :, 3] > 0)
    ymin = y.min()
    ymax = y.max()
    xmin = x.min()
    xmax = x.max()
    img = img[ymin:ymax, xmin:xmax]
    img = np.pad(img, ((margin, margin), (margin, margin), (0, 0)))
    return img


def blend(img, background):
    alpha = (img[:, :, 3] / 255)[:, :, np.newaxis]
    img = img[:, :, :3]
    img = (background * (1 - alpha) + img * alpha).astype(np.uint8)
    return img


def rounded_rectangle(src, top_left, bottom_right, radius=1, color=255, thickness=1, line_type=cv2.LINE_AA):
    """From https://stackoverflow.com/a/60210706"""

    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3

    p1 = top_left
    p2 = (bottom_right[0], top_left[1])
    p3 = bottom_right
    p4 = (top_left[0], bottom_right[1])

    height = abs(bottom_right[1] - top_left[1])
    width = abs(bottom_right[0] - top_left[0])

    if radius > 1:
        radius = 1

    corner_radius = int(radius * (min(height, width) / 2))

    if thickness < 0:
        # big rect
        top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
        bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

        top_left_rect_left = (p1[0], p1[1] + corner_radius)
        bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

        top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
        bottom_right_rect_right = (p3[0], p3[1] - corner_radius)

        all_rects = [
            [top_left_main_rect, bottom_right_main_rect],
            [top_left_rect_left, bottom_right_rect_left],
            [top_left_rect_right, bottom_right_rect_right],
        ]

        [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

    # draw straight lines
    cv2.line(
        src,
        (p1[0] + corner_radius, p1[1]),
        (p2[0] - corner_radius, p2[1]),
        color,
        abs(thickness),
        line_type,
    )
    cv2.line(
        src,
        (p2[0], p2[1] + corner_radius),
        (p3[0], p3[1] - corner_radius),
        color,
        abs(thickness),
        line_type,
    )
    cv2.line(
        src,
        (p3[0] - corner_radius, p4[1]),
        (p4[0] + corner_radius, p3[1]),
        color,
        abs(thickness),
        line_type,
    )
    cv2.line(
        src,
        (p4[0], p4[1] - corner_radius),
        (p1[0], p1[1] + corner_radius),
        color,
        abs(thickness),
        line_type,
    )

    # draw arcs
    cv2.ellipse(
        src,
        (p1[0] + corner_radius, p1[1] + corner_radius),
        (corner_radius, corner_radius),
        180.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )
    cv2.ellipse(
        src,
        (p2[0] - corner_radius, p2[1] + corner_radius),
        (corner_radius, corner_radius),
        270.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )
    cv2.ellipse(
        src,
        (p3[0] - corner_radius, p3[1] - corner_radius),
        (corner_radius, corner_radius),
        0.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )
    cv2.ellipse(
        src,
        (p4[0] + corner_radius, p4[1] - corner_radius),
        (corner_radius, corner_radius),
        90.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )

    return src


def get_css(
    font_size,
    font_path,
    vertical=True,
    background_color="white",
    text_color="black",
    shadow_size=0,
    shadow_color="black",
    stroke_size=0,
    stroke_color="black",
    letter_spacing=None,
    line_height=0.5,
    text_orientation=None,
):
    styles = [
        f"background-color: {background_color};",
        f"font-size: {font_size}px;",
        f"color: {text_color};",
        "font-family: custom;",
        f"line-height: {line_height};",
        "margin: 20px;",
    ]

    if text_orientation:
        styles.append(f"text-orientation: {text_orientation};")

    if vertical:
        styles.append("writing-mode: vertical-rl;")

    if shadow_size > 0:
        styles.append(f"text-shadow: 0 0 {shadow_size}px {shadow_color};")

    if stroke_size > 0:
        # stroke is simulated by shadow overlaid multiple times
        styles.extend(
            [
                "text-shadow: " + ",".join([f"0 0 {stroke_size}px {stroke_color}"] * 10 * stroke_size) + ";",
                "-webkit-font-smoothing: antialiased;",
            ]
        )

    if letter_spacing:
        styles.append(f"letter-spacing: {letter_spacing}em;")

    font_path = font_path.replace("\\", "/")

    styles_str = "\n".join(styles)
    css = ""
    css += '\n@font-face {\nfont-family: custom;\nsrc: url("' + font_path + '");\n}\n'
    css += "body {\n" + styles_str + "\n}"
    return css
