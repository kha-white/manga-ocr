from pathlib import Path
from setuptools import setup

long_description = (Path(__file__).parent / "README.md").read_text('utf-8').split('# Installation')[0]

setup(
    name="manga-ocr",
    version='0.1.8',
    description="OCR for Japanese manga",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kha-white/manga-ocr",
    author="Maciej Budyś",
    author_email="kha-white@mail.com",
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=['manga_ocr'],
    include_package_data=True,
    install_requires=[
        "fire",
        "fugashi",
        "jaconv",
        "loguru",
        "numpy",
        "Pillow",
        "pyperclip",
        "sentencepiece",
        "torch>=1.0",
        "transformers>=4.12.5",
        "unidic_lite",
    ],
    entry_points={
        "console_scripts": [
            "manga_ocr=manga_ocr.__main__:main",
        ]
    },
)
