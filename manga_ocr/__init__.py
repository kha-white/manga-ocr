"""Manga OCR: optical character recognition for Japanese text.

MangaOcr is imported lazily (PEP 562), so importing a helper module such as
manga_ocr.preprocessing does not load transformers.
"""

import importlib

from manga_ocr._version import __version__ as __version__

__all__ = ["MangaOcr", "__version__"]

_LAZY_ATTRS = {"MangaOcr": ("manga_ocr.ocr", "MangaOcr")}


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
