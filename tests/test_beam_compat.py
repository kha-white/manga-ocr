"""manga_ocr/beam.py must work on transformers 4.x (no ``scaling`` attribute) and 5.x (``scaling``)."""

import math

import torch

from manga_ocr.beam import _attn_scale


class _Tf5Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_head_size = 64
        self.scaling = self.attention_head_size**-0.5


class _Tf4Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_head_size = 64


def test_attn_scale_versions():
    assert _attn_scale(_Tf5Attn(), 64) == 64**-0.5
    assert _attn_scale(_Tf4Attn(), 64) == 64**-0.5 == 1 / math.sqrt(64)


def test_attn_scale_matches_sdpa_default():
    """4.x lets SDPA pick its default scale; the fallback must reproduce it bit-for-bit."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 12, 5, 64) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=_attn_scale(_Tf4Attn(), 64))
    assert torch.equal(ref, out)
