"""Skip the cross-attention KV cache reorder in model.generate()'s beam search.

_beam_search reorders both KV caches after every decoder step, but beam indices stay inside
an item's beam group and the cross-attention rows of that group are identical, so the
cross-attention gather returns its input and only wastes memory bandwidth. This attaches an
instance-level _reorder_cache that reorders only the self-attention cache, and falls back to
the stock reorder when the cache layout is unexpected. It is only used when the custom beam
search does not apply.
"""

import types

from loguru import logger


def install_skip_cross_attn_cache_reorder(model) -> bool:
    """Attach an instance-level _reorder_cache that skips the cross-attention gather."""
    try:
        from transformers.cache_utils import EncoderDecoderCache
    except ImportError as e:  # very old/new transformers
        logger.debug(f"skip-cross-attn-cache-reorder: EncoderDecoderCache unavailable ({e}); not patching")
        return False

    if hasattr(model, "_reorder_cache"):
        logger.debug("skip-cross-attn-cache-reorder: model already defines _reorder_cache; not patching")
        return False

    def _cross_batch(cache):
        """Batch size held by the cross-attention cache, or None if unknown/empty.

        transformers >= 4.56 / 5.x expose DynamicCache.layers[i].keys, 4.42-4.55 the
        DynamicCache.key_cache list; anything else returns None (the cross-attention rows are
        then assumed unchanged, which beam search guarantees).
        """
        try:
            cross = cache.cross_attention_cache
            layers = getattr(cross, "layers", None)
            if layers is not None:
                for layer in layers:
                    keys = getattr(layer, "keys", None)
                    if keys is not None and keys.dim() >= 1 and layer.get_seq_length() > 0:
                        return keys.shape[0]
                return None
            for keys in getattr(cross, "key_cache", []) or []:
                if keys is not None and keys.dim() == 4 and keys.shape[-2] > 0:
                    return keys.shape[0]
        except (AttributeError, IndexError, TypeError):  # unknown cache layout -> rows assumed unchanged
            return None
        return None

    def _reorder_cache(self, past_key_values, beam_idx):
        if (
            isinstance(past_key_values, EncoderDecoderCache)
            and hasattr(past_key_values, "self_attention_cache")
            and hasattr(past_key_values, "cross_attention_cache")
        ):
            n = _cross_batch(past_key_values)
            if n is None or n == beam_idx.numel():
                # Beam search keeps batch*num_beams rows for the whole run and only
                # permutes inside each item's group -> cross rows are unchanged.
                past_key_values.self_attention_cache.reorder_cache(beam_idx)
                return past_key_values
        # Anything unexpected: stock behaviour.
        if hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
        return past_key_values

    model._reorder_cache = types.MethodType(_reorder_cache, model)
    logger.debug("OCR generate(): cross-attention KV cache reorder skipped (identical output)")
    return True
