"""Hand-rolled beam search for the manga-ocr model (ViT encoder, 2-layer BERT decoder).

Reproduces transformers' GenerationMixin._beam_search token for token for the model's
generation config, but keeps the self-attention K/V in a preallocated cache that is reordered
in place and projects the cross-attention K/V once per crop instead of once per beam. It calls
the model's own decoder modules with the same shapes, so the numerics are the same kernels.
Only attribute names stable across transformers 4.x and 5.x are read from the decoder.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# What MangaOcr uses: on CUDA the host-side beam bookkeeping overlaps with GPU execution, at the
# cost of at most one wasted decoder step. It cannot change the result.
DEFAULT_SYNC_LAG = 1


def _attn_scale(attn_module, head_size: int) -> float:
    """Softmax scale of a BERT attention module: 5.x stores it, 4.x uses SDPA's default."""
    scaling = getattr(attn_module, "scaling", None)
    if scaling is None:
        return float(head_size) ** -0.5
    return float(scaling)


def _gather_beams(tensor: torch.Tensor, beam_indices: torch.Tensor) -> torch.Tensor:
    # same as GenerationMixin._gather_beams
    while beam_indices.dim() < tensor.dim():
        beam_indices = beam_indices.unsqueeze(-1)
    return torch.take_along_dim(tensor, beam_indices, dim=1)


def _no_repeat_ngram(input_ids: torch.Tensor, scores: torch.Tensor, ngram_size: int) -> torch.Tensor:
    # same as transformers>=5.16 NoRepeatNGramLogitsProcessor.__call__ (vectorised)
    cur_len = input_ids.shape[-1]
    if cur_len < ngram_size:
        return scores
    prefix = input_ids[:, cur_len + 1 - ngram_size :]
    windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)
    matches = (windows[..., :-1] == prefix.unsqueeze(1)).all(dim=-1)
    vocab_size = scores.shape[-1]
    banned_mask = scores.new_zeros((scores.shape[0], vocab_size + 1), dtype=torch.bool)
    banned_mask.scatter_(1, torch.where(matches, windows[..., -1], vocab_size), True)
    return scores.masked_fill(banned_mask[:, :vocab_size], -float("inf"))


class BeamSearchOCR:
    """Beam-search decoder for a VisionEncoderDecoderModel with a BERT decoder.

    generate(pixel_values) returns LongTensor[B, out_len] like model.generate (best beam per item,
    right-padded with pad_token_id).
    """

    _GEN_KEYS = frozenset({"max_length", "num_beams", "do_sample", "use_cache"})

    def __init__(self, model, sync_lag: int = 0, cache_len: int = 64):
        self.model = model
        gc = model.generation_config
        self.num_beams = int(gc.num_beams)
        self.ngram = int(gc.no_repeat_ngram_size or 0)
        self.length_penalty = float(gc.length_penalty)
        self.early_stopping = gc.early_stopping
        self.max_length = int(gc.max_length)
        self.eos = int(gc.eos_token_id)
        self.pad = int(gc.pad_token_id)
        self.start = int(gc.decoder_start_token_id)
        self.sync_lag = int(sync_lag)

        dec = model.decoder
        self.emb = dec.bert.embeddings
        self.layers = list(dec.bert.encoder.layer)
        self.cls = dec.cls
        self.vocab = int(dec.config.vocab_size)
        sa = self.layers[0].attention.self
        self.H = int(sa.num_attention_heads)
        self.D = int(sa.attention_head_size)
        # per layer: (self-attention scale, cross-attention scale); see _attn_scale
        self.scales = [
            (_attn_scale(layer.attention.self, self.D), _attn_scale(layer.crossattention.self, self.D))
            for layer in self.layers
        ]
        self.proj = getattr(model, "enc_to_dec_proj", None)

        self._cache = []  # per layer: [k, v] each [rows, H, cap, D]
        self._flags = None  # pinned host buffer for sync_lag
        self._rows = 0
        self._cap = 0
        self._init_cap = max(8, int(cache_len))

    def matches(self, gen_args: dict) -> bool:
        """True if model.generate(pixel_values, **gen_args) would run the beam search we replicate."""
        if not set(gen_args) <= self._GEN_KEYS:
            return False
        if gen_args.get("do_sample", False):
            return False
        if not gen_args.get("use_cache", True):
            return False
        if int(gen_args.get("num_beams", self.num_beams)) != self.num_beams or self.num_beams < 2:
            return False
        if int(gen_args.get("max_length", self.max_length)) != self.max_length:
            return False
        gc = self.model.generation_config
        if gc.num_return_sequences != 1 or gc.num_beam_groups != 1 or gc.repetition_penalty != 1.0:
            return False
        return gc.min_length in (None, 0) and gc.encoder_no_repeat_ngram_size in (None, 0)

    def _ensure_cache(self, rows: int, need_len: int, ref: torch.Tensor):
        if rows <= self._rows and need_len <= self._cap and self._cache:
            return
        cap = max(self._cap, self._init_cap)
        while cap < need_len:
            cap = min(self.max_length, cap * 2)
        rows_alloc = max(rows, self._rows)
        new = []
        for i in range(len(self.layers)):
            k = ref.new_empty((rows_alloc, self.H, cap, self.D))
            v = ref.new_empty((rows_alloc, self.H, cap, self.D))
            if self._cache:
                ok, ov = self._cache[i]
                k[: self._rows, :, : self._cap] = ok
                v[: self._rows, :, : self._cap] = ov
            new.append([k, v])
        self._cache = new
        self._rows, self._cap = rows_alloc, cap

    def _step(self, tokens: torch.Tensor, pos: int, cross, B: int) -> torch.Tensor:
        """tokens: [R, 1] int64 (R = B * num_beams); returns fp logits [R, vocab] for position pos."""
        R = tokens.shape[0]
        H, D, nb = self.H, self.D, self.num_beams
        h = self.emb(input_ids=tokens, past_key_values_length=pos)  # [R, 1, 768]
        for i, layer in enumerate(self.layers):
            scale_self, scale_cross = self.scales[i]
            # self-attention over the preallocated cache, rows 0..pos
            sa = layer.attention.self
            q = sa.query(h).view(R, 1, H, D).transpose(1, 2)
            k = sa.key(h).view(R, 1, H, D).transpose(1, 2)
            v = sa.value(h).view(R, 1, H, D).transpose(1, 2)
            kc, vc = self._cache[i]
            kc[:R, :, pos : pos + 1].copy_(k)
            vc[:R, :, pos : pos + 1].copy_(v)
            a = F.scaled_dot_product_attention(
                q,
                kc[:R, :, : pos + 1],
                vc[:R, :, : pos + 1],
                attn_mask=None,
                dropout_p=0.0,
                scale=scale_self,
                is_causal=False,
            )
            a = a.transpose(1, 2).contiguous().reshape(R, 1, H * D)
            h = layer.attention.output(a, h)
            # cross-attention: the beams of one crop are folded into the query-length axis
            ca = layer.crossattention.self
            q = ca.query(h).view(B, nb, H, D).transpose(1, 2)  # [B, H, nb, D]
            kx, vx = cross[i]  # [B, H, S, D]
            a = F.scaled_dot_product_attention(
                q, kx, vx, attn_mask=None, dropout_p=0.0, scale=scale_cross, is_causal=False
            )
            a = a.transpose(1, 2).contiguous().reshape(R, 1, H * D)
            h = layer.crossattention.output(a, h)
            h = layer.output(layer.intermediate(h), h)
        return self.cls(h)[:, -1, :]

    def _reorder(self, beam_idx: torch.Tensor, length: int, R: int):
        for kc, vc in self._cache:
            kc[:R, :, :length].copy_(kc[:R, :, :length].index_select(0, beam_idx))
            vc[:R, :, :length].copy_(vc[:R, :, :length].index_select(0, beam_idx))

    @torch.inference_mode()
    def generate(self, pixel_values: torch.Tensor) -> torch.Tensor:
        model = self.model
        nb, vocab, max_length, lp = self.num_beams, self.vocab, self.max_length, self.length_penalty
        early_stopping = self.early_stopping is True
        B = pixel_values.shape[0]
        R = B * nb
        dev = pixel_values.device

        enc = model.encoder(pixel_values=pixel_values, return_dict=True).last_hidden_state  # [B, S, 768]
        if self.proj is not None and enc.shape[-1] != self.layers[0].attention.self.query.in_features:
            enc = self.proj(enc)
        S = enc.shape[1]
        cross = []
        for layer in self.layers:
            ca = layer.crossattention.self
            kx = ca.key(enc).view(B, S, self.H, self.D).transpose(1, 2)
            vx = ca.value(enc).view(B, S, self.H, self.D).transpose(1, 2)
            cross.append((kx, vx))
        self._ensure_cache(R, self._init_cap, enc)

        beams_to_keep = 2 * nb
        top_num_beam_mask = torch.cat(
            (torch.ones(nb, dtype=torch.bool), torch.zeros(beams_to_keep - nb, dtype=torch.bool))
        ).to(dev)
        batch_offset = (torch.arange(B, device=dev) * nb).view(-1, 1)

        running_sequences = torch.full((B, nb, max_length), self.pad, dtype=torch.int64, device=dev)
        running_sequences[:, :, 0] = self.start
        sequences = running_sequences.clone()
        running_beam_scores = torch.zeros((B, nb), dtype=torch.float32, device=dev)
        running_beam_scores[:, 1:] = -1e9
        beam_scores = torch.full((B, nb), -1e9, dtype=torch.float32, device=dev)
        is_sent_finished = torch.zeros((B, nb), dtype=torch.bool, device=dev)
        finished_len = torch.zeros((B, nb), dtype=torch.int64, device=dev)
        unsatisfied = torch.ones((B, 1), dtype=torch.bool, device=dev)

        # sync_lag reads the previous step's keep_going flag from pinned memory so the host does not
        # block on the GPU; a late stop costs at most one wasted step and cannot change the output.
        use_lag = self.sync_lag > 0 and dev.type == "cuda"
        if use_lag and self._flags is None:
            self._flags = torch.empty(max_length + 1, dtype=torch.bool, pin_memory=True)
        pending = []  # (event, index into self._flags) for sync_lag
        cur_len = 1
        tokens = running_sequences[:, :, 0].reshape(R, 1)
        while True:
            pos = cur_len - 1
            if pos + 1 > self._cap:
                self._ensure_cache(R, pos + 1, enc)
            logits = self._step(tokens, pos, cross, B).to(dtype=torch.float32, copy=True)
            log_probs = F.log_softmax(logits, dim=-1)
            if self.ngram:
                log_probs = _no_repeat_ngram(
                    running_sequences[:, :, :cur_len].reshape(R, cur_len), log_probs, self.ngram
                )
            log_probs = log_probs.view(B, nb, vocab) + running_beam_scores[:, :, None]
            log_probs = log_probs.reshape(B, nb * vocab)

            # top-K continuations
            topk_log_probs, topk_indices = torch.topk(log_probs, k=beams_to_keep)
            topk_current_beam_indices = topk_indices // vocab
            topk_running_sequences = _gather_beams(running_sequences, topk_current_beam_indices)
            topk_ids = topk_indices % vocab
            topk_running_sequences[:, :, cur_len] = topk_ids
            hits = topk_ids == self.eos
            if cur_len + 1 >= max_length:
                hits = torch.ones_like(hits)

            # running beams for next iteration
            topk_running_log_probs = topk_log_probs + hits.to(torch.float32) * -1.0e9
            next_topk_indices = torch.topk(topk_running_log_probs, k=nb)[1]
            running_sequences = _gather_beams(topk_running_sequences, next_topk_indices)
            running_beam_scores = _gather_beams(topk_running_log_probs, next_topk_indices)
            beam_src = _gather_beams(topk_current_beam_indices, next_topk_indices)  # [B, nb]

            # finished beams
            did_finish = hits & top_num_beam_mask[None, :]
            fin_log_probs = topk_log_probs / ((cur_len + 1 - 1) ** lp)
            beams_full = torch.all(is_sent_finished, dim=-1, keepdim=True) & early_stopping
            fin_log_probs += beams_full.to(torch.float32) * -1.0e9
            fin_log_probs += (~unsatisfied).to(torch.float32) * -1.0e9
            fin_log_probs += (~did_finish) * -1.0e9
            merged_sequences = torch.cat((sequences, topk_running_sequences), dim=1)
            merged_scores = torch.cat((beam_scores, fin_log_probs), dim=1)
            merged_len = torch.cat((finished_len, torch.full_like(hits, cur_len, dtype=torch.int64)), dim=1)
            merged_finished = torch.cat((is_sent_finished, did_finish), dim=1)
            topk_merged = torch.topk(merged_scores, k=nb)[1]
            sequences = _gather_beams(merged_sequences, topk_merged)
            beam_scores = _gather_beams(merged_scores, topk_merged)
            finished_len = _gather_beams(merged_len, topk_merged)
            is_sent_finished = _gather_beams(merged_finished, topk_merged)

            # reorder the self-attention cache prefix (rows 0..cur_len-1) to the surviving beams
            self._reorder((beam_src + batch_offset).view(-1), cur_len, R)

            cur_len += 1
            best_possible = running_beam_scores[:, :1] / ((cur_len - 1) ** lp)
            worst_finished = torch.where(is_sent_finished, torch.min(beam_scores, dim=1, keepdim=True)[0], -1.0e9)
            unsatisfied = unsatisfied & torch.any(best_possible > worst_finished, dim=-1, keepdim=True)
            improvement_possible = torch.any(unsatisfied)
            exists_open_beam = ~(torch.all(is_sent_finished) & early_stopping)
            valid_continuations = ~torch.all(hits)
            keep_going = improvement_possible & exists_open_beam & valid_continuations

            if cur_len >= max_length:
                break
            if use_lag:
                self._flags[cur_len].copy_(keep_going, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()
                pending.append((ev, cur_len))
                if len(pending) > self.sync_lag:
                    ev0, i0 = pending.pop(0)
                    ev0.synchronize()
                    if not bool(self._flags[i0]):
                        break
            elif not bool(keep_going):
                break
            tokens = running_sequences[:, :, cur_len - 1].reshape(R, 1)

        out_len = 1 + int(finished_len[:, 0].max())
        return sequences[:, 0, :out_len]
