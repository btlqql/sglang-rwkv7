# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""RWKV-7 (Goose) linear-attention backend for sglang.

Unlike GDN, the RWKV-7 model module does ALL projections / LoRAs / gating in plain
torch and hands the backend already-projected per-head tensors. This backend owns
the two pieces of recurrent state in the MambaPool:

  * conv[0] / conv[1] : the two width-2 (prev-token) token-shift states
                        (attn / ffn), shape (size+1, hidden, 1), fp32.
  * temporal          : the WKV recurrent state S, shape (size+1, H, K, V), fp32.

The model calls `token_shift(...)` (before projections) and `recurrence(...)`
(after projections) directly on this backend instance, which it obtains via
`forward_batch.attn_backend.linear_attn_backend`. We therefore bypass the
RadixLinearAttention / HybridLinearAttnBackend.forward dispatch (whose fixed
mixed_qkv/a/b signature does not fit RWKV-7's r/w/k/v/kk/a). `init_forward_metadata`
is still driven through HybridLinearAttnBackend, so `self.forward_metadata`
(query_start_loc + mamba_cache_indices) is populated normally.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata

# The WKV recurrence is a self-contained triton kernel for BOTH the decode
# (T==1) AND the extend/prefill (packed varlen via cu_seqlens) path, with no
# dependency on the flash-linear-attention package.
from sglang.srt.layers.attention.rwkv7_kernels import (
    layernorm_token_shift_lerp1_decode,
    layernorm_token_shift_lerp6_decode,
    token_shift_lerp6_decode,
    token_shift_lerp6_packed_varlen,
    token_shift_packed_varlen,
    wkv_recurrent,
)
from sglang.srt.layers.attention.rwkv7_kernels.fused import fused_lerp6
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class Rwkv7NoOpFullAttnBackend(AttentionBackend):
    """A trivial full-attention backend for the all-linear RWKV-7 case.

    RWKV-7 has ZERO full-attention layers, so HybridLinearAttnBackend never routes
    any layer to the full backend. But it still calls `init_forward_metadata` (and a
    few cuda-graph hooks) on the full backend each step. Real backends
    (triton/flashinfer) either probe the empty full KV pool at construction or reject
    fp32 planning, so we substitute this no-op instead.
    """

    # HybridLinearAttnBackend (sglang main) copies these off the full backend at
    # construction; there is no full-attn KV pool here, and the req/token pools
    # come from the runner when one is given.
    token_to_kv_pool = None
    req_to_token_pool = None
    needs_cpu_seq_lens = False

    def __init__(self, model_runner: Optional[ModelRunner] = None):
        if model_runner is not None:
            self.req_to_token_pool = model_runner.req_to_token_pool
            self.token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
            self.max_context_len = model_runner.model_config.context_len

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def init_cpu_graph_state(self, *args, **kwargs):
        pass

    def init_forward_metadata_capture_cpu_graph(self, *args, **kwargs):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError("RWKV-7 has no full-attention layers.")

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError("RWKV-7 has no full-attention layers.")


class Rwkv7AttnBackend(MambaAttnBackendBase):
    """Linear-attention backend for RWKV-7.

    Both the decode and the extend/prefill paths run through the same FLA-free
    `wkv_recurrent` triton kernel (exact, ~1e-6 vs the numpy oracle). scale=1.0 to
    match the numpy oracle, which applies no scaling to r before GroupNorm.
    """

    supports_draft_extend_v2 = True

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        if self.topk > 1:
            raise ValueError(
                "RWKV-7 speculative decoding currently supports a linear draft "
                "chain only; set --speculative-eagle-topk 1."
            )
        # Base-class field (tuple[int, int]); RWKV-7's token-shift window is
        # (hidden, conv_kernel-1=1). The base reads [-1] as the conv window length.
        self.conv_states_shape = tuple(
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape[-2:]
        )
        self.scale = 1.0
        self.verify_intermediate_state_indices = torch.arange(
            self.req_to_token_pool.size,
            dtype=torch.int32,
            device=model_runner.device,
        )
        # Set by Rwkv7MultiStepDraftBackend for the topk=1 candidate chain.
        # Normal target/draft-extend backends leave this as None.
        self.draft_step_index: Optional[int] = None
        # Full prefill CUDA graphs use a fixed request-slot count and refresh
        # these buffers in place before replay. Every token bucket captures the
        # same addresses, so one max-bs allocation is sufficient for all graphs.
        self.prefill_query_start_loc: Optional[torch.Tensor] = None
        self.prefill_state_indices: Optional[torch.Tensor] = None

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        self._ensure_prefill_cuda_graph_state(max_bs)

    def _ensure_prefill_cuda_graph_state(self, min_bs: int):
        """Allocate once: prefill capture runs before decode graph setup.

        Decode graph initialization must not replace these tensors after full
        prefill graphs have captured their addresses. Allocate up to the request
        pool capacity on first use and preserve the allocation thereafter.
        """
        if self.prefill_state_indices is not None:
            if min_bs > self.prefill_state_indices.numel():
                raise RuntimeError(
                    "RWKV-7 prefill CUDA-graph request capacity cannot grow after "
                    f"capture: requested {min_bs}, allocated "
                    f"{self.prefill_state_indices.numel()}."
                )
            return
        capacity = max(min_bs, self.req_to_token_pool.size)
        self.prefill_query_start_loc = torch.zeros(
            (capacity + 1,), dtype=torch.int32, device=self.device
        )
        self.prefill_state_indices = torch.full(
            (capacity,), self.pad_slot_id, dtype=torch.int32, device=self.device
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        """Refresh graph-stable RWKV metadata for decode or full prefill.

        The shared Mamba implementation's CUDA-graph metadata planner is
        decode/verify-only and indexes its buffers by token batch size. Full
        prefill graphs instead carry a fixed number of request slots and packed
        token offsets, including zero-length sentinel slots. Handle that case
        explicitly while retaining the shared implementation for decode and
        speculative verify.
        """
        if not (
            forward_batch.forward_mode.is_extend(include_draft_extend_v2=False)
            and not forward_batch.forward_mode.is_target_verify()
        ):
            return super().init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

        bs = forward_batch.batch_size
        if (
            self.prefill_query_start_loc is None
            or self.prefill_state_indices is None
            or bs > self.prefill_state_indices.numel()
        ):
            self._ensure_prefill_cuda_graph_state(bs)

        query_start_loc = self.prefill_query_start_loc[: bs + 1]
        query_start_loc[:bs].copy_(forward_batch.extend_start_loc[:bs])
        query_start_loc[bs].copy_(
            forward_batch.extend_start_loc[bs - 1]
            + forward_batch.extend_seq_lens[bs - 1]
        )

        mamba_indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices[:bs]
        )
        mamba_indices = self._translate_mamba_indices(mamba_indices)
        # Full-prefill capture/replay pads unused request slots with zero-length
        # sentinels. Mark them as invalid so state reads/writes are strict no-ops.
        live = forward_batch.extend_seq_lens[:bs] > 0
        mamba_indices = torch.where(
            live,
            mamba_indices,
            torch.full_like(mamba_indices, self.pad_slot_id),
        )
        state_indices = self.prefill_state_indices[:bs]
        state_indices.copy_(mamba_indices)
        self.forward_metadata = ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=state_indices,
        )

    # ---- token-shift (width-2 causal shift via the conv state) ----
    def token_shift(
        self,
        x: torch.Tensor,
        layer_id: int,
        conv_idx: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Return the previous-token hidden state for each position in x.

        x: [num_tokens, hidden] (post attn_norm / ffn_norm). Updates the stored
        conv state with the last token of each sequence for the next step.
        """
        cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv = cache.conv[conv_idx]  # [size+1, hidden, 1]
        md = self.forward_metadata
        cache_indices = md.mamba_cache_indices

        if (
            self.is_draft_worker
            and self.draft_step_index is not None
            and forward_batch.forward_mode.is_decode_or_idle()
        ):
            if self.topk != 1:
                raise RuntimeError("RWKV-7 draft-state chaining requires topk=1.")
            batch_size = x.shape[0]
            scratch_rows = self.verify_intermediate_state_indices[:batch_size]
            intermediate_conv = cache.intermediate_conv_window[conv_idx]
            if self.draft_step_index == 0:
                safe_idx = torch.clamp_min(cache_indices[:batch_size], 0)
                prev = conv[safe_idx, :, 0]
            else:
                prev = intermediate_conv[scratch_rows, self.draft_step_index - 1, :, 0]
            intermediate_conv[scratch_rows, self.draft_step_index, :, 0] = x.to(
                intermediate_conv.dtype
            )
            return prev.to(x.dtype)

        if forward_batch.forward_mode.is_draft_extend_v2():
            width = forward_batch.spec_info.num_tokens_per_req
            if x.shape[0] % width != 0:
                raise RuntimeError(
                    "RWKV-7 DRAFT_EXTEND_V2 expects a fixed token width per "
                    f"request, got {x.shape[0]} tokens and width {width}."
                )
            batch_size = x.shape[0] // width
            safe_idx = torch.clamp_min(cache_indices[:batch_size], 0)
            x_by_req = x.view(batch_size, width, x.shape[-1])
            shifted = torch.empty_like(x_by_req)
            shifted[:, 0] = conv[safe_idx, :, 0].to(x.dtype)
            if width > 1:
                shifted[:, 1:] = x_by_req[:, :-1]
            accepted_step = (
                forward_batch.spec_info.num_accept_tokens[:batch_size]
                .to(torch.long)
                .sub(1)
                .clamp_(min=0, max=width - 1)
            )
            req_rows = torch.arange(batch_size, device=x.device)
            conv[safe_idx, :, 0] = x_by_req[req_rows, accepted_step].to(conv.dtype)
            return shifted.view_as(x)

        if forward_batch.forward_mode.is_target_verify():
            draft_token_num = forward_batch.spec_info.draft_token_num
            if x.shape[0] % draft_token_num != 0:
                raise RuntimeError(
                    "RWKV-7 TARGET_VERIFY expects an equal linear draft width per "
                    f"request, got {x.shape[0]} tokens and width {draft_token_num}."
                )
            batch_size = x.shape[0] // draft_token_num
            safe_idx = torch.clamp_min(cache_indices[:batch_size], 0)
            x_by_req = x.view(batch_size, draft_token_num, x.shape[-1])
            shifted = torch.empty_like(x_by_req)
            shifted[:, 0] = conv[safe_idx, :, 0].to(x.dtype)
            if draft_token_num > 1:
                shifted[:, 1:] = x_by_req[:, :-1]

            # The persistent token-shift state must not move before acceptance.
            # Save the post-token value for every candidate step into the same
            # request-major scratch protocol used by Mamba/GDN/KDA.
            intermediate_conv = cache.intermediate_conv_window[conv_idx]
            scratch_rows = self.verify_intermediate_state_indices[:batch_size]
            intermediate_conv[scratch_rows, :draft_token_num, :, 0] = x_by_req.to(
                intermediate_conv.dtype
            )
            return shifted.view_as(x)

        if forward_batch.forward_mode.is_decode_or_idle():
            # one token per request: shifted = stored prev; store current.
            # Padded cuda-graph replay fills the tail of mamba_cache_indices with
            # PAD_SLOT_ID = -1; torch advanced indexing would WRAP -1 to row `size`
            # (an allocatable live slot) and corrupt a real request's shift state.
            # Route pads to row 0, the MambaPool's reserved never-allocated slot
            # (free_slots = arange(1, size+1)); pad outputs are discarded anyway.
            safe_idx = torch.clamp_min(cache_indices, 0)
            # advanced indexing already materializes a fresh tensor (no aliasing)
            prev = conv[safe_idx, :, 0]  # [n, hidden]
            conv[safe_idx, :, 0] = x.to(conv.dtype)
            return prev.to(x.dtype)

        # extend (packed B=1, varlen via query_start_loc). The Triton boundary
        # update masks full-CUDA-graph zero-length sentinel slots instead of
        # forming one-past-the-end torch advanced indices.
        return token_shift_packed_varlen(
            x,
            conv,
            md.query_start_loc,
            cache_indices,
        )

    def token_shift_lerp6(
        self,
        x: torch.Tensor,
        mix6: torch.Tensor,
        layer_id: int,
        conv_idx: int,
        forward_batch: ForwardBatch,
        norm_weight: Optional[torch.Tensor] = None,
        norm_bias: Optional[torch.Tensor] = None,
        norm_eps: float = 1e-5,
    ) -> torch.Tensor:
        """Fuse normal serving token shift/state update with six time mixes.

        Speculative modes retain ``token_shift`` because their state protocol
        writes acceptance-window scratch rather than the persistent state pool.
        """
        mode = forward_batch.forward_mode
        has_norm = norm_weight is not None
        if (
            has_norm
            and x.dtype in (torch.float16, torch.bfloat16)
            and not self.is_draft_worker
            and not mode.is_draft_extend_v2()
            and not mode.is_target_verify()
            and mode.is_decode_or_idle()
        ):
            cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
            return layernorm_token_shift_lerp6_decode(
                x,
                cache.conv[conv_idx],
                mix6,
                norm_weight,
                norm_bias,
                norm_eps,
                self.forward_metadata.mamba_cache_indices,
            )
        if has_norm:
            x = F.layer_norm(
                x,
                (x.shape[-1],),
                norm_weight,
                norm_bias,
                norm_eps,
            )
        if self.is_draft_worker or mode.is_draft_extend_v2() or mode.is_target_verify():
            return fused_lerp6(
                x,
                self.token_shift(x, layer_id, conv_idx, forward_batch),
                mix6,
            )

        cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv = cache.conv[conv_idx]
        md = self.forward_metadata
        cache_indices = md.mamba_cache_indices
        if mode.is_decode_or_idle():
            return token_shift_lerp6_decode(x, conv, mix6, cache_indices)
        return token_shift_lerp6_packed_varlen(
            x,
            conv,
            mix6,
            md.query_start_loc,
            cache_indices,
        )

    def token_shift_lerp1(
        self,
        x: torch.Tensor,
        mix: torch.Tensor,
        layer_id: int,
        conv_idx: int,
        forward_batch: ForwardBatch,
        norm_weight: Optional[torch.Tensor] = None,
        norm_bias: Optional[torch.Tensor] = None,
        norm_eps: float = 1e-5,
    ) -> torch.Tensor:
        """Fuse the FFN decode LayerNorm, token shift and one time mix."""
        mode = forward_batch.forward_mode
        has_norm = norm_weight is not None
        if (
            has_norm
            and x.dtype in (torch.float16, torch.bfloat16)
            and not self.is_draft_worker
            and not mode.is_draft_extend_v2()
            and not mode.is_target_verify()
            and mode.is_decode_or_idle()
        ):
            cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
            return layernorm_token_shift_lerp1_decode(
                x,
                cache.conv[conv_idx],
                mix,
                norm_weight,
                norm_bias,
                norm_eps,
                self.forward_metadata.mamba_cache_indices,
            )
        if has_norm:
            x = F.layer_norm(
                x,
                (x.shape[-1],),
                norm_weight,
                norm_bias,
                norm_eps,
            )
        shifted = self.token_shift(x, layer_id, conv_idx, forward_batch)
        return x + mix * (shifted - x)

    # ---- WKV recurrence (decode + extend both -> the wkv_recurrent kernel) ----
    def recurrence(
        self,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        layer_id: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """r,w,k,kk,a: [num_tokens, H, K]; v: [num_tokens, H, V]. Returns [num_tokens, H, V]."""
        cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        temporal = cache.temporal  # [size+1, H, K, V] fp32
        md = self.forward_metadata
        cache_indices = md.mamba_cache_indices

        if (
            self.is_draft_worker
            and self.draft_step_index is not None
            and forward_batch.forward_mode.is_decode_or_idle()
        ):
            if self.topk != 1:
                raise RuntimeError("RWKV-7 draft-state chaining requires topk=1.")
            batch_size = r.shape[0]
            scratch_rows = self.verify_intermediate_state_indices[:batch_size]
            intermediate_state = cache.intermediate_ssm
            if intermediate_state is None:
                raise RuntimeError(
                    "RWKV-7 STANDALONE draft requires speculative state scratch."
                )
            if self.draft_step_index == 0:
                safe_idx = torch.clamp_min(cache_indices[:batch_size], 0)
                initial_state = temporal[safe_idx].contiguous().float()
            else:
                initial_state = intermediate_state[
                    scratch_rows, self.draft_step_index - 1
                ].contiguous()
            o, final_state = wkv_recurrent(
                r.unsqueeze(1).contiguous(),
                w.unsqueeze(1).contiguous(),
                k.unsqueeze(1).contiguous(),
                v.unsqueeze(1).contiguous(),
                kk.unsqueeze(1).contiguous(),
                a.unsqueeze(1).contiguous(),
                scale=self.scale,
                initial_state=initial_state,
                output_final_state=True,
            )
            intermediate_state[scratch_rows, self.draft_step_index] = final_state.to(
                intermediate_state.dtype
            )
            return o.squeeze(1)

        if forward_batch.forward_mode.is_draft_extend_v2():
            width = forward_batch.spec_info.num_tokens_per_req
            if r.shape[0] % width != 0:
                raise RuntimeError(
                    "RWKV-7 DRAFT_EXTEND_V2 expects a fixed token width per "
                    f"request, got {r.shape[0]} tokens and width {width}."
                )
            batch_size = r.shape[0] // width
            intermediate_state = cache.intermediate_ssm
            if intermediate_state is None:
                raise RuntimeError(
                    "RWKV-7 STANDALONE draft extend requires speculative state scratch."
                )
            scratch_rows = self.verify_intermediate_state_indices[:batch_size]
            o, _ = wkv_recurrent(
                r.unsqueeze(0).contiguous(),
                w.unsqueeze(0).contiguous(),
                k.unsqueeze(0).contiguous(),
                v.unsqueeze(0).contiguous(),
                kk.unsqueeze(0).contiguous(),
                a.unsqueeze(0).contiguous(),
                scale=self.scale,
                cu_seqlens=md.query_start_loc.to(torch.int64),
                state_pool=temporal,
                cache_indices=cache_indices[:batch_size],
                update_state_pool=False,
                intermediate_state=intermediate_state,
                intermediate_state_indices=scratch_rows,
            )
            accepted_step = (
                forward_batch.spec_info.num_accept_tokens[:batch_size]
                .to(torch.long)
                .sub(1)
                .clamp_(min=0, max=width - 1)
            )
            safe_idx = torch.clamp_min(cache_indices[:batch_size], 0)
            req_rows = torch.arange(batch_size, device=r.device)
            temporal[safe_idx] = intermediate_state[req_rows, accepted_step].to(
                temporal.dtype
            )
            return o.squeeze(0)

        if forward_batch.forward_mode.is_target_verify():
            draft_token_num = forward_batch.spec_info.draft_token_num
            if r.shape[0] % draft_token_num != 0:
                raise RuntimeError(
                    "RWKV-7 TARGET_VERIFY expects an equal linear draft width per "
                    f"request, got {r.shape[0]} tokens and width {draft_token_num}."
                )
            batch_size = r.shape[0] // draft_token_num
            intermediate_state = cache.intermediate_ssm
            if intermediate_state is None:
                raise RuntimeError(
                    "RWKV-7 TARGET_VERIFY requires a speculative recurrent-state "
                    "scratch buffer."
                )
            o, _ = wkv_recurrent(
                r.unsqueeze(0).contiguous(),
                w.unsqueeze(0).contiguous(),
                k.unsqueeze(0).contiguous(),
                v.unsqueeze(0).contiguous(),
                kk.unsqueeze(0).contiguous(),
                a.unsqueeze(0).contiguous(),
                scale=self.scale,
                cu_seqlens=md.query_start_loc.to(torch.int64),
                state_pool=temporal,
                cache_indices=cache_indices[:batch_size],
                update_state_pool=False,
                intermediate_state=intermediate_state,
                intermediate_state_indices=self.verify_intermediate_state_indices[
                    :batch_size
                ],
            )
            return o.squeeze(0)

        if forward_batch.forward_mode.is_decode_or_idle():
            # [bs, H, *] -> [bs, 1, H, *]
            r4 = r.unsqueeze(1).contiguous()
            w4 = w.unsqueeze(1).contiguous()
            k4 = k.unsqueeze(1).contiguous()
            v4 = v.unsqueeze(1).contiguous()
            kk4 = kk.unsqueeze(1).contiguous()
            a4 = a.unsqueeze(1).contiguous()
            # In-place indexed state: the kernel reads/writes temporal[cache_indices]
            # directly (skips the gather+scatter copies; ~3x less state traffic at large
            # bsz). Same reduction math + bits -> greedy decoding stays token-exact.
            o, _ = wkv_recurrent(
                r4,
                w4,
                k4,
                v4,
                kk4,
                a4,
                scale=self.scale,
                state_pool=temporal,
                cache_indices=cache_indices,
            )
            return o.squeeze(1)  # [bs, H, V]

        # extend: packed B=1, varlen -> in-place indexed recurrent kernel. This
        # removes a full [N,H,K,V] gather, temporary final-state allocation and
        # scatter per layer. Negative cache indices from graph padding are masked
        # inside the kernel and therefore do not touch the state pool.
        cu = md.query_start_loc
        r1 = r.unsqueeze(0).contiguous()
        w1 = w.unsqueeze(0).contiguous()
        k1 = k.unsqueeze(0).contiguous()
        v1 = v.unsqueeze(0).contiguous()
        kk1 = kk.unsqueeze(0).contiguous()
        a1 = a.unsqueeze(0).contiguous()
        o, final_state = wkv_recurrent(
            r1,
            w1,
            k1,
            v1,
            kk1,
            a1,
            scale=self.scale,
            cu_seqlens=cu,
            state_pool=temporal,
            cache_indices=cache_indices,
        )
        assert final_state is None
        return o.squeeze(0)  # [total_T, H, V]

    # The model calls token_shift/recurrence directly; these are not used.
    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "Rwkv7AttnBackend.recurrence/token_shift are called directly by the model."
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "Rwkv7AttnBackend.recurrence/token_shift are called directly by the model."
        )


class Rwkv7MultiStepDraftBackend:
    """Multi-step wrapper for a topk=1 STANDALONE RWKV draft model.

    Full-attention multi-step wrappers plan temporary KV branches. RWKV has no
    KV cache; each child backend instead owns one draft-step id and writes its
    post-step recurrent state into shared speculative scratch.
    """

    needs_cpu_seq_lens = False

    def __init__(self, model_runner: ModelRunner, speculative_num_steps: int):
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for step in range(speculative_num_steps - 1):
            backend = model_runner._get_attention_backend()
            backend.linear_attn_backend.draft_step_index = step
            self.attn_backends.append(backend)
        self.max_context_len = model_runner.model_config.context_len

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for backend in self.attn_backends:
            backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for backend in self.attn_backends:
            backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        for backend in self.attn_backends:
            backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        for backend in self.attn_backends:
            backend.init_forward_metadata_in_graph(forward_batch)

    def on_after_cuda_graph_warmup(self):
        for backend in self.attn_backends:
            backend.on_after_cuda_graph_warmup()
