# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Self-contained RWKV-7 (Goose) WKV recurrence Triton kernel.

One Triton kernel serves both the DECODE (T==1 fast path) and the recurrent
varlen (cu_seqlens) path. It carries zero FLA dependency: it is
authored directly from the published RWKV-7 recurrence, not copied from FLA.

Math (matches the reference RWKV-LM numpy `time_mixing` exactly):

    # per (request n, head h), at each time step t, with
    #   r,w,k,kk,a : [K]   (w is LOG-decay; kk already L2-normalized)
    #   v          : [V]
    # numpy oracle keeps the state as S_np[V, K] and does:
    #   S_np = S_np * exp(w)            # decay along K
    #        - (S_np @ kk) (kk*a)^T     # rank-1 "remove" term
    #        + v k^T                    # rank-1 "write" term
    #   y    = S_np @ (scale * r)       # output [V]

We store the state TRANSPOSED as S[K, V] (== S_np^T), which matches the
temporal cache layout [N, H, K, V] used by the backend. In that layout the
step becomes (k indexes K, v indexes V):

    decay[k] = exp(w[k])
    sa[v]    = sum_k (-kk[k]) * S[k, v]          # = -(S_np @ kk)[v]
    S[k, v]  = decay[k] * S[k, v]
             + (kk[k]*a[k]) * sa[v]              # rank-1 remove
             + k[k] * v[v]                       # rank-1 write
    o[v]     = sum_k S[k, v] * (scale * r[k])    # = (S_np @ (scale*r))[v]

Both reductions contract only the K axis, so tiling the V axis (BV) is exact
and embarrassingly parallel. The default state/accumulation contract is FP32;
an explicitly FP16 state pool selects the lower-precision performance lane.

Conventions match the (now-retired) FLA `fused_mul_recurrent_rwkv7` so the
backend call sites are unchanged:
  * layout [B, T, H, K] for r/w/k/kk/a and [B, T, H, V] for v/o (K==V==head_dim)
  * `w` is log-decay; the kernel applies exp(w)
  * `kk` is L2-normalized by the caller; `a_kernel=-kk`, `b_kernel=kk*a` formed here
  * `scale` is applied to r
  * state [N, H, K, V] in the configured state dtype; returns
    (o[B, T, H, V], final_state[N, H, K, V] or None)
"""

import os

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.rwkv7_kernels.wkv_cuda import (
    can_use_wkv_varlen_fp16_cuda,
    wkv_varlen_fp16_cuda,
)


@triton.jit(do_not_specialize=["T"])
def _wkv_recurrent_kernel(
    r_ptr,
    w_ptr,
    k_ptr,
    v_ptr,
    kk_ptr,
    a_ptr,
    o_ptr,
    h0_ptr,
    ht_ptr,
    hi_ptr,
    cu_ptr,
    ci_ptr,
    ii_ptr,
    n_slots,
    n_intermediate_slots,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STORE_INTERMEDIATE_STATE: tl.constexpr,
    INTERMEDIATE_STEPS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INDEXED_STATE: tl.constexpr,
    FP16_COMPUTE: tl.constexpr,
):
    # program grid: (V-tile, request*head). One program owns the full K axis and
    # a BV-slice of V for a single (request, head); it walks the sequence in time.
    i_nh = tl.program_id(0).to(tl.int64)
    i_v = tl.program_id(1).to(tl.int64)
    i_n = i_nh // H
    i_h = i_nh % H

    # State row in the h0/ht buffer. Default: init/final are packed [N,H,K,V] (row=i_nh).
    # INDEXED_STATE: h0_ptr==ht_ptr is the paged pool [size+1,H,K,V]; read/write the slot
    # cache_indices[i_n] IN PLACE — same state values + reduction order (bit-identical),
    # but skips the temporal[ci] gather+scatter copies the backend used to do.
    if INDEXED_STATE:
        cidx = tl.load(ci_ptr + i_n).to(tl.int64)
        state_nh = cidx * H + i_h
    else:
        state_nh = i_nh

    if IS_VARLEN:
        bos = tl.load(cu_ptr + i_n).to(tl.int64)
        eos = tl.load(cu_ptr + i_n + 1).to(tl.int64)
        seqlen = eos - bos
    else:
        bos = i_n * T
        seqlen = T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_s = mask_k[:, None] & mask_v[None, :]
    # In-place mode: mask off cuda-graph pad/sentinel slots (cache_indices >= pool size)
    # so their state read/write is a no-op (their output row is discarded by the caller);
    # this is what the temporal[ci] gather/scatter did robustly for free.
    if INDEXED_STATE:
        s_mask = mask_s & (cidx >= 0) & (cidx < n_slots)
    else:
        s_mask = mask_s

    # S[K, V] fp32 == numpy oracle state transposed.
    S = tl.zeros([BK, BV], dtype=tl.float16 if FP16_COMPUTE else tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0_ptr + state_nh * K * V + o_k[:, None] * V + o_v[None, :]
        S += tl.load(p_h0, mask=s_mask, other=0.0).to(
            tl.float16 if FP16_COMPUTE else tl.float32
        )

    # token-0 pointers into the packed [., T, H, .] layout (B folded into bos).
    base_k = bos * H * K + i_h * K
    base_v = bos * H * V + i_h * V
    p_r = r_ptr + base_k + o_k
    p_w = w_ptr + base_k + o_k
    p_k = k_ptr + base_k + o_k
    p_a = a_ptr + base_k + o_k
    p_kk = kk_ptr + base_k + o_k
    p_v = v_ptr + base_v + o_v
    p_o = o_ptr + base_v + o_v

    i_t = 0
    for _ in range(0, seqlen):
        compute_dtype = tl.float16 if FP16_COMPUTE else tl.float32
        b_r = tl.load(p_r, mask=mask_k, other=0.0).to(compute_dtype) * scale
        b_w = tl.load(p_w, mask=mask_k, other=0.0).to(compute_dtype)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(compute_dtype)
        b_a = tl.load(p_a, mask=mask_k, other=0.0).to(compute_dtype)
        b_kk = tl.load(p_kk, mask=mask_k, other=0.0).to(compute_dtype)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(compute_dtype)

        # Keep exp itself in fp32, then round to the selected recurrent compute
        # dtype. The fp16-state fast path mirrors Albatross's half2 recurrence;
        # the default fp32-state path retains the original numerical contract.
        decay = tl.exp(b_w.to(tl.float32)).to(compute_dtype)
        b_b = b_kk * b_a  # [K]  b_kernel = kk * a
        # sa[v] = sum_k (-kk[k]) * S[k, v]  (uses PRE-update S)
        sa = tl.sum((-b_kk)[:, None] * S, axis=0).to(compute_dtype)  # [V]
        # state update (RHS fully evaluated before assign -> all-old-S, matches oracle)
        S = (
            decay[:, None] * S
            + b_b[:, None] * sa[None, :]
            + b_k[:, None] * b_v[None, :]
        ).to(compute_dtype)
        # o[v] = sum_k S[k, v] * (scale * r[k])
        b_o = tl.sum(S * b_r[:, None], axis=0).to(compute_dtype)  # [V]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # TARGET_VERIFY must leave the persistent state untouched until the
        # acceptance decision is known.  Snapshot every post-token state into
        # request-major speculative scratch; the generic SGLang commit kernel
        # later selects the last accepted step and scatters it into the pool.
        if STORE_INTERMEDIATE_STATE:
            intermediate_idx = tl.load(ii_ptr + i_n).to(tl.int64)
            p_hi = (
                hi_ptr
                + (((intermediate_idx * INTERMEDIATE_STEPS + i_t) * H + i_h) * K * V)
                + o_k[:, None] * V
                + o_v[None, :]
            )
            hi_mask = (
                mask_s
                & (intermediate_idx >= 0)
                & (intermediate_idx < n_intermediate_slots)
                & (i_t < INTERMEDIATE_STEPS)
            )
            tl.store(p_hi, S.to(p_hi.dtype.element_ty), mask=hi_mask)

        p_r += H * K
        p_w += H * K
        p_k += H * K
        p_a += H * K
        p_kk += H * K
        p_v += H * V
        p_o += H * V
        i_t += 1

    if STORE_FINAL_STATE:
        p_ht = ht_ptr + state_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, S.to(p_ht.dtype.element_ty), mask=s_mask)


def wkv_recurrent(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    state_pool: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    update_state_pool: bool = True,
    intermediate_state: torch.Tensor | None = None,
    intermediate_state_indices: torch.Tensor | None = None,
    _bv: int | None = None,
    _nw: int | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """RWKV-7 WKV recurrence (decode T==1 fast path + varlen recurrent-prefill).

    Drop-in replacement for the FLA `fused_mul_recurrent_rwkv7` decode/recurrent
    path. See module docstring for the math and conventions.

    Args:
        r, w, k, kk, a: `[B, T, H, K]`; `w` is log-decay, `kk` is L2-normalized.
        v: `[B, T, H, V]` (K == V == head_dim).
        scale: scalar applied to `r` (1.0 to match the numpy oracle).
        initial_state: `[N, H, K, V]` state tensor, or None (zero state).
        output_final_state: also return the final `[N, H, K, V]` state.
        cu_seqlens: `[N+1]` token offsets for packed varlen (B must be 1). None
            for the batched decode path (then N == B, T per request).
        reverse: must be False (the decode/recurrent paths never run in reverse).
        update_state_pool: when ``False``, read indexed initial states without
            advancing the persistent pool. Used by speculative target verify.
        intermediate_state: optional ``[S, D, H, K, V]`` scratch receiving each
            post-token state for acceptance-time rollback/commit.
        intermediate_state_indices: ``[N]`` rows in ``intermediate_state``.
    """
    if reverse:
        raise NotImplementedError("wkv_recurrent does not support reverse=True")
    if cu_seqlens is not None and r.shape[0] != 1:
        raise ValueError(
            f"batch size must be 1 with cu_seqlens, got {r.shape[0]}; "
            "flatten varlen inputs before calling."
        )

    B, T, H, K = r.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else (cu_seqlens.numel() - 1)
    if scale is None:
        scale = K**-0.5

    if can_use_wkv_varlen_fp16_cuda(
        r,
        v,
        state_pool,
        cu_seqlens,
        cache_indices,
        update_state_pool,
        intermediate_state,
    ):
        output = wkv_varlen_fp16_cuda(
            r,
            w,
            k,
            v,
            kk,
            a,
            state_pool,
            cu_seqlens.to(torch.int32),
            cache_indices.to(torch.int32),
            scale,
        )
        return output, None

    BK = triton.next_power_of_2(K)
    NV = triton.next_power_of_2(V)
    # Per-path launch config. BV (the V tile) and num_warps fix the thread layout
    # of the axis-0 (K) reductions, hence their float summation ORDER, hence the
    # exact bits of the WKV output (and carried state). Different summation orders
    # are each individually correct but differ by ~1 ULP in fp32, which the bf16
    # output cast amplifies (~1e-2) enough to flip an argmax on a knife-edge
    # (gibberish) continuation. We pin the configs that reproduce, BIT-FOR-BIT in
    # bf16 (output AND fp32 state), the WKV summation order the greedy
    # token-exactness checks were validated against (head_dim<=64):
    #   * decode (T==1):            BV=32, num_warps=4
    #   * varlen recurrent-prefill: BV=16, num_warps=4
    # That bit-stability is what keeps dynamic batching token-exact (
    # batched == B=1) exact on knife-edge continuations under bf16.
    if cu_seqlens is None:
        BV, num_warps = min(32, NV), 4
    elif (
        r.dtype == torch.float16
        and K == 64
        and V == 64
        and os.getenv("SGLANG_RWKV7_FAST_FP16_PREFILL", "0") == "1"
    ):
        # Ada B8 sweep (T=128/512/2048): BV=16, 1 warp is 2.43x-2.85x
        # faster than the strict BV=16, 4-warp layout. The largest observed
        # delta was one fp16 quarter-ULP in the output (2.44e-4) and <=5.97e-8
        # in fp32 state. BF16 keeps the strict token-exact layout because its
        # wider ULP can amplify a reduction-order change on knife-edge logits.
        # Long-sequence greedy generation can amplify the different reduction
        # order, so production defaults to the strict layout. Set
        # SGLANG_RWKV7_FAST_FP16_PREFILL=1 only when the workload has passed
        # the application's logit/token acceptance gate.
        BV, num_warps = min(16, NV), 1
    else:
        BV, num_warps = min(16, NV), 4
    if _bv is not None:
        BV = _bv
    if _nw is not None:
        num_warps = _nw

    o = torch.empty_like(v)
    # In-place indexed-state mode: the WKV kernel reads AND writes the paged pool slots
    # cache_indices[i] directly (state_pool is [size+1, H, K, V] fp32), skipping the
    # temporal[ci] gather + scatter the backend used to do. Same reduction math + bits.
    indexed = state_pool is not None
    fp16_compute = bool(
        indexed
        and state_pool.dtype == torch.float16
        and r.dtype == torch.float16
        and os.getenv("SGLANG_RWKV7_FP16_STATE_COMPUTE", "1") != "0"
    )
    store_intermediate = intermediate_state is not None
    if store_intermediate:
        if not indexed:
            raise ValueError("intermediate_state requires state_pool")
        if intermediate_state_indices is None:
            raise ValueError(
                "intermediate_state_indices are required with intermediate_state"
            )
        if intermediate_state.ndim != 5:
            raise ValueError(
                "intermediate_state must have shape [S, D, H, K, V], got "
                f"{tuple(intermediate_state.shape)}"
            )
        if tuple(intermediate_state.shape[2:]) != (H, K, V):
            raise ValueError(
                "intermediate_state trailing dimensions do not match the WKV state: "
                f"got {tuple(intermediate_state.shape[2:])}, expected {(H, K, V)}"
            )
        if intermediate_state_indices.numel() != N:
            raise ValueError(
                "intermediate_state_indices length must match the number of sequences: "
                f"got {intermediate_state_indices.numel()}, expected {N}"
            )
        if cu_seqlens is None and T > intermediate_state.shape[1]:
            raise ValueError(
                f"sequence length {T} exceeds speculative scratch width "
                f"{intermediate_state.shape[1]}"
            )
    if indexed:
        h0 = state_pool
        ht_out = state_pool
        use_init = True
        store_final = update_state_pool
    else:
        h0 = initial_state
        ht_out = (
            r.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
        )
        use_init = initial_state is not None
        store_final = output_final_state

    # N*H on grid axis 0 (2^31 range); axis 1 (max 65535) would overflow at
    # batch x heads >= 64Ki (e.g. large-batch eager decode on big models).
    grid = (N * H, triton.cdiv(V, BV))
    _wkv_recurrent_kernel[grid](
        r,
        w,
        k,
        v,
        kk,
        a,
        o,
        h0,
        ht_out,
        intermediate_state if store_intermediate else o,
        cu_seqlens,
        cache_indices,
        intermediate_state_indices,
        (state_pool.shape[0] if indexed else 0),
        (intermediate_state.shape[0] if store_intermediate else 0),
        scale,
        T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=use_init,
        STORE_FINAL_STATE=store_final,
        STORE_INTERMEDIATE_STATE=store_intermediate,
        INTERMEDIATE_STEPS=(intermediate_state.shape[1] if store_intermediate else 0),
        IS_VARLEN=cu_seqlens is not None,
        INDEXED_STATE=indexed,
        FP16_COMPUTE=fp16_compute,
        num_warps=num_warps,
    )
    return o, (None if indexed else ht_out)
