# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Fused elementwise + low-rank kernels for the RWKV-7 decode/extend hot path.

Profiling found ~20% of the 1.5B bsz1 decode step is
elementwise "glue" spread across ~40 tiny CUDA kernels, plus 8 pathological tiny
LoRA GEMVs (15.3%). These triton kernels collapse that glue into a handful of
launches, lifting decode bandwidth utilization.

BIT-EXACTNESS (the hard gate for kernels A-C). The deployed reference computes
everything in plain torch where every binary op on a low-precision tensor
(bf16/fp16) rounds its result back to that dtype (`at::opmath_type<bf16> == float`,
so compute-in-fp32 → round-to-bf16). Kernels A-C reproduce that EXACT rounding
sequence: each sub-expression is evaluated in fp32 then immediately `.to(DT)`
(round to the storage dtype) before being consumed by the next op. The single trick
that makes this work for both bf16 AND fp32 with one kernel:
``x.to(DT).to(tl.float32)`` where ``DT == float32`` is the identity (no precision
change), while ``DT == bfloat16`` rounds.

CRITICAL: every launch passes ``enable_fp_fusion=False`` (-> ptxas ``--fmad=false``).
Without it triton/LLVM contracts ``x + m*d`` into one FMA and folds away the
intermediate ``.to(DT)`` round, making the result ~1 ULP MORE accurate than torch --
bit-DIFFERENT, which can flip a knife-edge bf16 argmax. With it the kernels are
bit-identical to the torch reference (verified max_abs_diff == 0.0 at fp32/bf16/fp16).
The hd-axis reductions in kernels B-C accumulate in fp32 exactly like torch's
reductions; the final round-to-DT absorbs any reduction-order ULP.

Kernel D additionally folds ATen GroupNorm into kernel C. Its Triton reduction tree
is not bit-identical to ATen GroupNorm, so it is used only by the low-precision
serving lane and is guarded by kernel tolerance plus end-to-end logit/greedy gates.
The fp32 correctness lane retains ATen GroupNorm and kernel C.

All kernels are cuda-graph safe: static shapes, no host syncs, output into
caller-allocated buffers.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# ----------------------------------------------------------------------------
# Kernel A: token-shift lerp (6x). xr/xw/xk/xv/xa/xg = x + x_*·(shifted - x).
# Replaces ~12 tiny torch kernels (1 sub + 6 mul + 6 add) with one launch.
# ----------------------------------------------------------------------------
@triton.jit
def _lerp6_kernel(
    x_ptr,
    sh_ptr,
    mix_ptr,
    out_ptr,
    T,
    H,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    offs = hb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H
    DT = out_ptr.dtype.element_ty

    row = t * H + offs
    x = tl.load(x_ptr + row, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(sh_ptr + row, mask=mask, other=0.0).to(tl.float32)
    # d = shifted - x  (one rounded torch op)
    d = (sh - x).to(DT).to(tl.float32)

    for i in tl.static_range(6):
        m = tl.load(mix_ptr + i * H + offs, mask=mask, other=0.0).to(tl.float32)
        # x_*·d  (rounded)  then  x + (...)  (rounded)
        prod = (m * d).to(DT).to(tl.float32)
        o = (x + prod).to(DT)
        tl.store(out_ptr + i * (T * H) + row, o, mask=mask)


def fused_lerp6(x, shifted, mix6):
    """x,shifted: [T,H]; mix6: [6,H] (stacked xr,xk,xw,xa,xg,xv mix vectors).

    Returns [6,T,H] (same dtype as x): xr,xk,xw,xa,xg,xv in that order (matches
    Rwkv7Attention._mix6_buf and the forward unpack).
    """
    T, H = x.shape
    # flat pointer math below assumes contiguous rows (no-op when already so)
    x, shifted, mix6 = x.contiguous(), shifted.contiguous(), mix6.contiguous()
    out = torch.empty(6, T, H, dtype=x.dtype, device=x.device)
    BLOCK = 1024
    grid = (T, triton.cdiv(H, BLOCK))
    _lerp6_kernel[grid](
        x, shifted, mix6, out, T, H, BLOCK=BLOCK, enable_fp_fusion=False
    )
    return out


# ----------------------------------------------------------------------------
# Kernel B: kk = L2norm(k·k_k) over head_dim, and k <- k + k·(a-1)·k_a.
# One launch over (T, n_head) replaces ~7 tiny torch kernels + a reduction.
# ----------------------------------------------------------------------------
@triton.jit
def _kk_kmix_kernel(
    k_ptr,
    a_ptr,
    kk_param_ptr,
    ka_param_ptr,
    kk_out_ptr,
    knew_out_ptr,
    T,
    H,
    NH,
    BK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    j = tl.arange(0, BK)
    HD = H // NH
    mask = j < HD
    DT = kk_out_ptr.dtype.element_ty

    base = t * H + h * HD + j
    pbase = h * HD + j
    k = tl.load(k_ptr + base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + base, mask=mask, other=0.0).to(tl.float32)
    kkp = tl.load(kk_param_ptr + pbase, mask=mask, other=0.0).to(tl.float32)
    kap = tl.load(ka_param_ptr + pbase, mask=mask, other=0.0).to(tl.float32)

    # kk = k * k_k   (rounded)
    kk = (k * kkp).to(DT)
    kk_f = kk.to(tl.float32)
    # k_new = k + k*(a-1.0)*k_a   (each sub-op rounded, matches torch eval order)
    am = (a - 1.0).to(DT).to(tl.float32)
    t1 = (k * am).to(DT).to(tl.float32)
    t2 = (t1 * kap).to(DT).to(tl.float32)
    knew = (k + t2).to(DT)

    # L2 normalize kk over head_dim. torch: kk.norm(dim=-1) -> DT, clamp_min(1e-12),
    # then kk / norm (rounded). Accumulate sum-of-squares in fp32.
    ss = tl.sum(tl.where(mask, kk_f * kk_f, 0.0), axis=0)
    norm = tl.sqrt(ss)
    norm = norm.to(DT).to(tl.float32)  # torch norm returns storage dtype
    clamp = tl.full((), 1e-12, tl.float32).to(DT).to(tl.float32)
    norm = tl.maximum(norm, clamp)
    kk_n = (kk_f / norm).to(DT)

    tl.store(kk_out_ptr + base, kk_n, mask=mask)
    tl.store(knew_out_ptr + base, knew, mask=mask)


def fused_kk_kmix(k, a, kk_param, ka_param, num_heads):
    """k,a: [T,H]; kk_param,ka_param: [H]; returns (kk_norm [T,nh,hd], k_new [T,H]).

    kk_norm is the L2-normalized (k·k_k); k_new is the a-gated k. Bit-identical to
    the deployed torch sequence.
    """
    k, a = k.contiguous(), a.contiguous()
    kk_param, ka_param = kk_param.contiguous(), ka_param.contiguous()
    T, H = k.shape
    HD = H // num_heads
    BK = triton.next_power_of_2(HD)
    kk_out = torch.empty(T, H, dtype=k.dtype, device=k.device)
    knew_out = torch.empty(T, H, dtype=k.dtype, device=k.device)
    grid = (T, num_heads)
    _kk_kmix_kernel[grid](
        k,
        a,
        kk_param.reshape(-1),
        ka_param.reshape(-1),
        kk_out,
        knew_out,
        T,
        H,
        num_heads,
        BK=BK,
        enable_fp_fusion=False,
    )
    return kk_out.view(T, num_heads, HD), knew_out


# ----------------------------------------------------------------------------
# Kernel C: gate-correction + residual add + gate multiply.
#   o = o_norm + ((r*k*r_k).sum(-1,keepdim) * v);  o = o * g
# One launch over (T, n_head) replaces 3 muls + sum + add + reshape + mul.
# (g_norm/GroupNorm stays a torch op upstream — see model.)
# ----------------------------------------------------------------------------
@triton.jit
def _gate_corr_kernel(
    onorm_ptr,
    r_ptr,
    k_ptr,
    rk_ptr,
    v_ptr,
    g_ptr,
    out_ptr,
    T,
    H,
    NH,
    BK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    j = tl.arange(0, BK)
    HD = H // NH
    mask = j < HD
    DT = out_ptr.dtype.element_ty

    base = t * H + h * HD + j
    pbase = h * HD + j
    r = tl.load(r_ptr + base, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + base, mask=mask, other=0.0).to(tl.float32)
    rk = tl.load(rk_ptr + pbase, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + base, mask=mask, other=0.0).to(tl.float32)
    onorm = tl.load(onorm_ptr + base, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + base, mask=mask, other=0.0).to(tl.float32)

    # (r*k*r_k)  -> each mul rounded
    p1 = (r * k).to(DT).to(tl.float32)
    p2 = (p1 * rk).to(DT).to(tl.float32)
    s = tl.sum(tl.where(mask, p2, 0.0), axis=0)
    s = s.to(DT).to(tl.float32)  # .sum() returns storage dtype
    gc = (s * v).to(DT).to(tl.float32)  # broadcast scalar over head_dim
    o = (onorm + gc).to(DT).to(tl.float32)
    o = (o * g).to(DT)
    tl.store(out_ptr + base, o, mask=mask)


def fused_gate_corr(o_norm, r, k, r_k, v, g, num_heads):
    """o_norm,g: [T,H]; r,k,v: [T,nh,hd] (or [T,H]); r_k: [nh,hd]. Returns [T,H]."""
    T, H = o_norm.shape
    HD = H // num_heads
    BK = triton.next_power_of_2(HD)
    # flat pointer math assumes contiguous rows; .reshape() may return a strided
    # view, and a strided o_norm/g would be silently mis-read —
    # normalize here once (no-op when already contiguous).
    o_norm, g = o_norm.contiguous(), g.contiguous()
    out = torch.empty(T, H, dtype=o_norm.dtype, device=o_norm.device)
    grid = (T, num_heads)
    _gate_corr_kernel[grid](
        o_norm,
        r.reshape(T, H).contiguous(),
        k.reshape(T, H).contiguous(),
        r_k.reshape(-1).contiguous(),
        v.reshape(T, H).contiguous(),
        g,
        out,
        T,
        H,
        num_heads,
        BK=BK,
        enable_fp_fusion=False,
    )
    return out


# ----------------------------------------------------------------------------
# Kernel D: per-head GroupNorm + gate-correction + output gate.
# This removes the materialized GroupNorm output from the low-precision serving
# path. Each RWKV head is exactly one GroupNorm group (normally 64 channels).
# ----------------------------------------------------------------------------
@triton.jit
def _groupnorm_gate_corr_kernel(
    o_ptr,
    r_ptr,
    k_ptr,
    rk_ptr,
    v_ptr,
    g_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    T,
    H,
    NH,
    EPS: tl.constexpr,
    BK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    j = tl.arange(0, BK)
    HD = H // NH
    mask = j < HD
    DT = out_ptr.dtype.element_ty
    base = t * H + h * HD + j
    pbase = h * HD + j

    raw = tl.load(o_ptr + base, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(tl.where(mask, raw, 0.0), axis=0) / HD
    centered = tl.where(mask, raw - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / HD
    weight = tl.load(weight_ptr + pbase, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + pbase, mask=mask, other=0.0).to(tl.float32)
    # PyTorch's CUDA GroupNorm accumulates in fp32 and casts the affine result
    # once to the input dtype.
    o_norm = (centered * tl.rsqrt(variance + EPS) * weight + bias).to(DT)
    o_norm = o_norm.to(tl.float32)

    r = tl.load(r_ptr + base, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + base, mask=mask, other=0.0).to(tl.float32)
    rk = tl.load(rk_ptr + pbase, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + base, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + base, mask=mask, other=0.0).to(tl.float32)
    p1 = (r * k).to(DT).to(tl.float32)
    p2 = (p1 * rk).to(DT).to(tl.float32)
    corr_sum = tl.sum(tl.where(mask, p2, 0.0), axis=0)
    corr_sum = corr_sum.to(DT).to(tl.float32)
    correction = (corr_sum * v).to(DT).to(tl.float32)
    result = (o_norm + correction).to(DT).to(tl.float32)
    result = (result * g).to(DT)
    tl.store(out_ptr + base, result, mask=mask)


def fused_groupnorm_gate_corr(
    o,
    r,
    k,
    r_k,
    v,
    g,
    weight,
    bias,
    num_heads,
    eps,
):
    """Fuse RWKV per-head GroupNorm, recurrent correction, and output gate."""
    T = o.shape[0]
    H = o.numel() // T
    HD = H // num_heads
    BK = triton.next_power_of_2(HD)
    o = o.reshape(T, H).contiguous()
    r = r.reshape(T, H).contiguous()
    k = k.reshape(T, H).contiguous()
    v = v.reshape(T, H).contiguous()
    g = g.reshape(T, H).contiguous()
    out = torch.empty_like(o)
    _groupnorm_gate_corr_kernel[(T, num_heads)](
        o,
        r,
        k,
        r_k.reshape(-1).contiguous(),
        v,
        g,
        weight.reshape(-1).contiguous(),
        bias.reshape(-1).contiguous(),
        out,
        T,
        H,
        num_heads,
        EPS=float(eps),
        BK=BK,
        num_warps=1 if T <= 8 else 2,
        enable_fp_fusion=False,
    )
    return out


# ----------------------------------------------------------------------------
# Kernels E/F: jointly evaluate the W/A/G/V low-rank control branches.
#
# Decode has four independent, poorly shaped low-rank MMs per half of the
# control block.  At small batch sizes their launch cost is comparable to the
# useful work.  The kernels below keep the canonical, separately named module
# weights (so checkpoint loading, TP and weight updates retain their normal
# contract), but dispatch all rank-in work in one launch and all rank-out work
# in one launch.  No padded or transposed weight copy is materialized.
# ----------------------------------------------------------------------------
@triton.jit
def _lowrank4_down_kernel(
    xw_ptr,
    xa_ptr,
    xg_ptr,
    xv_ptr,
    ww_ptr,
    wa_ptr,
    wg_ptr,
    wv_ptr,
    out_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    RW: tl.constexpr,
    RA: tl.constexpr,
    RG: tl.constexpr,
    RV: tl.constexpr,
    HAS_V: tl.constexpr,
    BK: tl.constexpr,
):
    group = tl.program_id(0)
    rank = tl.program_id(1)
    k = tl.arange(0, BK)
    mask = k < H
    total_rank = RW + RA + RG + RV

    if group == 0:
        if rank < RW:
            weight = tl.load(ww_ptr + rank * H + k, mask=mask, other=0.0).to(tl.float32)
            for row in tl.static_range(M):
                value = tl.load(xw_ptr + row * H + k, mask=mask, other=0.0).to(
                    tl.float32
                )
                tl.store(
                    out_ptr + row * total_rank + rank,
                    tl.sum(value * weight, axis=0),
                )
    elif group == 1:
        if rank < RA:
            weight = tl.load(wa_ptr + rank * H + k, mask=mask, other=0.0).to(tl.float32)
            for row in tl.static_range(M):
                value = tl.load(xa_ptr + row * H + k, mask=mask, other=0.0).to(
                    tl.float32
                )
                tl.store(
                    out_ptr + row * total_rank + RW + rank,
                    tl.sum(value * weight, axis=0),
                )
    elif group == 2:
        if rank < RG:
            weight = tl.load(wg_ptr + rank * H + k, mask=mask, other=0.0).to(tl.float32)
            for row in tl.static_range(M):
                value = tl.load(xg_ptr + row * H + k, mask=mask, other=0.0).to(
                    tl.float32
                )
                tl.store(
                    out_ptr + row * total_rank + RW + RA + rank,
                    tl.sum(value * weight, axis=0),
                )
    elif HAS_V:
        if rank < RV:
            weight = tl.load(wv_ptr + rank * H + k, mask=mask, other=0.0).to(tl.float32)
            for row in tl.static_range(M):
                value = tl.load(xv_ptr + row * H + k, mask=mask, other=0.0).to(
                    tl.float32
                )
                tl.store(
                    out_ptr + row * total_rank + RW + RA + RG + rank,
                    tl.sum(value * weight, axis=0),
                )


@triton.jit
def _lowrank4_up_kernel(
    rank_ptr,
    ww_ptr,
    wa_ptr,
    wg_ptr,
    wv_ptr,
    bw_ptr,
    ba_ptr,
    bv_ptr,
    value_ptr,
    v_first_ptr,
    out_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    RW: tl.constexpr,
    RA: tl.constexpr,
    RG: tl.constexpr,
    RV: tl.constexpr,
    HAS_V: tl.constexpr,
    BR: tl.constexpr,
):
    group = tl.program_id(0)
    channel = tl.program_id(1)
    rank = tl.arange(0, BR)
    total_rank = RW + RA + RG + RV
    DT = out_ptr.dtype.element_ty

    if group == 0:
        mask = rank < RW
        weight = tl.load(ww_ptr + channel * RW + rank, mask=mask, other=0.0).to(
            tl.float32
        )
        for row in tl.static_range(M):
            control = tl.load(
                rank_ptr + row * total_rank + rank, mask=mask, other=0.0
            ).to(tl.float32)
            # The rank-in linear returns the storage dtype before tanh.
            control = control.to(DT).to(tl.float32)
            control = libdevice.tanh(control).to(DT).to(tl.float32)
            raw = tl.sum(control * weight, axis=0)
            raw += tl.load(bw_ptr + channel).to(tl.float32)
            # Rank-out+bias rounds before the outer sigmoid and scale.
            raw = raw.to(DT).to(tl.float32)
            gate = (1.0 / (1.0 + tl.exp(-raw))).to(DT).to(tl.float32)
            gate = (-gate).to(DT).to(tl.float32)
            result = (gate * 0.6065306597126334).to(DT)
            tl.store(out_ptr + row * H + channel, result)
    elif group == 1:
        mask = rank < RA
        weight = tl.load(wa_ptr + channel * RA + rank, mask=mask, other=0.0).to(
            tl.float32
        )
        for row in tl.static_range(M):
            control = tl.load(
                rank_ptr + row * total_rank + RW + rank,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            control = control.to(DT).to(tl.float32)
            raw = tl.sum(control * weight, axis=0)
            raw += tl.load(ba_ptr + channel).to(tl.float32)
            raw = raw.to(DT).to(tl.float32)
            result = (1.0 / (1.0 + tl.exp(-raw))).to(DT)
            tl.store(out_ptr + (M + row) * H + channel, result)
    elif group == 2:
        mask = rank < RG
        weight = tl.load(wg_ptr + channel * RG + rank, mask=mask, other=0.0).to(
            tl.float32
        )
        for row in tl.static_range(M):
            control = tl.load(
                rank_ptr + row * total_rank + RW + RA + rank,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            control = control.to(DT).to(tl.float32)
            control = (1.0 / (1.0 + tl.exp(-control))).to(DT).to(tl.float32)
            result = tl.sum(control * weight, axis=0).to(DT)
            tl.store(out_ptr + (2 * M + row) * H + channel, result)
    elif HAS_V:
        mask = rank < RV
        weight = tl.load(wv_ptr + channel * RV + rank, mask=mask, other=0.0).to(
            tl.float32
        )
        for row in tl.static_range(M):
            control = tl.load(
                rank_ptr + row * total_rank + RW + RA + RG + rank,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            control = control.to(DT).to(tl.float32)
            raw = tl.sum(control * weight, axis=0)
            raw += tl.load(bv_ptr + channel).to(tl.float32)
            raw = raw.to(DT).to(tl.float32)
            gate = (1.0 / (1.0 + tl.exp(-raw))).to(DT).to(tl.float32)
            value = tl.load(value_ptr + row * H + channel).to(tl.float32)
            v_first = tl.load(v_first_ptr + row * H + channel).to(tl.float32)
            delta = (v_first - value).to(DT).to(tl.float32)
            update = (delta * gate).to(DT).to(tl.float32)
            result = (value + update).to(DT)
            tl.store(out_ptr + (3 * M + row) * H + channel, result)


def _plain_linear_weight(module):
    weight = getattr(module, "weight", None)
    if (
        weight is None
        or weight.ndim != 2
        or not weight.is_contiguous()
        or weight.dtype not in (torch.float16, torch.bfloat16)
    ):
        return None
    return weight


def can_fuse_lowrank_controls(w_lora, a_lora, g_lora, v_lora=None) -> bool:
    """Return whether four canonical dense RWKV low-rank modules are fusible."""
    modules = [w_lora, a_lora, g_lora]
    if v_lora is not None:
        modules.append(v_lora)
    for module in modules:
        if len(module.lora) != 3:
            return False
        down = _plain_linear_weight(module.lora[0])
        up = _plain_linear_weight(module.lora[2])
        if (
            down is None
            or up is None
            or down.shape[0] != up.shape[1]
            or down.device != up.device
            or down.dtype != up.dtype
        ):
            return False
    # W and A require rank-out biases. V does too when that branch exists;
    # G is intentionally bias-free in the canonical RWKV-7 checkpoint.
    if w_lora.lora[2].bias is None or a_lora.lora[2].bias is None:
        return False
    if v_lora is not None and v_lora.lora[2].bias is None:
        return False
    return True


def is_profitable_fused_lowrank_shape(hidden_size: int, num_tokens: int) -> bool:
    """Conservative measured dispatch policy for the grouped decode kernel.

    The no-copy kernel wins for all decode batches through hidden size 2560.
    At hidden size 4096 it still wins for B=1/2/4, while B=8 has enough GEMM
    work for the canonical modules to be faster on Ada. Unknown larger shapes
    stay on the portable module path until they have hardware evidence.
    """
    return num_tokens in (1, 2, 4, 8) and (
        hidden_size <= 2560 or (hidden_size <= 4096 and num_tokens <= 4)
    )


def fused_lowrank_controls(
    xw,
    xa,
    xg,
    xv,
    value,
    v_first,
    w_lora,
    a_lora,
    g_lora,
    v_lora=None,
):
    """Evaluate W/A/G[/V] controls in two launches for decode batches <= 8.

    Returns already activated ``w_log``, ``a``, ``g`` and (when ``v_lora`` is
    present) the value-residual-mixed ``v``.  The caller owns dispatch and keeps
    the ordinary module path as the universal fallback.
    """
    inputs = (xw, xa, xg, xv)
    if any(not tensor.is_contiguous() for tensor in inputs):
        inputs = tuple(tensor.contiguous() for tensor in inputs)
    M, hidden = xw.shape
    modules = (w_lora, a_lora, g_lora, v_lora)
    has_v = v_lora is not None
    downs = [module.lora[0].weight for module in modules if module is not None]
    ups = [module.lora[2].weight for module in modules if module is not None]
    ranks = [weight.shape[0] for weight in downs]
    if not has_v:
        ranks.append(0)
        # Triton still requires valid pointers for compile-time-dead operands.
        downs.append(downs[0])
        ups.append(ups[0])
    rw, ra, rg, rv = ranks
    rank_values = torch.empty((M, rw + ra + rg + rv), dtype=xw.dtype, device=xw.device)
    _lowrank4_down_kernel[(4, max(ranks))](
        *inputs,
        *downs,
        rank_values,
        M=M,
        H=hidden,
        RW=rw,
        RA=ra,
        RG=rg,
        RV=rv,
        HAS_V=has_v,
        BK=triton.next_power_of_2(hidden),
        num_warps=4 if hidden <= 2560 else 8,
        num_stages=1,
    )

    bw = w_lora.lora[2].bias
    ba = a_lora.lora[2].bias
    bv = bw if not has_v else v_lora.lora[2].bias
    if bw is None or ba is None or (has_v and bv is None):
        raise ValueError("RWKV-7 fused low-rank controls require W/A/V biases")
    if not has_v:
        v_first = value
    outputs = torch.empty(
        (4 if has_v else 3, M, hidden), dtype=xw.dtype, device=xw.device
    )
    _lowrank4_up_kernel[(4, hidden)](
        rank_values,
        *ups,
        bw,
        ba,
        bv,
        value,
        v_first,
        outputs,
        M=M,
        H=hidden,
        RW=rw,
        RA=ra,
        RG=rg,
        RV=rv,
        HAS_V=has_v,
        BR=triton.next_power_of_2(max(ranks)),
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    if has_v:
        return outputs[0], outputs[1], outputs[2], outputs[3]
    return outputs[0], outputs[1], outputs[2], value
