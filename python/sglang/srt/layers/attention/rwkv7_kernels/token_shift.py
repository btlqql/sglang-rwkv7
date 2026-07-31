# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Graph-safe RWKV-7 packed-varlen token shift.

The regular extend path has one recurrent request slot per pair of entries in
``query_start_loc``. Full prefill CUDA graphs keep that slot count static and
pad unused slots with zero-length sequences. Torch advanced indexing cannot
safely write those sentinels: their start/end offset is the flat token count,
which is one past the last valid row. This kernel masks zero-length and invalid
cache slots while updating the real sequence boundaries in place.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_token_shift_lerp6_decode_kernel(
    x_ptr,
    conv_ptr,
    mix_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    cache_indices_ptr,
    hidden_size: tl.constexpr,
    conv_slots,
    conv_stride_slot: tl.constexpr,
    conv_stride_h: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fuse decode LayerNorm, state shift/update and all six time mixes."""
    request_index = tl.program_id(0)
    hidden_offset = tl.arange(0, BLOCK_H)
    mask = hidden_offset < hidden_size
    row = request_index * hidden_size
    raw = tl.load(x_ptr + row + hidden_offset, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(tl.where(mask, raw, 0.0), axis=0) / hidden_size
    centered = tl.where(mask, raw - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / hidden_size
    weight = tl.load(weight_ptr + hidden_offset, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + hidden_offset, mask=mask, other=0.0).to(tl.float32)
    else:
        bias = 0.0
    DT = out_ptr.dtype.element_ty
    normalized = (centered * tl.rsqrt(variance + EPS) * weight + bias).to(DT)
    normalized_fp32 = normalized.to(tl.float32)

    raw_cache_index = tl.load(cache_indices_ptr + request_index)
    valid_cache = (raw_cache_index >= 0) & (raw_cache_index < conv_slots)
    cache_index = tl.minimum(tl.maximum(raw_cache_index, 0), conv_slots - 1).to(
        tl.int64
    )
    conv_offset = cache_index * conv_stride_slot + hidden_offset * conv_stride_h
    shifted = (
        tl.load(conv_ptr + conv_offset, mask=mask, other=0.0).to(DT).to(tl.float32)
    )
    delta = (shifted - normalized_fp32).to(DT).to(tl.float32)
    token_count = tl.num_programs(0)
    for i in tl.static_range(6):
        mix = tl.load(
            mix_ptr + i * hidden_size + hidden_offset,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        product = (mix * delta).to(DT).to(tl.float32)
        value = (normalized_fp32 + product).to(DT)
        tl.store(
            out_ptr + i * token_count * hidden_size + row + hidden_offset,
            value,
            mask=mask,
        )
    tl.store(
        conv_ptr + conv_offset,
        normalized,
        mask=mask & valid_cache,
    )


@triton.jit
def _layernorm_token_shift_lerp1_decode_kernel(
    x_ptr,
    conv_ptr,
    mix_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    cache_indices_ptr,
    hidden_size: tl.constexpr,
    conv_slots,
    conv_stride_slot: tl.constexpr,
    conv_stride_h: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fuse decode LayerNorm, FFN token shift/update and its single mix."""
    request_index = tl.program_id(0)
    hidden_offset = tl.arange(0, BLOCK_H)
    mask = hidden_offset < hidden_size
    row = request_index * hidden_size
    raw = tl.load(x_ptr + row + hidden_offset, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(tl.where(mask, raw, 0.0), axis=0) / hidden_size
    centered = tl.where(mask, raw - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / hidden_size
    weight = tl.load(weight_ptr + hidden_offset, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + hidden_offset, mask=mask, other=0.0).to(tl.float32)
    else:
        bias = 0.0
    DT = out_ptr.dtype.element_ty
    normalized = (centered * tl.rsqrt(variance + EPS) * weight + bias).to(DT)
    normalized_fp32 = normalized.to(tl.float32)

    raw_cache_index = tl.load(cache_indices_ptr + request_index)
    valid_cache = (raw_cache_index >= 0) & (raw_cache_index < conv_slots)
    cache_index = tl.minimum(tl.maximum(raw_cache_index, 0), conv_slots - 1).to(
        tl.int64
    )
    conv_offset = cache_index * conv_stride_slot + hidden_offset * conv_stride_h
    shifted = (
        tl.load(conv_ptr + conv_offset, mask=mask, other=0.0).to(DT).to(tl.float32)
    )
    delta = (shifted - normalized_fp32).to(DT).to(tl.float32)
    mix = tl.load(mix_ptr + hidden_offset, mask=mask, other=0.0).to(tl.float32)
    product = (mix * delta).to(DT).to(tl.float32)
    value = (normalized_fp32 + product).to(DT)
    tl.store(out_ptr + row + hidden_offset, value, mask=mask)
    tl.store(
        conv_ptr + conv_offset,
        normalized,
        mask=mask & valid_cache,
    )


def _validate_layernorm_decode_inputs(x, conv, mix, weight, bias, cache_indices):
    if x.ndim != 2 or x.shape[0] != cache_indices.numel():
        raise ValueError(
            "decode x must be [batch, hidden] with one cache index per row; "
            f"got x={tuple(x.shape)}, indices={cache_indices.numel()}"
        )
    if conv.ndim != 3 or conv.shape[-1] != 1 or conv.shape[1] != x.shape[1]:
        raise ValueError(
            f"conv must have shape [slots, {x.shape[1]}, 1], got {tuple(conv.shape)}"
        )
    if weight.shape != (x.shape[1],):
        raise ValueError(f"LayerNorm weight must have shape ({x.shape[1]},)")
    if bias is not None and bias.shape != weight.shape:
        raise ValueError("LayerNorm bias and weight must have identical shapes")
    if mix.shape[-1] != x.shape[1]:
        raise ValueError(f"time-mix hidden size must be {x.shape[1]}")


def layernorm_token_shift_lerp6_decode(
    x: torch.Tensor,
    conv: torch.Tensor,
    mix6: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """Decode-only fused LayerNorm + six-way time mix and state update."""
    _validate_layernorm_decode_inputs(x, conv, mix6, weight, bias, cache_indices)
    if mix6.shape != (6, x.shape[1]):
        raise ValueError(f"mix6 must have shape (6, {x.shape[1]})")
    x, mix6, weight = x.contiguous(), mix6.contiguous(), weight.contiguous()
    bias_ptr = weight if bias is None else bias.contiguous()
    out = torch.empty(6, *x.shape, dtype=x.dtype, device=x.device)
    block_h = triton.next_power_of_2(x.shape[1])
    _layernorm_token_shift_lerp6_decode_kernel[(x.shape[0],)](
        x,
        conv,
        mix6,
        weight,
        bias_ptr,
        out,
        cache_indices,
        x.shape[1],
        conv.shape[0],
        conv.stride(0),
        conv.stride(1),
        EPS=float(eps),
        HAS_BIAS=bias is not None,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 2048 else 4,
        enable_fp_fusion=False,
    )
    return out


def layernorm_token_shift_lerp1_decode(
    x: torch.Tensor,
    conv: torch.Tensor,
    mix: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """Decode-only fused LayerNorm + FFN time mix and state update."""
    mix = mix.reshape(-1)
    _validate_layernorm_decode_inputs(x, conv, mix, weight, bias, cache_indices)
    x, mix, weight = x.contiguous(), mix.contiguous(), weight.contiguous()
    bias_ptr = weight if bias is None else bias.contiguous()
    out = torch.empty_like(x)
    block_h = triton.next_power_of_2(x.shape[1])
    _layernorm_token_shift_lerp1_decode_kernel[(x.shape[0],)](
        x,
        conv,
        mix,
        weight,
        bias_ptr,
        out,
        cache_indices,
        x.shape[1],
        conv.shape[0],
        conv.stride(0),
        conv.stride(1),
        EPS=float(eps),
        HAS_BIAS=bias is not None,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 2048 else 4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _store_lerp6(
    x,
    shifted,
    mix_ptr,
    out_ptr,
    row,
    hidden_offset,
    hidden_size: tl.constexpr,
    token_count,
    mask,
):
    """Store six torch-order rounded lerps for one token/hidden tile."""
    DT = out_ptr.dtype.element_ty
    delta = (shifted - x).to(DT).to(tl.float32)
    for i in tl.static_range(6):
        mix = tl.load(
            mix_ptr + i * hidden_size + hidden_offset,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        product = (mix * delta).to(DT).to(tl.float32)
        value = (x + product).to(DT)
        tl.store(
            out_ptr + i * token_count * hidden_size + row + hidden_offset,
            value,
            mask=mask,
        )


@triton.jit
def _token_shift_lerp6_adjacent_kernel(
    x_ptr,
    mix_ptr,
    out_ptr,
    token_count,
    hidden_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fast interior path; request starts are overwritten by the boundary kernel."""
    token = tl.program_id(0) + 1
    hidden_offset = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = hidden_offset < hidden_size
    row = token * hidden_size
    x = tl.load(x_ptr + row + hidden_offset, mask=mask, other=0.0).to(tl.float32)
    shifted = tl.load(
        x_ptr + row - hidden_size + hidden_offset, mask=mask, other=0.0
    ).to(tl.float32)
    _store_lerp6(
        x,
        shifted,
        mix_ptr,
        out_ptr,
        row,
        hidden_offset,
        hidden_size,
        token_count,
        mask,
    )


@triton.jit
def _token_shift_lerp6_boundaries_kernel(
    x_ptr,
    conv_ptr,
    mix_ptr,
    out_ptr,
    query_start_loc_ptr,
    cache_indices_ptr,
    token_count,
    hidden_size: tl.constexpr,
    conv_slots,
    conv_stride_slot: tl.constexpr,
    conv_stride_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Patch request starts with cached state and persist each request's last row."""
    request_index = tl.program_id(0)
    hidden_offset = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)

    start = tl.load(query_start_loc_ptr + request_index).to(tl.int64)
    end = tl.load(query_start_loc_ptr + request_index + 1).to(tl.int64)
    cache_index = tl.load(cache_indices_ptr + request_index).to(tl.int64)
    valid_request = (
        (end > start)
        & (start >= 0)
        & (end <= token_count)
        & (cache_index >= 0)
        & (cache_index < conv_slots)
    )
    mask = valid_request & (hidden_offset < hidden_size)
    row = start * hidden_size
    conv_offset = cache_index * conv_stride_slot + hidden_offset * conv_stride_h
    x = tl.load(x_ptr + row + hidden_offset, mask=mask, other=0.0).to(tl.float32)
    # token_shift returns the fp32 cache converted to x.dtype before lerp.
    DT = out_ptr.dtype.element_ty
    shifted = (
        tl.load(conv_ptr + conv_offset, mask=mask, other=0.0).to(DT).to(tl.float32)
    )
    _store_lerp6(
        x,
        shifted,
        mix_ptr,
        out_ptr,
        row,
        hidden_offset,
        hidden_size,
        token_count,
        mask,
    )

    last = tl.load(
        x_ptr + (end - 1) * hidden_size + hidden_offset,
        mask=mask,
        other=0.0,
    )
    tl.store(conv_ptr + conv_offset, last, mask=mask)


@triton.jit
def _token_shift_lerp6_decode_kernel(
    x_ptr,
    conv_ptr,
    mix_ptr,
    out_ptr,
    cache_indices_ptr,
    token_count,
    hidden_size: tl.constexpr,
    conv_slots,
    conv_stride_slot: tl.constexpr,
    conv_stride_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    request_index = tl.program_id(0)
    hidden_offset = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = hidden_offset < hidden_size
    raw_cache_index = tl.load(cache_indices_ptr + request_index)
    valid_cache = (raw_cache_index >= 0) & (raw_cache_index < conv_slots)
    cache_index = tl.maximum(raw_cache_index, 0).to(tl.int64)
    row = request_index * hidden_size
    conv_offset = cache_index * conv_stride_slot + hidden_offset * conv_stride_h
    x = tl.load(x_ptr + row + hidden_offset, mask=mask, other=0.0).to(tl.float32)
    # Match conv[index].to(x.dtype) in the unfused decode path.
    DT = out_ptr.dtype.element_ty
    shifted = (
        tl.load(conv_ptr + conv_offset, mask=mask, other=0.0).to(DT).to(tl.float32)
    )
    _store_lerp6(
        x,
        shifted,
        mix_ptr,
        out_ptr,
        row,
        hidden_offset,
        hidden_size,
        token_count,
        mask,
    )
    tl.store(conv_ptr + conv_offset, x, mask=mask & valid_cache)


@triton.jit
def _token_shift_boundaries_kernel(
    x_ptr,
    shifted_ptr,
    conv_ptr,
    query_start_loc_ptr,
    cache_indices_ptr,
    token_count,
    hidden_size: tl.constexpr,
    conv_slots,
    x_stride_t: tl.constexpr,
    x_stride_h: tl.constexpr,
    shifted_stride_t: tl.constexpr,
    shifted_stride_h: tl.constexpr,
    conv_stride_slot: tl.constexpr,
    conv_stride_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    request_index = tl.program_id(0)
    hidden_offset = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)

    start = tl.load(query_start_loc_ptr + request_index).to(tl.int64)
    end = tl.load(query_start_loc_ptr + request_index + 1).to(tl.int64)
    cache_index = tl.load(cache_indices_ptr + request_index).to(tl.int64)
    valid_request = (
        (end > start)
        & (start >= 0)
        & (end <= token_count)
        & (cache_index >= 0)
        & (cache_index < conv_slots)
    )
    mask = valid_request & (hidden_offset < hidden_size)

    conv_offset = cache_index * conv_stride_slot + hidden_offset * conv_stride_h
    previous = tl.load(conv_ptr + conv_offset, mask=mask, other=0.0)
    tl.store(
        shifted_ptr + start * shifted_stride_t + hidden_offset * shifted_stride_h,
        previous,
        mask=mask,
    )

    last = tl.load(
        x_ptr + (end - 1) * x_stride_t + hidden_offset * x_stride_h,
        mask=mask,
        other=0.0,
    )
    tl.store(conv_ptr + conv_offset, last, mask=mask)


def token_shift_packed_varlen(
    x: torch.Tensor,
    conv: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """Shift packed sequences and persist each live sequence's final token.

    Args:
        x: Flat packed hidden states, ``[T, D]``.
        conv: Token-shift state pool, ``[S, D, 1]``.
        query_start_loc: Static-address packed offsets, ``[N + 1]``.
        cache_indices: Physical state-pool rows, ``[N]``; negative rows are pads.
    """
    if x.ndim != 2:
        raise ValueError(f"x must have shape [T, D], got {tuple(x.shape)}")
    if conv.ndim != 3 or conv.shape[-1] != 1:
        raise ValueError(f"conv must have shape [S, D, 1], got {tuple(conv.shape)}")
    if query_start_loc.numel() != cache_indices.numel() + 1:
        raise ValueError(
            "query_start_loc must have exactly one more entry than cache_indices"
        )
    if x.shape[1] != conv.shape[1]:
        raise ValueError(f"hidden-size mismatch: x={x.shape[1]}, conv={conv.shape[1]}")

    shifted = torch.empty_like(x)
    if x.shape[0] > 1:
        shifted[1:].copy_(x[:-1])

    block_h = 256
    grid = (cache_indices.numel(), triton.cdiv(x.shape[1], block_h))
    _token_shift_boundaries_kernel[grid](
        x,
        shifted,
        conv,
        query_start_loc,
        cache_indices,
        x.shape[0],
        x.shape[1],
        conv.shape[0],
        x.stride(0),
        x.stride(1),
        shifted.stride(0),
        shifted.stride(1),
        conv.stride(0),
        conv.stride(1),
        BLOCK_H=block_h,
        num_warps=4,
    )
    return shifted


def token_shift_lerp6_packed_varlen(
    x: torch.Tensor,
    conv: torch.Tensor,
    mix6: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """Fuse packed token shift, six time mixes, and boundary-state updates.

    Interior tokens read the preceding normalized row directly. A second small
    launch replaces every request boundary with its cached previous row and
    writes the request's final normalized row back to the state pool. This
    removes the full-size ``shifted`` tensor and its extra global-memory pass.
    """
    if x.ndim != 2:
        raise ValueError(f"x must have shape [T, D], got {tuple(x.shape)}")
    if conv.ndim != 3 or conv.shape[-1] != 1:
        raise ValueError(f"conv must have shape [S, D, 1], got {tuple(conv.shape)}")
    if mix6.shape != (6, x.shape[1]):
        raise ValueError(
            f"mix6 must have shape [6, {x.shape[1]}], got {tuple(mix6.shape)}"
        )
    if query_start_loc.numel() != cache_indices.numel() + 1:
        raise ValueError(
            "query_start_loc must have exactly one more entry than cache_indices"
        )
    if x.shape[1] != conv.shape[1]:
        raise ValueError(f"hidden-size mismatch: x={x.shape[1]}, conv={conv.shape[1]}")

    x, mix6 = x.contiguous(), mix6.contiguous()
    out = torch.empty(6, *x.shape, dtype=x.dtype, device=x.device)
    # Six output streams make this bandwidth-bound. A one-warp 128-channel
    # tile gave the best occupancy across the measured packed prefill buckets.
    block_h = 128
    if x.shape[0] > 1:
        _token_shift_lerp6_adjacent_kernel[
            (x.shape[0] - 1, triton.cdiv(x.shape[1], block_h))
        ](
            x,
            mix6,
            out,
            x.shape[0],
            x.shape[1],
            BLOCK_H=block_h,
            num_warps=1,
            enable_fp_fusion=False,
        )
    _token_shift_lerp6_boundaries_kernel[
        (cache_indices.numel(), triton.cdiv(x.shape[1], block_h))
    ](
        x,
        conv,
        mix6,
        out,
        query_start_loc,
        cache_indices,
        x.shape[0],
        x.shape[1],
        conv.shape[0],
        conv.stride(0),
        conv.stride(1),
        BLOCK_H=block_h,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out


def token_shift_lerp6_decode(
    x: torch.Tensor,
    conv: torch.Tensor,
    mix6: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """Fuse the one-token-per-request shift/state update with six time mixes."""
    if x.ndim != 2 or x.shape[0] != cache_indices.numel():
        raise ValueError(
            "decode x must be [batch, hidden] with one cache index per row; "
            f"got x={tuple(x.shape)}, indices={cache_indices.numel()}"
        )
    if conv.ndim != 3 or conv.shape[-1] != 1 or conv.shape[1] != x.shape[1]:
        raise ValueError(
            f"conv must have shape [slots, {x.shape[1]}, 1], got {tuple(conv.shape)}"
        )
    if mix6.shape != (6, x.shape[1]):
        raise ValueError(
            f"mix6 must have shape [6, {x.shape[1]}], got {tuple(mix6.shape)}"
        )
    x, mix6 = x.contiguous(), mix6.contiguous()
    out = torch.empty(6, *x.shape, dtype=x.dtype, device=x.device)
    block_h = 256
    _token_shift_lerp6_decode_kernel[(x.shape[0], triton.cdiv(x.shape[1], block_h))](
        x,
        conv,
        mix6,
        out,
        cache_indices,
        x.shape[0],
        x.shape[1],
        conv.shape[0],
        conv.stride(0),
        conv.stride(1),
        BLOCK_H=block_h,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
