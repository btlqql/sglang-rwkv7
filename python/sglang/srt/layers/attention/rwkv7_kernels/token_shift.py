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
