"""Shape- and backend-based dispatch for native RWKV-7 quantized linears.

The selectors are pure Python and cached.  Hardware-specific policy belongs
here instead of being spread through Triton launch sites; kernels remain
responsible only for executing the selected plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

W8A8_PREFILL_MIN_ROWS = 512
WEIGHT_STREAMING_MAX_ROWS = 8
ROCM_W4_DEQUANT_MIN_ROWS = 1024


@dataclass(frozen=True)
class RWKV7KernelCapabilities:
    """Backend capabilities that affect portable W8/W4 dispatch."""

    backend: Literal["cuda", "rocm"]
    large_w4_prefill: Literal["packed", "dequant_mm"]


@dataclass(frozen=True)
class RWKV7LinearKernelPlan:
    """A launch plan shared by the Python wrapper and CPU-only tests."""

    kernel: Literal[
        "w8_row",
        "w8_tiled",
        "w8a8_tiled",
        "w4_row",
        "w4_tiled",
        "w4_dequant_mm",
    ]
    block_m: int = 0
    block_n: int = 0
    block_k: int = 0
    num_warps: int = 0
    num_stages: int = 0


@lru_cache(maxsize=2)
def rwkv7_kernel_capabilities(is_hip: bool) -> RWKV7KernelCapabilities:
    if is_hip:
        return RWKV7KernelCapabilities(backend="rocm", large_w4_prefill="dequant_mm")
    return RWKV7KernelCapabilities(backend="cuda", large_w4_prefill="packed")


def _validate_shape(m: int, k: int) -> None:
    if m <= 0 or k <= 0:
        raise ValueError(f"RWKV-7 linear dimensions must be positive; got M={m}, K={k}")


def _decode_block_k(k: int, bits: int) -> int:
    """Match Triton's next-power-of-two reduction tile selection."""
    _validate_shape(1, k)
    limit = 4096 if bits == 8 else 2048
    return min(limit, 1 << (k - 1).bit_length())


@lru_cache(maxsize=128)
def select_rwkv7_w8_kernel(m: int, k: int) -> RWKV7LinearKernelPlan:
    _validate_shape(m, k)
    if m >= W8A8_PREFILL_MIN_ROWS:
        return RWKV7LinearKernelPlan(
            "w8a8_tiled",
            block_m=128,
            block_n=128,
            block_k=64,
            num_warps=8,
            num_stages=3,
        )
    if m <= WEIGHT_STREAMING_MAX_ROWS:
        return RWKV7LinearKernelPlan(
            "w8_row",
            block_m=8,
            block_k=_decode_block_k(k, 8),
            num_warps=1,
        )
    if m <= 32:
        block_m, block_k, num_warps = 16, 64, 4
    elif m <= 128 and k >= 4096:
        block_m, block_k, num_warps = 64, 128, 8
    else:
        block_m, block_k, num_warps = 32, 64, 4
    return RWKV7LinearKernelPlan(
        "w8_tiled",
        block_m=block_m,
        block_n=64,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=2,
    )


@lru_cache(maxsize=256)
def select_rwkv7_w4_kernel(
    m: int, k: int, capabilities: RWKV7KernelCapabilities
) -> RWKV7LinearKernelPlan:
    _validate_shape(m, k)
    if capabilities.large_w4_prefill == "dequant_mm" and m >= ROCM_W4_DEQUANT_MIN_ROWS:
        return RWKV7LinearKernelPlan("w4_dequant_mm")
    if m <= WEIGHT_STREAMING_MAX_ROWS:
        block_k = _decode_block_k(k, 4)
        return RWKV7LinearKernelPlan(
            "w4_row",
            block_m=8,
            block_k=block_k,
            num_warps=2 if block_k == 2048 else 1,
        )
    block_m = 16 if m <= 32 else (64 if m < 128 else 128)
    return RWKV7LinearKernelPlan(
        "w4_tiled",
        block_m=block_m,
        block_n=64,
        block_k=32,
        num_warps=4 if block_m < 128 else 8,
        num_stages=2,
    )
