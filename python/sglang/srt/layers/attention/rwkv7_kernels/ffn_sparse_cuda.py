# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Optional zero-skipping CUDA path for RWKV-7 SqReLU FFN down."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_TILE_F = 128
_TILE_C = 256


def sparse_ffn_enabled() -> bool:
    return os.getenv("SGLANG_RWKV7_CUDA_SPARSE_FFN", "1") != "0"


def sparse_ffn_max_rows() -> int:
    return max(1, int(os.getenv("SGLANG_RWKV7_CUDA_SPARSE_FFN_MAX_ROWS", "16")))


@lru_cache(maxsize=1)
def _load_sparse_ffn_extension() -> bool:
    if not sparse_ffn_enabled():
        return False
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    if torch.cuda.get_device_capability()[0] < 7:
        return False
    try:
        from torch.utils.cpp_extension import load

        source_dir = Path(__file__).resolve().parent / "cuda"
        load(
            name="sglang_rwkv7_sparse_ffn_fp16_v2",
            sources=[
                str(source_dir / "ffn_sparse.cpp"),
                str(source_dir / "ffn_sparse.cu"),
            ],
            extra_cflags=["-O3", "-std=c++17", "-DNDEBUG"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "--extra-device-vectorization",
                "-DNDEBUG",
            ],
            is_python_module=False,
            verbose=os.getenv("SGLANG_RWKV7_CUDA_BUILD_VERBOSE", "0") == "1",
        )
        logger.info("Loaded the optional RWKV-7 sparse SqReLU FFN CUDA path.")
        return True
    except Exception:
        logger.warning(
            "Failed to build the optional RWKV-7 sparse SqReLU FFN CUDA path; "
            "falling back to the dense projection.",
            exc_info=True,
        )
        return False


def is_profitable_sparse_ffn_shape(rows: int, hidden_size: int) -> bool:
    """Measured small-M policy; larger batches retain Tensor Core GEMM."""
    if hidden_size <= 2048:
        return rows <= 16
    if hidden_size <= 3072:
        return rows <= 4
    return rows <= 2


def can_use_sparse_sqrelu_down(
    preact: torch.Tensor,
    value_weight_t: torch.Tensor | None,
) -> bool:
    if value_weight_t is None:
        return False
    if (
        not preact.is_cuda
        or preact.dtype != torch.float16
        or value_weight_t.dtype != torch.float16
        or preact.ndim != 2
        or value_weight_t.ndim != 2
        or not preact.is_contiguous()
        or not value_weight_t.is_contiguous()
    ):
        return False
    rows, intermediate = preact.shape
    if (
        rows < 1
        or rows > sparse_ffn_max_rows()
        or not is_profitable_sparse_ffn_shape(rows, value_weight_t.shape[1])
        or value_weight_t.shape[0] != intermediate
        or intermediate % _TILE_F
        or value_weight_t.shape[1] % _TILE_C
    ):
        return False
    return _load_sparse_ffn_extension()


def sparse_sqrelu_down(
    preact: torch.Tensor,
    value_weight_t: torch.Tensor,
) -> torch.Tensor:
    """Compute SqReLU(preact) times value_weight_t, skipping exact zeros."""
    if not _load_sparse_ffn_extension():
        raise RuntimeError("RWKV-7 sparse FFN CUDA extension is unavailable")
    return torch.ops.sglang_rwkv7_sparse_ffn.sqrelu_down_fp16(preact, value_weight_t)
