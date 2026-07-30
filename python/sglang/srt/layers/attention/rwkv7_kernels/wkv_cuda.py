# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Optional JIT CUDA fast path for fp16-state RWKV-7 prefill."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_wkv_varlen_fp16_extension() -> bool:
    if os.getenv("SGLANG_RWKV7_CUDA_FP16_WKV", "1") == "0":
        return False
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    if torch.cuda.get_device_capability()[0] < 8:
        return False
    try:
        from torch.utils.cpp_extension import load

        source_dir = Path(__file__).resolve().parent / "cuda"
        load(
            name="sglang_rwkv7_wkv_varlen_fp16_v3",
            sources=[
                str(source_dir / "wkv_varlen_fp16.cpp"),
                str(source_dir / "wkv_varlen_fp16.cu"),
            ],
            extra_cflags=["-O3", "-std=c++17", "-DNDEBUG"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-DNDEBUG"],
            is_python_module=False,
            verbose=os.getenv("SGLANG_RWKV7_CUDA_BUILD_VERBOSE", "0") == "1",
        )
        logger.info("Loaded the optional RWKV-7 fp16-state CUDA WKV fast path.")
        return True
    except Exception:
        logger.warning(
            "Failed to build the optional RWKV-7 fp16-state CUDA WKV; "
            "falling back to the portable Triton kernel.",
            exc_info=True,
        )
        return False


def can_use_wkv_varlen_fp16_cuda(
    r: torch.Tensor,
    v: torch.Tensor,
    state_pool: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    cache_indices: torch.Tensor | None,
    update_state_pool: bool,
    intermediate_state: torch.Tensor | None,
) -> bool:
    return bool(
        state_pool is not None
        and cu_seqlens is not None
        and cache_indices is not None
        and update_state_pool
        and intermediate_state is None
        and r.is_cuda
        and r.dtype == torch.float16
        and v.dtype == torch.float16
        and state_pool.dtype == torch.float16
        and r.shape[-1] == 64
        and v.shape[-1] == 64
        and _load_wkv_varlen_fp16_extension()
    )


def wkv_varlen_fp16_cuda(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cache_indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    output = torch.empty_like(v)
    torch.ops.sglang_rwkv7_cuda.wkv_varlen_fp16(
        state_pool,
        r,
        w,
        k,
        v,
        kk,
        a,
        output,
        cu_seqlens,
        cache_indices,
        scale,
    )
    return output
