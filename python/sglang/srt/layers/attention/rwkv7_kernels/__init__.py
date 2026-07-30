# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Self-contained RWKV-7 Triton kernels (no FLA dependency).

wkv_recurrent -> DECODE (T==1 fast path) + recurrent varlen (cu_seqlens) WKV.
"""

from sglang.srt.layers.attention.rwkv7_kernels.token_shift import (
    token_shift_lerp6_decode,
    token_shift_lerp6_packed_varlen,
    token_shift_packed_varlen,
)
from sglang.srt.layers.attention.rwkv7_kernels.wkv_recurrent import wkv_recurrent

__all__ = [
    "token_shift_lerp6_decode",
    "token_shift_lerp6_packed_varlen",
    "token_shift_packed_varlen",
    "wkv_recurrent",
]
