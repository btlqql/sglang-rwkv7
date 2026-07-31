"""Deterministic small-M projection for RWKV-7's online W4 Marlin path."""

from __future__ import annotations

import torch

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8


def rwkv7_w4_int8_shadow_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """Apply the per-channel INT8 shadow with fixed-shape fused accumulation.

    ``group_size`` is accepted to keep the Marlin dispatch hook uniform; the
    shadow intentionally uses one W8 scale per output channel. Decode-sized
    inputs use a fixed 32-row fused INT8 kernel. Larger short prefills
    reconstruct the dense quantized weight for chunk-boundary-stable GEMM
    without retaining an FP16 copy in model memory.
    """
    del group_size
    if x.ndim != 2 or qweight.ndim != 2 or scales.ndim != 2:
        raise ValueError("RWKV-7 W4 shadow GEMM expects 2D tensors")
    rows, size_k = x.shape
    size_n, shadow_k = qweight.shape
    if shadow_k != size_k:
        raise ValueError(f"W4 shadow K mismatch: input={size_k}, weight={shadow_k}")
    if scales.shape != (size_n, 1):
        raise ValueError(
            f"W4 shadow scale shape mismatch: expected {(size_n, 1)}, "
            f"got {tuple(scales.shape)}"
        )
    if qweight.dtype != torch.int8:
        raise TypeError(f"W4 shadow weight must be int8, got {qweight.dtype}")

    if rows > 8:
        dense_weight = (qweight.float() * scales).to(dtype=x.dtype)
        return torch.nn.functional.linear(x, dense_weight)

    x_quantized, x_scale = per_token_quant_int8(x)
    padded_rows = max(32, ((rows + 31) // 32) * 32)
    if padded_rows != rows:
        x_quantized = torch.nn.functional.pad(
            x_quantized, (0, 0, 0, padded_rows - rows)
        )
    if padded_rows != rows:
        x_scale = torch.nn.functional.pad(
            x_scale.view(rows, 1), (0, 0, 0, padded_rows - rows), value=1.0
        )
    from sgl_kernel import int8_scaled_mm

    output = int8_scaled_mm(
        x_quantized,
        qweight.t(),
        x_scale,
        scales,
        out_dtype=x.dtype,
        bias=None,
    )
    return output[:rows]
