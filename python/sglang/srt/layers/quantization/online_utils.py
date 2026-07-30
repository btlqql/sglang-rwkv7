"""Device-independent helpers for quantizing ordinary checkpoint weights."""

import torch


def online_quantize_w8a8_int8_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an ordinary ``[out, in]`` weight to symmetric per-row INT8."""
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError(
            "Online W8A8 quantization requires a floating-point 2D weight, "
            f"got shape={weight.shape} dtype={weight.dtype}"
        )
    weight_fp32 = weight.float()
    scale = (weight_fp32.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(
        torch.finfo(torch.float32).eps
    )
    quantized = (weight_fp32 / scale).round().clamp_(-127, 127).to(torch.int8)
    return quantized, scale
