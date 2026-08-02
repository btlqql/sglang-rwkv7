"""CUDA/HIP correctness tests for RWKV-7 portable W8/W4 kernels."""

import unittest

import torch

from sglang.srt.layers.quantization.online_utils import online_quantize_w8a8_int8_weight
from sglang.srt.layers.quantization.rwkv7_native import (
    online_quantize_rwkv7_w4_weight,
    rwkv7_w4_linear,
    rwkv7_w8_linear,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
class TestRwkv7NativeQuant(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0x7A18)
        self.weight = torch.randn(256, 256, device="cuda", dtype=torch.float16) * 0.02

    def test_w8_decode_and_prefill_match_dequantized_weight(self):
        quant, scale = online_quantize_w8a8_int8_weight(self.weight)
        dequant = quant.float() * scale.float()

        for rows, tolerance in ((8, 2e-2), (512, 4e-2)):
            inputs = torch.randn(
                rows, self.weight.shape[1], device="cuda", dtype=torch.float16
            )
            actual = rwkv7_w8_linear(inputs, quant, scale)
            expected = inputs.float() @ dequant.t()
            torch.testing.assert_close(
                actual.float(), expected, atol=tolerance, rtol=tolerance
            )

    def test_w4_decode_and_prefill_match_dequantized_weight(self):
        packed, scale = online_quantize_rwkv7_w4_weight(self.weight)
        low = (packed & 15).to(torch.int16) - 8
        high = ((packed >> 4) & 15).to(torch.int16) - 8
        quant = torch.stack((low, high), dim=-1).reshape_as(self.weight)
        dequant = quant.float() * scale.float().repeat_interleave(32, dim=1)

        for rows in (8, 512):
            inputs = torch.randn(
                rows, self.weight.shape[1], device="cuda", dtype=torch.float16
            )
            actual = rwkv7_w4_linear(inputs, packed, scale, group_size=32)
            expected = inputs.float() @ dequant.t()
            torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
