"""CPU correctness tests for online W8A8 weight quantization."""

import unittest

import torch

from sglang.srt.layers.quantization.online_utils import (
    online_quantize_w8a8_int8_weight,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestW8A8OnlineWeight(unittest.TestCase):
    def test_per_output_channel_symmetric_quantization(self):
        weight = torch.tensor(
            [
                [-2.0, -0.5, 0.0, 1.0],
                [-0.25, 0.0, 0.125, 0.5],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float16,
        )
        quantized, scale = online_quantize_w8a8_int8_weight(weight)

        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(tuple(scale.shape), (3, 1))
        self.assertEqual(int(quantized[0].abs().max()), 127)
        self.assertEqual(int(quantized[1].abs().max()), 127)
        self.assertGreater(float(scale[2]), 0.0)

        reconstructed = quantized.float() * scale
        error = (reconstructed - weight.float()).abs()
        self.assertTrue(torch.all(error <= scale / 2 + 1e-6))

    def test_rejects_non_matrix_weights(self):
        with self.assertRaisesRegex(ValueError, "floating-point 2D weight"):
            online_quantize_w8a8_int8_weight(torch.ones(8))


if __name__ == "__main__":
    unittest.main()
