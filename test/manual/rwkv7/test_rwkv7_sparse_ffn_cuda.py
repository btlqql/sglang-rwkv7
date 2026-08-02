# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.rwkv7_kernels.ffn_sparse_cuda import (
    can_use_sparse_sqrelu_down,
    sparse_sqrelu_down,
)


@unittest.skipUnless(
    torch.cuda.is_available()
    and (torch.version.hip is not None or torch.cuda.get_device_capability()[0] >= 7),
    "requires CUDA sm70+ or ROCm",
)
class TestRWKV7SparseFFNCUDA(unittest.TestCase):
    def test_sqrelu_down_matches_dense(self):
        torch.manual_seed(17)
        for rows in (1, 8, 16):
            preact = torch.randn(rows, 2048, device="cuda", dtype=torch.float16)
            weight_t = (
                torch.randn(2048, 1024, device="cuda", dtype=torch.float16) * 0.01
            ).contiguous()
            self.assertTrue(can_use_sparse_sqrelu_down(preact, weight_t))
            actual = sparse_sqrelu_down(preact, weight_t)
            expected = F.relu(preact).square() @ weight_t
            torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)

    def test_shape_guard_keeps_fallback(self):
        preact = torch.randn(8, 2000, device="cuda", dtype=torch.float16)
        weight_t = torch.randn(2000, 1024, device="cuda", dtype=torch.float16)
        self.assertFalse(can_use_sparse_sqrelu_down(preact, weight_t))


if __name__ == "__main__":
    unittest.main()
