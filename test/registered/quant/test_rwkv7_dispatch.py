import unittest

from sglang.srt.layers.quantization.rwkv7_dispatch import (
    rwkv7_kernel_capabilities,
    select_rwkv7_w4_kernel,
    select_rwkv7_w8_kernel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRwkv7NativeDispatch(unittest.TestCase):
    def test_w8_boundaries_preserve_decode_and_prefill_paths(self):
        self.assertEqual(select_rwkv7_w8_kernel(8, 2048).kernel, "w8_row")
        self.assertEqual(select_rwkv7_w8_kernel(9, 2048).kernel, "w8_tiled")
        self.assertEqual(select_rwkv7_w8_kernel(511, 2048).kernel, "w8_tiled")
        self.assertEqual(select_rwkv7_w8_kernel(512, 2048).kernel, "w8a8_tiled")

    def test_w8_wide_mid_batch_uses_large_reduction_tile(self):
        plan = select_rwkv7_w8_kernel(64, 4096)
        self.assertEqual((plan.block_m, plan.block_k, plan.num_warps), (64, 128, 8))
        narrow = select_rwkv7_w8_kernel(64, 2048)
        self.assertEqual(
            (narrow.block_m, narrow.block_k, narrow.num_warps), (32, 64, 4)
        )

    def test_w4_backend_capability_owns_large_prefill_choice(self):
        cuda = rwkv7_kernel_capabilities(False)
        rocm = rwkv7_kernel_capabilities(True)
        self.assertEqual(select_rwkv7_w4_kernel(1024, 2048, cuda).kernel, "w4_tiled")
        self.assertEqual(select_rwkv7_w4_kernel(1023, 2048, rocm).kernel, "w4_tiled")
        self.assertEqual(
            select_rwkv7_w4_kernel(1024, 2048, rocm).kernel, "w4_dequant_mm"
        )

    def test_decode_reduction_tiles_match_previous_limits(self):
        w8 = select_rwkv7_w8_kernel(8, 6144)
        w4 = select_rwkv7_w4_kernel(8, 6144, rwkv7_kernel_capabilities(False))
        self.assertEqual(w8.block_k, 4096)
        self.assertEqual(w4.block_k, 2048)
        self.assertEqual(w4.num_warps, 2)

    def test_invalid_shapes_fail_before_launch(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            select_rwkv7_w8_kernel(0, 2048)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            select_rwkv7_w4_kernel(8, 0, rwkv7_kernel_capabilities(False))


if __name__ == "__main__":
    unittest.main()
