"""CUDA correctness tests for RWKV-7's online Marlin W4 path."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def _marlin_is_available():
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] >= 8
    )


if _marlin_is_available():
    from sglang.srt.layers.quantization.marlin_utils import (
        MarlinConfig,
        MarlinLinearMethod,
        online_marlin_quantize_weight,
        online_marlin_quantize_weight_with_int8_shadow,
        scalar_types,
    )
    from sglang.test.test_marlin_utils import marlin_quantize


@unittest.skipUnless(_marlin_is_available(), "Marlin W4 requires NVIDIA SM80+")
class TestRwkv7OnlineMarlin(unittest.TestCase):
    def test_online_pack_and_gemm_match_reference(self):
        torch.manual_seed(0x7A11)
        size_m, size_k, size_n = 8, 256, 256
        group_size = 128
        weight = torch.randn(size_n, size_k, device="cuda", dtype=torch.float16) / 20

        reference = marlin_quantize(
            weight.t().contiguous(),
            scalar_types.uint4b8,
            group_size,
            False,
        )
        packed_weight, scales = online_marlin_quantize_weight(weight, group_size)
        torch.testing.assert_close(packed_weight, reference[1], rtol=0, atol=0)
        torch.testing.assert_close(scales, reference[2], rtol=0, atol=0)

        workspace_size = max(
            (size_n // 64) * 16,
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count,
        )
        layer = SimpleNamespace(
            B=packed_weight,
            s=scales,
            workspace=torch.zeros(workspace_size, device="cuda", dtype=torch.int32),
        )
        inputs = torch.randn(size_m, size_k, device="cuda", dtype=torch.float16) / 10
        actual = MarlinLinearMethod(MarlinConfig(group_size, False)).apply(
            layer, inputs
        )
        expected = inputs @ reference[0]
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)

    def test_int8_shadow_is_batch_layout_invariant(self):
        torch.manual_seed(0x7A12)
        size_k, size_n = 256, 256
        group_size = 128
        weight = torch.randn(size_n, size_k, device="cuda", dtype=torch.float16) / 20
        marlin_reference = marlin_quantize(
            weight.t().contiguous(),
            scalar_types.uint4b8,
            group_size,
            False,
        )
        packed_weight, scales, shadow, shadow_scales = (
            online_marlin_quantize_weight_with_int8_shadow(weight, group_size)
        )
        workspace_size = max(
            (size_n // 64) * 16,
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count,
        )
        layer = SimpleNamespace(
            B=packed_weight,
            s=scales,
            workspace=torch.zeros(workspace_size, device="cuda", dtype=torch.int32),
            _rwkv7_decode_qweight=shadow,
            _rwkv7_decode_scales=shadow_scales,
            _rwkv7_marlin_fallback_max_tokens=512,
        )
        method = MarlinLinearMethod(MarlinConfig(group_size, False))
        inputs = torch.randn(8, size_k, device="cuda", dtype=torch.float16) / 10
        batched = method.apply(layer, inputs)
        shadow_reference = shadow.float() * shadow_scales
        expected = torch.nn.functional.linear(inputs.float(), shadow_reference).to(
            inputs.dtype
        )
        torch.testing.assert_close(batched, expected, rtol=0.03, atol=0.03)
        for row in range(inputs.shape[0]):
            single = method.apply(layer, inputs[row : row + 1])
            torch.testing.assert_close(single, batched[row : row + 1], rtol=0, atol=0)

        prefill = torch.randn(17, size_k, device="cuda", dtype=torch.float16) / 10
        prefill_expected = torch.nn.functional.linear(
            prefill.float(), shadow_reference
        ).to(prefill.dtype)
        torch.testing.assert_close(
            method.apply(layer, prefill), prefill_expected, rtol=0.03, atol=0.03
        )
        chunked_prefill = (
            torch.randn(65, size_k, device="cuda", dtype=torch.float16) / 10
        )
        whole_prefill = method.apply(layer, chunked_prefill)
        split_prefill = torch.cat(
            (
                method.apply(layer, chunked_prefill[:17]),
                method.apply(layer, chunked_prefill[17:]),
            )
        )
        torch.testing.assert_close(split_prefill, whole_prefill, rtol=0, atol=0)
        with patch(
            "sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager.get_tc_piecewise_forward_context",
            return_value=SimpleNamespace(full_graph=True),
        ):
            captured = method.apply(layer, prefill)
        torch.testing.assert_close(
            captured,
            prefill @ marlin_reference[0],
            rtol=0.03,
            atol=0.03,
        )

    def test_dense_shadow_uses_exact_small_m_and_marlin_in_full_graph(self):
        torch.manual_seed(0x7A13)
        size_k, size_n = 256, 256
        group_size = 128
        weight = torch.randn(size_n, size_k, device="cuda", dtype=torch.float16) / 20
        marlin_reference = marlin_quantize(
            weight.t().contiguous(),
            scalar_types.uint4b8,
            group_size,
            False,
        )
        packed_weight, scales = online_marlin_quantize_weight(weight, group_size)
        workspace_size = max(
            (size_n // 64) * 16,
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count,
        )
        layer = SimpleNamespace(
            B=packed_weight,
            s=scales,
            workspace=torch.zeros(workspace_size, device="cuda", dtype=torch.int32),
            _rwkv7_decode_weight=weight,
            _rwkv7_marlin_fallback_max_tokens=512,
        )
        method = MarlinLinearMethod(MarlinConfig(group_size, False))
        inputs = torch.randn(17, size_k, device="cuda", dtype=torch.float16) / 10
        torch.testing.assert_close(
            method.apply(layer, inputs),
            torch.nn.functional.linear(inputs, weight),
            rtol=0,
            atol=0,
        )
        with patch(
            "sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager.get_tc_piecewise_forward_context",
            return_value=SimpleNamespace(full_graph=True),
        ):
            captured = method.apply(layer, inputs)
        torch.testing.assert_close(
            captured,
            inputs @ marlin_reference[0],
            rtol=0.03,
            atol=0.03,
        )


if __name__ == "__main__":
    unittest.main()
