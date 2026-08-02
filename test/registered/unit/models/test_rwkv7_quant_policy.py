import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.bitsandbytes import BitsAndBytesLinearMethod
from sglang.srt.models.rwkv7 import (
    _rwkv7_bnb_target_modules,
    _rwkv7_int8_exact_max_tokens,
    _rwkv7_marlin_fallback_max_tokens,
    _rwkv7_projection_quant_config,
    _rwkv7_w4_shadow_mode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeQuantConfig:
    def __init__(self, name, load_in_8bit=False):
        self.name = name
        self.load_in_8bit = load_in_8bit

    def get_name(self):
        return self.name


class TestRwkv7QuantPolicy(unittest.TestCase):
    def test_bnb_4bit_honors_configured_compute_dtype(self):
        method = object.__new__(BitsAndBytesLinearMethod)
        method.quant_config = SimpleNamespace(bnb_4bit_compute_dtype="float16")
        weight = torch.empty(8, 1, dtype=torch.uint8)
        weight.bnb_quant_state = {0: torch.empty(3, 4)}
        weight.bnb_shard_offsets = torch.tensor([0, 8])
        layer = SimpleNamespace(weight=weight)

        def fake_apply(x, _weight, _offsets, out):
            self.assertEqual(x.dtype, torch.float16)
            self.assertEqual(out.dtype, torch.float16)
            out.zero_()

        with patch(
            "sglang.srt.layers.quantization.bitsandbytes.apply_bnb_4bit",
            side_effect=fake_apply,
        ):
            output = method._apply_4bit_weight(
                layer, torch.ones(2, 4, dtype=torch.bfloat16)
            )

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(tuple(output.shape), (2, 3))

    def test_accuracy_is_the_default_policy(self):
        w8 = FakeQuantConfig("w8a8_int8")
        w4 = FakeQuantConfig("marlin")
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(_rwkv7_projection_quant_config(w8, "attention", 12, 24))
            self.assertIsNone(_rwkv7_projection_quant_config(w4, "ffn_key", 0, 24))

    def test_w8_accuracy_protects_recurrence_sqrelu_and_edge_values(self):
        quant = FakeQuantConfig("w8a8_int8")
        with patch.dict(os.environ, {"SGLANG_RWKV7_W8_POLICY": "accuracy"}):
            self.assertIsNone(
                _rwkv7_projection_quant_config(quant, "attention", 12, 24)
            )
            self.assertIsNone(_rwkv7_projection_quant_config(quant, "ffn_key", 12, 24))
            self.assertIsNone(_rwkv7_projection_quant_config(quant, "ffn_value", 0, 24))
            self.assertIs(
                _rwkv7_projection_quant_config(quant, "ffn_value", 12, 24),
                quant,
            )

    def test_w4_accuracy_uses_w4_expansion_and_w8_contraction(self):
        quant = FakeQuantConfig("marlin")
        with (
            patch.dict(os.environ, {"SGLANG_RWKV7_W4_POLICY": "accuracy"}),
            patch(
                "sglang.srt.models.rwkv7._rwkv7_w8a8_config",
                return_value=FakeQuantConfig("w8a8_int8"),
            ),
        ):
            self.assertIsNone(
                _rwkv7_projection_quant_config(quant, "attention", 12, 24)
            )
            for layer_id in (0, 3, 20, 23):
                self.assertIsNone(
                    _rwkv7_projection_quant_config(quant, "ffn_value", layer_id, 24)
                )
            self.assertIs(
                _rwkv7_projection_quant_config(quant, "ffn_key", 12, 24), quant
            )
            self.assertEqual(
                _rwkv7_projection_quant_config(quant, "ffn_value", 12, 24).get_name(),
                "w8a8_int8",
            )
            self.assertEqual(
                _rwkv7_projection_quant_config(quant, "ffn_key", 13, 24).get_name(),
                "w8a8_int8",
            )
            self.assertEqual(
                _rwkv7_projection_quant_config(quant, "ffn_value", 13, 24).get_name(),
                "w8a8_int8",
            )

    def test_speed_policies_quantize_every_large_projection(self):
        for env_name, quant_name in (
            ("SGLANG_RWKV7_W8_POLICY", "w8a8_int8"),
            ("SGLANG_RWKV7_W4_POLICY", "marlin"),
        ):
            quant = FakeQuantConfig(quant_name)
            with patch.dict(os.environ, {env_name: "speed"}):
                for projection in ("attention", "ffn_key", "ffn_value"):
                    self.assertIs(
                        _rwkv7_projection_quant_config(
                            quant, projection, layer_id=0, num_hidden_layers=24
                        ),
                        quant,
                    )

    def test_balanced_policies_keep_attention_dense(self):
        for env_name, quant_name in (
            ("SGLANG_RWKV7_W8_POLICY", "w8a8_int8"),
            ("SGLANG_RWKV7_W4_POLICY", "marlin"),
        ):
            quant = FakeQuantConfig(quant_name)
            with patch.dict(os.environ, {env_name: "balanced"}):
                self.assertIsNone(
                    _rwkv7_projection_quant_config(quant, "attention", 0, 24)
                )
                for projection in ("ffn_key", "ffn_value"):
                    self.assertIs(
                        _rwkv7_projection_quant_config(
                            quant, projection, layer_id=0, num_hidden_layers=24
                        ),
                        quant,
                    )

    def test_w4_sparse_policy_quantizes_selected_ffn_expansions_only(self):
        quant = FakeQuantConfig("marlin")
        with patch.dict(os.environ, {"SGLANG_RWKV7_W4_POLICY": "sparse"}):
            self.assertIsNone(_rwkv7_projection_quant_config(quant, "attention", 0, 24))
            for layer_id in (4, 6, 8, 10, 12, 16):
                self.assertIs(
                    _rwkv7_projection_quant_config(quant, "ffn_key", layer_id, 24),
                    quant,
                )
            for layer_id in (0, 5, 14, 18, 23):
                self.assertIsNone(
                    _rwkv7_projection_quant_config(quant, "ffn_key", layer_id, 24)
                )
            self.assertIsNone(_rwkv7_projection_quant_config(quant, "ffn_value", 0, 24))

    def test_bnb_accuracy_w8_quantizes_only_ffn_value(self):
        quant = FakeQuantConfig("bitsandbytes", load_in_8bit=True)
        with patch.dict(os.environ, {"SGLANG_RWKV7_BNB_POLICY": "accuracy"}):
            self.assertIsNone(
                _rwkv7_projection_quant_config(quant, "attention", 12, 24)
            )
            self.assertIsNone(_rwkv7_projection_quant_config(quant, "ffn_key", 12, 24))
            self.assertIs(
                _rwkv7_projection_quant_config(quant, "ffn_value", 12, 24), quant
            )
            self.assertEqual(_rwkv7_bnb_target_modules(quant, 24), [".value."])

    def test_bnb_accuracy_w4_uses_sparse_middle_value_lane(self):
        quant = FakeQuantConfig("bitsandbytes", load_in_8bit=False)
        with patch.dict(os.environ, {"SGLANG_RWKV7_BNB_POLICY": "accuracy"}):
            for layer_id in (4, 8, 12, 16):
                self.assertIs(
                    _rwkv7_projection_quant_config(quant, "ffn_value", layer_id, 24),
                    quant,
                )
            for layer_id in (0, 5, 18, 23):
                self.assertIsNone(
                    _rwkv7_projection_quant_config(quant, "ffn_value", layer_id, 24)
                )
            self.assertEqual(
                _rwkv7_bnb_target_modules(quant, 24),
                [
                    ".layers.4.ffn.value.",
                    ".layers.8.ffn.value.",
                    ".layers.12.ffn.value.",
                    ".layers.16.ffn.value.",
                ],
            )

    def test_bnb_speed_and_balanced_policies_match_loader_targets(self):
        quant = FakeQuantConfig("bitsandbytes", load_in_8bit=True)
        with patch.dict(os.environ, {"SGLANG_RWKV7_BNB_POLICY": "balanced"}):
            self.assertIsNone(_rwkv7_projection_quant_config(quant, "attention", 0, 24))
            self.assertEqual(_rwkv7_bnb_target_modules(quant, 24), [".key.", ".value."])
        with patch.dict(os.environ, {"SGLANG_RWKV7_BNB_POLICY": "speed"}):
            for projection in ("attention", "ffn_key", "ffn_value"):
                self.assertIs(
                    _rwkv7_projection_quant_config(quant, projection, 0, 24), quant
                )
            self.assertEqual(len(_rwkv7_bnb_target_modules(quant, 24)), 6)

    def test_invalid_policy_is_rejected(self):
        quant = FakeQuantConfig("w8a8_int8")
        with patch.dict(os.environ, {"SGLANG_RWKV7_W8_POLICY": "unknown"}):
            with self.assertRaisesRegex(ValueError, "accuracy, balanced, or speed"):
                _rwkv7_projection_quant_config(quant, "attention", 0, 24)

    def test_marlin_fallback_limit_is_configurable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_rwkv7_marlin_fallback_max_tokens(), 512)
        with patch.dict(
            os.environ, {"SGLANG_RWKV7_MARLIN_FALLBACK_MAX_TOKENS": "2048"}
        ):
            self.assertEqual(_rwkv7_marlin_fallback_max_tokens(), 2048)
        with patch.dict(os.environ, {"SGLANG_RWKV7_MARLIN_FALLBACK_MAX_TOKENS": "-1"}):
            with self.assertRaisesRegex(ValueError, "must be non-negative"):
                _rwkv7_marlin_fallback_max_tokens()
        with patch.dict(
            os.environ, {"SGLANG_RWKV7_MARLIN_FALLBACK_MAX_TOKENS": "many"}
        ):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                _rwkv7_marlin_fallback_max_tokens()

    def test_w4_shadow_mode_is_size_aware_and_overrideable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_rwkv7_w4_shadow_mode(2048), "fp16")
            self.assertEqual(_rwkv7_w4_shadow_mode(2560), "int8")
        for mode in ("fp16", "int8"):
            with patch.dict(os.environ, {"SGLANG_RWKV7_W4_SHADOW": mode}):
                self.assertEqual(_rwkv7_w4_shadow_mode(4096), mode)
        with patch.dict(os.environ, {"SGLANG_RWKV7_W4_SHADOW": "unknown"}):
            with self.assertRaisesRegex(ValueError, "auto, fp16, or int8"):
                _rwkv7_w4_shadow_mode(2048)

    def test_int8_exact_limit_is_configurable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_rwkv7_int8_exact_max_tokens(), 512)
        with patch.dict(os.environ, {"SGLANG_RWKV7_INT8_EXACT_MAX_TOKENS": "2048"}):
            self.assertEqual(_rwkv7_int8_exact_max_tokens(), 2048)
        with patch.dict(os.environ, {"SGLANG_RWKV7_INT8_EXACT_MAX_TOKENS": "-1"}):
            with self.assertRaisesRegex(ValueError, "must be non-negative"):
                _rwkv7_int8_exact_max_tokens()
        with patch.dict(os.environ, {"SGLANG_RWKV7_INT8_EXACT_MAX_TOKENS": "many"}):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                _rwkv7_int8_exact_max_tokens()


if __name__ == "__main__":
    unittest.main()
