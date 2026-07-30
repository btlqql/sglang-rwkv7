import os
import unittest
from unittest.mock import patch

from sglang.srt.models.rwkv7 import _rwkv7_projection_quant_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeQuantConfig:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class TestRwkv7QuantPolicy(unittest.TestCase):
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

    def test_w4_accuracy_interleaves_w4_and_w8_in_middle_ffn_layers(self):
        quant = FakeQuantConfig("marlin")
        with patch.dict(os.environ, {"SGLANG_RWKV7_W4_POLICY": "accuracy"}):
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

    def test_unrecognized_quantizer_uses_the_generic_policy(self):
        quant = FakeQuantConfig("bitsandbytes")
        for projection in ("attention", "ffn_key", "ffn_value"):
            self.assertIs(
                _rwkv7_projection_quant_config(
                    quant, projection, layer_id=0, num_hidden_layers=24
                ),
                quant,
            )

    def test_invalid_policy_is_rejected(self):
        quant = FakeQuantConfig("w8a8_int8")
        with patch.dict(os.environ, {"SGLANG_RWKV7_W8_POLICY": "unknown"}):
            with self.assertRaisesRegex(ValueError, "accuracy, balanced, or speed"):
                _rwkv7_projection_quant_config(quant, "attention", 0, 24)


if __name__ == "__main__":
    unittest.main()
