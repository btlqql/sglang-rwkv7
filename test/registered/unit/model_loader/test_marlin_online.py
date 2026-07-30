"""CPU contract tests for online Marlin W4 configuration."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.quantization import QUANTIZATION_METHODS
from sglang.srt.layers.quantization.marlin_utils import MarlinConfig
from sglang.srt.model_loader.weight_utils import get_quant_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _model_config():
    return SimpleNamespace(
        quantization="marlin",
        hf_config=SimpleNamespace(quantization_config=None, text_config=None),
        model_path="unused",
    )


class TestMarlinOnlineConfig(unittest.TestCase):
    def test_marlin_is_registered(self):
        self.assertIs(QUANTIZATION_METHODS["marlin"], MarlinConfig)

    def test_online_defaults(self):
        config = get_quant_config(
            _model_config(),
            SimpleNamespace(model_loader_extra_config={"online_quantization": True}),
            packed_modules_mapping={},
        )

        self.assertIsInstance(config, MarlinConfig)
        self.assertEqual(config.group_size, 128)
        self.assertFalse(config.lm_head_quantized)

    def test_online_overrides(self):
        config = get_quant_config(
            _model_config(),
            SimpleNamespace(
                model_loader_extra_config={
                    "online_quantization": True,
                    "group_size": -1,
                    "lm_head_quantized": True,
                }
            ),
            packed_modules_mapping={},
        )

        self.assertEqual(config.group_size, -1)
        self.assertTrue(config.lm_head_quantized)


if __name__ == "__main__":
    unittest.main()
