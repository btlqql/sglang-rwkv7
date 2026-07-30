"""CPU contract tests for online BitsAndBytes loading."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.bitsandbytes import BitsAndBytesConfig
from sglang.srt.model_loader.loader import BitsAndBytesModelLoader
from sglang.srt.model_loader.weight_utils import get_quant_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeWeight:
    is_cuda = True

    def size(self, dim):
        return (8, 16)[dim]

    def __getitem__(self, _key):
        return self

    def is_contiguous(self):
        return True

    def contiguous(self):
        return self

    def cpu(self):
        return self


def _model_config():
    return SimpleNamespace(
        quantization="bitsandbytes",
        hf_config=SimpleNamespace(quantization_config=None, text_config=None),
        model_path="unused",
    )


class TestBitsAndBytesOnlineConfig(unittest.TestCase):
    def test_non_linear_layer_does_not_import_moe_stack(self):
        config = BitsAndBytesConfig()
        moe_module = "sglang.srt.layers.moe.fused_moe_triton.layer"
        with patch.dict(sys.modules, {moe_module: None}):
            self.assertIsNone(
                config.get_quant_method(torch.nn.Embedding(8, 16), prefix="embed")
            )

    def test_w4_remains_the_default(self):
        config = get_quant_config(
            _model_config(),
            SimpleNamespace(model_loader_extra_config={}),
            packed_modules_mapping={},
        )

        self.assertIsInstance(config, BitsAndBytesConfig)
        self.assertTrue(config.load_in_4bit)
        self.assertFalse(config.load_in_8bit)

    def test_w8_defaults_to_cuda_graph_compatible_threshold(self):
        config = get_quant_config(
            _model_config(),
            SimpleNamespace(
                model_loader_extra_config={
                    "load_in_8bit": True,
                    "load_in_4bit": False,
                }
            ),
            packed_modules_mapping={},
        )

        self.assertTrue(config.load_in_8bit)
        self.assertFalse(config.load_in_4bit)
        self.assertEqual(config.llm_int8_threshold, 0.0)

    def test_explicit_w8_outlier_threshold_is_preserved(self):
        config = get_quant_config(
            _model_config(),
            SimpleNamespace(
                model_loader_extra_config={
                    "load_in_8bit": True,
                    "load_in_4bit": False,
                    "llm_int8_threshold": 5.5,
                }
            ),
            packed_modules_mapping={},
        )

        self.assertEqual(config.llm_int8_threshold, 5.5)


class TestBitsAndBytesOnlineWeightNames(unittest.TestCase):
    def _loader(self):
        loader = BitsAndBytesModelLoader.__new__(BitsAndBytesModelLoader)
        loader.target_modules = [".r_proj."]
        loader.column_parallel_weights_modules = []
        loader._hf_weight_iter = lambda *_args: iter(
            [("model.layers.0.attn.r_proj.weight", _FakeWeight())]
        )
        return loader

    def _fake_modules(self, quantize_4bit, int8_params):
        package = types.ModuleType("bitsandbytes")
        package.__path__ = []
        functional = types.ModuleType("bitsandbytes.functional")
        functional.quantize_4bit = quantize_4bit
        nn_module = types.ModuleType("bitsandbytes.nn")
        nn_module.Int8Params = int8_params
        return {
            "bitsandbytes": package,
            "bitsandbytes.functional": functional,
            "bitsandbytes.nn": nn_module,
        }

    def test_w4_iterator_keeps_model_parameter_name(self):
        def quantize_4bit(_weight, **kwargs):
            self.assertEqual(kwargs["quant_type"], "nf4")
            return "packed-w4", "w4-state"

        class UnusedInt8Params:
            pass

        state = {}
        with (
            patch.dict(
                sys.modules,
                self._fake_modules(quantize_4bit, UnusedInt8Params),
            ),
            patch(
                "sglang.srt.model_loader.loader.get_parallel",
                return_value=SimpleNamespace(tp_size=1, tp_rank=0),
            ),
        ):
            result = list(
                self._loader()._unquantized_generator(
                    ["unused"], True, state, load_8bit=False
                )
            )

        self.assertEqual(result, [("model.layers.0.attn.r_proj.weight", "packed-w4")])
        self.assertEqual(state, {"model.layers.0.attn.r_proj.weight": "w4-state"})

    def test_w8_iterator_keeps_model_parameter_name(self):
        class FakeInt8Params:
            def __init__(self, _weight, **_kwargs):
                self.data = "packed-w8"
                self.SCB = "w8-scale"

            def cuda(self):
                return self

        state = {}
        with (
            patch.dict(
                sys.modules,
                self._fake_modules(lambda *_args, **_kwargs: None, FakeInt8Params),
            ),
            patch(
                "sglang.srt.model_loader.loader.get_parallel",
                return_value=SimpleNamespace(tp_size=1, tp_rank=0),
            ),
        ):
            result = list(
                self._loader()._unquantized_generator(
                    ["unused"], True, state, load_8bit=True
                )
            )

        self.assertEqual(result, [("model.layers.0.attn.r_proj.weight", "packed-w8")])
        self.assertEqual(state, {"model.layers.0.attn.r_proj.weight": "w8-scale"})


if __name__ == "__main__":
    unittest.main()
