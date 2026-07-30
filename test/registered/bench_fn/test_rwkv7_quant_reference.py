import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cpu_ci(est_time=2, suite="base-c-test-cpu")


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "benchmark" / "rwkv7" / "verify_quant_alignment.py"
SPEC = importlib.util.spec_from_file_location("rwkv7_quant_alignment", MODULE_PATH)
assert SPEC and SPEC.loader
ALIGN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ALIGN
SPEC.loader.exec_module(ALIGN)


class TestRwkv7QuantReference(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            model="rwkv7-7.2b",
            dtype="float16",
            max_new_tokens=2,
            top_k=3,
        )
        self.references = [
            ALIGN.Reference(
                prompt="hello",
                prompt_ids=[1, 2],
                output_ids=[3, 4],
                token_logprobs=[-0.1, -0.2],
                top_ids=[[3, 8, 9], [4, 7, 6]],
            )
        ]

    def test_reference_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            ALIGN.save_references(path, self.args, self.references)
            actual = ALIGN.load_references(path, self.args)
        self.assertEqual(actual, self.references)

    def test_reference_configuration_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            ALIGN.save_references(path, self.args, self.references)
            wrong = SimpleNamespace(**vars(self.args))
            wrong.top_k = 5
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                ALIGN.load_references(path, wrong)

    def test_inconsistent_token_arrays_are_rejected(self):
        broken = [
            ALIGN.Reference(
                prompt="hello",
                prompt_ids=[1],
                output_ids=[2, 3],
                token_logprobs=[-0.1],
                top_ids=[[2, 4, 5], [3, 5, 6]],
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            ALIGN.save_references(path, self.args, broken)
            with self.assertRaisesRegex(ValueError, "inconsistent reference lengths"):
                ALIGN.load_references(path, self.args)


if __name__ == "__main__":
    unittest.main()
