import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cpu_ci(est_time=2, suite="base-c-test-cpu")


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "benchmark" / "rwkv7" / "analyze_acceptance_matrix.py"
SPEC = importlib.util.spec_from_file_location("rwkv7_acceptance_report", MODULE_PATH)
assert SPEC and SPEC.loader
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def row(model, mode, *, prefill, decode, e2e, ttft=0.1, tpot=0.01):
    return {
        "schema": "rwkv7-serving-acceptance-v1",
        "model": model,
        "mode": mode,
        "batch_size": 1,
        "prompt_tokens_per_request": 128,
        "decode_tokens_per_request": 128,
        "median": {
            "prefill_tok_s": prefill,
            "decode_tok_s": decode,
            "e2e_output_tok_s": e2e,
            "ttft_s": ttft,
            "tpot_s": tpot,
        },
    }


class TestRwkv7AcceptanceReport(unittest.TestCase):
    def test_albatross_result_parser(self):
        text = (
            "noise\n"
            "RESULT B=4 T=512 iters=5 p10_ms=1.0 p50_ms=2.0 "
            "p90_ms=3.0 tok_s_p50=1024.5\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "albatross.log"
            path.write_text(text, encoding="utf-8")
            parsed = REPORT.parse_albatross_log(path)
        self.assertEqual(parsed[(4, 512)]["iters"], 5)
        self.assertEqual(parsed[(4, 512)]["tok_s_p50"], 1024.5)

    def test_joined_matrix_passes_all_speed_gates(self):
        model = "rwkv7-g1-1.5b"
        dense = row(model, "dense", prefill=200, decode=200, e2e=180)
        quant = row(model, "w8", prefill=220, decode=210, e2e=190)
        qwen = row("qwen3.5-2b", "dense", prefill=100, decode=100, e2e=90)
        albatross = {
            (1, 1): {"tok_s_p50": 100},
            (1, 128): {"tok_s_p50": 100},
        }
        report = REPORT.analyze(
            candidate_rows=[dense, quant],
            qwen_rows={model: [qwen]},
            albatross_rows={model: albatross},
            models=[model],
            modes=["dense", "w8"],
            batch_sizes=[1],
            prompt_lengths=[128],
            decode_lengths=[128],
            dense_mode="dense",
            qwen_minimum=1.0,
            albatross_minimum=1.0,
            quant_minimum=1.0,
            active_work_factors={model: 1.0},
            active_work_minimums={model: 1.5},
            require_active_work=True,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["summary"], {"passed": 2, "failed": 0, "missing": 0})

    def test_missing_and_failed_cells_remain_in_denominator(self):
        model = "rwkv7-g1-1.5b"
        dense = row(model, "dense", prefill=80, decode=80, e2e=80)
        qwen = row("qwen3.5-2b", "dense", prefill=100, decode=100, e2e=100)
        albatross = {
            (1, 1): {"tok_s_p50": 100},
            (1, 128): {"tok_s_p50": 100},
        }
        report = REPORT.analyze(
            candidate_rows=[dense],
            qwen_rows={model: [qwen]},
            albatross_rows={model: albatross},
            models=[model],
            modes=["dense", "w8"],
            batch_sizes=[1],
            prompt_lengths=[128],
            decode_lengths=[128],
            dense_mode="dense",
            qwen_minimum=1.0,
            albatross_minimum=1.0,
            quant_minimum=1.0,
            active_work_factors={},
            active_work_minimums={},
            require_active_work=False,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["summary"], {"passed": 0, "failed": 1, "missing": 1})
        self.assertIn("qwen_prefill_ratio", report["cells"][0]["failed_gates"])

    def test_jsonl_loader_rejects_wrong_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps({"schema": "wrong"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported schema"):
                REPORT.load_jsonl([path])

    def test_quant_memory_must_be_strictly_lower_than_dense(self):
        model = "rwkv7-g1-1.5b"
        dense = row(model, "dense", prefill=200, decode=200, e2e=180)
        quant = row(model, "w8", prefill=220, decode=210, e2e=190)
        qwen = row("qwen3.5-2b", "dense", prefill=100, decode=100, e2e=90)
        albatross = {
            (1, 1): {"tok_s_p50": 100},
            (1, 128): {"tok_s_p50": 100},
        }
        memory = [
            {
                "schema": "rwkv7-serving-memory-v1",
                "model": model,
                "mode": "dense",
                "model_weight_memory_gb": 3.0,
            },
            {
                "schema": "rwkv7-serving-memory-v1",
                "model": model,
                "mode": "w8",
                "model_weight_memory_gb": 2.5,
            },
        ]
        report = REPORT.analyze(
            candidate_rows=[dense, quant],
            qwen_rows={model: [qwen]},
            albatross_rows={model: albatross},
            models=[model],
            modes=["dense", "w8"],
            batch_sizes=[1],
            prompt_lengths=[128],
            decode_lengths=[128],
            dense_mode="dense",
            qwen_minimum=1.0,
            albatross_minimum=1.0,
            quant_minimum=1.0,
            active_work_factors={},
            active_work_minimums={},
            require_active_work=False,
            memory_rows=memory,
            require_memory=True,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["memory_summary"], {"passed": 2, "failed": 0, "missing": 0}
        )
        gate = report["memory_cells"][1]["gates"]["quant_dense_model_memory_ratio"]
        self.assertAlmostEqual(gate["value"], 2.5 / 3.0)

        memory[1]["model_weight_memory_gb"] = 3.0
        failed = REPORT.analyze(
            candidate_rows=[dense, quant],
            qwen_rows={model: [qwen]},
            albatross_rows={model: albatross},
            models=[model],
            modes=["dense", "w8"],
            batch_sizes=[1],
            prompt_lengths=[128],
            decode_lengths=[128],
            dense_mode="dense",
            qwen_minimum=1.0,
            albatross_minimum=1.0,
            quant_minimum=1.0,
            active_work_factors={},
            active_work_minimums={},
            require_active_work=False,
            memory_rows=memory,
            require_memory=True,
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["memory_summary"]["failed"], 1)

    def test_memory_loader_rejects_wrong_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            path.write_text(json.dumps({"schema": "wrong"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported memory schema"):
                REPORT.load_memory_jsonl([path])


if __name__ == "__main__":
    unittest.main()
