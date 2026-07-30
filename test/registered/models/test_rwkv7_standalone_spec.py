"""End-to-end correctness gate for RWKV-7 STANDALONE speculation."""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-small")

MODEL_PATH = os.getenv("SGLANG_RWKV7_TEST_MODEL", "fla-hub/rwkv7-191M-world")
BASELINE_URL = "http://127.0.0.1:31080"
SPEC_URL = "http://127.0.0.1:31081"

PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "1 + 2 + 3 =",
    "The quick brown fox",
]


class TestRwkv7StandaloneSpec(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "triton",
            "--sampling-backend",
            "pytorch",
            "--grammar-backend",
            "none",
            "--dtype",
            "bfloat16",
            "--log-level",
            "warning",
            "--disable-overlap-schedule",
            "--chunked-prefill-size",
            "64",
            "--max-running-requests",
            "4",
            "--max-mamba-cache-size",
            "8",
        ]
        cls.baseline_process = popen_launch_server(
            MODEL_PATH,
            BASELINE_URL,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=common_args,
        )
        cls.spec_process = popen_launch_server(
            MODEL_PATH,
            SPEC_URL,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=common_args
            + [
                "--speculative-algorithm",
                "STANDALONE",
                "--speculative-draft-model-path",
                MODEL_PATH,
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        for process in (
            getattr(cls, "baseline_process", None),
            getattr(cls, "spec_process", None),
        ):
            if process is not None:
                kill_process_tree(process.pid)

    @staticmethod
    def _generate(base_url):
        response = requests.post(
            f"{base_url}/generate",
            json={
                "text": PROMPTS,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 32,
                    "ignore_eos": True,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _generate_ids(input_ids, max_new_tokens):
        response = requests.post(
            f"{SPEC_URL}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _flush_spec_cache():
        response = requests.post(
            f"{SPEC_URL}/flush_cache",
            params={"timeout": 30},
            timeout=40,
        )
        response.raise_for_status()

    def test_matches_non_speculative_greedy_generation(self):
        baseline = self._generate(BASELINE_URL)
        speculative = self._generate(SPEC_URL)

        self.assertEqual(
            [item["output_ids"] for item in speculative],
            [item["output_ids"] for item in baseline],
        )
        for item in speculative:
            meta = item["meta_info"]
            self.assertGreater(meta["spec_num_proposed_drafts"], 0)
            self.assertGreaterEqual(meta["spec_accept_rate"], 0.0)
            self.assertLessEqual(meta["spec_accept_rate"], 1.0)

    def test_recurrent_state_cache_hit_matches_cold_generation(self):
        prefix_ids = [100 + i for i in range(128)]
        full_ids = prefix_ids + [1000 + i for i in range(64)]

        self._flush_spec_cache()
        cold = self._generate_ids(full_ids, max_new_tokens=16)
        self.assertEqual(cold["meta_info"]["cached_tokens"], 0)

        self._flush_spec_cache()
        self._generate_ids(prefix_ids, max_new_tokens=0)
        warm = self._generate_ids(full_ids, max_new_tokens=16)

        self.assertEqual(warm["meta_info"]["cached_tokens"], len(prefix_ids))
        self.assertEqual(warm["output_ids"], cold["output_ids"])
        self.assertGreater(warm["meta_info"]["spec_num_proposed_drafts"], 0)


if __name__ == "__main__":
    unittest.main()
