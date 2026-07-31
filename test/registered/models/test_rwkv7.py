"""End-to-end server test for RWKV-7 (Goose) — a pure-recurrent (attention-free)
model on the native sglang serving stack.

Boots a real ``fla-hub/rwkv7-191M-world`` server via ``popen_launch_server``
(small checkpoint, ~0.4 GB) and checks the properties that matter for a
constant-state recurrent model:

  1. basic generation through ``/generate``,
  2. greedy decoding is deterministic call-to-call,
  3. per-request state isolation under dynamic batching — duplicates of a
     prompt inside one batch must decode identically (same GEMM shapes, so
     any divergence is leaked/mixed recurrent state, not numerics), and a
     batched request must reproduce the single-request tokens over the
     leading window (state corruption shows up at the first tokens;
     batch-size-dependent cuBLAS reduction-order drift needs many tokens
     of autoregressive accumulation to surface),
  4. chunked prefill and recurrent-state prefix-cache restoration preserve
     greedy output while reporting a real cache hit,
  5. mixed output lengths can compact a running batch without moving another
     request's recurrent state,
  6. aborting an in-flight request releases its recurrent state before the
     scheduler reuses that pool slot.
"""

import os
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-small")

MODEL_PATH = os.getenv("SGLANG_RWKV7_TEST_MODEL", "fla-hub/rwkv7-191M-world")

PROMPTS = [
    "The capital of France is",
    "1 + 2 + 3 + 4 + 5 =",
    "Once upon a time, in a small village by the sea,",
    "The quick brown fox",
]


class TestRwkv7Server(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "triton",
                # Pin bf16: the checkpoint's torch_dtype is float32, and fp32
                # matmuls take M-dependent cuBLAS/TF32 reduction paths whose
                # ulp-level differences between batched and single GEMMs get
                # amplified autoregressively — that is a GEMM-backend property,
                # not a model one. The batch==single guarantee below is
                # exercised (and extensively validated) at bf16/fp16.
                "--dtype",
                "bfloat16",
                # Exercise recurrent state hand-off between prefill chunks.
                "--chunked-prefill-size",
                "64",
                "--mem-fraction-static",
                "0.5",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)

    def _generate(self, text, max_new_tokens=32):
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data[0] if isinstance(data, list) else data

    def _generate_batch(self, texts, max_new_tokens=32):
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": texts,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def _generate_ids(self, input_ids, max_new_tokens=16, rid=None):
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        }
        if rid is not None:
            payload["rid"] = rid
        resp = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data[0] if isinstance(data, list) else data

    def _flush_cache(self):
        resp = requests.post(
            f"{self.base_url}/flush_cache",
            params={"timeout": 30},
            timeout=40,
        )
        resp.raise_for_status()

    def test_generation_basic(self):
        for prompt in PROMPTS[:3]:
            out = self._generate(prompt)
            self.assertTrue(
                len(out["output_ids"]) > 0,
                f"empty completion for prompt {prompt!r}",
            )

    def test_greedy_deterministic(self):
        first = self._generate(PROMPTS[0])["output_ids"]
        second = self._generate(PROMPTS[0])["output_ids"]
        self.assertEqual(
            first,
            second,
            "greedy decoding must be deterministic across identical requests",
        )

    def test_batch_state_isolation(self):
        # (a) Duplicates inside one batch must decode identically, token for
        # token. All duplicate rows share every GEMM/kernel launch, so the
        # numerics are bit-identical by construction; the only way they can
        # diverge is a per-request state bug (a reused pool slot that was not
        # reset, or recurrence reading a neighbor's state).
        a, b = PROMPTS[0], PROMPTS[1]
        batched = self._generate_batch([a, b, a, b])
        outs = [item["output_ids"] for item in batched]
        self.assertEqual(
            outs[0], outs[2], f"duplicate requests of {a!r} diverged in one batch"
        )
        self.assertEqual(
            outs[1], outs[3], f"duplicate requests of {b!r} diverged in one batch"
        )
        # (b) Batched output must reproduce the single-request output over the
        # leading tokens. Corrupted carry-in state breaks greedy decoding at
        # token 0-1, while bf16 reduction-order differences between
        # batch-size-dependent GEMM kernel choices need autoregressive
        # accumulation before they can flip a near-tie token — a short prefix
        # window discriminates between the two. Kept to 4 tokens to stay far
        # from the accumulation regime across CI GPU/cuBLAS variants.
        prefix = 4
        for prompt, got in ((a, outs[0]), (b, outs[1])):
            single = self._generate(prompt)["output_ids"]
            self.assertEqual(
                single[:prefix],
                got[:prefix],
                f"batched greedy output diverged from single-request output "
                f"within the first {prefix} tokens for prompt {prompt!r}: "
                f"single={single[:prefix]} batched={got[:prefix]}",
            )

    def test_chunked_prefill_state_cache_hit(self):
        # Use explicit token ids so the seeded prefix is guaranteed to be an
        # exact token prefix of the continuation. Keep the checkpoint on a
        # 64-token prefill boundary: otherwise the cold and warm paths use
        # different GEMM M dimensions for the same prefix. On larger models,
        # normal cuBLAS reduction-order drift can then flip a later near-tie
        # greedy token even though the restored recurrent state is correct.
        prefix_ids = [100 + i for i in range(128)]
        full_ids = prefix_ids + [1000 + i for i in range(64)]

        # Establish the cold-cache greedy reference.
        self._flush_cache()
        cold = self._generate_ids(full_ids)
        self.assertEqual(cold["meta_info"]["cached_tokens"], 0)

        # Seed a state checkpoint exactly at the branch point, then continue
        # from it. MambaRadixCache must restore all token-shift and WKV states.
        self._flush_cache()
        self._generate_ids(prefix_ids, max_new_tokens=0)
        warm = self._generate_ids(full_ids)

        self.assertEqual(
            warm["meta_info"]["cached_tokens"],
            len(prefix_ids),
            "RWKV-7 should restore the recurrent state at the cached prefix",
        )
        self.assertEqual(
            warm["output_ids"],
            cold["output_ids"],
            "cached-state continuation must match cold chunked prefill",
        )

    def test_unaligned_state_cache_first_token(self):
        # Prefix caching is also valid away from a prefill boundary. Compare
        # the immediate continuation, which catches a missing/stale RWKV state
        # without conflating cache correctness with shape-dependent GEMM drift
        # accumulated over later autoregressive steps.
        prefix_ids = [200 + i for i in range(96)]
        full_ids = prefix_ids + [1200 + i for i in range(64)]

        self._flush_cache()
        cold = self._generate_ids(full_ids, max_new_tokens=1)

        self._flush_cache()
        self._generate_ids(prefix_ids, max_new_tokens=0)
        warm = self._generate_ids(full_ids, max_new_tokens=1)

        self.assertEqual(warm["meta_info"]["cached_tokens"], len(prefix_ids))
        self.assertEqual(warm["output_ids"], cold["output_ids"])

    def test_mixed_length_dynamic_batch_compaction(self):
        """Finishing short rows must not move a neighbor's recurrent state."""

        prompt_a = [300 + i for i in range(64)]
        prompt_b = [1300 + i for i in range(96)]
        prompt_c = [2300 + i for i in range(128)]
        work = [
            (prompt_a, 16),
            (prompt_b, 8),
            (prompt_a, 64),
            (prompt_c, 32),
            (prompt_a, 64),
        ]
        barrier = threading.Barrier(len(work))

        def generate(item):
            input_ids, max_new_tokens = item
            barrier.wait(timeout=10)
            return self._generate_ids(input_ids, max_new_tokens)["output_ids"]

        self._flush_cache()
        with ThreadPoolExecutor(max_workers=len(work)) as executor:
            outputs = list(executor.map(generate, work))

        self.assertEqual([len(output) for output in outputs], [16, 8, 64, 32, 64])
        self.assertEqual(
            outputs[0],
            outputs[2][:16],
            "the duplicate prompt diverged before its short request completed",
        )
        self.assertEqual(
            outputs[2],
            outputs[4],
            "surviving duplicate requests diverged while the batch compacted",
        )

        # Once the other three rows have finished and the running batch has
        # compacted, the surviving row must still agree with a clean request.
        self._flush_cache()
        clean = self._generate_ids(prompt_a, 64)["output_ids"]
        self.assertEqual(
            outputs[2][:4],
            clean[:4],
            "mixed-length batch compaction corrupted the surviving RWKV state",
        )

    def test_abort_releases_recurrent_state(self):
        """An aborted request must not leak state into the next pool occupant."""

        reference_ids = [100 + i for i in range(32)]
        self._flush_cache()
        reference = self._generate_ids(reference_ids, 1)["output_ids"]
        # Remove the reference prefix from radix cache. The post-abort request
        # must start from a newly initialized state rather than a cache hit.
        self._flush_cache()

        rid = f"rwkv7-abort-{uuid.uuid4().hex}"
        result: dict[str, Any] = {}

        def run_long_request():
            try:
                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "rid": rid,
                        "input_ids": [4300 + (i % 2048) for i in range(2048)],
                        "sampling_params": {
                            "temperature": 0.0,
                            "max_new_tokens": 4096,
                            "ignore_eos": True,
                        },
                    },
                    timeout=120,
                )
                result["status_code"] = response.status_code
                try:
                    result["body"] = response.json()
                except ValueError:
                    result["body"] = response.text
            except requests.RequestException as exc:
                result["exception"] = repr(exc)

        thread = threading.Thread(target=run_long_request)
        thread.start()
        time.sleep(0.5)
        deadline = time.monotonic() + 8
        while thread.is_alive() and time.monotonic() < deadline:
            response = requests.post(
                f"{self.base_url}/abort_request",
                json={"rid": rid, "abort_all": False},
                timeout=10,
            )
            response.raise_for_status()
            time.sleep(0.2)

        thread.join(timeout=30)
        self.assertFalse(thread.is_alive(), "aborted RWKV request did not terminate")
        self.assertNotIn("exception", result, result.get("exception"))
        self.assertEqual(result.get("status_code"), 200, result)
        body = result.get("body")
        if isinstance(body, list):
            body = body[0]
        finish_reason = (
            body.get("meta_info", {}).get("finish_reason", {})
            if isinstance(body, dict)
            else {}
        )
        self.assertEqual(finish_reason.get("type"), "abort", result)

        after_abort = self._generate_ids(reference_ids, 1)
        self.assertEqual(after_abort["meta_info"]["cached_tokens"], 0)
        self.assertEqual(
            after_abort["output_ids"],
            reference,
            "the request after abort inherited stale recurrent state",
        )


if __name__ == "__main__":
    unittest.main()
