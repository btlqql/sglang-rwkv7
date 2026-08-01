#!/usr/bin/env python3
"""Verify RWKV-7 serving invariants against an already-running SGLang server."""

import argparse
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--single-batch-prefix-tokens",
        type=int,
        default=4,
        help=(
            "Require this many leading tokens to match between single and batched "
            "generation; use 0 for quantized paths with shape-dependent drift."
        ),
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        help="Compare the deterministic output with a JSON reference file.",
    )
    parser.add_argument(
        "--write-reference",
        type=Path,
        help="Write the deterministic output IDs for a later TP/PP comparison.",
    )
    parser.add_argument(
        "--skip-lifecycle",
        action="store_true",
        help="Skip mixed-length compaction and explicit-abort state-reuse gates.",
    )
    parser.add_argument(
        "--repeat-prefix-tokens",
        type=int,
        default=0,
        help=(
            "Compare only this many leading greedy tokens across separate "
            "requests. Zero requires the complete output to match. Quantized "
            "acceptance may use a short prefix because near-tied logits can "
            "amplify a one-ULP kernel difference over long recurrence."
        ),
    )
    return parser.parse_args()


class Client:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def flush_cache(self):
        response = requests.post(
            f"{self.base_url}/flush_cache",
            params={"timeout": min(30, self.timeout)},
            timeout=self.timeout,
        )
        response.raise_for_status()

    def generate(self, input_ids, max_new_tokens: int, rid: str | None = None):
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        }
        if rid is not None:
            payload["rid"] = rid
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def abort(self, rid: str):
        response = requests.post(
            f"{self.base_url}/abort_request",
            json={"rid": rid, "abort_all": False},
            timeout=min(10, self.timeout),
        )
        response.raise_for_status()


def as_list(response):
    return response if isinstance(response, list) else [response]


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def equal_ids(left, right, prefix_tokens: int = 0):
    if prefix_tokens:
        return left[:prefix_tokens] == right[:prefix_tokens]
    return left == right


def matching_prefix_length(left, right) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))


def decode_response(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def is_abort_result(status_code: int, body: Any) -> bool:
    if status_code == 200:
        item = as_list(body)[0] if isinstance(body, (dict, list)) else {}
        reason = item.get("meta_info", {}).get("finish_reason", {})
        return isinstance(reason, dict) and reason.get("type") == "abort"
    if status_code not in (500, 503):
        return False
    return "abort" in str(body).lower()


def verify_lifecycle(client: Client, compare_prefix_tokens: int = 0):
    prompt_a = [300 + index for index in range(64)]
    prompt_b = [1300 + index for index in range(96)]
    prompt_c = [2300 + index for index in range(128)]
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
        return as_list(client.generate(input_ids, max_new_tokens))[0]["output_ids"]

    client.flush_cache()
    with ThreadPoolExecutor(max_workers=len(work)) as executor:
        outputs = list(executor.map(generate, work))
    require(
        [len(output) for output in outputs] == [16, 8, 64, 32, 64],
        "mixed-length requests returned unexpected output lengths",
    )
    require(
        equal_ids(outputs[0], outputs[2][:16], compare_prefix_tokens),
        "duplicate prompt diverged before mixed-length batch compaction",
    )
    require(
        equal_ids(outputs[2], outputs[4], compare_prefix_tokens),
        "surviving duplicate requests diverged while the batch compacted",
    )

    reference_ids = [100 + index for index in range(32)]
    client.flush_cache()
    reference = as_list(client.generate(reference_ids, 1))[0]["output_ids"]
    client.flush_cache()

    rid = f"rwkv7-serving-abort-{uuid.uuid4().hex}"
    result: dict[str, Any] = {}

    def run_long_request():
        try:
            response = requests.post(
                f"{client.base_url}/generate",
                json={
                    "rid": rid,
                    "input_ids": [4300 + (index % 2048) for index in range(2048)],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 4096,
                        "ignore_eos": True,
                    },
                },
                timeout=client.timeout,
            )
            result["status_code"] = response.status_code
            result["body"] = decode_response(response)
        except requests.RequestException as exc:
            result["exception"] = repr(exc)

    thread = threading.Thread(target=run_long_request)
    thread.start()
    time.sleep(0.5)
    deadline = time.monotonic() + 8
    while thread.is_alive() and time.monotonic() < deadline:
        client.abort(rid)
        time.sleep(0.2)
    thread.join(timeout=30)
    require(not thread.is_alive(), "aborted request did not terminate")
    require("exception" not in result, f"aborted request failed: {result}")
    require(
        is_abort_result(result.get("status_code", 0), result.get("body")),
        f"request did not report a clean abort: {result}",
    )

    after_abort = as_list(client.generate(reference_ids, 1))[0]
    require(
        after_abort["meta_info"]["cached_tokens"] == 0,
        "post-abort control request unexpectedly hit the state cache",
    )
    require(
        after_abort["output_ids"] == reference,
        "post-abort request inherited stale recurrent state",
    )
    return {
        "mixed_length_dynamic_batch_compaction": True,
        "abort_terminated": True,
        "post_abort_state_reuse_clean": True,
    }


def main():
    args = parse_args()
    client = Client(args.base_url, args.timeout)

    prompt_a = [100 + index for index in range(32)]
    prompt_b = [1000 + index for index in range(32)]

    client.flush_cache()
    first = as_list(client.generate(prompt_a, args.max_new_tokens))[0]
    second = as_list(client.generate(prompt_a, args.max_new_tokens))[0]
    deterministic_ids = first["output_ids"]
    repeat_tokens = args.repeat_prefix_tokens
    require(
        equal_ids(deterministic_ids, second["output_ids"], repeat_tokens),
        "identical greedy requests produced different token IDs",
    )

    if args.reference_output:
        expected = json.loads(args.reference_output.read_text())
        require(
            deterministic_ids == expected,
            "output IDs differ from the supplied TP/PP reference",
        )
    if args.write_reference:
        args.write_reference.write_text(json.dumps(deterministic_ids) + "\n")

    batched = as_list(
        client.generate([prompt_a, prompt_b, prompt_a, prompt_b], args.max_new_tokens)
    )
    batch_ids = [item["output_ids"] for item in batched]
    require(
        equal_ids(batch_ids[0], batch_ids[2], repeat_tokens),
        "duplicate prompt A leaked recurrent state",
    )
    require(
        equal_ids(batch_ids[1], batch_ids[3], repeat_tokens),
        "duplicate prompt B leaked recurrent state",
    )
    prefix_tokens = args.single_batch_prefix_tokens
    single_batch_prefix_match = (
        deterministic_ids[:prefix_tokens] == batch_ids[0][:prefix_tokens]
        if prefix_tokens
        else None
    )
    if prefix_tokens:
        require(
            single_batch_prefix_match,
            f"single and batched prompt A diverged in the first {prefix_tokens} tokens",
        )

    prefix_ids = [200 + index for index in range(128)]
    full_ids = prefix_ids + [1200 + index for index in range(64)]
    client.flush_cache()
    cold = as_list(client.generate(full_ids, args.max_new_tokens))[0]
    require(cold["meta_info"]["cached_tokens"] == 0, "cold request hit cache")

    client.flush_cache()
    client.generate(prefix_ids, 0)
    warm = as_list(client.generate(full_ids, args.max_new_tokens))[0]
    require(
        warm["meta_info"]["cached_tokens"] == len(prefix_ids),
        "state cache did not restore the complete prefix",
    )
    cache_matching_prefix = matching_prefix_length(
        cold["output_ids"], warm["output_ids"]
    )
    require(
        cold["output_ids"] == warm["output_ids"],
        "cached continuation differs from cold chunked prefill "
        f"after {cache_matching_prefix} matching tokens",
    )

    lifecycle = {} if args.skip_lifecycle else verify_lifecycle(client, repeat_tokens)

    result = {
        "base_url": args.base_url,
        "max_new_tokens": args.max_new_tokens,
        "deterministic": repeat_tokens == 0,
        "repeat_prefix_match": True,
        "repeat_prefix_tokens": repeat_tokens,
        "dynamic_batch_duplicate_isolation": True,
        "single_batch_prefix_tokens": prefix_tokens,
        "single_batch_prefix_match": single_batch_prefix_match,
        "chunked_prefill_cold_warm_match": True,
        "chunked_prefill_cold_warm_matching_prefix": cache_matching_prefix,
        "state_cache_hit_tokens": warm["meta_info"]["cached_tokens"],
        "reference_match": args.reference_output is not None,
        "reference_output_ids": deterministic_ids,
        **lifecycle,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
