#!/usr/bin/env python3
"""Verify RWKV-7 serving invariants against an already-running SGLang server."""

import argparse
import json
from pathlib import Path

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

    def generate(self, input_ids, max_new_tokens: int):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


def as_list(response):
    return response if isinstance(response, list) else [response]


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def main():
    args = parse_args()
    client = Client(args.base_url, args.timeout)

    prompt_a = [100 + index for index in range(32)]
    prompt_b = [1000 + index for index in range(32)]

    client.flush_cache()
    first = as_list(client.generate(prompt_a, args.max_new_tokens))[0]
    second = as_list(client.generate(prompt_a, args.max_new_tokens))[0]
    deterministic_ids = first["output_ids"]
    require(
        deterministic_ids == second["output_ids"],
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
    require(batch_ids[0] == batch_ids[2], "duplicate prompt A leaked recurrent state")
    require(batch_ids[1] == batch_ids[3], "duplicate prompt B leaked recurrent state")
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
    require(
        cold["output_ids"] == warm["output_ids"],
        "cached continuation differs from cold chunked prefill",
    )

    result = {
        "base_url": args.base_url,
        "max_new_tokens": args.max_new_tokens,
        "deterministic": True,
        "dynamic_batch_duplicate_isolation": True,
        "single_batch_prefix_tokens": prefix_tokens,
        "single_batch_prefix_match": single_batch_prefix_match,
        "chunked_prefill_cold_warm_match": True,
        "state_cache_hit_tokens": warm["meta_info"]["cached_tokens"],
        "reference_match": args.reference_output is not None,
        "reference_output_ids": deterministic_ids,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
