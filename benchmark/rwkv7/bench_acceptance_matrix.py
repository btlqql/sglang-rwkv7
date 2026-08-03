#!/usr/bin/env python3
"""Run the repository-standard RWKV-7 serving workload and emit JSONL evidence.

The client uses SGLang's streaming endpoint so prefill/TTFT and steady-state
decode can be measured separately. A batch is considered ready when every
request has produced its first token; decode covers the remaining N-1 steps.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests


def _csv_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--batch-sizes", type=_csv_ints, default=[8])
    parser.add_argument("--prompt-lengths", type=_csv_ints, default=[128, 512, 2048])
    parser.add_argument("--decode-lengths", type=_csv_ints, default=[128, 512])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="dense")
    parser.add_argument("--gpu", default="unknown")
    parser.add_argument("--implementation", default="sglang-rwkv7")
    parser.add_argument("--repo-sha", default="unknown")
    parser.add_argument("--contract-revision", default="rwkv7-acceptance-v1")
    parser.add_argument("--no-flush-cache", action="store_true")
    return parser.parse_args()


def make_payload(batch_size: int, input_len: int, output_len: int, vocab_size: int):
    modulus = vocab_size - 2
    input_ids = [
        [2 + ((1000 + row * 257 + column) % modulus) for column in range(input_len)]
        for row in range(batch_size)
    ]
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
    }


def flush_cache(base_url: str, timeout: float) -> None:
    response = requests.post(
        f"{base_url}/flush_cache", params={"timeout": 30}, timeout=timeout
    )
    response.raise_for_status()


def run_once(
    args: argparse.Namespace,
    batch_size: int,
    input_len: int,
    output_len: int,
) -> dict[str, float]:
    if not args.no_flush_cache:
        flush_cache(args.base_url, args.timeout)
    payload = make_payload(batch_size, input_len, output_len, args.vocab_size)
    first_token_at: dict[int, float] = {}
    completed_at: dict[int, float] = {}
    final_token_counts: dict[int, int] = {}

    start = time.perf_counter()
    response = requests.post(
        f"{args.base_url}/generate",
        json=payload,
        timeout=args.timeout,
        stream=True,
    )
    response.raise_for_status()
    for raw_line in response.iter_lines():
        if not raw_line or raw_line == b"data: [DONE]":
            continue
        if not raw_line.startswith(b"data: "):
            continue
        now = time.perf_counter()
        event: dict[str, Any] = json.loads(raw_line[6:])
        index = int(event.get("index", 0))
        meta = event["meta_info"]
        completion_tokens = int(meta.get("completion_tokens", 0))
        if completion_tokens >= 1:
            first_token_at.setdefault(index, now)
        final_token_counts[index] = completion_tokens
        if meta.get("finish_reason") is not None:
            completed_at[index] = now

    end = time.perf_counter()
    expected_indexes = set(range(batch_size))
    if set(first_token_at) != expected_indexes or set(completed_at) != expected_indexes:
        raise RuntimeError(
            "Incomplete streaming response: "
            f"first={sorted(first_token_at)} completed={sorted(completed_at)}"
        )
    if any(final_token_counts.get(index) != output_len for index in expected_indexes):
        raise RuntimeError(
            f"Expected {output_len} tokens per request, got {final_token_counts}"
        )

    first_batch_ready = max(first_token_at.values())
    last_batch_done = max(completed_at.values())
    ttft_s = first_batch_ready - start
    e2e_s = max(end, last_batch_done) - start
    decode_steps = output_len - 1
    decode_s = last_batch_done - first_batch_ready if decode_steps else 0.0
    result = {
        "ttft_s": ttft_s,
        "prefill_tok_s": batch_size * input_len / ttft_s,
        "e2e_s": e2e_s,
        "e2e_output_tok_s": batch_size * output_len / e2e_s,
    }
    if decode_steps:
        result.update(
            {
                "decode_s": decode_s,
                "tpot_s": decode_s / decode_steps,
                "decode_tok_s": batch_size * decode_steps / decode_s,
            }
        )
    return result


def median_metrics(samples: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: statistics.median(sample[name] for sample in samples)
        for name in samples[0]
    }


def main() -> None:
    args = parse_args()
    rows = []
    for batch_size in args.batch_sizes:
        for input_len in args.prompt_lengths:
            for output_len in args.decode_lengths:
                for _ in range(args.warmups):
                    run_once(args, batch_size, input_len, output_len)
                samples = [
                    run_once(args, batch_size, input_len, output_len)
                    for _ in range(args.repeats)
                ]
                row = {
                    "schema": "rwkv7-serving-acceptance-v1",
                    "implementation": args.implementation,
                    "model": args.model,
                    "mode": args.mode,
                    "gpu": args.gpu,
                    "repo_sha": args.repo_sha,
                    "contract_revision": args.contract_revision,
                    "batch_size": batch_size,
                    "prompt_tokens_per_request": input_len,
                    "decode_tokens_per_request": output_len,
                    "warmups": args.warmups,
                    "repeats": args.repeats,
                    "cache_flushed_each_run": not args.no_flush_cache,
                    "median": median_metrics(samples),
                    "samples": samples,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
