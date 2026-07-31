#!/usr/bin/env python3
"""Compare RWKV-7 STANDALONE speculation with a non-speculative server.

The two endpoints are measured with the same deterministic fixed-batch input.
Each measured speculative response must exactly match its baseline response.
The JSONL artifact records prefill, decode and end-to-end throughput together
with per-request draft acceptance counters and the server-wide accept length.
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
    parser.add_argument("--baseline-url", default="http://127.0.0.1:30000")
    parser.add_argument("--spec-url", default="http://127.0.0.1:30001")
    parser.add_argument("--batch-sizes", type=_csv_ints, default=[8])
    parser.add_argument("--prompt-lengths", type=_csv_ints, default=[128, 512, 2048])
    parser.add_argument("--decode-lengths", type=_csv_ints, default=[128, 512])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--mode", default="fp16")
    parser.add_argument("--gpu", default="unknown")
    parser.add_argument("--repo-sha", default="unknown")
    return parser.parse_args()


def make_payload(batch_size: int, input_len: int, output_len: int, vocab_size: int):
    modulus = vocab_size - 2
    return {
        "input_ids": [
            [2 + ((1000 + row * 257 + column) % modulus) for column in range(input_len)]
            for row in range(batch_size)
        ],
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
    base_url: str,
    payload: dict[str, Any],
    batch_size: int,
    input_len: int,
    output_len: int,
    timeout: float,
) -> dict[str, Any]:
    flush_cache(base_url, timeout)
    first_token_at: dict[int, float] = {}
    completed_at: dict[int, float] = {}
    output_ids: dict[int, list[int]] = {}
    final_meta: dict[int, dict[str, Any]] = {}

    start = time.perf_counter()
    response = requests.post(
        f"{base_url}/generate", json=payload, timeout=timeout, stream=True
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
        if int(meta.get("completion_tokens", 0)) >= 1:
            first_token_at.setdefault(index, now)
        if "output_ids" in event:
            output_ids[index] = event["output_ids"]
        if meta.get("finish_reason") is not None:
            completed_at[index] = now
            final_meta[index] = meta

    end = time.perf_counter()
    indexes = set(range(batch_size))
    if set(first_token_at) != indexes or set(completed_at) != indexes:
        raise RuntimeError(
            f"incomplete response from {base_url}: "
            f"first={sorted(first_token_at)}, completed={sorted(completed_at)}"
        )
    if any(len(output_ids.get(i, [])) != output_len for i in indexes):
        raise RuntimeError(
            f"expected {output_len} output tokens per request from {base_url}"
        )

    first_batch_ready = max(first_token_at.values())
    last_batch_done = max(completed_at.values())
    ttft_s = first_batch_ready - start
    e2e_s = max(end, last_batch_done) - start
    decode_steps = output_len - 1
    decode_s = last_batch_done - first_batch_ready if decode_steps else 0.0
    metrics: dict[str, float] = {
        "ttft_s": ttft_s,
        "prefill_tok_s": batch_size * input_len / ttft_s,
        "e2e_s": e2e_s,
        "e2e_output_tok_s": batch_size * output_len / e2e_s,
    }
    if decode_steps:
        metrics.update(
            {
                "decode_s": decode_s,
                "tpot_s": decode_s / decode_steps,
                "decode_tok_s": batch_size * decode_steps / decode_s,
            }
        )
    return {
        "metrics": metrics,
        "output_ids": [output_ids[i] for i in range(batch_size)],
        "meta_info": [final_meta[i] for i in range(batch_size)],
    }


def median_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
    metrics = [sample["metrics"] for sample in samples]
    return {
        name: statistics.median(sample[name] for sample in metrics)
        for name in metrics[0]
    }


def acceptance_summary(samples: list[dict[str, Any]]) -> dict[str, float]:
    metas = [meta for sample in samples for meta in sample["meta_info"]]
    proposed = sum(int(meta.get("spec_num_proposed_drafts", 0)) for meta in metas)
    correct = sum(int(meta.get("spec_num_correct_drafts", 0)) for meta in metas)
    verify_ct = sum(int(meta.get("spec_verify_ct", 0)) for meta in metas)
    completion = sum(int(meta.get("completion_tokens", 0)) for meta in metas)
    return {
        "num_proposed_drafts": proposed,
        "num_correct_drafts": correct,
        "accept_rate": correct / proposed if proposed else 0.0,
        "accept_length": completion / verify_ct if verify_ct else 0.0,
    }


def compact_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep evidence small after exact IDs and metadata have been checked."""
    return [
        {
            "metrics": sample["metrics"],
            "acceptance": acceptance_summary([sample]),
        }
        for sample in samples
    ]


def server_accept_length(base_url: str, timeout: float) -> float | None:
    response = requests.get(f"{base_url}/server_info", timeout=timeout)
    response.raise_for_status()
    for state in response.json().get("internal_states", []):
        if state.get("avg_spec_accept_length") is not None:
            return float(state["avg_spec_accept_length"])
    return None


def main() -> None:
    args = parse_args()
    rows = []
    for batch_size in args.batch_sizes:
        for input_len in args.prompt_lengths:
            for output_len in args.decode_lengths:
                payload = make_payload(
                    batch_size, input_len, output_len, args.vocab_size
                )
                for _ in range(args.warmups):
                    baseline = run_once(
                        args.baseline_url,
                        payload,
                        batch_size,
                        input_len,
                        output_len,
                        args.timeout,
                    )
                    speculative = run_once(
                        args.spec_url,
                        payload,
                        batch_size,
                        input_len,
                        output_len,
                        args.timeout,
                    )
                    if speculative["output_ids"] != baseline["output_ids"]:
                        raise RuntimeError("speculative warmup diverged from baseline")

                baseline_samples = []
                speculative_samples = []
                for _ in range(args.repeats):
                    baseline = run_once(
                        args.baseline_url,
                        payload,
                        batch_size,
                        input_len,
                        output_len,
                        args.timeout,
                    )
                    speculative = run_once(
                        args.spec_url,
                        payload,
                        batch_size,
                        input_len,
                        output_len,
                        args.timeout,
                    )
                    if speculative["output_ids"] != baseline["output_ids"]:
                        raise RuntimeError("speculative output diverged from baseline")
                    baseline_samples.append(baseline)
                    speculative_samples.append(speculative)

                baseline_median = median_metrics(baseline_samples)
                speculative_median = median_metrics(speculative_samples)
                row = {
                    "schema": "rwkv7-speculative-acceptance-v1",
                    "target_model": args.target_model,
                    "draft_model": args.draft_model,
                    "mode": args.mode,
                    "gpu": args.gpu,
                    "repo_sha": args.repo_sha,
                    "batch_size": batch_size,
                    "prompt_tokens_per_request": input_len,
                    "decode_tokens_per_request": output_len,
                    "warmups": args.warmups,
                    "repeats": args.repeats,
                    "greedy_output_exact": True,
                    "baseline_median": baseline_median,
                    "speculative_median": speculative_median,
                    "speedup": {
                        name: speculative_median[name] / baseline_median[name]
                        for name in (
                            "prefill_tok_s",
                            "decode_tok_s",
                            "e2e_output_tok_s",
                        )
                        if name in baseline_median
                    },
                    "acceptance": acceptance_summary(speculative_samples),
                    "server_avg_accept_length": server_accept_length(
                        args.spec_url, args.timeout
                    ),
                    "baseline_samples": [
                        sample["metrics"] for sample in baseline_samples
                    ],
                    "speculative_samples": compact_samples(speculative_samples),
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
