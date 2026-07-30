#!/usr/bin/env python3
"""Reproducible fixed-batch serving benchmark for RWKV-7."""

import argparse
import json
import statistics
import time

import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--no-flush-cache", action="store_true")
    return parser.parse_args()


def make_payload(args):
    # Every row is different, deterministic, and contains no special token 0/1.
    modulus = args.vocab_size - 2
    input_ids = [
        [
            2 + ((1000 + row * 257 + column) % modulus)
            for column in range(args.input_len)
        ]
        for row in range(args.batch_size)
    ]
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.output_len,
            "ignore_eos": True,
        },
    }


def flush_cache(base_url, timeout):
    response = requests.post(
        f"{base_url}/flush_cache", params={"timeout": 30}, timeout=timeout
    )
    response.raise_for_status()


def run_once(args, payload):
    if not args.no_flush_cache:
        flush_cache(args.base_url, args.timeout)

    start = time.perf_counter()
    response = requests.post(
        f"{args.base_url}/generate", json=payload, timeout=args.timeout
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    outputs = response.json()
    if not isinstance(outputs, list):
        outputs = [outputs]

    output_tokens = sum(len(item["output_ids"]) for item in outputs)
    expected_tokens = args.batch_size * args.output_len
    if output_tokens != expected_tokens:
        raise RuntimeError(
            f"Expected {expected_tokens} output tokens, got {output_tokens}"
        )
    return {
        "elapsed_s": elapsed,
        "output_tokens": output_tokens,
        "output_throughput_tok_s": output_tokens / elapsed,
    }


def main():
    args = parse_args()
    payload = make_payload(args)

    for _ in range(args.warmups):
        run_once(args, payload)
    samples = [run_once(args, payload) for _ in range(args.repeats)]
    throughputs = [item["output_throughput_tok_s"] for item in samples]
    result = {
        "base_url": args.base_url,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "cache_flushed_each_run": not args.no_flush_cache,
        "median_output_throughput_tok_s": statistics.median(throughputs),
        "samples": samples,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
