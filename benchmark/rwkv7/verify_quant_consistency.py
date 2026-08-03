#!/usr/bin/env python3
"""Measure quantized SGLang logits on a fixed dense reference continuation.

Free-running greedy sequences are a poor quantization metric: one early token
change puts the two implementations on different contexts. This harness first
generates a continuation with an independent dense model, then asks the
already-running SGLang server to score that exact continuation. It reports
chosen-token log-probability error, top-k overlap, and teacher-forced top-1
agreement without hiding the free-running matching-prefix diagnostic.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = [
    "The capital of France is",
    "1 + 2 + 3 + 4 + 5 =",
    "Once upon a time, in a small village by the sea,",
    "The quick brown fox",
]


@dataclass
class Reference:
    prompt: str
    prompt_ids: list[int]
    output_ids: list[int]
    token_logprobs: list[float]
    top_ids: list[list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-token-logprob-error", type=float, default=0.25)
    parser.add_argument("--min-top-k-overlap", type=float, default=0.80)
    parser.add_argument("--min-top1-agreement", type=float, default=0.90)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument(
        "--reference-input",
        type=Path,
        help="Reuse a dense reference generated in a separate process.",
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        help="Persist the dense reference continuation and logits for later scoring.",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Generate --reference-output without contacting an SGLang server.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reference_only and args.reference_output is None:
        parser.error("--reference-only requires --reference-output")
    if args.reference_only and args.reference_input is not None:
        parser.error("--reference-only cannot be combined with --reference-input")
    return args


def load_reference_model(model_path: str, dtype: torch.dtype):
    kwargs = {"trust_remote_code": True, "dtype": dtype}
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except TypeError:
        kwargs.pop("dtype")
        kwargs["torch_dtype"] = dtype
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def build_references(args: argparse.Namespace) -> list[Reference]:
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_reference_model(args.model, dtype).cuda().eval()
    references = []
    for prompt in args.prompts or DEFAULT_PROMPTS:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.cuda()
        attention_mask = encoded.attention_mask.cuda()
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=0,
            )
        generated = output.sequences[0, input_ids.shape[1] :].tolist()
        token_logprobs = []
        top_ids = []
        for token_id, scores in zip(generated, output.scores):
            logprobs = scores[0].float().log_softmax(dim=-1)
            token_logprobs.append(float(logprobs[token_id].cpu()))
            top_ids.append(logprobs.topk(args.top_k).indices.cpu().tolist())
        references.append(
            Reference(
                prompt=prompt,
                prompt_ids=input_ids[0].cpu().tolist(),
                output_ids=generated,
                token_logprobs=token_logprobs,
                top_ids=top_ids,
            )
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return references


def save_references(
    path: Path, args: argparse.Namespace, references: list[Reference]
) -> None:
    payload = {
        "schema": "rwkv7-quant-reference-v1",
        "model": args.model,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "references": [reference.__dict__ for reference in references],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_references(path: Path, args: argparse.Namespace) -> list[Reference]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "rwkv7-quant-reference-v1":
        raise ValueError(
            f"{path}: unsupported reference schema {payload.get('schema')!r}"
        )
    expected = {
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{path}: reference configuration mismatch: {mismatches}")
    references = [Reference(**row) for row in payload.get("references", [])]
    if not references:
        raise ValueError(f"{path}: reference set is empty")
    for reference in references:
        count = len(reference.output_ids)
        if count != len(reference.token_logprobs) or count != len(reference.top_ids):
            raise ValueError(
                f"{path}: inconsistent reference lengths for prompt {reference.prompt!r}"
            )
        if any(len(ids) != args.top_k for ids in reference.top_ids):
            raise ValueError(
                f"{path}: top-k width mismatch for prompt {reference.prompt!r}"
            )
    return references


def score_reference(
    args: argparse.Namespace, reference: Reference
) -> dict[str, object]:
    teacher_ids = reference.prompt_ids + reference.output_ids
    response = requests.post(
        f"{args.base_url}/generate",
        json={
            "input_ids": teacher_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "top_logprobs_num": args.top_k,
            "logprob_start_len": 0,
        },
        timeout=args.timeout,
    )
    response.raise_for_status()
    meta = response.json()["meta_info"]
    offset = len(reference.prompt_ids)
    scored_logprobs = [row[0] for row in meta["input_token_logprobs"][offset:]]
    scored_top_ids = [
        [entry[1] for entry in row] for row in meta["input_top_logprobs"][offset:]
    ]
    if not (len(scored_logprobs) == len(scored_top_ids) == len(reference.output_ids)):
        raise RuntimeError(
            "Teacher-forced response length mismatch: "
            f"tokens={len(reference.output_ids)} logprobs={len(scored_logprobs)} "
            f"top_ids={len(scored_top_ids)}"
        )

    errors = [
        abs(expected - actual)
        for expected, actual in zip(reference.token_logprobs, scored_logprobs)
    ]
    overlaps = [
        len(set(expected) & set(actual)) / args.top_k
        for expected, actual in zip(reference.top_ids, scored_top_ids)
    ]
    top1 = [
        bool(actual and actual[0] == expected)
        for expected, actual in zip(reference.output_ids, scored_top_ids)
    ]

    free_response = requests.post(
        f"{args.base_url}/generate",
        json={
            "input_ids": reference.prompt_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
                "ignore_eos": True,
            },
        },
        timeout=args.timeout,
    )
    free_response.raise_for_status()
    free_ids = free_response.json()["output_ids"]
    matching_prefix = 0
    for expected, actual in zip(reference.output_ids, free_ids):
        if expected != actual:
            break
        matching_prefix += 1

    return {
        "prompt": reference.prompt,
        "tokens": len(reference.output_ids),
        "max_token_logprob_error": max(errors),
        "mean_token_logprob_error": sum(errors) / len(errors),
        "mean_top_k_overlap": sum(overlaps) / len(overlaps),
        "teacher_forced_top1_agreement": sum(top1) / len(top1),
        "free_running_matching_prefix": matching_prefix,
        "free_running_exact": free_ids == reference.output_ids,
    }


def main() -> None:
    args = parse_args()
    if args.reference_input:
        references = load_references(args.reference_input, args)
    else:
        references = build_references(args)
    if args.reference_output:
        save_references(args.reference_output, args, references)
    if args.reference_only:
        print(
            json.dumps(
                {
                    "schema": "rwkv7-quant-reference-result-v1",
                    "model": args.model,
                    "reference_output": str(args.reference_output),
                    "prompts": len(references),
                    "tokens": sum(
                        len(reference.output_ids) for reference in references
                    ),
                },
                indent=2,
            )
        )
        return
    results = [score_reference(args, reference) for reference in references]
    total_tokens = sum(int(row["tokens"]) for row in results)
    summary = {
        "max_token_logprob_error": max(
            float(row["max_token_logprob_error"]) for row in results
        ),
        "mean_token_logprob_error": sum(
            float(row["mean_token_logprob_error"]) * int(row["tokens"])
            for row in results
        )
        / total_tokens,
        "mean_top_k_overlap": sum(
            float(row["mean_top_k_overlap"]) * int(row["tokens"]) for row in results
        )
        / total_tokens,
        "teacher_forced_top1_agreement": sum(
            float(row["teacher_forced_top1_agreement"]) * int(row["tokens"])
            for row in results
        )
        / total_tokens,
    }
    passed = bool(
        summary["max_token_logprob_error"] <= args.max_token_logprob_error
        and summary["mean_top_k_overlap"] >= args.min_top_k_overlap
        and summary["teacher_forced_top1_agreement"] >= args.min_top1_agreement
    )
    report = {
        "schema": "rwkv7-quant-consistency-v1",
        "model": args.model,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "thresholds": {
            "max_token_logprob_error": args.max_token_logprob_error,
            "min_top_k_overlap": args.min_top_k_overlap,
            "min_top1_agreement": args.min_top1_agreement,
        },
        "passed": passed,
        "summary": summary,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
