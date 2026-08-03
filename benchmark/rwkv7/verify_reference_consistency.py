#!/usr/bin/env python3
"""Compare an independent dense RWKV-7 reference with SGLang serving."""

import argparse
import gc
import json
from dataclasses import dataclass

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
class ReferenceResult:
    prompt: str
    input_ids: list[int]
    output_ids: list[int]
    token_logprobs: list[float]
    top_ids: list[list[int]]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--chunked-prefill-size", type=int, default=64)
    parser.add_argument("--max-token-logprob-error", type=float, default=0.05)
    parser.add_argument("--min-top-k-overlap", type=float, default=0.8)
    parser.add_argument("--prompt", action="append", dest="prompts")
    return parser.parse_args()


def load_reference_model(model_path, dtype):
    kwargs = {"trust_remote_code": True, "dtype": dtype}
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except TypeError:
        kwargs.pop("dtype")
        kwargs["torch_dtype"] = dtype
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def build_references(args, tokenizer, dtype):
    model = load_reference_model(args.model, dtype).cuda().eval()
    results = []
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
        results.append(
            ReferenceResult(
                prompt=prompt,
                input_ids=input_ids[0].cpu().tolist(),
                output_ids=generated,
                token_logprobs=token_logprobs,
                top_ids=top_ids,
            )
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def compare_one(args, base_url, reference):
    response = requests.post(
        f"{base_url}/generate",
        json={
            "input_ids": reference.input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "top_logprobs_num": args.top_k,
            "logprob_start_len": -1,
        },
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()
    server_ids = result["output_ids"]
    meta = result["meta_info"]
    server_logprobs = [row[0] for row in meta["output_token_logprobs"]]
    server_top_ids = [
        [entry[1] for entry in row] for row in meta["output_top_logprobs"]
    ]

    prefix = 0
    for reference_token, server_token in zip(reference.output_ids, server_ids):
        if reference_token != server_token:
            break
        prefix += 1

    comparable_steps = min(
        prefix,
        len(reference.token_logprobs),
        len(server_logprobs),
        len(server_top_ids),
    )
    errors = [
        abs(reference.token_logprobs[i] - server_logprobs[i])
        for i in range(comparable_steps)
    ]
    overlaps = [
        len(set(reference.top_ids[i]) & set(server_top_ids[i])) / args.top_k
        for i in range(comparable_steps)
    ]
    return {
        "prompt": reference.prompt,
        "exact_greedy_match": server_ids == reference.output_ids,
        "matching_prefix_tokens": prefix,
        "reference_output_ids": reference.output_ids,
        "server_output_ids": server_ids,
        "comparable_steps": comparable_steps,
        "max_token_logprob_error": max(errors, default=None),
        "mean_token_logprob_error": sum(errors) / len(errors) if errors else None,
        "mean_top_k_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
    }


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    references = build_references(args, tokenizer, dtype)

    # Import SGLang only after the independent reference has been constructed.
    # SGLang registers its optimized rwkv7_native config globally; importing it
    # first would intentionally replace the remote config class and make
    # AutoModel's remote-code consistency check reject the reference model.
    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import popen_launch_server

    base_url = f"http://127.0.0.1:{args.port}"
    process = None
    try:
        process = popen_launch_server(
            args.model,
            base_url,
            timeout=900,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "triton",
                "--dtype",
                args.dtype,
                "--chunked-prefill-size",
                str(args.chunked_prefill_size),
                "--mem-fraction-static",
                "0.55",
                "--max-mamba-cache-size",
                "16",
                "--cuda-graph-max-bs-decode",
                "8",
                "--max-running-requests",
                "16",
                "--port",
                str(args.port),
            ],
        )
        results = [compare_one(args, base_url, item) for item in references]
    finally:
        if process is not None:
            kill_process_tree(process.pid)

    passed = all(
        item["exact_greedy_match"]
        and item["max_token_logprob_error"] is not None
        and item["max_token_logprob_error"] <= args.max_token_logprob_error
        and item["mean_top_k_overlap"] is not None
        and item["mean_top_k_overlap"] >= args.min_top_k_overlap
        for item in results
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "dtype": args.dtype,
                "max_new_tokens": args.max_new_tokens,
                "top_k": args.top_k,
                "passed": passed,
                "results": results,
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
