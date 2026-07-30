#!/usr/bin/env python3
"""Build a fail-closed RWKV-7 serving acceptance report.

Join SGLang candidate JSONL with matched Qwen3.5 serving JSONL, raw Albatross
``RESULT B=... T=...`` logs, and dense RWKV-7 rows used by quantization gates.
Missing candidate or baseline data remains visible in the denominator.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

FLOAT_RE = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?"
ALBATROSS_RESULT_RE = re.compile(
    r"RESULT\s+B=(?P<batch>\d+)\s+T=(?P<tokens>\d+)\s+"
    r"iters=(?P<iters>\d+)\s+"
    rf"p10_ms=(?P<p10>{FLOAT_RE})\s+"
    rf"p50_ms=(?P<p50>{FLOAT_RE})\s+"
    rf"p90_ms=(?P<p90>{FLOAT_RE})\s+"
    rf"tok_s_p50=(?P<tokps>{FLOAT_RE})"
)


@dataclass(frozen=True, order=True)
class CellKey:
    model: str
    mode: str
    batch_size: int
    prompt_tokens: int
    decode_tokens: int


def csv_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def csv_ints(value: str) -> list[int]:
    try:
        values = [int(item) for item in csv_strings(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("matrix dimensions must be positive")
    return values


def parse_mapping(values: Iterable[str], *, value_name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected MODEL={value_name}, got {value!r}")
        model, mapped = (item.strip() for item in value.split("=", 1))
        if not model or not mapped:
            raise ValueError(f"expected MODEL={value_name}, got {value!r}")
        if model in result:
            raise ValueError(f"duplicate mapping for model {model!r}")
        result[model] = mapped
    return result


def parse_float_mapping(values: Iterable[str], *, value_name: str) -> dict[str, float]:
    raw = parse_mapping(values, value_name=value_name)
    result = {model: float(value) for model, value in raw.items()}
    if any(value <= 0 for value in result.values()):
        raise ValueError(f"{value_name} values must be positive")
    return result


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if row.get("schema") != "rwkv7-serving-acceptance-v1":
                raise ValueError(
                    f"{path}:{line_number}: unsupported schema {row.get('schema')!r}"
                )
            rows.append(row)
    return rows


def serving_key(row: dict[str, Any]) -> CellKey:
    return CellKey(
        model=str(row["model"]),
        mode=str(row["mode"]),
        batch_size=int(row["batch_size"]),
        prompt_tokens=int(row["prompt_tokens_per_request"]),
        decode_tokens=int(row["decode_tokens_per_request"]),
    )


def index_serving(rows: Iterable[dict[str, Any]]) -> dict[CellKey, dict[str, Any]]:
    """Index rows with last-row-wins semantics for intentional reruns."""
    return {serving_key(row): row for row in rows}


def parse_albatross_log(path: Path) -> dict[tuple[int, int], dict[str, float | int]]:
    rows: dict[tuple[int, int], dict[str, float | int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ALBATROSS_RESULT_RE.search(line)
        if not match:
            continue
        data = match.groupdict()
        batch, tokens = int(data["batch"]), int(data["tokens"])
        rows[(batch, tokens)] = {
            "batch_size": batch,
            "tokens_per_sequence": tokens,
            "iters": int(data["iters"]),
            "latency_p50_ms": float(data["p50"]),
            "tok_s_p50": float(data["tokps"]),
        }
    if not rows:
        raise ValueError(f"{path}: no Albatross RESULT rows found")
    return rows


def median(row: dict[str, Any], metric: str) -> float:
    return float(row["median"][metric])


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError(f"ratio denominator must be positive, got {denominator}")
    return numerator / denominator


def add_gate(
    gates: dict[str, dict[str, float | bool]],
    name: str,
    value: float,
    minimum: float,
) -> None:
    gates[name] = {"value": value, "minimum": minimum, "pass": value >= minimum}


def expected_keys(
    *,
    models: Iterable[str],
    modes: Iterable[str],
    batch_sizes: Iterable[int],
    prompt_lengths: Iterable[int],
    decode_lengths: Iterable[int],
) -> list[CellKey]:
    return [
        CellKey(model, mode, batch, prompt, decode)
        for model in models
        for mode in modes
        for batch in batch_sizes
        for prompt in prompt_lengths
        for decode in decode_lengths
    ]


def _matched_qwen_row(
    rows: dict[CellKey, dict[str, Any]], key: CellKey
) -> dict[str, Any] | None:
    matches = [
        row
        for qwen_key, row in rows.items()
        if qwen_key.batch_size == key.batch_size
        and qwen_key.prompt_tokens == key.prompt_tokens
        and qwen_key.decode_tokens == key.decode_tokens
    ]
    return matches[0] if len(matches) == 1 else None


def analyze(
    *,
    candidate_rows: Iterable[dict[str, Any]],
    qwen_rows: dict[str, Iterable[dict[str, Any]]],
    albatross_rows: dict[str, dict[tuple[int, int], dict[str, float | int]]],
    models: list[str],
    modes: list[str],
    batch_sizes: list[int],
    prompt_lengths: list[int],
    decode_lengths: list[int],
    dense_mode: str,
    qwen_minimum: float,
    albatross_minimum: float,
    quant_minimum: float,
    active_work_factors: dict[str, float],
    active_work_minimums: dict[str, float],
    require_active_work: bool,
) -> dict[str, Any]:
    candidate = index_serving(candidate_rows)
    qwen = {model: index_serving(rows) for model, rows in qwen_rows.items()}
    keys = expected_keys(
        models=models,
        modes=modes,
        batch_sizes=batch_sizes,
        prompt_lengths=prompt_lengths,
        decode_lengths=decode_lengths,
    )

    report_rows: list[dict[str, Any]] = []
    for key in keys:
        row_report: dict[str, Any] = {
            "model": key.model,
            "mode": key.mode,
            "batch_size": key.batch_size,
            "prompt_tokens_per_request": key.prompt_tokens,
            "decode_tokens_per_request": key.decode_tokens,
        }
        current = candidate.get(key)
        if current is None:
            row_report.update(status="missing", missing=["candidate"])
            report_rows.append(row_report)
            continue

        missing: list[str] = []
        gates: dict[str, dict[str, float | bool]] = {}
        qwen_index = qwen.get(key.model)
        qwen_row = None if qwen_index is None else _matched_qwen_row(qwen_index, key)
        if qwen_index is None:
            missing.append("qwen_mapping")
        elif qwen_row is None:
            missing.append("qwen_cell")

        if qwen_row is not None:
            qwen_prefill = ratio(
                median(current, "prefill_tok_s"), median(qwen_row, "prefill_tok_s")
            )
            qwen_decode = ratio(
                median(current, "decode_tok_s"), median(qwen_row, "decode_tok_s")
            )
            add_gate(gates, "qwen_prefill_ratio", qwen_prefill, qwen_minimum)
            add_gate(gates, "qwen_decode_ratio", qwen_decode, qwen_minimum)
            add_gate(
                gates,
                "qwen_e2e_ratio",
                ratio(
                    median(current, "e2e_output_tok_s"),
                    median(qwen_row, "e2e_output_tok_s"),
                ),
                qwen_minimum,
            )
            add_gate(
                gates,
                "qwen_ttft_ratio",
                ratio(median(qwen_row, "ttft_s"), median(current, "ttft_s")),
                qwen_minimum,
            )
            add_gate(
                gates,
                "qwen_tpot_ratio",
                ratio(median(qwen_row, "tpot_s"), median(current, "tpot_s")),
                qwen_minimum,
            )
            factor = active_work_factors.get(key.model)
            minimum = active_work_minimums.get(key.model)
            if factor is not None and minimum is not None:
                add_gate(gates, "active_work_decode", qwen_decode * factor, minimum)
            elif require_active_work:
                missing.append("active_work_config")

        albatross = albatross_rows.get(key.model)
        if albatross is None:
            missing.append("albatross_mapping")
        else:
            prefill_reference = albatross.get((key.batch_size, key.prompt_tokens))
            decode_reference = albatross.get((key.batch_size, 1))
            if prefill_reference is None:
                missing.append("albatross_prefill_cell")
            else:
                add_gate(
                    gates,
                    "albatross_prefill_ratio",
                    ratio(
                        median(current, "prefill_tok_s"),
                        float(prefill_reference["tok_s_p50"]),
                    ),
                    albatross_minimum,
                )
            if decode_reference is None:
                missing.append("albatross_decode_cell")
            else:
                add_gate(
                    gates,
                    "albatross_decode_ratio",
                    ratio(
                        median(current, "decode_tok_s"),
                        float(decode_reference["tok_s_p50"]),
                    ),
                    albatross_minimum,
                )

        if key.mode != dense_mode:
            dense = candidate.get(
                CellKey(
                    key.model,
                    dense_mode,
                    key.batch_size,
                    key.prompt_tokens,
                    key.decode_tokens,
                )
            )
            if dense is None:
                missing.append("dense_quant_reference")
            else:
                for metric, gate_name in (
                    ("prefill_tok_s", "quant_dense_prefill_ratio"),
                    ("decode_tok_s", "quant_dense_decode_ratio"),
                    ("e2e_output_tok_s", "quant_dense_e2e_ratio"),
                ):
                    add_gate(
                        gates,
                        gate_name,
                        ratio(median(current, metric), median(dense, metric)),
                        quant_minimum,
                    )

        failures = sorted(name for name, gate in gates.items() if not gate["pass"])
        status = "missing" if missing else "failed" if failures else "passed"
        row_report.update(
            status=status,
            missing=sorted(set(missing)),
            failed_gates=failures,
            gates=gates,
        )
        report_rows.append(row_report)

    counts = {status: 0 for status in ("passed", "failed", "missing")}
    for row in report_rows:
        counts[row["status"]] += 1
    return {
        "schema": "rwkv7-serving-acceptance-report-v1",
        "matrix": {
            "models": models,
            "modes": modes,
            "batch_sizes": batch_sizes,
            "prompt_lengths": prompt_lengths,
            "decode_lengths": decode_lengths,
            "expected_cells": len(keys),
        },
        "thresholds": {
            "qwen_minimum": qwen_minimum,
            "albatross_minimum": albatross_minimum,
            "quant_dense_minimum": quant_minimum,
            "active_work_factors": active_work_factors,
            "active_work_minimums": active_work_minimums,
            "require_active_work": require_active_work,
        },
        "summary": counts,
        "passed": counts["failed"] == 0 and counts["missing"] == 0,
        "cells": report_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument(
        "--qwen", action="append", default=[], metavar="RWKV_MODEL=JSONL"
    )
    parser.add_argument(
        "--albatross", action="append", default=[], metavar="RWKV_MODEL=LOG"
    )
    parser.add_argument(
        "--models",
        type=csv_strings,
        default=["rwkv7-g1-1.5b", "rwkv7-g1-2.9b", "rwkv7-g1-7.2b"],
    )
    parser.add_argument(
        "--modes",
        type=csv_strings,
        default=["dense", "w8-accuracy", "w4-hybrid-accuracy"],
    )
    parser.add_argument("--batch-sizes", type=csv_ints, default=[1, 2, 4, 8])
    parser.add_argument("--prompt-lengths", type=csv_ints, default=[128, 512, 2048])
    parser.add_argument("--decode-lengths", type=csv_ints, default=[128, 512])
    parser.add_argument("--dense-mode", default="dense")
    parser.add_argument("--qwen-minimum", type=float, default=1.0)
    parser.add_argument("--albatross-minimum", type=float, default=1.0)
    parser.add_argument("--quant-minimum", type=float, default=1.0)
    parser.add_argument(
        "--active-work-factor", action="append", default=[], metavar="MODEL=FACTOR"
    )
    parser.add_argument(
        "--active-work-minimum", action="append", default=[], metavar="MODEL=MINIMUM"
    )
    parser.add_argument("--require-active-work", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qwen_paths = parse_mapping(args.qwen, value_name="JSONL")
    albatross_paths = parse_mapping(args.albatross, value_name="LOG")
    report = analyze(
        candidate_rows=load_jsonl(args.candidate),
        qwen_rows={
            model: load_jsonl([Path(path)]) for model, path in qwen_paths.items()
        },
        albatross_rows={
            model: parse_albatross_log(Path(path))
            for model, path in albatross_paths.items()
        },
        models=args.models,
        modes=args.modes,
        batch_sizes=args.batch_sizes,
        prompt_lengths=args.prompt_lengths,
        decode_lengths=args.decode_lengths,
        dense_mode=args.dense_mode,
        qwen_minimum=args.qwen_minimum,
        albatross_minimum=args.albatross_minimum,
        quant_minimum=args.quant_minimum,
        active_work_factors=parse_float_mapping(
            args.active_work_factor, value_name="FACTOR"
        ),
        active_work_minimums=parse_float_mapping(
            args.active_work_minimum, value_name="MINIMUM"
        ),
        require_active_work=args.require_active_work,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 1 if args.strict and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
