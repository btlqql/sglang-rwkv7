"""RWKV-7 projection-level mixed-precision policy.

This module owns policy selection and its environment overrides.  Keeping the
decision table separate from the model graph makes the effective FP16/W8/W4
layout testable without constructing an RWKV model or loading GPU kernels.
"""

from __future__ import annotations

import os
from typing import Optional

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.rwkv7_native import RWKV7W8Config

_RWKV7_W8A8_CONFIG: Optional[QuantizationConfig] = None
_RWKV7_NATIVE_W8_CONFIG: Optional[QuantizationConfig] = None


def _choice_env(
    name: str, default: str, choices: tuple[str, ...], choices_text: str
) -> str:
    value = os.getenv(name, default).lower()
    if value not in choices:
        raise ValueError(f"{name} must be {choices_text}; got {value!r}.")
    return value


def _non_negative_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}.") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative; got {value}.")
    return value


def _rwkv7_w8_policy() -> str:
    """Select the model-specific online W8A8 accuracy/compression trade-off."""
    return _choice_env(
        "SGLANG_RWKV7_W8_POLICY",
        "accuracy",
        ("accuracy", "balanced", "speed"),
        "accuracy, balanced, or speed",
    )


def _rwkv7_w4_policy() -> str:
    """Select the model-specific online Marlin accuracy/compression trade-off."""
    return _choice_env(
        "SGLANG_RWKV7_W4_POLICY",
        "accuracy",
        ("accuracy", "balanced", "speed", "sparse"),
        "accuracy, balanced, speed, or sparse",
    )


def _rwkv7_bnb_policy() -> str:
    """Select the BitsAndBytes accuracy/compression trade-off.

    Quantizing every recurrent projection is useful as a throughput-oriented
    lane, but it is a poor default for RWKV because projection error is carried
    in the recurrent state.  The accuracy lane quantizes only the large FFN
    contraction, while balanced additionally quantizes the FFN expansion.
    """
    return _choice_env(
        "SGLANG_RWKV7_BNB_POLICY",
        "accuracy",
        ("accuracy", "balanced", "speed"),
        "accuracy, balanced, or speed",
    )


def _rwkv7_bnb_target_modules(
    quant_config: QuantizationConfig, num_hidden_layers: int
) -> list[str]:
    """Return loader targets matching the projection policy."""
    policy = _rwkv7_bnb_policy()
    if policy == "speed":
        return [
            ".r_proj.",
            ".k_proj.",
            ".v_proj.",
            ".o_proj.",
            ".key.",
            ".value.",
        ]
    if policy == "balanced":
        return [".key.", ".value."]
    if quant_config.load_in_8bit:
        return [".value."]
    edge_layers = min(4, max(1, num_hidden_layers // 6))
    return [
        f".layers.{layer_id}.ffn.value."
        for layer_id in range(edge_layers, num_hidden_layers - edge_layers, 4)
    ]


def _rwkv7_marlin_fallback_max_tokens() -> int:
    """Token-count limit for the batch-invariant W4 accuracy shadow."""
    return _non_negative_int_env("SGLANG_RWKV7_MARLIN_FALLBACK_MAX_TOKENS", 512)


def _rwkv7_w4_shadow_mode(hidden_size: int) -> str:
    """Choose the small-M W4 accuracy shadow.

    The 2048-wide checkpoint has several near-tied greedy logits for which a
    per-channel W8 projection can amplify chunk-boundary rounding through the
    recurrence. Keep its exact FP16 shard by default. Wider checkpoints use
    the compact fused INT8 shadow; callers may override either choice for an
    explicit memory/accuracy experiment.
    """
    mode = os.getenv("SGLANG_RWKV7_W4_SHADOW", "auto").lower()
    if mode == "auto":
        return "fp16" if hidden_size <= 2048 else "int8"
    if mode not in ("fp16", "int8"):
        raise ValueError(
            f"SGLANG_RWKV7_W4_SHADOW must be auto, fp16, or int8; got {mode!r}."
        )
    return mode


def _rwkv7_int8_exact_max_tokens() -> int:
    """Token-count limit for batch-invariant INT32 W8A8 accumulation."""
    return _non_negative_int_env("SGLANG_RWKV7_INT8_EXACT_MAX_TOKENS", 512)


def _rwkv7_w8a8_config() -> QuantizationConfig:
    """Lazily import W8A8 so legacy CUDA dense/BnB paths need no sgl-kernel."""
    global _RWKV7_W8A8_CONFIG
    if _RWKV7_W8A8_CONFIG is None:
        from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

        _RWKV7_W8A8_CONFIG = W8A8Int8Config()
    return _RWKV7_W8A8_CONFIG


def _rwkv7_native_w8_config() -> QuantizationConfig:
    global _RWKV7_NATIVE_W8_CONFIG
    if _RWKV7_NATIVE_W8_CONFIG is None:
        _RWKV7_NATIVE_W8_CONFIG = RWKV7W8Config()
    return _RWKV7_NATIVE_W8_CONFIG


def _rwkv7_projection_quant_config(
    quant_config: Optional[QuantizationConfig],
    projection: str,
    layer_id: Optional[int] = None,
    num_hidden_layers: Optional[int] = None,
) -> Optional[QuantizationConfig]:
    """Apply RWKV-specific mixed-precision policy to large projections.

    ``projection`` is one of ``attention``, ``ffn_key``, or ``ffn_value``.
    Speed policies quantize every large projection. Balanced policies preserve
    recurrent attention. Accuracy policies additionally protect SqReLU inputs
    and sensitive edge layers, with native W4 using W8 for its contraction and
    intervening expansion projections.
    """
    if quant_config is None:
        return None
    if projection not in ("attention", "ffn_key", "ffn_value"):
        raise ValueError(f"Unknown RWKV-7 projection class: {projection!r}")

    name = quant_config.get_name()
    if name == "bitsandbytes":
        policy = _rwkv7_bnb_policy()
        if policy == "speed":
            return quant_config
        if projection == "attention":
            return None
        if policy == "accuracy" and projection == "ffn_key":
            return None
        if (
            policy == "accuracy"
            and not quant_config.load_in_8bit
            and projection == "ffn_value"
        ):
            if layer_id is None or num_hidden_layers is None:
                raise ValueError("BitsAndBytes W4 FFN policy requires layer metadata")
            edge_layers = min(4, max(1, num_hidden_layers // 6))
            if (
                layer_id < edge_layers
                or layer_id >= num_hidden_layers - edge_layers
                or (layer_id - edge_layers) % 4
            ):
                return None

    if name in ("w8a8_int8", "rwkv7_w8"):
        policy = _rwkv7_w8_policy()
        if policy == "speed":
            return quant_config
        if projection == "attention":
            return None
        if policy == "accuracy" and projection == "ffn_key" and name == "w8a8_int8":
            return None
        if policy == "accuracy" and name == "rwkv7_w8":
            if layer_id is None or num_hidden_layers is None:
                raise ValueError("Native W8 FFN policy requires layer metadata")
            edge_layers = min(4, max(1, num_hidden_layers // 6))
            if layer_id < edge_layers or layer_id >= num_hidden_layers - edge_layers:
                return None
        elif policy == "accuracy" and projection == "ffn_value":
            if layer_id is None or num_hidden_layers is None:
                raise ValueError("FFN value policy requires layer metadata")
            if layer_id in (0, num_hidden_layers - 1):
                return None

    if name in ("marlin", "rwkv7_w4"):
        policy = _rwkv7_w4_policy()
        if policy == "sparse":
            if projection != "ffn_key":
                return None
            if layer_id is None or num_hidden_layers is None:
                raise ValueError("W4 sparse policy requires layer metadata")
            edge_layers = min(4, max(1, num_hidden_layers // 6))
            interior_offset = layer_id - edge_layers
            primary_lane = interior_offset % 4 == 0
            early_secondary_lane = (
                interior_offset % 4 == 2 and layer_id < num_hidden_layers // 2
            )
            if not (
                edge_layers <= layer_id < num_hidden_layers - edge_layers
                and (primary_lane or early_secondary_lane)
            ):
                return None
            return quant_config
        if policy == "speed":
            return quant_config
        if projection == "attention":
            return None
        if policy == "accuracy":
            if layer_id is None or num_hidden_layers is None:
                raise ValueError("W4 accuracy policy requires layer metadata")
            edge_layers = min(4, max(1, num_hidden_layers // 6))
            if layer_id < edge_layers or layer_id >= num_hidden_layers - edge_layers:
                return None
            if name == "rwkv7_w4":
                if projection == "ffn_value":
                    return _rwkv7_native_w8_config()
                return (
                    quant_config
                    if (layer_id - edge_layers) % 4 == 0
                    else _rwkv7_native_w8_config()
                )
            if projection == "ffn_value" or layer_id % 2:
                return _rwkv7_w8a8_config()

    return quant_config


def rwkv7_quantization_plan(
    quant_config: Optional[QuantizationConfig], num_hidden_layers: int
) -> list[dict[str, str | int]]:
    """Return a serializable per-layer projection plan for diagnostics."""

    def name_or_fp16(config: Optional[QuantizationConfig]) -> str:
        return "fp16" if config is None else config.get_name()

    return [
        {
            "layer": layer_id,
            "attention": name_or_fp16(
                _rwkv7_projection_quant_config(
                    quant_config, "attention", layer_id, num_hidden_layers
                )
            ),
            "ffn_key": name_or_fp16(
                _rwkv7_projection_quant_config(
                    quant_config, "ffn_key", layer_id, num_hidden_layers
                )
            ),
            "ffn_value": name_or_fp16(
                _rwkv7_projection_quant_config(
                    quant_config, "ffn_value", layer_id, num_hidden_layers
                )
            ),
        }
        for layer_id in range(num_hidden_layers)
    ]
