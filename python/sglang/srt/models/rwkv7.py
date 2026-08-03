# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""RWKV-7 (Goose) model for sglang.

The portable path matches the RWKV-LM reference directly. Decode adds measured
Triton fast paths for token-shift/control/gating math and a small-batch batched
R/K/V projection, while retaining the canonical modules as universal fallbacks.
The WKV recurrence runs in a dedicated backend kernel. Module and parameter
names mirror the fla-format checkpoint, so ``load_weights`` needs no remapping.

Quantization: the large r/k/v/o_proj and FFN key/value projections are SGLang
quant-aware linears. RWKV-specific policies, including a W4 lane that preserves
the sparse-FFN kernel, choose which projections are quantized. BitsAndBytes and
the legacy CUDA W8/W4 paths keep low-rank controls and lm_head dense. The
portable native W8/W4 kernels cover lm_head as well because their row-streaming
layout is efficient for the wide vocabulary projection and passes the strict
alignment gate. The WKV recurrence/state and per-channel parameters (x_*, k_k,
k_a, r_k, g_norm) are never weight-quantized.

Tensor parallelism is head-parallel: head_dim stays whole and whole heads are
split across ranks (r/k/v + LoRA-up column-parallel with no gather, per-channel
params / g_norm / WKV state on the local head slice, o_proj and ffn.value
row-parallel with a single allreduce each). The token-shift mix vectors and the
conv (prev-token) state stay full-width — they act on the replicated hidden
before the column-parallel projections. tp=1 keeps the exact original path.

Pipeline parallelism partitions the layer stack into contiguous per-rank slices
(llama-style make_layers + PPMissingLayer): the first rank owns the embeddings
(+ ln0 inside layer 0), the last rank owns the final norm + lm_head, and stages
hand off {hidden_states, v_first} as PPProxyTensors — v_first (layer 0's value
projection, under tp>1 the LOCAL head slice) must ride along because every later
layer's v-residual mix consumes it. Backend state stays indexed by GLOBAL
layer_id; the mamba/linear-state pool allocates only this rank's layer slice
(the runner filters by model.start_layer/end_layer). pp=1 keeps the exact
original path.

Per-layer time-mix (att):
  shifted = prev_token(x);  x* = x + x_*·(shifted - x)
  r = r_proj(xr); k = k_proj(xk); v = v_proj(xv)
  w_log = -e^-0.5 * sigmoid( w_up(tanh(w_down(xw))) + w_bias )       # log decay
  a = sigmoid( a_up(a_down(xa)) + a_bias )
  g = g_up( sigmoid(g_down(xg)) )                                    # no bias
  v-residual (layer>0): v += (v_first - v) * sigmoid( v_up(v_down(xv)) + v_bias )
  kk = k * k_k ; k = k + k*(a-1)*k_a ; kk = L2norm(kk) over head_dim
  y = WKV(r, w_log, k, v, kk, a)                                     # backend kernel
  y = g_norm(y) + (r*k*r_k).sum * v ; out = o_proj(y * g)
Channel-mix (ffn): shifted=prev(x); xk = x + x_k·(shifted-x); out = value(relu(key(xk))**2)
"""

import logging
import os
from typing import Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn

from sglang.srt.configs.rwkv7 import Rwkv7Config
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.layers.activation import ReLU2
from sglang.srt.layers.attention.rwkv7_kernels.ffn_sparse_cuda import (
    can_use_sparse_sqrelu_down,
    sparse_ffn_enabled,
    sparse_sqrelu_down,
)
from sglang.srt.layers.attention.rwkv7_kernels.fused import (
    can_fuse_lowrank_controls,
    fused_groupnorm_gate_corr,
    fused_kk_kmix,
    fused_lowrank_controls,
    is_profitable_fused_lowrank_shape,
)
from sglang.srt.layers.attention.rwkv7_kernels.token_shift import layernorm_residual
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.online_utils import online_quantize_w8a8_int8_weight
from sglang.srt.layers.quantization.rwkv7_native import (
    RWKV7W8Config,
    online_quantize_rwkv7_w4_weight,
)
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)

_RWKV7_W8A8_CONFIG: Optional[QuantizationConfig] = None
_RWKV7_NATIVE_W8_CONFIG: Optional[QuantizationConfig] = None
_online_marlin_quantize_weight = None
_online_marlin_quantize_weight_with_int8_shadow = None
if (
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability() >= (8, 0)
):
    from sglang.srt.layers.quantization.marlin_utils import (
        online_marlin_quantize_weight as _online_marlin_quantize_weight,
    )
    from sglang.srt.layers.quantization.marlin_utils import (
        online_marlin_quantize_weight_with_int8_shadow as _online_marlin_quantize_weight_with_int8_shadow,
    )


def _tp_size() -> int:
    """TP world size, tolerating uninitialized distributed state (standalone
    tools may build layers without an engine)."""
    try:
        return get_tensor_model_parallel_world_size()
    except (AssertionError, ValueError):
        return 1


def _tp_rank() -> int:
    try:
        return get_tensor_model_parallel_rank()
    except (AssertionError, ValueError):
        return 0


# e^-0.5 = 1/sqrt(e); w_log = -this * sigmoid(w_raw)  =>  decay = exp(w_log).
_INV_SQRT_E = 0.6065306597126334


def _rwkv7_w8_policy() -> str:
    """Select the model-specific online W8A8 accuracy/compression trade-off."""
    policy = os.getenv("SGLANG_RWKV7_W8_POLICY", "accuracy").lower()
    if policy not in ("accuracy", "balanced", "speed"):
        raise ValueError(
            "SGLANG_RWKV7_W8_POLICY must be accuracy, balanced, or speed; "
            f"got {policy!r}."
        )
    return policy


def _rwkv7_w4_policy() -> str:
    """Select the model-specific online Marlin accuracy/compression trade-off."""
    policy = os.getenv("SGLANG_RWKV7_W4_POLICY", "accuracy").lower()
    if policy not in ("accuracy", "balanced", "speed", "sparse"):
        raise ValueError(
            "SGLANG_RWKV7_W4_POLICY must be accuracy, balanced, speed, or sparse; "
            f"got {policy!r}."
        )
    return policy


def _rwkv7_bnb_policy() -> str:
    """Select the BitsAndBytes accuracy/compression trade-off.

    BitsAndBytes is also the portable W8/W4 path on ROCm.  Quantizing every
    recurrent projection is useful as a throughput-oriented lane, but it is a
    poor default for RWKV because projection error is carried in the recurrent
    state.  The accuracy lane quantizes only the large FFN contraction, while
    balanced additionally quantizes the FFN expansion.
    """
    policy = os.getenv("SGLANG_RWKV7_BNB_POLICY", "accuracy").lower()
    if policy not in ("accuracy", "balanced", "speed"):
        raise ValueError(
            "SGLANG_RWKV7_BNB_POLICY must be accuracy, balanced, or speed; "
            f"got {policy!r}."
        )
    return policy


def _rwkv7_bnb_target_modules(
    quant_config: QuantizationConfig, num_hidden_layers: int
) -> list[str]:
    """Return loader targets matching the projection policy above."""
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
    raw = os.getenv("SGLANG_RWKV7_MARLIN_FALLBACK_MAX_TOKENS", "512")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"SGLANG_RWKV7_MARLIN_FALLBACK_MAX_TOKENS must be an integer; got {raw!r}."
        ) from exc
    if value < 0:
        raise ValueError(
            "SGLANG_RWKV7_MARLIN_FALLBACK_MAX_TOKENS must be non-negative; "
            f"got {value}."
        )
    return value


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
    raw = os.getenv("SGLANG_RWKV7_INT8_EXACT_MAX_TOKENS", "512")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"SGLANG_RWKV7_INT8_EXACT_MAX_TOKENS must be an integer; got {raw!r}."
        ) from exc
    if value < 0:
        raise ValueError(
            f"SGLANG_RWKV7_INT8_EXACT_MAX_TOKENS must be non-negative; got {value}."
        )
    return value


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
    The speed policies quantize every large projection. The W4 sparse policy
    keeps recurrent attention and the FFN contraction dense so the zero-skipping
    SqReLU decode kernel can be used while a conservative subset of middle FFN
    expansion weights remains quantized. Balanced policies keep recurrent
    attention dense. Accuracy policies additionally protect the W8 sqReLU
    expansion and the most sensitive edge FFN layers. The W4 accuracy lane uses
    W4 only for alternating middle FFN expansion projections. The contraction
    projection stays W8: repeated-request testing found that its small-M Marlin
    reduction can flip near-tied greedy logits, while W4 on the expansion
    remains deterministic and retains a real W4 memory/speed lane.
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
                # Keep a real W4 lane while using the faster native W8 kernel
                # for the intervening expansion projections.
                return (
                    quant_config
                    if (layer_id - edge_layers) % 4 == 0
                    else _rwkv7_native_w8_config()
                )
            if projection == "ffn_value" or layer_id % 2:
                return (
                    _rwkv7_native_w8_config()
                    if name == "rwkv7_w4"
                    else _rwkv7_w8a8_config()
                )

    return quant_config


def _make_proj(
    in_f: int, out_f: int, quant_config, prefix: str, parallel: str = "column"
):
    """A bias-free projection: the quant-aware ReplicatedLinear at tp=1. Under tp>1
    the projection is head-parallel instead: ColumnParallelLinear (output = this
    rank's head slice, no gather) or RowParallelLinear (local-slice input,
    allreduce inside)."""
    if _tp_size() > 1:
        if parallel == "row":
            m = RowParallelLinear(
                in_f,
                out_f,
                bias=False,
                input_is_parallel=True,
                reduce_results=True,
                quant_config=quant_config,
                prefix=prefix,
            )
        else:
            m = ColumnParallelLinear(
                in_f,
                out_f,
                bias=False,
                gather_output=False,
                quant_config=quant_config,
                prefix=prefix,
            )
    else:
        m = ReplicatedLinear(
            in_f, out_f, bias=False, quant_config=quant_config, prefix=prefix
        )
    if quant_config is not None and quant_config.get_name() == "w8a8_int8":
        m._rwkv7_int8_exact_max_tokens = _rwkv7_int8_exact_max_tokens()
    return m


def _linear_backend(forward_batch: ForwardBatch):
    """The RWKV-7 linear-attention backend (HybridLinearAttnBackend's linear half).

    Normally read from the global forward context; batches that carry their own
    attn_backend (e.g. cuda-graph capture paths) take precedence."""
    ab = getattr(forward_batch, "attn_backend", None)
    if ab is None:
        ab = get_attn_backend()
    return ab.linear_attn_backend


class Rwkv7LoRA(nn.Module):
    """fla low-rank block: up(act(down(x))) [+ bias].

    Keys: lora.0.weight (down), lora.2.weight (up), lora.2.bias (up bias).

    The down/up projections are sglang ``ReplicatedLinear`` (tp=1) so they are
    quant-aware: with ``quant_config=None`` they fall through to an
    unquantized ``F.linear`` (bit-identical to ``nn.Linear``); with a quant
    config they carry int8/4-bit weights. The ``nn.Sequential`` is kept purely as
    a name container so checkpoint keys stay ``lora.0`` / ``lora.2`` (we drive the
    forward manually because ReplicatedLinear returns a ``(out, bias)`` tuple).

    Under tp>1 the down proj stays replicated (its input is the full replicated
    hidden and the rank-dim output is tiny, so every rank computes it locally,
    no comm) while the up proj is ColumnParallelLinear (no gather): its output —
    and its bias, sharded by the ColumnParallelLinear bias loader — is exactly
    this rank's head slice, matching the head-parallel r/k/v projections.
    """

    def __init__(
        self,
        hidden_size: int,
        low_rank: int,
        activation: str,
        bias: bool,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if activation == "tanh":
            act = nn.Tanh()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        else:
            act = nn.Identity()
        if _tp_size() > 1:
            up = ColumnParallelLinear(
                low_rank,
                hidden_size,
                bias=bias,
                gather_output=False,
                quant_config=quant_config,
                prefix=add_prefix("lora.2", prefix),
            )
        else:
            up = ReplicatedLinear(
                low_rank,
                hidden_size,
                bias=bias,
                quant_config=quant_config,
                prefix=add_prefix("lora.2", prefix),
            )
        self.lora = nn.Sequential(
            ReplicatedLinear(
                hidden_size,
                low_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("lora.0", prefix),
            ),
            act,
            up,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lora[0](x)
        h = self.lora[1](h)
        out, _ = self.lora[2](h)
        return out


class Rwkv7Attention(nn.Module):
    """RWKV-7 time-mixing block."""

    def __init__(
        self,
        config: Rwkv7Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        # WKV heads tile the channel dim exactly; g_norm(num_groups=num_heads,
        # num_channels=H) and every [T, nh, hd] reshape below silently corrupt if
        # this is violated, so fail loudly at construction instead.
        assert self.num_heads * self.head_dim == self.hidden_size, (
            f"RWKV-7 head geometry mismatch: num_heads({self.num_heads}) * "
            f"head_dim({self.head_dim}) != hidden_size({self.hidden_size})"
        )
        # Head-parallel TP: head_dim stays whole, whole heads are split across
        # ranks. Everything downstream of the r/k/v/LoRA-up projections (per-
        # channel params, g_norm, the WKV recurrence and its state) lives on
        # this rank's head slice; o_proj (row-parallel) restores the full H.
        tp_size = _tp_size()
        assert self.num_heads % tp_size == 0, (
            f"RWKV-7 TP requires num_heads({self.num_heads}) divisible by "
            f"tp_size({tp_size})"
        )
        self.local_num_heads = self.num_heads // tp_size
        self.local_hidden_size = self.local_num_heads * self.head_dim

        H = self.hidden_size
        Hl = self.local_hidden_size
        # token-shift mix vectors (lerp coefficients)
        self.x_r = nn.Parameter(torch.zeros(1, 1, H))
        self.x_w = nn.Parameter(torch.zeros(1, 1, H))
        self.x_k = nn.Parameter(torch.zeros(1, 1, H))
        self.x_v = nn.Parameter(torch.zeros(1, 1, H))
        self.x_a = nn.Parameter(torch.zeros(1, 1, H))
        self.x_g = nn.Parameter(torch.zeros(1, 1, H))

        # Dynamic W8A8 activation quantization inside the recurrent time-mix
        # compounds its error through the persistent state. Keep these four
        # attention projections in the activation dtype for the accuracy lane;
        # the larger FFN key/value matrices still use INT8 and retain most of
        # the memory/throughput benefit. Weight-only W4 avoids activation
        # quantization, but its accuracy policy also keeps recurrent attention
        # dense because these projections feed the persistent state.
        attention_quant_config = _rwkv7_projection_quant_config(
            quant_config,
            "attention",
            layer_id=layer_id,
            num_hidden_layers=config.num_hidden_layers,
        )
        # Projections are quant-aware ReplicatedLinear (tp=1) / parallel linears (tp>1).
        self.r_proj = _make_proj(
            H, H, attention_quant_config, add_prefix("r_proj", prefix)
        )
        self.k_proj = _make_proj(
            H, H, attention_quant_config, add_prefix("k_proj", prefix)
        )
        self.v_proj = _make_proj(
            H, H, attention_quant_config, add_prefix("v_proj", prefix)
        )
        self.o_proj = _make_proj(
            H,
            H,
            attention_quant_config,
            add_prefix("o_proj", prefix),
            parallel="row",
        )
        # The 1.5B decode shape is launch-bound: one strided-batched GEMM is
        # measurably faster than three independent R/K/V GEMMs at B=1/2/4/8.
        # Re-home the three canonical parameters as views into one storage so
        # the fast path needs neither a per-forward stack nor a duplicate
        # weight copy.  Checkpoint names and per-parameter weight loaders stay
        # unchanged; in-place RL/weight updates update the shared storage too.
        self._rkv_stacked_weight = None
        self._rkv_pack_enabled = (
            H == 2048
            and _tp_size() == 1
            and attention_quant_config is None
            and all(
                getattr(module, "weight", None) is not None and module.weight.ndim == 2
                for module in (self.r_proj, self.k_proj, self.v_proj)
            )
        )
        if self._rkv_pack_enabled:
            self._pack_rkv_weights()

        low_rank_quant_config = quant_config
        if quant_config is not None and quant_config.get_name() in (
            "bitsandbytes",
            "marlin",
            "w8a8_int8",
            "rwkv7_w8",
            "rwkv7_w4",
        ):
            low_rank_quant_config = None
        self.w_lora = Rwkv7LoRA(
            H,
            config.decay_low_rank_dim,
            "tanh",
            bias=True,
            quant_config=low_rank_quant_config,
            prefix=add_prefix("w_lora", prefix),
        )
        self.a_lora = Rwkv7LoRA(
            H,
            config.a_low_rank_dim,
            "identity",
            bias=True,
            quant_config=low_rank_quant_config,
            prefix=add_prefix("a_lora", prefix),
        )
        self.g_lora = Rwkv7LoRA(
            H,
            config.gate_low_rank_dim,
            "sigmoid",
            bias=False,
            quant_config=low_rank_quant_config,
            prefix=add_prefix("g_lora", prefix),
        )
        if layer_id > 0:
            self.v_lora = Rwkv7LoRA(
                H,
                config.v_low_rank_dim,
                "identity",
                bias=True,
                quant_config=low_rank_quant_config,
                prefix=add_prefix("v_lora", prefix),
            )

        self.k_k = nn.Parameter(torch.zeros(Hl))
        self.k_a = nn.Parameter(torch.zeros(Hl))
        self.r_k = nn.Parameter(torch.zeros(self.local_num_heads, self.head_dim))

        self.g_norm = nn.GroupNorm(
            num_groups=self.local_num_heads,
            num_channels=Hl,
            eps=self.head_dim * config.norm_eps,
            affine=True,
        )

        # Stacked token-shift mix vectors, lazily built (post weight-load)
        # on first forward and cached. R/K/V occupy the first three contiguous
        # slices so the small-batch strided-batched projection consumes a view.
        # Order [x_r, x_k, x_v, x_w, x_a, x_g].
        self._mix6 = None

    def _pack_rkv_weights(self) -> None:
        """Re-home canonical R/K/V parameters as views of one 3-D storage."""
        modules = (self.r_proj, self.k_proj, self.v_proj)
        original = [module.weight for module in modules]
        base = torch.empty(
            (3, *original[0].shape),
            dtype=original[0].dtype,
            device=original[0].device,
        )
        with torch.no_grad():
            for old_weight, new_storage in zip(original, base):
                new_storage.copy_(old_weight)
        for module, old_weight, new_storage in zip(modules, original, base):
            new_weight = nn.Parameter(
                new_storage, requires_grad=old_weight.requires_grad
            )
            # Preserve SGLang's per-parameter weight-loader metadata.
            new_weight.__dict__.update(old_weight.__dict__)
            module.weight = new_weight
        # A plain tensor attribute (not a parameter/buffer) owns the common
        # 3-D view without adding a duplicate state-dict/load obligation.
        self._rkv_stacked_weight = base

    def _apply(self, fn, recurse=True):
        # ``Module.to`` transforms registered parameters independently. Repack
        # afterward so R/K/V retain shared storage instead of silently keeping
        # an obsolete duplicate base and falling off the optimized path.
        result = super()._apply(fn, recurse=recurse)
        if self._rkv_pack_enabled:
            self._pack_rkv_weights()
        return result

    def _mix6_buf(self) -> torch.Tensor:
        if self._mix6 is None:
            self._mix6 = torch.stack(
                [
                    self.x_r.reshape(-1),
                    self.x_k.reshape(-1),
                    self.x_v.reshape(-1),
                    self.x_w.reshape(-1),
                    self.x_a.reshape(-1),
                    self.x_g.reshape(-1),
                ],
                dim=0,
            ).contiguous()
        return self._mix6

    def forward(
        self,
        forward_batch: ForwardBatch,
        x: torch.Tensor,
        v_first: Optional[torch.Tensor],
        norm: Optional[nn.LayerNorm] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        T = x.shape[0]
        if T == 0:
            return x, v_first, x

        be = _linear_backend(forward_batch)
        # Local (per-rank) head slice; == the full width at tp=1.
        H, hd, nh = self.local_hidden_size, self.head_dim, self.local_num_heads

        # Fused Triton serving path: the elementwise pieces are bit-identical at
        # bf16/fp16; fused GroupNorm has a bounded reduction-order delta covered
        # by end-to-end logit/greedy gates. fp32 keeps the strict torch path.
        fused = x.dtype != torch.float32

        if fused:
            lp, x = be.token_shift_lerp6(
                x,
                self._mix6_buf(),
                self.layer_id,
                0,
                forward_batch,
                norm_weight=None if norm is None else norm.weight,
                norm_bias=None if norm is None else norm.bias,
                norm_eps=1e-5 if norm is None else norm.eps,
                residual=residual,
            )
            xr, xk, xv, xw, xa, xg = lp[0], lp[1], lp[2], lp[3], lp[4], lp[5]
        else:
            if residual is not None:
                x = x + residual
            residual_base = x
            if norm is not None:
                x = norm(x)
            shifted = be.token_shift(x, self.layer_id, 0, forward_batch)
            d = shifted - x
            xr = x + self.x_r.view(-1) * d
            xw = x + self.x_w.view(-1) * d
            xk = x + self.x_k.view(-1) * d
            xv = x + self.x_v.view(-1) * d
            xa = x + self.x_a.view(-1) * d
            xg = x + self.x_g.view(-1) * d

        stacked = self._rkv_stacked_weight
        if (
            stacked is not None
            and fused
            and (T in (1, 2, 4, 8) or T >= 256)
            and stacked.device == xr.device
            and stacked.dtype == xr.dtype
            and self.r_proj.weight.data_ptr() == stacked[0].data_ptr()
            and self.k_proj.weight.data_ptr() == stacked[1].data_ptr()
            and self.v_proj.weight.data_ptr() == stacked[2].data_ptr()
        ):
            rkv = torch.bmm(lp[:3], stacked.transpose(1, 2))
            r, k, v = rkv[0], rkv[1], rkv[2]
        else:
            r = self.r_proj(xr)[0]
            k = self.k_proj(xk)[0]
            v = self.v_proj(xv)[0]

        if self.layer_id == 0:
            v_first = v

        # LoRA gates: w=decay, a=in-context-lr, g=output-gate, v=v-residual
        # (layer>0). Decode batches <=8 use two grouped Triton launches while
        # retaining separately named canonical weights and the generic path.
        v_lora = self.v_lora if self.layer_id != 0 else None
        use_fused_lowrank = (
            fused
            and is_profitable_fused_lowrank_shape(H, T)
            and forward_batch.forward_mode.is_decode_or_idle()
            and can_fuse_lowrank_controls(self.w_lora, self.a_lora, self.g_lora, v_lora)
        )
        if use_fused_lowrank:
            w_log, a, g, v = fused_lowrank_controls(
                xw,
                xa,
                xg,
                xv,
                v,
                v_first,
                self.w_lora,
                self.a_lora,
                self.g_lora,
                v_lora,
            )
        else:
            w_log = -torch.sigmoid(self.w_lora(xw)) * _INV_SQRT_E
            a = torch.sigmoid(self.a_lora(xa))
            g = self.g_lora(xg)
            if self.layer_id != 0:
                v = v + (v_first - v) * torch.sigmoid(self.v_lora(xv))

        if fused:
            # kk = L2norm(k·k_k) over hd; k <- k + k·(a-1)·k_a  (one launch)
            kk, k = fused_kk_kmix(k, a, self.k_k, self.k_a, nh)
            r = r.view(T, nh, hd)
            w_log = w_log.view(T, nh, hd)
            k = k.view(T, nh, hd)
            v = v.view(T, nh, hd)
            a = a.view(T, nh, hd)
        else:
            kk = k * self.k_k
            k = k + k * (a - 1.0) * self.k_a
            r = r.view(T, nh, hd)
            w_log = w_log.view(T, nh, hd)
            k = k.view(T, nh, hd)
            v = v.view(T, nh, hd)
            a = a.view(T, nh, hd)
            kk = kk.view(T, nh, hd)
            kk = kk / kk.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        o = be.recurrence(r, w_log, k, v, kk, a, self.layer_id, forward_batch)
        # o: [T, nh, hd]
        if fused:
            # GroupNorm + recurrent correction + output gate (one launch).
            o = fused_groupnorm_gate_corr(
                o,
                r,
                k,
                self.r_k,
                v,
                g,
                self.g_norm.weight,
                self.g_norm.bias,
                nh,
                self.g_norm.eps,
            )
        else:
            o = self.g_norm(o.reshape(T, H))
            gate_corr = ((r * k * self.r_k).sum(dim=-1, keepdim=True) * v).reshape(T, H)
            o = o + gate_corr
            o = o * g
        out = self.o_proj(o)[0]
        return out, v_first, x if fused else residual_base


class Rwkv7FeedForward(nn.Module):
    """RWKV-7 channel-mixing block (sqrelu)."""

    def __init__(
        self,
        config: Rwkv7Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        H = config.hidden_size
        self.hidden_size = H
        inter = config.intermediate_size
        self.x_k = nn.Parameter(torch.zeros(H))
        self.activation = ReLU2()
        # W8A8 on the expansion projection perturbs values before sqReLU,
        # which squares that activation error. Keep ``key`` dense and quantize
        # the equally large contraction projection (``value``); this is the
        # conservative W8 accuracy lane. W4 accuracy protects edge layers and
        # interleaves W4/W8 FFN blocks in the middle of the stack.
        key_quant_config = _rwkv7_projection_quant_config(
            quant_config,
            "ffn_key",
            layer_id=layer_id,
            num_hidden_layers=config.num_hidden_layers,
        )
        # tp>1: key is column-parallel (local inter slice; sqrelu is elementwise so
        # it acts per-slice), value is row-parallel (allreduce restores the full H).
        self.key = _make_proj(H, inter, key_quant_config, add_prefix("key", prefix))
        if (
            key_quant_config is not None
            and key_quant_config.get_name() == "marlin"
            and _rwkv7_w4_policy() == "accuracy"
        ):
            # Marlin remains the throughput path for large prefills. The
            # checkpoint-sensitive small-M shadow is exact FP16 at width 2048
            # and compact fused INT8 on wider models (overridable by env).
            local_inter = self.key.s.shape[1]
            if _rwkv7_w4_shadow_mode(H) == "fp16":
                self.key.register_buffer(
                    "_rwkv7_decode_weight",
                    torch.empty(
                        local_inter,
                        H,
                        dtype=self.key.s.dtype,
                        device=self.key.s.device,
                    ),
                    persistent=False,
                )
            else:
                self.key.register_buffer(
                    "_rwkv7_decode_qweight",
                    torch.empty(
                        local_inter,
                        H,
                        dtype=torch.int8,
                        device=self.key.s.device,
                    ),
                    persistent=False,
                )
                self.key.register_buffer(
                    "_rwkv7_decode_scales",
                    torch.empty(
                        local_inter,
                        1,
                        dtype=torch.float32,
                        device=self.key.s.device,
                    ),
                    persistent=False,
                )
            self.key._rwkv7_marlin_fallback_max_tokens = (
                _rwkv7_marlin_fallback_max_tokens()
            )
        value_quant_config = _rwkv7_projection_quant_config(
            quant_config,
            "ffn_value",
            layer_id=layer_id,
            num_hidden_layers=config.num_hidden_layers,
        )
        self.value = _make_proj(
            inter,
            H,
            value_quant_config,
            add_prefix("value", prefix),
            parallel="row",
        )
        # Decode SqReLU is naturally sparse (roughly half of the expansion is
        # exactly zero). Re-home the dense value weight in transposed contiguous
        # storage so a zero-skipping CUDA kernel can read rows coalesced without
        # retaining a second copy. The canonical value.weight parameter remains
        # a correctly shaped view, preserving checkpoint and dense fallback
        # behavior. Quantized and TP paths retain their native layouts.
        self._value_transposed_weight = None
        self._sparse_value_enabled = bool(
            sparse_ffn_enabled()
            and torch.cuda.is_available()
            and _tp_size() == 1
            and value_quant_config is None
            and getattr(self.value, "weight", None) is not None
            and self.value.weight.ndim == 2
        )
        if self._sparse_value_enabled:
            self._pack_value_weight_for_sparse()

    def _pack_value_weight_for_sparse(self) -> None:
        old_weight = self.value.weight
        transposed = torch.empty(
            (old_weight.shape[1], old_weight.shape[0]),
            dtype=old_weight.dtype,
            device=old_weight.device,
        )
        with torch.no_grad():
            transposed.copy_(old_weight.t())
        new_weight = nn.Parameter(
            transposed.t(), requires_grad=old_weight.requires_grad
        )
        new_weight.__dict__.update(old_weight.__dict__)
        self.value.weight = new_weight
        self._value_transposed_weight = transposed

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if self._sparse_value_enabled:
            self._pack_value_weight_for_sparse()
        return result

    def forward(
        self,
        forward_batch: ForwardBatch,
        x: torch.Tensor,
        norm: Optional[nn.LayerNorm] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] == 0:
            return x, x
        be = _linear_backend(forward_batch)
        xk, x = be.token_shift_lerp1(
            x,
            self.x_k,
            self.layer_id,
            1,
            forward_batch,
            norm_weight=None if norm is None else norm.weight,
            norm_bias=None if norm is None else norm.bias,
            norm_eps=1e-5 if norm is None else norm.eps,
            residual=residual,
        )
        k = self.key(xk)[0]
        value_weight_t = self._value_transposed_weight
        use_sparse_value = bool(
            forward_batch.forward_mode.is_decode_or_idle()
            and value_weight_t is not None
            and self.value.weight.data_ptr() == value_weight_t.data_ptr()
            and can_use_sparse_sqrelu_down(k, value_weight_t)
        )
        if use_sparse_value:
            out = sparse_sqrelu_down(k, value_weight_t)
        else:
            act = self.activation(k)
            out = self.value(act)[0]
        return out, x


class Rwkv7DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Rwkv7Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        H = config.hidden_size
        eps = config.norm_eps
        bias = config.norm_bias
        if layer_id == 0:
            # ln0: applied ONCE to the embeddings (driven from Rwkv7Model.forward).
            self.pre_norm = nn.LayerNorm(H, eps=eps, bias=bias)
        self.attn_norm = nn.LayerNorm(H, eps=eps, bias=bias)
        self.ffn_norm = nn.LayerNorm(H, eps=eps, bias=bias)
        self.attn = Rwkv7Attention(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.ffn = Rwkv7FeedForward(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=add_prefix("ffn", prefix),
        )

    def forward(
        self,
        forward_batch: ForwardBatch,
        x: torch.Tensor,
        v_first: Optional[torch.Tensor],
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        attn_out, v_first, x = self.attn(
            forward_batch,
            x,
            v_first,
            norm=self.attn_norm,
            residual=residual,
        )
        ffn_out, x = self.ffn(
            forward_batch,
            x,
            norm=self.ffn_norm,
            residual=attn_out,
        )
        return x, ffn_out, v_first


class Rwkv7Model(nn.Module):
    def __init__(
        self,
        config: Rwkv7Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        # PP: the first rank owns the embeddings (ln0 lives inside layer 0, which
        # make_layers also puts on the first rank), the last rank owns the final
        # norm; every other position is a PPMissingLayer placeholder. pp=1 (all
        # ranks first AND last) constructs exactly the original module tree.
        if self.pp_group.is_first_rank:
            self.embeddings = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )
        else:
            self.embeddings = PPMissingLayer()
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Rwkv7DecoderLayer(
                config, idx, quant_config=quant_config, prefix=prefix
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
            )
        else:
            self.norm = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                x = inputs_embeds
            else:
                x = self.embeddings(input_ids)

            if x.shape[0] > 0:
                # ln0 on the embeddings (once), then the recurrent stack.
                x = self.layers[0].pre_norm(x)
            v_first = None
        else:
            assert pp_proxy_tensors is not None
            x = pp_proxy_tensors["hidden_states"]
            v_first = pp_proxy_tensors["v_first"]
            # v_first crosses the stage boundary FULL-WIDTH (see the send side:
            # sglang's pp tensor-dict transfer chunk-sends over the tp group and
            # all-gathers on receive, which is only lossless for tp-replicated
            # tensors) — slice back to this rank's head slice.
            tp_size = _tp_size()
            if tp_size > 1 and v_first.shape[0] > 0:
                Hl = v_first.shape[-1] // tp_size
                r = _tp_rank()
                v_first = v_first[:, r * Hl : (r + 1) * Hl].contiguous()

        residual = None
        for i in range(self.start_layer, self.end_layer):
            x, residual, v_first = self.layers[i](
                forward_batch, x, v_first, residual=residual
            )
        final_norm_fused = bool(
            residual is not None
            and self.pp_group.is_last_rank
            and x.dtype in (torch.float16, torch.bfloat16)
            and forward_batch.forward_mode.is_decode_or_idle()
        )
        if final_norm_fused:
            x = layernorm_residual(
                x,
                residual,
                self.norm.weight,
                self.norm.bias,
                self.norm.eps,
            )
        elif residual is not None:
            x = x + residual

        if not self.pp_group.is_last_rank:
            # v_first (layer 0's value projection — under tp>1 the LOCAL head
            # slice, same layout on the matching tp rank of the next stage) rides
            # along with the hidden state: every later layer's v-residual mix
            # consumes it. It is None only for empty batches (T==0 skips every
            # layer); send a same-width empty placeholder so the p2p tensor dict
            # stays uniform.
            if v_first is None:
                v_first = x.new_zeros(
                    x.shape[0], self.layers[self.start_layer].attn.local_hidden_size
                )
            # sglang's pp transfer chunk-sends each tensor across the tp group and
            # reassembles rank-by-rank on receive — lossless ONLY for tp-replicated
            # tensors (#30015). v_first is the LOCAL head slice under tp>1, so
            # gather it to full width here (the receiver slices its head range
            # back out). Once send_tensor_dict honors the model-declared
            # `pp_proxy_tensors_all_gather_exclude` (#30095), this transit can send
            # the per-rank slice whole instead — kept for now so PP×TP is correct
            # regardless of which PR lands first.
            if _tp_size() > 1 and v_first.shape[0] > 0:
                v_first = tensor_model_parallel_all_gather(v_first.contiguous())
            return PPProxyTensors({"hidden_states": x, "v_first": v_first})

        if not final_norm_fused:
            x = self.norm(x)
        return x


class Rwkv7ForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    # PP proxy tensors that are NOT replicated across the attention-TP group:
    # v_first is a per-rank head slice. send_tensor_dict's slice/all-gather
    # optimization must send these whole (#30015 / #30095). Today the model
    # additionally all-gathers v_first to full width before the stage boundary
    # (see Rwkv7Model.forward), so correctness does not depend on #30095;
    # declaring the key here lets that transit be dropped once it lands.
    pp_proxy_tensors_all_gather_exclude = frozenset({"v_first"})

    # ---- BitsAndBytes (4-bit nf4 / 8-bit) support metadata ----
    # RWKV-7 has no fused/stacked projections (r/k/v/o are separate linears), so
    # the stacked-params mapping is empty. The target modules list the linear
    # sub-modules the bnb loader should quantize on the fly (substring match on
    # the checkpoint weight name). Keep RWKV's small low-rank control paths in
    # the activation dtype: quantizing them saves little memory, adds eight
    # poorly-shaped BNB matmuls per layer, and reduces numerical fidelity.
    bitsandbytes_stacked_params_mapping = {}
    default_bitsandbytes_target_modules = [
        ".r_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
        ".key.",
        ".value.",
    ]

    def __init__(
        self,
        config: Rwkv7Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        if quant_config is not None and quant_config.get_name() == "bitsandbytes":
            self.default_bitsandbytes_target_modules = _rwkv7_bnb_target_modules(
                quant_config, config.num_hidden_layers
            )
        self.pp_group = get_pp_group()
        self.model = Rwkv7Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        # Legacy online quantizers keep the vocabulary projection in the
        # activation dtype. The RWKV-native row-streaming kernels deliberately
        # cover lm_head: it is a meaningful memory/decode cost and the native
        # W8/W4 accuracy policies pass the independent alignment gate.
        lm_head_quant_config = quant_config
        if quant_config is not None and quant_config.get_name() in (
            "bitsandbytes",
            "marlin",
            "w8a8_int8",
        ):
            lm_head_quant_config = None
        # lm_head exists on every pp rank (llama pattern; only the last rank uses
        # it — the logits_processor runs there).
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=lm_head_quant_config,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            inputs_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        # Non-last pp rank: hand the PPProxyTensors (hidden_states + v_first) to
        # the next stage; logits only exist on the last rank.
        return hidden_states

    def get_embed_and_head(self):
        return self.model.embeddings.weight, self.lm_head.weight

    # The runner reads model.start_layer/end_layer (llama pattern) to size the
    # per-rank mamba/linear-state pool: under pp>1 only this rank's layer slice
    # is allocated and mamba2_layer_cache maps the GLOBAL layer_id to it.
    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))
        tp_size = _tp_size()
        tp_rank = _tp_rank()
        # Head-sharded per-channel params (tp>1): the checkpoint stores the full
        # tensor; narrow dim 0 (channels resp. heads) to this rank's head slice
        # before the plain copy. Parallel linears shard via their own weight_loader.
        _head_sharded = (".k_k", ".k_a", ".r_k", ".g_norm.weight", ".g_norm.bias")
        loaded_params: Set[str] = set()
        pp_skipped = 0
        for name, loaded_weight in weights:
            marlin_prefix = name.removesuffix(".weight")
            marlin_weight_name = marlin_prefix + ".B"
            marlin_scale_name = marlin_prefix + ".s"
            if (
                name.endswith(".weight")
                and loaded_weight.is_floating_point()
                and marlin_weight_name in params_dict
                and marlin_scale_name in params_dict
            ):
                if _online_marlin_quantize_weight is None:
                    raise RuntimeError(
                        "Online Marlin quantization requires NVIDIA SM80 or newer."
                    )
                shadow_name = marlin_prefix + "._rwkv7_decode_qweight"
                shadow_scale_name = marlin_prefix + "._rwkv7_decode_scales"
                dense_shadow_name = marlin_prefix + "._rwkv7_decode_weight"
                if dense_shadow_name in params_dict:
                    packed_weight, weight_scale = _online_marlin_quantize_weight(
                        loaded_weight,
                        group_size=self.quant_config.group_size,
                    )
                    dense_shadow = params_dict[dense_shadow_name]
                    fallback_weight = loaded_weight
                    if fallback_weight.shape != dense_shadow.shape:
                        shard = dense_shadow.shape[0]
                        fallback_weight = fallback_weight.narrow(
                            0, tp_rank * shard, shard
                        )
                    dense_shadow.copy_(
                        fallback_weight.to(
                            device=dense_shadow.device, dtype=dense_shadow.dtype
                        )
                    )
                    loaded_params.add(dense_shadow_name)
                elif shadow_name in params_dict:
                    if _online_marlin_quantize_weight_with_int8_shadow is None:
                        raise RuntimeError(
                            "RWKV-7 online W4 shadow requires NVIDIA SM80 or newer."
                        )
                    (
                        packed_weight,
                        weight_scale,
                        shadow_weight,
                        shadow_scales,
                    ) = _online_marlin_quantize_weight_with_int8_shadow(
                        loaded_weight,
                        group_size=self.quant_config.group_size,
                    )
                    shadow = params_dict[shadow_name]
                    fallback_scales = params_dict[shadow_scale_name]
                    if shadow_weight.shape != shadow.shape:
                        shard = shadow.shape[0]
                        shadow_weight = shadow_weight.narrow(0, tp_rank * shard, shard)
                        shadow_scales = shadow_scales.narrow(0, tp_rank * shard, shard)
                    shadow.copy_(shadow_weight.to(device=shadow.device))
                    fallback_scales.copy_(
                        shadow_scales.to(
                            device=fallback_scales.device,
                            dtype=fallback_scales.dtype,
                        )
                    )
                    loaded_params.update((shadow_name, shadow_scale_name))
                else:
                    packed_weight, weight_scale = _online_marlin_quantize_weight(
                        loaded_weight,
                        group_size=self.quant_config.group_size,
                    )
                for target_name, tensor in (
                    (marlin_weight_name, packed_weight),
                    (marlin_scale_name, weight_scale),
                ):
                    target = params_dict[target_name]
                    weight_loader = getattr(
                        target, "weight_loader", default_weight_loader
                    )
                    weight_loader(target, tensor)
                    loaded_params.add(target_name)
                continue
            if name not in params_dict:
                # pp>1: keys for another stage's slice (layers outside
                # [start_layer, end_layer), the embeddings off the first rank,
                # the final norm off the last rank) are PPMissingLayer here —
                # skip them. Anything else is still a hard error, and at pp=1
                # every miss raises exactly as before.
                if self.pp_group.world_size > 1 and self._on_other_pp_rank(name):
                    pp_skipped += 1
                    continue
                raise KeyError(
                    f"[rwkv7.load_weights] unexpected checkpoint key: {name}"
                )
            param = params_dict[name]
            scale_name = name.removesuffix(".weight") + ".weight_scale"
            if (
                name.endswith(".weight")
                and param.dtype == torch.uint8
                and loaded_weight.is_floating_point()
                and scale_name in params_dict
            ):
                loaded_weight, weight_scale = online_quantize_rwkv7_w4_weight(
                    loaded_weight,
                    group_size=getattr(self.quant_config, "group_size", 128),
                )
                scale_param = params_dict[scale_name]
                scale_loader = getattr(
                    scale_param, "weight_loader", default_weight_loader
                )
                scale_loader(scale_param, weight_scale)
                loaded_params.add(scale_name)
            if (
                name.endswith(".weight")
                and param.dtype == torch.int8
                and loaded_weight.is_floating_point()
                and scale_name in params_dict
            ):
                # Online W8A8: convert an ordinary HF weight to static
                # per-output-channel symmetric INT8. The layer dynamically
                # quantizes activations per token and sgl-kernel executes the
                # scaled INT8 GEMM. Quantize before the normal weight loaders
                # so TP column/row sharding is applied to both tensors.
                loaded_weight, weight_scale = online_quantize_w8a8_int8_weight(
                    loaded_weight
                )
                scale_param = params_dict[scale_name]
                scale_loader = getattr(
                    scale_param, "weight_loader", default_weight_loader
                )
                scale_loader(scale_param, weight_scale.float())
                loaded_params.add(scale_name)
            if tp_size > 1 and name.endswith(_head_sharded):
                shard = param.shape[0]
                loaded_weight = loaded_weight.narrow(0, tp_rank * shard, shard)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if pp_skipped:
            logger.info(
                "[rwkv7.load_weights] pp rank %s: skipped %d checkpoint keys "
                "owned by other pp ranks",
                self.pp_group.rank_in_group,
                pp_skipped,
            )

        # Assert every model parameter was loaded (catches naming mismatches).
        missing = {
            name
            for name in set(params_dict.keys()) - loaded_params
            if not name.endswith(".workspace")
        }
        if missing:
            raise RuntimeError(
                f"[rwkv7.load_weights] {len(missing)} params not loaded, e.g. "
                f"{sorted(missing)[:8]}"
            )
        # Weight updates (update_weights_from_disk / RL sync) copy into x_* in
        # place; drop the stacked mix buffers so the next forward rebuilds them.
        for module in self.modules():
            if hasattr(module, "_mix6"):
                module._mix6 = None
        return loaded_params

    def _on_other_pp_rank(self, name: str) -> bool:
        """True iff this checkpoint key belongs to a module another pp rank owns
        (so this rank holds a PPMissingLayer for it and must skip the key)."""
        layer_id = get_layer_id(name)
        if layer_id is not None:
            return not (self.model.start_layer <= layer_id < self.model.end_layer)
        if name.startswith("model.embeddings."):
            return not self.pp_group.is_first_rank
        if name.startswith("model.norm."):
            return not self.pp_group.is_last_rank
        return False


# config.json architectures = ["RWKV7ForCausalLM"]; the registry keys by class
# __name__, so expose that spelling too (thin subclass).
class RWKV7ForCausalLM(Rwkv7ForCausalLM):
    pass


class NativeRWKV7ForCausalLM(Rwkv7ForCausalLM):
    """Architecture alias used by native rwkv-rs/hf-adapter checkpoints."""

    pass


EntryClass = [Rwkv7ForCausalLM, RWKV7ForCausalLM, NativeRWKV7ForCausalLM]
