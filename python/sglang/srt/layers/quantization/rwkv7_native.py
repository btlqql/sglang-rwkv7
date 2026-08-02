"""Native portable weight-only quantization for RWKV-7 projections.

The kernels in this module are intentionally self contained.  CUDA keeps the
existing W8A8/Marlin paths, while ROCm can use these kernels without depending
on a particular AITER, Quark, or bitsandbytes build.  Small decode batches use
one program per output row so a packed weight row is read only once for all
active requests.  Larger prefills use a tiled matrix kernel.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import triton
import triton.language as tl
from torch.nn.parameter import Parameter

from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)

_W4_GROUP_SIZE = 32


@triton.jit
def _w8_row_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    """Weight-streaming GEMV/GEMM for serving decode batches."""
    n = tl.program_id(0)
    rows = tl.arange(0, BM)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        weight = tl.load(weight_ptr + n * K + k0 + kk).to(tl.float32)
        x = tl.load(
            x_ptr + rows[:, None] * K + k0 + kk[None, :],
            mask=rows[:, None] < M,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x * weight[None, :], axis=1)
    scale = tl.load(scale_ptr + n).to(tl.float32)
    tl.store(out_ptr + rows * N + n, acc * scale, mask=rows < M)


@triton.jit
def _w4_row_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    """Group-scaled packed-INT4 decode kernel, one program per output row."""
    n = tl.program_id(0)
    rows = tl.arange(0, BM)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM,), dtype=tl.float32)
    packed_k = K // 2
    groups_k = K // GROUP_SIZE
    for k0 in range(0, K, BK):
        k = k0 + kk
        packed = tl.load(weight_ptr + n * packed_k + k // 2).to(tl.int32)
        q = ((packed >> ((k & 1) * 4)) & 15) - 8
        scale = tl.load(scale_ptr + n * groups_k + k // GROUP_SIZE).to(tl.float32)
        x = tl.load(
            x_ptr + rows[:, None] * K + k[None, :],
            mask=rows[:, None] < M,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x * (q.to(tl.float32) * scale)[None, :], axis=1)
    tl.store(out_ptr + rows * N + n, acc, mask=rows < M)


@triton.jit
def _w8_matmul_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(
            x_ptr + m[:, None] * K + k0 + k[None, :],
            mask=(m[:, None] < M) & (k0 + k[None, :] < K),
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + n[None, :] * K + k0 + k[:, None],
            mask=(n[None, :] < N) & (k0 + k[:, None] < K),
            other=0,
        ).to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, weight)
    scale = tl.load(scale_ptr + n, mask=n < N, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + m[:, None] * N + n[None, :],
        acc * scale[None, :],
        mask=(m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _w8a8_matmul_kernel(
    x_ptr,
    weight_ptr,
    x_scale_ptr,
    weight_scale_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Per-token/per-channel INT8 tensor-core path for large prefills."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, K, BK):
        x = tl.load(
            x_ptr + m[:, None] * K + k0 + k[None, :],
            mask=(m[:, None] < M) & (k0 + k[None, :] < K),
            other=0,
        )
        weight = tl.load(
            weight_ptr + n[None, :] * K + k0 + k[:, None],
            mask=(n[None, :] < N) & (k0 + k[:, None] < K),
            other=0,
        )
        acc += tl.dot(x, weight, out_dtype=tl.int32)

    x_scale = tl.load(x_scale_ptr + m, mask=m < M, other=0.0).to(tl.float32)
    weight_scale = tl.load(weight_scale_ptr + n, mask=n < N, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + m[:, None] * N + n[None, :],
        acc.to(tl.float32) * x_scale[:, None] * weight_scale[None, :],
        mask=(m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _w4_matmul_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    packed_k = K // 2
    groups_k = K // GROUP_SIZE
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ko = k0 + k
        x = tl.load(
            x_ptr + m[:, None] * K + ko[None, :],
            mask=(m[:, None] < M) & (ko[None, :] < K),
            other=0.0,
        )
        packed = tl.load(
            weight_ptr + n[None, :] * packed_k + ko[:, None] // 2,
            mask=(n[None, :] < N) & (ko[:, None] < K),
            other=0,
        ).to(tl.int32)
        q = ((packed >> ((ko[:, None] & 1) * 4)) & 15) - 8
        scale = tl.load(
            scale_ptr + n[None, :] * groups_k + ko[:, None] // GROUP_SIZE,
            mask=(n[None, :] < N) & (ko[:, None] < K),
            other=0.0,
        ).to(x_ptr.dtype.element_ty)
        weight = q.to(x_ptr.dtype.element_ty) * scale
        acc += tl.dot(x, weight)
    tl.store(
        out_ptr + m[:, None] * N + n[None, :],
        acc,
        mask=(m[:, None] < M) & (n[None, :] < N),
    )


def _decode_block_k(k: int, bits: int) -> int:
    # Large reduction tiles amortize the one-program-per-output launch without
    # re-reading a packed row. INT4 needs more registers for unpacking/scales.
    limit = 4096 if bits == 8 else 2048
    return min(limit, triton.next_power_of_2(k))


def rwkv7_w8_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    original_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x_2d.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m >= 512:
        from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8

        x_quant, x_scale = per_token_quant_int8(x_2d)
        bm = bn = 128
        _w8a8_matmul_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            x_quant,
            weight,
            x_scale,
            scale,
            out,
            M=m,
            N=n,
            K=k,
            BM=bm,
            BN=bn,
            BK=64,
            GROUP_M=8,
            num_warps=8,
            num_stages=3,
        )
    elif m <= 8:
        _w8_row_kernel[(n,)](
            x_2d,
            weight,
            scale,
            out,
            M=m,
            N=n,
            K=k,
            BM=8,
            BK=_decode_block_k(k, 8),
            num_warps=1,
        )
    else:
        if m <= 32:
            bm, bn, bk, num_warps = 16, 64, 64, 4
        elif m <= 128 and k >= 4096:
            bm, bn, bk, num_warps = 64, 64, 128, 8
        elif m <= 512:
            bm, bn, bk, num_warps = 32, 64, 64, 4
        else:
            bm, bn, bk, num_warps = 64, 64, 64, 4
        _w8_matmul_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            x_2d,
            weight,
            scale,
            out,
            M=m,
            N=n,
            K=k,
            BM=bm,
            BN=bn,
            BK=bk,
            num_warps=num_warps,
            num_stages=2,
        )
    return out.view(*original_shape, n)


def rwkv7_w4_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    original_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x_2d.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m <= 8:
        block_k = _decode_block_k(k, 4)
        _w4_row_kernel[(n,)](
            x_2d,
            weight,
            scale,
            out,
            M=m,
            N=n,
            K=k,
            GROUP_SIZE=group_size,
            BM=8,
            BK=block_k,
            num_warps=2 if block_k == 2048 else 1,
        )
    else:
        bm = 16 if m <= 32 else (64 if m < 128 else 128)
        _w4_matmul_kernel[(triton.cdiv(m, bm), triton.cdiv(n, 64))](
            x_2d,
            weight,
            scale,
            out,
            M=m,
            N=n,
            K=k,
            GROUP_SIZE=group_size,
            BM=bm,
            BN=64,
            BK=32,
            num_warps=4 if bm < 128 else 8,
            num_stages=2,
        )
    return out.view(*original_shape, n)


def online_quantize_rwkv7_w4_weight(
    weight: torch.Tensor, group_size: int = _W4_GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack an ordinary ``[out, in]`` weight as symmetric groupwise INT4."""
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("RWKV-7 W4 requires a floating-point rank-2 weight")
    out_features, in_features = weight.shape
    if in_features % group_size or in_features % 2:
        raise ValueError(
            f"RWKV-7 W4 requires input width divisible by {group_size}; "
            f"got {in_features}"
        )
    grouped = weight.float().view(out_features, in_features // group_size, group_size)
    scale = (grouped.abs().amax(dim=-1) / 7.0).clamp_min(torch.finfo(torch.float32).eps)
    q = (
        (grouped / scale[..., None])
        .round()
        .clamp_(-7, 7)
        .to(torch.int16)
        .view(out_features, in_features)
    )
    unsigned = (q + 8).to(torch.uint8)
    packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    return packed.contiguous(), scale.to(torch.float16).contiguous()


class RWKV7W8LinearMethod(LinearMethodBase):
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        scale = ChannelQuantScaleParameter(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

    def apply(self, layer, x, bias=None):
        out = rwkv7_w8_linear(x, layer.weight, layer.weight_scale)
        return out if bias is None else out + bias


class RWKV7W4LinearMethod(LinearMethodBase):
    def __init__(self, group_size: int = _W4_GROUP_SIZE):
        self.group_size = group_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if input_size_per_partition % self.group_size:
            raise ValueError(
                "RWKV-7 W4 input partition must be divisible by group_size"
            )
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        scale = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.float16,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

    def apply(self, layer, x, bias=None):
        out = rwkv7_w4_linear(x, layer.weight, layer.weight_scale, self.group_size)
        return out if bias is None else out + bias


class _RWKV7NativeConfig(QuantizationConfig):
    bits: int

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}
        self.group_size = int(config.get("group_size", _W4_GROUP_SIZE))

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(config)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase

        if not isinstance(layer, LinearBase):
            return None
        if self.bits == 8:
            return RWKV7W8LinearMethod()
        return RWKV7W4LinearMethod(self.group_size)

    def get_scaled_act_names(self) -> List[str]:
        return []


class RWKV7W8Config(_RWKV7NativeConfig):
    bits = 8

    @classmethod
    def get_name(cls) -> str:
        return "rwkv7_w8"


class RWKV7W4Config(_RWKV7NativeConfig):
    bits = 4

    @classmethod
    def get_name(cls) -> str:
        return "rwkv7_w4"
