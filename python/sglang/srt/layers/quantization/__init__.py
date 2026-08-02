# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
from __future__ import annotations

import builtins
import inspect
import logging
from typing import TYPE_CHECKING, Dict, Optional, Type

import torch

logger = logging.getLogger(__name__)


# Define empty classes as placeholders when vllm is not available
class DummyConfig:
    def override_quantization_method(self, *args, **kwargs):
        return None


CompressedTensorsConfig = DummyConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.platforms import current_platform
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_mps,
    is_npu,
    mxfp_supported,
)

_cuda_compute_capability = (
    torch.cuda.get_device_capability()
    if is_cuda() and torch.cuda.is_available() and torch.version.hip is None
    else None
)
_legacy_cuda = _cuda_compute_capability is not None and _cuda_compute_capability < (
    7,
    5,
)

# A dense model must still be able to start when the optional kernel wheel is
# absent or when a wheel for the wrong CUDA/SM variant was installed. Importing
# several quantizer modules eagerly imports ``sgl_kernel``; a loader error there
# used to abort model-config construction before quantization was even selected.
# Keep BitsAndBytes and the unquantized path available, and expose native kernel
# quantizers only after the extension has proved loadable on the current GPU.
_sgl_kernel_available = True
if is_cuda() and not _legacy_cuda:
    try:
        import sgl_kernel as _sgl_kernel  # noqa: F401
    except (ImportError, OSError) as exc:
        _sgl_kernel_available = False
        logger.warning(
            "Native GPU quantizers are unavailable (%s). Dense and "
            "BitsAndBytes model loading remain enabled; install the matching "
            "sglang-kernel wheel to enable native quantization.",
            exc,
        )

# Current sgl-kernel wheels start at SM75. Importing quantizers backed by that
# package on Pascal/Volta fails before an unquantized or BitsAndBytes model can
# even be constructed. Keep the legacy registry intentionally small; unsupported
# methods fail at configuration time instead of breaking every model import.
AutoRoundConfig = AWQConfig = AWQCPUConfig = AWQMarlinConfig = DummyConfig
BlockInt8Config = Fp8Config = GGUFConfig = DummyConfig
CPUGPTQConfig = GPTQAscendConfig = GPTQConfig = GPTQMarlinConfig = DummyConfig
HummingConfig = MarlinConfig = MlxQuantizationConfig = DummyConfig
ModelOptFp4Config = ModelOptFp8Config = ModelOptMixedPrecisionConfig = DummyConfig
ModelSlimConfig = MoeWNA16Config = Mxfp4Config = DummyConfig
Mxfp4W4A8Config = Mxfp4W4A4Config = NvFp4OnlineConfig = DummyConfig
PetitNvFp4Config = QuarkConfig = QuarkInt4Fp8Config = DummyConfig
W4AFp8Config = W8A8Fp8Config = W8A8Int8Config = DummyConfig

from sglang.srt.layers.quantization.bitsandbytes import BitsAndBytesConfig

if not _legacy_cuda and (not is_cuda() or _sgl_kernel_available):
    from sglang.srt.layers.quantization.auto_round import AutoRoundConfig
    from sglang.srt.layers.quantization.awq import (
        AWQConfig,
        AWQCPUConfig,
        AWQMarlinConfig,
    )
    from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8Config
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.layers.quantization.gguf import GGUFConfig
    from sglang.srt.layers.quantization.gptq import (
        CPUGPTQConfig,
        GPTQAscendConfig,
        GPTQConfig,
        GPTQMarlinConfig,
    )
    from sglang.srt.layers.quantization.humming import HummingConfig
    from sglang.srt.layers.quantization.marlin_utils import MarlinConfig
    from sglang.srt.layers.quantization.mlx import MlxQuantizationConfig
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptFp4Config,
        ModelOptFp8Config,
        ModelOptMixedPrecisionConfig,
    )
    from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
    from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
    from sglang.srt.layers.quantization.npu_mxfp4 import Mxfp4W4A8Config
    from sglang.srt.layers.quantization.npu_mxfp4_w4a4 import Mxfp4W4A4Config
    from sglang.srt.layers.quantization.nvfp4_online import NvFp4OnlineConfig
    from sglang.srt.layers.quantization.petit import PetitNvFp4Config

    try:
        from sglang.srt.layers.quantization.quark.quark import QuarkConfig
    except (ImportError, OSError) as exc:
        if not is_hip():
            raise
        logger.warning(
            "Quark quantization is unavailable in this ROCm environment (%s). "
            "Other ROCm quantizers remain enabled.",
            exc,
        )
    from sglang.srt.layers.quantization.quark_int4fp8_moe import QuarkInt4Fp8Config
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config
    from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config
    from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

_is_mxfp_supported = mxfp_supported()

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput

# Base quantization methods
BASE_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "fp8": Fp8Config,
    "mxfp8": Fp8Config,
    "blockwise_int8": BlockInt8Config,
    "modelopt": ModelOptFp8Config,  # Auto-detect, defaults to FP8
    "modelopt_fp8": ModelOptFp8Config,
    "modelopt_fp4": ModelOptFp4Config,
    "nvfp4_online": NvFp4OnlineConfig,
    "modelopt_mixed": ModelOptMixedPrecisionConfig,
    "w8a8_int8": W8A8Int8Config,
    "w8a8_fp8": W8A8Fp8Config,
    "awq": AWQConfig,
    "awq_marlin": AWQMarlinConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "gguf": GGUFConfig,
    "gptq": GPTQConfig,
    "gptq_marlin": GPTQMarlinConfig,
    "marlin": MarlinConfig,
    "moe_wna16": MoeWNA16Config,
    "compressed-tensors": CompressedTensorsConfig,
    "w4afp8": W4AFp8Config,
    "petit_nvfp4": PetitNvFp4Config,
    "quark": QuarkConfig,
    "quark_mxfp4": QuarkConfig,
    "auto-round": AutoRoundConfig,
    "auto-round-int8": W8A8Int8Config,
    "modelslim": ModelSlimConfig,
    "quark_int4fp8_moe": QuarkInt4Fp8Config,
    "humming": HummingConfig,
    "mxfp_w4a8": Mxfp4W4A8Config,
}

if QuarkConfig is DummyConfig:
    BASE_QUANTIZATION_METHODS.pop("quark", None)
    BASE_QUANTIZATION_METHODS.pop("quark_mxfp4", None)

if _legacy_cuda or (is_cuda() and not _sgl_kernel_available):
    BASE_QUANTIZATION_METHODS = {
        name: config
        for name, config in BASE_QUANTIZATION_METHODS.items()
        if config is not DummyConfig
    }


if (
    is_cpu()
    or (is_cuda() and not _legacy_cuda and _sgl_kernel_available)
    or (_is_mxfp_supported and is_hip())
):
    BASE_QUANTIZATION_METHODS.update(
        {
            "mxfp4": Mxfp4Config,
        }
    )


if is_npu():
    BASE_QUANTIZATION_METHODS.update(
        {
            "gptq": GPTQAscendConfig,
            # On NPU, `mxfp4` means single-level W4A4 MXFP4 for dense LLM (the
            # upstream `Mxfp4Config` OCP-MoE path is only registered on
            # cpu/cuda/hip above, so there is no collision here).
            "mxfp4": Mxfp4W4A4Config,
        }
    )


if is_mps():
    BASE_QUANTIZATION_METHODS.update(
        {
            "mlx_q4": MlxQuantizationConfig,
            "mlx_q8": MlxQuantizationConfig,
        }
    )

# subset of above quant methods, supported on CPU
CPU_QUANTIZATION_METHODS = {
    "fp8": Fp8Config,
    "w8a8_int8": W8A8Int8Config,
    "compressed-tensors": CompressedTensorsConfig,
    "awq": AWQCPUConfig,
    "gptq": CPUGPTQConfig,
    "mxfp4": Mxfp4Config,
}

if _legacy_cuda:
    CPU_QUANTIZATION_METHODS = {}

QUANTIZATION_METHODS = {**BASE_QUANTIZATION_METHODS}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    from sglang.srt.utils import is_cpu

    if is_cpu() and cpu_has_amx_support():
        if quantization not in CPU_QUANTIZATION_METHODS:
            raise ValueError(
                f"Invalid quantization method on CPU: {quantization}. "
                f"Available methods on CPU: {list(QUANTIZATION_METHODS.keys())}"
            )
        else:
            return CPU_QUANTIZATION_METHODS[quantization]

    if current_platform.is_out_of_tree():
        config = current_platform.get_quantization_config(quantization)

        # If the platform has a quantization config, use it else use the default
        if config is not None:
            return config

    return QUANTIZATION_METHODS[quantization]


original_isinstance = builtins.isinstance
