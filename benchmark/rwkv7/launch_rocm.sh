#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODEL_PATH [dense|w8|w4|native-w8|native-w4] [additional sglang arguments...]" >&2
  exit 2
fi

model_path=$1
mode=${2:-dense}
shift
if [[ $# -gt 0 ]]; then
  shift
fi

# RDNA installations often do not ship an AITER build for their exact gfx
# target.  Keep the portable Triton path as the default while allowing CDNA
# users with a complete AITER installation to override either variable.
export SGLANG_USE_AITER=${SGLANG_USE_AITER:-0}
export USE_ROCM_AITER_ROPE_BACKEND=${USE_ROCM_AITER_ROPE_BACKEND:-0}
export SGLANG_RWKV7_BNB_POLICY=${SGLANG_RWKV7_BNB_POLICY:-accuracy}
python_bin=${SGLANG_PYTHON:-python}
chunked_prefill_size=${SGLANG_RWKV7_ROCM_CHUNKED_PREFILL_SIZE:-8192}

quant_args=()
state_dtype=${SGLANG_RWKV7_SSM_DTYPE:-}
case "$mode" in
  dense)
    state_dtype=${state_dtype:-float32}
    ;;
  w8)
    state_dtype=${state_dtype:-float32}
    quant_args=(
      --quantization bitsandbytes
      --load-format bitsandbytes
      --model-loader-extra-config
      '{"load_in_8bit":true,"load_in_4bit":false}'
    )
    ;;
  w4)
    state_dtype=${state_dtype:-float32}
    quant_args=(
      --quantization bitsandbytes
      --load-format bitsandbytes
      --model-loader-extra-config
      '{"load_in_8bit":false,"load_in_4bit":true,"bnb_4bit_compute_dtype":"float16","bnb_4bit_quant_type":"nf4"}'
    )
    ;;
  native-w8)
    state_dtype=${state_dtype:-float16}
    quant_args=(--quantization rwkv7_w8)
    ;;
  native-w4)
    state_dtype=${state_dtype:-float16}
    quant_args=(--quantization rwkv7_w4)
    ;;
  *)
    echo "unsupported mode: $mode (expected dense, w8, w4, native-w8, or native-w4)" >&2
    exit 2
    ;;
esac

exec "$python_bin" -m sglang.launch_server \
  --model-path "$model_path" \
  --trust-remote-code \
  --attention-backend triton \
  --dtype float16 \
  --mamba-ssm-dtype "$state_dtype" \
  --max-running-requests 8 \
  --chunked-prefill-size "$chunked_prefill_size" \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 8 \
  --cuda-graph-backend-prefill disabled \
  "${quant_args[@]}" \
  "$@"
