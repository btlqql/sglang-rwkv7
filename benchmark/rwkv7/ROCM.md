# RWKV-7 on ROCm

RWKV-7 can use SGLang's native CUDA/HIP and portable Triton serving paths on
AMD GPUs without `sgl-kernel`, bitsandbytes, or a device-specific AITER build.
The fallback covers dense and native W8/W4 generation, decode graphs, dynamic
batching, chunked prefill, recurrent state cache, abort/reuse lifecycle
handling, xgrammar masking, and the imports used by the speculative scheduler.

## Environment

The initial RDNA validation used:

- GPU target: `gfx1100`, 48 GB VRAM;
- ROCm 7.2.1;
- PyTorch 2.9.1 (HIP 7.2 build);
- Triton 3.5.1;
- bitsandbytes 0.49.2 for the legacy optional W8/W4 lanes.

ROCm support in bitsandbytes is still marked preview by that project. Install
0.49 or newer only when using the legacy `w8` or `w4` launch modes:

```bash
python -m pip install 'bitsandbytes>=0.49,<0.50'
python -m bitsandbytes
```

The diagnostic must report that ROCm is callable. Dense serving and the
`native-w8`/`native-w4` modes do not depend on bitsandbytes.

## Launch

The helper defaults to the portable Triton backend and decode graphs through
batch size 8:

```bash
benchmark/rwkv7/launch_rocm.sh /path/to/rwkv7-hf dense
benchmark/rwkv7/launch_rocm.sh /path/to/rwkv7-hf w8
benchmark/rwkv7/launch_rocm.sh /path/to/rwkv7-hf w4
benchmark/rwkv7/launch_rocm.sh /path/to/rwkv7-hf native-w8
benchmark/rwkv7/launch_rocm.sh /path/to/rwkv7-hf native-w4
```

Additional SGLang arguments can follow the mode.  PyTorch retains the CUDA API
names for HIP graph capture, so SGLang's flags are still named
`--cuda-graph-*` on ROCm.

The validated gfx1100 default uses an 8192-token chunked-prefill budget. Set a
smaller budget for a lower-memory card or a larger checkpoint:

```bash
SGLANG_RWKV7_ROCM_CHUNKED_PREFILL_SIZE=2048 \
  benchmark/rwkv7/launch_rocm.sh /path/to/rwkv7-hf native-w8
```

The helper uses `python` from `PATH`.  Set `SGLANG_PYTHON` when SGLang is in a
dedicated environment that is not activated in the current shell:

```bash
SGLANG_PYTHON=/opt/venv/bin/python \
  benchmark/rwkv7/launch_rocm.sh /path/to/rwkv7-hf dense
```

The native modes default recurrent state to FP16 so the packed-varlen HIP WKV
kernel can be selected. Override this explicitly when a stricter state lane is
required:

```bash
SGLANG_RWKV7_SSM_DTYPE=float32 \
  benchmark/rwkv7/launch_rocm.sh /path/to/rwkv7-hf native-w8
```

The script disables AITER by default because consumer RDNA wheels may omit the
required GEMM modules.  A complete CDNA installation can opt back in:

```bash
SGLANG_USE_AITER=1 USE_ROCM_AITER_ROPE_BACKEND=1 \
  benchmark/rwkv7/launch_rocm.sh /path/to/rwkv7-hf dense
```

## Validation

Run these against the server:

```bash
python benchmark/rwkv7/verify_serving_features.py \
  --base-url http://127.0.0.1:30000 --max-new-tokens 24

python benchmark/rwkv7/verify_quant_alignment.py \
  --model /path/to/rwkv7-hf \
  --base-url http://127.0.0.1:30000 \
  --dtype float16 --max-new-tokens 24 --top-k 10

python benchmark/rwkv7/bench_acceptance_matrix.py \
  --base-url http://127.0.0.1:30000 \
  --batch-sizes 1,8 --prompt-lengths 128,2048 \
  --decode-lengths 128 --warmups 2 --repeats 5
```

The dense correctness run covered 0.4B, 1.5B, and 2.9B checkpoints.  The 0.4B
and 2.9B models passed the complete serving-feature gate, including an exact
128-token state-cache continuation.  The 1.5B model passed HF teacher-forced
alignment, with maximum token log-probability error 0.0082 and top-1 agreement
1.0.  Its synthetic cache prompt has a near-tied third-token argmax: cold and
cached logits differ by normal low-precision reduction noise before the two
greedy sequences diverge, so that one strict token-exact fixture remains a
documented numerical edge case rather than a cache-isolation failure.

### Dense FP16 performance

Median server-side results with decode graphs enabled:

| Model | Batch | Prompt | Decode tok/s | Prefill tok/s |
| --- | ---: | ---: | ---: | ---: |
| 0.4B | 1 | 128 | 183.2 | 5,020 |
| 0.4B | 8 | 128 | 1,321.6 | 22,341 |
| 1.5B | 1 | 128 | 79.9 | 4,672 |
| 1.5B | 8 | 128 | 585.1 | 10,124 |
| 2.9B | 1 | 128 | 49.1 | 3,649 |
| 2.9B | 8 | 128 | 367.8 | 6,747 |

On the 0.4B checkpoint, enabling the decode graph improved batch-1 decode from
73.2 to 183.2 tok/s (2.50x) and batch-8 aggregate decode from 580.5 to 1,321.6
tok/s (2.28x).

### BitsAndBytes W8/W4

RWKV recurrent attention is more sensitive to projection error than a plain
transformer MLP.  `SGLANG_RWKV7_BNB_POLICY` therefore exposes three policies:

- `accuracy` (default): W8 quantizes FFN contraction weights; W4 quantizes a
  sparse lane of middle FFN contraction layers;
- `balanced`: quantizes both FFN projections;
- `speed`: quantizes all large attention and FFN projections.

The 1.5B accuracy lanes passed the quant alignment gate:

| Mode | Max logprob error | Mean top-10 overlap | Top-1 agreement | B1 decode | B8 decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense FP16 | 0.0082 | 0.9990 | 1.0000 | 79.9 | 585.1 |
| W8 accuracy | 0.2322 | 0.9417 | 0.9792 | 74.6 | 547.1 |
| W4 accuracy | 0.1903 | 0.9688 | 0.9896 | 80.5 | 534.3 |

The model-load telemetry measured 2.93 GB for dense FP16, 2.56 GB for W8
accuracy, and 2.87 GB for W4 accuracy.  Full-model W8/W4 reduced model memory
to 1.72/1.20 GB, but did not pass the same accuracy gate; full W4 was also much
slower at batch 8.  Consequently, full quantization and batch-8 quant speed are
not production acceptance claims for the legacy bitsandbytes path.

### Native W8/W4 kernels

The native modes avoid the bitsandbytes and AITER dependency:

- decode and small batches use a row-streaming weight-only Triton kernel;
- large prefills dynamically quantize activations per token and use an INT8
  dot-product kernel with INT32 accumulation;
- W8 weights use symmetric per-output-channel scales;
- W4 expansion weights use symmetric group-32 packing, while protected
  contraction projections use the native W8 path;
- recurrent attention, low-rank controls and edge FFN blocks stay FP16 under
  the default accuracy policy; the native row-streaming kernel also covers the
  wide LM head.

The CUDA/HIP WKV and sparse SqReLU extensions are local JIT fast paths. If the
installed compiler cannot build one of them, dispatch fails closed to the
portable Triton WKV or dense FFN implementation.

On ROCm, the JIT loader targets only the active gfx architecture unless the
operator explicitly sets `PYTORCH_ROCM_ARCH`. This avoids compiling every
architecture embedded in a broad PyTorch wheel. Sparse compaction supports
both RDNA wave32 and CDNA wave64. gfx1100 uses a compare-and-swap half2
accumulator because HIP does not provide CUDA's packed-half `atomicAdd`
overload on that target.

For large gfx1100 W4 prefills, packed weights remain the persistent model
storage. One projection is expanded into a transient fp16 buffer and dispatched
to the tuned rocBLAS GEMM, then released. Small batches continue to use the
row-streaming packed-W4 kernel. This thresholded path retains the model-memory
reduction while removing the repeated unpack/scale cost from every large GEMM
tile.

### Native gfx1100 acceptance

RWKV-7 G1h 1.5B was measured at batch 8 with 128 decode tokens, prompt lengths
128/512/2048, two warmups, five repeats, and a cache flush before each sample.
All three modes used the same 8192-token chunked-prefill budget.

| Prompt | Dense prefill | Native W8 | W8/dense | Native W4 | W4/dense |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 10,446.6 | 11,354.3 | 1.087x | 11,165.6 | 1.069x |
| 512 | 13,777.6 | 14,669.2 | 1.065x | 14,586.6 | 1.059x |
| 2048 | 13,855.5 | 14,127.7 | 1.020x | 14,059.5 | 1.015x |

Native W8 decode was 760.4-762.4 tok/s and native W4 was 759.2-760.1 tok/s,
versus 691.9-693.6 tok/s for dense: both quantized paths were about 1.10x dense
in every decode cell. Loaded weight memory fell from 2.86 GB to 2.42 GB for W8
and 2.40 GB for W4.

The strict quantized alignment gates passed. W8 reported maximum chosen-token
log-probability error 0.0784, mean top-10 overlap 0.9813, and top-1 agreement
1.0000. W4 reported 0.1486, 0.9633, and 0.9922 respectively. Dynamic batching,
chunked-prefill cold/warm matching, state-cache hits, mixed-length compaction,
abort, and post-abort reuse also passed. Raw evidence is under
`results/2026-08-03/gfx1100/`.

## Current limitations

- Native `sgl-kernel` quantizers require a matching ROCm wheel. Missing
  `sgl-kernel` no longer prevents dense, BitsAndBytes, or RWKV-native W8/W4
  startup.
- Quark remains unavailable when the installed AITER package lacks
  `aiter.ops.triton.gemm`; other available ROCm quantizers stay registered.
- gfx1100 RDNA has complete native dense/W8/W4 evidence. CDNA and other RDNA
  targets still require their own performance runs; successful HIP compilation
  alone is not treated as acceptance for another architecture.
