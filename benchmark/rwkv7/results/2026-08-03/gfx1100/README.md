# RWKV-7 SGLang ROCm acceptance: gfx1100

This directory contains the first native ROCm acceptance run for the RWKV-7
SGLang implementation. The run used RWKV-7 G1h 1.5B on one 48 GB gfx1100 GPU
with ROCm 7.2, PyTorch 2.9.1, Triton 3.5.1, batch size 8, 128 decode tokens,
two warmups, five measured repeats, and a cache flush before every sample.
Dense, W8, and W4 used the same 8192-token chunked-prefill setting.

## End-to-end results

| Prompt/request | Dense prefill | Native W8 | W8/dense | Native W4 | W4/dense |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 10,446.6 tok/s | 11,354.3 tok/s | 1.087x | 11,165.6 tok/s | 1.069x |
| 512 | 13,777.6 tok/s | 14,669.2 tok/s | 1.065x | 14,586.6 tok/s | 1.059x |
| 2048 | 13,855.5 tok/s | 14,127.7 tok/s | 1.020x | 14,059.5 tok/s | 1.015x |

| Prompt/request | Dense decode | Native W8 | W8/dense | Native W4 | W4/dense |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 693.6 tok/s | 762.4 tok/s | 1.099x | 760.1 tok/s | 1.096x |
| 512 | 692.7 tok/s | 761.4 tok/s | 1.099x | 759.5 tok/s | 1.096x |
| 2048 | 691.9 tok/s | 760.4 tok/s | 1.099x | 759.2 tok/s | 1.097x |

The native quantized modes are faster than dense fp16 in every measured cell.

## Weight memory and accuracy

| Mode | Loaded weight memory | Change from dense |
| --- | ---: | ---: |
| Dense fp16 | 2.86 GB | baseline |
| Native W8 | 2.42 GB | -15.4% |
| Native W4 | 2.40 GB | -16.1% |

Both quantized modes passed the teacher-forced alignment gate:

- Native W8: maximum chosen-token log-probability error 0.0784, mean top-10
  overlap 0.9813, teacher-forced top-1 agreement 1.0000.
- Native W4: maximum chosen-token log-probability error 0.1486, mean top-10
  overlap 0.9633, teacher-forced top-1 agreement 0.9922.

Dynamic-batch duplicate isolation, chunked-prefill cold/warm matching, state
cache hits, mixed-length compaction, request abort, and post-abort state reuse
also passed for both quantized modes.

## ROCm-specific implementation notes

- The WKV and sparse SqReLU FFN extensions compile only the active gfx target
  unless `PYTORCH_ROCM_ARCH` explicitly requests a multi-architecture build.
- Sparse FFN compaction handles both RDNA wave32 and CDNA wave64 execution.
- HIP uses a 32-bit compare-and-swap loop for packed half2 accumulation because
  gfx1100 does not expose CUDA's half2 `atomicAdd` overload.
- Large gfx1100 W4 prefills keep persistent weights packed and dequantize one
  projection into a transient fp16 buffer before rocBLAS GEMM. This preserves
  the model-memory reduction while avoiding repeated W4 unpack/scale work in
  every matrix tile.

See `environment.json`, `model-memory.jsonl`, and the raw JSON/JSONL files for
the complete samples and validation details.
