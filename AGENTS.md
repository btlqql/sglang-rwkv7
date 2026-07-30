# RWKV-7 SGLang maintenance instructions

## Primary objective

Treat `RWKV7_HF_PARITY.md` as the acceptance contract. The SGLang RWKV-7
inference path must match or exceed every applicable inference feature and
performance path in `rwkv-rs/hf-adapter`, then retain SGLang-specific serving
advantages.

## Source of truth

- Record the exact HF adapter commit for every comparison.
- The initial pinned baseline is
  `f1b49bc52a050d09a6739bc4859850f5dc50e7ef`.
- Mathematical correctness is checked against the HF adapter and official
  RWKV-7 behavior; serving behavior is checked through SGLang end to end.
- Do not copy experimental defaults blindly. Reuse algorithms and kernel ideas,
  but adapt layouts to SGLang state pools, scheduler metadata, CUDA Graph
  buffers, tensor parallelism, and pipeline parallelism.

## Engineering priorities

1. Matched HF/SGLang benchmark harness and red-cell analyzer.
2. Fused dense decode and graph replay.
3. Fixed-shape and chunked/DPLR prefill.
4. Native W8 speed and memory policies.
5. Native W4 speed and memory policies.
6. Dynamic batching, recurrent radix cache, TP/PP, and speculation integration.
7. Card-specific validation and conservative fallback promotion.

The performance route is native kernels and graph integration, not additional
Python wrapper layers.

## Mandatory promotion gates

- Exact dense greedy output; declared quantized tolerance and same-next-token
  evidence.
- End-to-end speedup on the target card and workload, not only an isolated
  microbenchmark.
- No regression at batch sizes 1, 2, 4, and 8.
- Prefill, decode, TTFT, TPOT, throughput, footprint, active state memory,
  configured pool memory, and peak VRAM are all reported.
- W8/W4 must reduce memory and be no slower than dense and matched HF quant.
- Dynamic batching, chunked prefill, state-cache cold/warm equality, and slot
  reuse remain correct.
- GPU-specific defaults require an exact-device result; all other devices retain
  a safe fallback.

## Test expectations

- CPU/import tests must not require CUDA-only optional packages.
- Kernel tests require reference outputs, adversarial shapes, non-contiguous
  inputs where supported, and explicit tolerances.
- Registered serving tests cover generation, batching, cache hits, quantized
  loading, graph replay, and speculation.
- Performance PRs add JSONL evidence and update the parity matrix or benchmark
  documentation.
- Missing hardware may be marked unverified, never passed.

## Scope boundary

SGLang does not reimplement HF Trainer, PEFT, TRL, or DeepSpeed training. It must
load checkpoints produced by those workflows, including base, merged adapter,
and supported quantized formats. Apple/MLX, AMD/ROCm, and CPU performance remain
explicit deliverables; they may not be silently removed from the overall goal.

## Repository hygiene

- Keep upstream SGLang attribution and license information intact.
- Keep `upstream` fetch-only and never force-push shared public branches.
- Use focused commits with tests and reproducible commands.
- Do not commit credentials, local paths, private machine addresses, or raw
  checkpoint files.
