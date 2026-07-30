# RWKV-7 HF-to-SGLang parity contract

This document is the performance contract for the RWKV-7 implementation in
this repository. The objective is not merely to load the same checkpoints. The
SGLang path must implement every production-relevant inference optimization in
the RWKV-7 Hugging Face adapter, meet or exceed it under matched conditions,
and retain SGLang's serving capabilities.

## Reference baseline

- Reference repository: <https://github.com/rwkv-rs/hf-adapter>
- Initial pinned performance baseline:
  [`f1b49bc52a050d09a6739bc4859850f5dc50e7ef`](https://github.com/rwkv-rs/hf-adapter/commit/f1b49bc52a050d09a6739bc4859850f5dc50e7ef)
- Checkpoint, tokenizer, prompt tokens, generated-token count, dtype, device,
  warm-up count, measurement count, and synchronization policy must match.
- A newer HF commit may replace the pin only when the benchmark result records
  the exact new SHA and both sides are rerun.

HF training implementations such as Trainer, PEFT, TRL, and DeepSpeed are not
embedded into SGLang. Their output checkpoints must load directly and produce
aligned inference results.

## Definition of complete parity

Parity is complete only when all of the following are true:

1. Every applicable HF inference optimization has an equivalent SGLang path or
   a measured SGLang-native replacement.
2. Every required benchmark cell contains a real result. Unsupported, missing,
   skipped, estimated, and microbenchmark-only cells do not pass.
3. Dense FP16/BF16 prefill, decode, and end-to-end serving meet or exceed the HF
   baseline on the same card.
4. W8 and W4 reduce memory and are no slower than both the corresponding HF
   quantized path and the SGLang dense reference.
5. Correctness, dynamic batching, chunked prefill, recurrent state caching,
   CUDA Graphs, TP/PP, and speculative decoding remain valid.
6. Card-specific promotion never regresses already-supported cards or the
   generic fallback.

## Optimization inventory

| HF performance surface | SGLang requirement | Current state |
| --- | --- | --- |
| `native_jit` / `native_graph` token decode | Use SGLang CUDA Graph runners with static state-pool addressing and no per-token host synchronization | Partial |
| Fused decode norm and time-mix | Fuse safe normalization, shift-mix, low-rank controls, and state preparation around the serving tensor layout | Partial |
| Fused recurrent update and output preparation | Join WKV state update/readout with normalization, correction, gate, and output preparation when end-to-end profiling proves a win | Partial |
| Fixed-shape whole-prefill graph | Capture profitable fixed prompt/batch shapes without breaking chunked prefill or cache handoff | Missing |
| Split recurrent prefill scan | Add card-specific scan layouts and autotuning for sequence mode | Partial |
| DPLR/chunked prefill | Implement the long-sequence path and preserve exact chunk-to-decode state handoff | Missing |
| Fused projection/LoRA experiments | Port only deeper fusions that beat vendor GEMM end to end; do not promote isolated slower kernels | Missing |
| Fused FFN experiments | Keep cuBLAS/vendor GEMM unless a larger graph/layer fusion wins on serving workloads | Missing |
| Native W8 speed policy | Add packed/fused projection kernels and card policy; footprint must fall and all-phase speed must pass | Partial: online W8A8 exists |
| Native W4 speed policy | Use Marlin/TorchAO/custom packed kernels as appropriate and select by measured card/shape policy | Partial: online Marlin exists |
| W8/W4 memory policy | Retain a maximum-compression lane separately from the speed lane | Partial: BitsAndBytes fallback exists |
| SM70 projection/FFN policy | Match the HF V100 fast path without assuming tensor-core features unavailable on Volta | Partial |
| Ada fixed-shape and quant policy | Reproduce the HF 4090 dense and W8/W4 wins, then validate 4080 independently | Missing matrix |
| Blackwell quant matrix | Reproduce the HF 5090 216-cell matrix and close every red cell | Missing |
| Apple MLX/Metal performance | Track as an explicit backend/bridge deliverable rather than silently excluding it | Missing |
| ROCm/AMD policy | Provide HIP-compatible recurrence, dense, and quantized paths with real AMD results | Missing |
| Dynamic batch state operations | Preserve state isolation under select, reorder, drop, compact, copy-on-write, and slot reuse | Implemented; performance gate missing |
| Recurrent radix cache | Require exact cold/warm continuation and measured hit rate/latency/footprint | Implemented; full matrix missing |
| STANDALONE speculation | Measure acceptance, rollback/commit cost, state scratch memory, and net throughput | Functional top-k 1; performance gate missing |
| TP/PP | Require exact output and useful scaling for dense and quantized serving | Functional V100 coverage; performance matrix missing |

`Partial` means that code exists but the complete matched HF matrix and
promotion evidence do not yet exist. It must not be reported as full parity.

## Core 216-cell matrix

The core matrix contains:

- models: 1.5B, 2.9B, and 7.2B;
- prompt lengths: 128, 512, and 2048 tokens;
- decode lengths: 128 and 512 tokens;
- active batch sizes: 1, 2, 4, and 8;
- modes: dense FP16/BF16, W8, and W4.

This produces `3 x 3 x 2 x 4 x 3 = 216` cells per hardware target. Add 0.4B
for kernel/debug coverage and 13.3B wherever memory permits. Serving sweeps must
also include changing active batches rather than fixed-batch replay only.

## Required metrics and gates

### Correctness

- Dense greedy token IDs: exact match.
- Generated-token log-probability maximum error: `<= 0.05`.
- Mean top-10 token-set overlap: `>= 0.80`.
- W8/W4: same next token for every measured decode step plus a recorded cosine
  or logit-error metric.
- Full prefill and equivalent chunked prefill must produce the same next token
  and recurrent state within the declared tolerance.

### Performance

For each cell, record at least:

- prefill tokens/s and latency;
- decode tokens/s, milliseconds/token, and aggregate output tokens/s;
- TTFT and TPOT;
- end-to-end request throughput;
- model footprint, active recurrent-state memory, configured state-pool memory,
  and peak device memory;
- CUDA Graph capture/replay status and graph hit rate;
- radix/state-cache hit rate and saved prefill tokens;
- speculative acceptance rate and net speedup when enabled.

Acceptance ratios:

```text
dense_prefill_ratio = sglang_dense_prefill / hf_dense_prefill
dense_decode_ratio  = sglang_dense_decode  / hf_dense_decode
quant_hf_ratio       = sglang_quant         / hf_quant
quant_dense_ratio    = sglang_quant         / sglang_dense
```

Every promoted cell must satisfy:

- `dense_prefill_ratio >= 1.00`;
- `dense_decode_ratio >= 1.00`;
- SGLang end-to-end throughput `>= 1.00x` HF;
- SGLang TTFT and TPOT no worse than HF;
- `quant_hf_ratio >= 1.00` and `quant_dense_ratio >= 1.00`;
- W8 and W4 model footprint lower than dense;
- no correctness or serving-feature regression.

Because SGLang reserves state for concurrent requests, memory reports must show
both configured server peak and normalized per-active-request memory. Comparing
an HF batch with an unreported oversized server pool is invalid.

## Hardware acceptance ladder

| Hardware family | Required role |
| --- | --- |
| Pascal / SM61 | Compatibility fallback and W8/W4 memory validation |
| Volta / SM70 (V100) | Legacy CUDA optimized regression baseline |
| Turing / SM75 | Consumer/datacenter compatibility and quantization |
| Ampere / SM80-SM86 | A100/A800/A6000 dense, quantized, TP/PP, and graph matrix |
| Hopper / SM90 | H100 dense, quantized, long-context, and distributed matrix |
| Ada / SM89 | RTX 4080/4090 fixed-shape prefill, decode, and quant speed matrix |
| Blackwell / SM100+ | RTX 5090 216-cell matrix and native W8/W4 policy |
| AMD ROCm | HIP recurrence, dense, W8/W4, batching, and state-cache matrix |
| Apple Silicon | MLX/Metal serving backend or a maintained bridge with matched evidence |
| CPU | Import, correctness, and documented fallback performance |

Kernel policy must dispatch by capability and measured shape. A block size or
fusion promoted on one exact GPU must stay opt-in elsewhere until that family
passes the same end-to-end gates.

## SGLang serving-superset gates

Matching single-process HF speed is necessary but not sufficient. The following
must also pass:

- continuous/dynamic batching with changing active batch sizes;
- chunked prefill under scheduler pressure;
- recurrent radix cache cold/warm equality and meaningful hit rates;
- state slot allocation, release, reuse, copy-on-write, and compaction;
- CUDA Graph replay for supported decode and prefill shapes;
- TP and PP correctness plus scaling evidence;
- dense and quantized OpenAI-compatible serving;
- STANDALONE speculative decoding with net positive throughput;
- cancellation, timeout, mixed prompt length, and long-running stability tests.

## Implementation order

1. **Matched harness:** emit one JSONL schema from HF and SGLang and generate a
   missing/red-cell report for the 216-cell matrix.
2. **Dense decode:** port/equivalently implement native graph and profitable
   recurrent-output fusion before adding more wrappers.
3. **Dense prefill:** add fixed-shape capture, split scan, and DPLR/chunked
   sequence kernels while preserving state handoff.
4. **W8 speed and memory lanes:** packed weights, fused dequant projection,
   graph-safe workspaces, and per-card policy.
5. **W4 speed and memory lanes:** Marlin/TorchAO/custom kernels selected by
   dtype, architecture, group size, and matrix shape.
6. **Serving integration:** rerun every kernel through dynamic batching, state
   cache, chunked prefill, TP/PP, and speculative decoding.
7. **Cross-card promotion:** validate each hardware family and keep conservative
   fallbacks for unmeasured devices.

## Evidence layout

Store reproducible artifacts under:

```text
benchmark/rwkv7/results/<date>/<gpu>/<model>/<mode>.jsonl
benchmark/rwkv7/results/<date>/<gpu>/environment.json
benchmark/rwkv7/results/<date>/<gpu>/summary.md
```

Every result must include repository SHAs for SGLang and the HF baseline,
checkpoint identifier and hash, hardware/software versions, exact command,
warm-up/repeat policy, raw measurements, and pass/fail reasons. A summary table
without raw machine-readable rows is not acceptance evidence.

## Completion rule

The project may claim **full HF inference parity** only when the core matrix has
zero missing cells and zero failed cells on the required hardware targets, all
serving-superset gates pass, and the results are reproducible from documented
commands. Individual kernels, cards, or modes may be marked complete earlier,
but they must not be presented as completion of the overall target.
