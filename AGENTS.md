# RWKV-7 SGLang maintenance instructions

## Primary objective

Treat `RWKV7_ACCEPTANCE.md` as the repository-owned acceptance contract. The
SGLang RWKV-7 inference path must follow the workload, correctness,
quantization, metric, and hardware rules maintained here. Performance
promotion is decided against matched Qwen3.5 and Albatross baselines while
SGLang-specific serving advantages remain enabled and correct.

## Source of truth

- Record the exact repository commit and acceptance-contract revision for every
  acceptance run.
- Mathematical correctness is checked against an independently generated
  dense reference and the declared RWKV-7 equations. Qwen3.5 is the
  matched-activation speed baseline and
  Albatross is the same-checkpoint RWKV speed baseline. Serving behavior is
  checked through SGLang end to end.
- Do not copy experimental defaults blindly. Reuse algorithms and kernel ideas,
  but adapt layouts to SGLang state pools, scheduler metadata, CUDA Graph
  buffers, tensor parallelism, and pipeline parallelism.

## Engineering priorities

1. Repository-owned benchmark harness with Qwen3.5/Albatross red-cell analysis.
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
- W8/W4 must reduce memory and be no slower than the SGLang dense path. Their
  accuracy policy follows the tolerances in `RWKV7_ACCEPTANCE.md`.
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
- Performance PRs add JSONL evidence and update the acceptance matrix or
  benchmark documentation.
- Missing hardware may be marked unverified, never passed.

## Scope boundary

Training frameworks are outside this serving repository's scope. The runtime
must load supported standard checkpoint layouts, including base, merged
adapter, and supported quantized formats. Apple/MLX, AMD/ROCm, and CPU
performance remain explicit deliverables; they may not be silently removed
from the overall goal.

## Repository hygiene

- Keep upstream SGLang attribution and license information intact.
- Keep `upstream` fetch-only and never force-push shared public branches.
- Use focused commits with tests and reproducible commands.
- Do not commit credentials, local paths, private machine addresses, or raw
  checkpoint files.

## GitHub identity isolation

- Run `scripts/install_git_identity_hooks.sh` once in every clone. It binds the
  repository-local Git author and committer to the active `gh` account and
  enables the versioned hooks in `.githooks/`.
- The pre-commit hook requires the author name, committer name, and verified or
  canonical GitHub noreply email to match the active GitHub account.
- The pre-push hook checks only commits newly introduced relative to the remote,
  so upstream history is not rejected. It requires the branch prefix to match
  the active account, blocks direct pushes to `main`, and blocks updates to an
  open PR created by another account.
- Switch accounts with `gh auth switch -h github.com -u <login>`, then rerun the
  installer before working on that account's branch or PR.
