# RWKV-7 serving benchmarks

This directory contains the reproducible correctness, serving-feature, and
performance harnesses for native RWKV-7 in SGLang.

- Acceptance contract: [`RWKV7_HF_PARITY.md`](../../RWKV7_HF_PARITY.md)
- Current RTX 4080 snapshot: [`RESULTS_4080.md`](RESULTS_4080.md)

A single microbenchmark is not an acceptance result. Promote a configuration
only after the same model, dtype, state precision, graph mode, batch, prompt,
and decode lengths pass correctness, memory, and end-to-end serving gates.

## HF-standard batch matrix

`bench_acceptance_matrix.py` drives the streaming API and records TTFT, prefill
throughput, TPOT, decode throughput, end-to-end output throughput, raw samples,
and provenance in JSONL.

```bash
python benchmark/rwkv7/bench_acceptance_matrix.py \
  --model rwkv7-g1-1.5b \
  --mode dense-fp16 \
  --gpu 'NVIDIA GeForce RTX 4080' \
  --repo-sha "$(git rev-parse HEAD)" \
  --standard-sha f1b49bc52a050d09a6739bc4859850f5dc50e7ef \
  --batch-sizes 1,2,4,8 \
  --prompt-lengths 128,512,2048 \
  --decode-lengths 128,512 \
  --warmups 2 --repeats 5 \
  --output /tmp/rwkv7-dense.jsonl
```

The full contract covers 1.5B/2.9B/7.2B and dense/W8/W4, for 216 cells per
hardware target. `--no-flush-cache` is diagnostic only; acceptance runs flush
the recurrent radix cache before every sample.

## Dense server

The strict state-cache lane keeps recurrent state in FP32:

```bash
python -m sglang.launch_server \
  --model-path /path/to/rwkv7-hf \
  --trust-remote-code \
  --attention-backend triton \
  --dtype float16 \
  --mamba-ssm-dtype float32 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --cuda-graph-max-bs-decode 8 \
  --max-running-requests 16 \
  --cuda-graph-config \
  '{"decode":{"backend":"full","bs":[1,2,4,8]},"prefill":{"backend":"full","bs":[1024,4096,16384],"full_prefill_max_req":8}}'
```

`--mamba-ssm-dtype float16` selects the Ada FP16-state performance lane and its
optional native packed-varlen CUDA WKV kernel. It lowers state memory and raises
throughput, but is not token-exact across every long chunk boundary. Use it only
when the target workload passes its logit/token gate. Disable the optional CUDA
kernel with `SGLANG_RWKV7_CUDA_FP16_WKV=0` to exercise the portable Triton
fallback.

`SGLANG_RWKV7_FAST_FP16_PREFILL=1` changes the Triton reduction layout for an
additional experimental FP16 prefill fast path. It is opt-in because long
free-running generation can amplify the reduction-order difference.

## Quantized servers

RWKV-7 applies model-specific mixed-precision policies to online W8 and W4.
The default is `accuracy`; the alternatives are explicit trade-offs:

```text
SGLANG_RWKV7_W8_POLICY=accuracy|balanced|speed
SGLANG_RWKV7_W4_POLICY=accuracy|balanced|speed
```

Low-rank controls and lm_head stay dense for these paths. The recurrent WKV
state is not weight-quantized.

### W8A8 INT8

```bash
SGLANG_RWKV7_W8_POLICY=accuracy \
python -m sglang.launch_server \
  --model-path /path/to/rwkv7-hf \
  --trust-remote-code \
  --attention-backend triton \
  --dtype float16 \
  --mamba-ssm-dtype float32 \
  --quantization w8a8_int8 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs-decode 8 \
  --max-running-requests 16
```

The RTX 4080 accuracy policy keeps recurrent attention and the sqReLU expansion
dense, and quantizes the middle FFN value projections. FP32 recurrent state is
required for the strict cold/warm cache-continuation gate in the current build.

### Online Marlin W4 (SM80+)

```bash
SGLANG_RWKV7_W4_POLICY=accuracy \
python -m sglang.launch_server \
  --model-path /path/to/rwkv7-hf \
  --trust-remote-code \
  --attention-backend triton \
  --dtype float16 \
  --quantization marlin \
  --model-loader-extra-config \
    '{"online_quantization":true,"group_size":128}' \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs-decode 8 \
  --max-running-requests 16
```

The W4 accuracy policy keeps attention and edge FFN layers dense. In the
middle stack it interleaves W4 Marlin and W8A8 FFN blocks. This hybrid keeps a
substantial W4 memory/decode benefit while avoiding Marlin's large-batch
prefill regression. `balanced` and `speed` remain pure-W4 policies. Group size
128 is the supported online Marlin setting in the validated environment.

## Quantized alignment

`verify_quant_alignment.py` generates a continuation with dense Hugging Face,
then teacher-forces that exact continuation through an already-running
quantized SGLang server. It reports chosen-token logprob error, top-k overlap,
teacher-forced top-1 agreement, and a free-running prefix diagnostic.

```bash
python benchmark/rwkv7/verify_quant_alignment.py \
  --model /path/to/rwkv7-hf \
  --base-url http://127.0.0.1:30000 \
  --dtype float16 \
  --max-new-tokens 32 \
  --top-k 10
```

The default W8 gate is max chosen-token logprob error `<= 0.25`, mean top-10
overlap `>= 0.80`, and teacher-forced top-1 agreement `>= 0.90`. A W4 run may
use a separately justified threshold, but a relaxed gate must be named in the
result rather than silently replacing the W8 threshold.

## Serving-feature gate

Run this against every promoted dense or quantized server:

```bash
python benchmark/rwkv7/verify_serving_features.py \
  --base-url http://127.0.0.1:30000
```

The harness checks deterministic replay, duplicate-request state isolation,
chunked-prefill cold/warm equality, and an exact 128-token recurrent cache hit.
For a diagnostic quantized run, `--single-batch-prefix-tokens 0` disables only
the single-vs-batched prefix check; it does not disable cache equality or state
isolation.

## Legacy CUDA validation (V100 / SM70)

The RWKV-7 path can run without a modern `sgl-kernel` or FlashInfer wheel. On
Volta, select the PyTorch sampler and disable CUDA graphs; the recurrent WKV
and fused RWKV elementwise kernels continue to use Triton:

```bash
python -m sglang.launch_server \
  --model-path /path/to/rwkv7-hf \
  --trust-remote-code \
  --dtype float16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --grammar-backend none \
  --chunked-prefill-size 64 \
  --max-mamba-cache-size 16 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled
```

The same command accepts `--tp-size 2` or `--pp-size 2` when two devices are
available. A two-card V100-32GB validation with a native
`rwkv7-hf-adapter` checkpoint covered the following matrix:

| Configuration | Greedy tokens vs. TP1 | Dynamic batch isolation | 64-token chunked prefill | 128-token state-cache hit |
| --- | --- | --- | --- | --- |
| TP1 | reference | exact | exact | exact |
| TP2 | exact | exact | exact | exact |
| PP2 | exact | exact | exact | exact |

Run the feature gate against an already-started server. The first run can save
the TP1 output, and later TP/PP runs can require the same token IDs:

```bash
python benchmark/rwkv7/verify_serving_features.py \
  --write-reference /tmp/rwkv7-tp1.json

python benchmark/rwkv7/verify_serving_features.py \
  --reference-output /tmp/rwkv7-tp1.json
```

For W8/W4, add `--single-batch-prefix-tokens 0` if shape-dependent quantized
GEMM rounding changes greedy tokens between single and batched execution. The
gate still requires deterministic replay, duplicate-request state isolation,
an exact state-cache hit, and cold/warm continuation equality.

This compatibility path is scoped to RWKV-7. Other SGLang architectures keep
the normal SM75-or-newer runtime requirement. Online W8A8 requires SM75 and
online Marlin W4 requires SM80; use a supported BitsAndBytes configuration if
weight-only quantization is required on older GPUs.

The following loader options were validated end to end on V100 for W8 and W4,
including dynamic batching, chunked prefill, and recurrent state-cache hits:

```bash
# W8
--quantization bitsandbytes --load-format bitsandbytes \
--model-loader-extra-config \
  '{"load_in_8bit":true,"load_in_4bit":false}'

# W4 NF4
--quantization bitsandbytes --load-format bitsandbytes \
--model-loader-extra-config \
  '{"load_in_8bit":false,"load_in_4bit":true,"bnb_4bit_compute_dtype":"float16","bnb_4bit_quant_type":"nf4"}'
```

BitsAndBytes is the legacy functional and memory-saving fallback; it is not the
optimized quantization speed path. Use online W8A8 or Marlin on their supported
architectures when throughput is the acceptance criterion.

## Hugging Face alignment

For native `rwkv-rs/hf-adapter` checkpoints, the alignment harness first runs
Hugging Face generation, releases that model, starts SGLang, and then compares:

- greedy output token IDs;
- generated-token log probabilities;
- top-k token-set overlap at each matched decode step.

```bash
python benchmark/rwkv7/verify_hf_alignment.py \
  --model /path/to/rwkv7-hf \
  --dtype float16 \
  --max-new-tokens 16 \
  --top-k 10
```

The command exits with a non-zero status if any greedy sequence diverges, the
maximum generated-token log-probability error exceeds `0.05`, or mean top-10
overlap falls below `0.8`.

## STANDALONE speculative decoding

RWKV-7 supports a recurrent STANDALONE draft with a topk-1 candidate chain.
The target and draft own separate recurrent state pools; target verification
and draft decoding write per-step snapshots to scratch and draft extend commits
only the accepted state.

```bash
python -m sglang.launch_server \
  --model-path /path/to/rwkv7-target \
  --trust-remote-code \
  --attention-backend triton \
  --speculative-algorithm STANDALONE \
  --speculative-draft-model-path /path/to/rwkv7-draft \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disable-overlap-schedule
```

The end-to-end gate is
`test/registered/models/test_rwkv7_standalone_spec.py`. It starts speculative
and non-speculative servers, requires exact greedy output IDs under dynamic
batching, checks that drafting is active, and verifies a 128-token recurrent
state-cache hit against cold generation.

Hardware validation snapshots:

| Device | Model | Batch sizes | Output window | Result |
| --- | --- | --- | --- | --- |
| RTX 4080 | RWKV-7 1.5B, FP16 | 1, 2, 4 | 32 tokens, repeated twice | exact vs. non-speculative |
| RTX 4080 | RWKV-7 1.5B, FP16 | 4 | 64 tokens | exact vs. non-speculative |
| RTX 4080 | RWKV-7 1.5B, FP16 | 1 | 128-token cache hit + 16-token decode | cold/warm exact |
| V100 | RWKV-7 0.1B, FP16 | 1, 2 | 32 tokens | exact vs. non-speculative |

For decode CUDA Graphs, target verify and multi-step draft decode are captured;
RWKV draft extend currently remains eager. On legacy CUDA systems without a
compatible `sgl_kernel`, speculative tree construction and greedy verification
fall back to the in-tree Triton kernels.
