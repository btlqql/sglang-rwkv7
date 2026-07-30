# RWKV-7 serving acceptance

This directory provides a fixed-shape benchmark for comparing RWKV-7 serving
configurations without changing the workload between runs.

## Reference workload

- batch size: 8;
- input length: 128 tokens per request;
- output length: 64 tokens per request;
- greedy decoding with EOS ignored;
- decode CUDA graph captured through batch size 8;
- two warm-up runs and the median of three measured runs;
- the recurrent radix cache is flushed before every run.

Start an unquantized server:

```bash
python -m sglang.launch_server \
  --model-path /path/to/rwkv7-hf \
  --trust-remote-code \
  --attention-backend triton \
  --dtype bfloat16 \
  --chunked-prefill-size 64 \
  --cuda-graph-max-bs-decode 8 \
  --max-running-requests 16
```

Run the benchmark:

```bash
python benchmark/rwkv7/bench_serving.py \
  --batch-size 8 --input-len 128 --output-len 64 \
  --warmups 2 --repeats 3
```

Use the same command after restarting the server with one of the online
quantization configurations below.

### W8A8

```bash
python -m sglang.launch_server \
  --model-path /path/to/rwkv7-hf \
  --trust-remote-code \
  --attention-backend triton \
  --dtype bfloat16 \
  --quantization w8a8_int8 \
  --chunked-prefill-size 64 \
  --cuda-graph-max-bs-decode 8 \
  --max-running-requests 16
```

### Marlin W4 (SM80+)

```bash
python -m sglang.launch_server \
  --model-path /path/to/rwkv7-hf \
  --trust-remote-code \
  --attention-backend triton \
  --dtype float16 \
  --quantization marlin \
  --model-loader-extra-config \
    '{"online_quantization":true,"group_size":128}' \
  --chunked-prefill-size 64 \
  --cuda-graph-max-bs-decode 8 \
  --max-running-requests 16
```

## RTX 4080 snapshot

The native RWKV-7 1.5B checkpoint produced the following medians with the
workload above on an RTX 4080. These values are a hardware snapshot; the ratio
is the acceptance signal.

| Mode | Model-weight memory reported while loading | Output tok/s | Ratio to BF16 |
| --- | ---: | ---: | ---: |
| BF16 | 3.03 GB | 679.47 | 1.000x |
| online W8A8 | about 1.7 GB | 711.15 | 1.047x |
| online Marlin W4 | 0.66 GB | 816.54 | 1.202x |

The online W8A8 and W4 paths leave RWKV's small low-rank control projections in
the activation dtype. Quantizing those projections saves little memory while
adding poorly shaped quantized matrix multiplications on every layer.

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
