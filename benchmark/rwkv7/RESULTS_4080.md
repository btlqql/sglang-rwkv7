# RTX 4080 RWKV-7 engineering snapshot

Date: 2026-07-31

This page records the current Ada optimization slice. It is intentionally
narrower than the full acceptance contract in `RWKV7_HF_PARITY.md`: one
RWKV-7 1.5B checkpoint, batch size 8, and one RTX 4080. It is an engineering
snapshot, not a claim that the 216-cell, multi-model, or multi-hardware matrix
is complete.

The matched dense/W8/W4 raw JSONL, alignment output, serving-feature output,
and public environment manifest are stored under
[`results/2026-07-31/rtx-4080`](results/2026-07-31/rtx-4080/). The earlier
strict FP32-state evidence remains under `results/2026-07-30/rtx-4080`.

## Workload

- model: RWKV-7 G1 1.5B;
- device: NVIDIA GeForce RTX 4080, 16 GB;
- activation dtype: FP16;
- batch size: 8;
- prompt lengths per request: 128, 512, and 2048 tokens;
- decode lengths per request: 128 and 512 tokens;
- greedy decoding with EOS ignored;
- recurrent radix cache flushed before each sample;
- two warm-ups and the median of five measured samples;
- full CUDA Graph decode at batch sizes 1/2/4/8;
- fixed-shape full-prefill CUDA Graph buckets at 1024/4096/16384 packed tokens.

`prefill tok/s` counts all input tokens until every request has emitted its
first token. `decode tok/s` counts aggregate output tokens after that first
batch-wide token. `E2E tok/s` counts all generated output tokens over complete
request wall time.

## Quantization policies

The default policy is `accuracy`.

| Mode | Quantized projections | Dense protections |
| --- | --- | --- |
| W8 accuracy | middle FFN value projections | attention, FFN key, first/last FFN value, low-rank controls, lm_head |
| W4 hybrid accuracy | middle-layer FFN key/value, interleaved W4/W8 by layer | attention, first/last four FFN blocks, low-rank controls, lm_head |

`balanced` and `speed` are explicit pure-W4 opt-in policies. The default hybrid
accuracy policy keeps meaningful W4 coverage but uses W8 in alternating middle
FFN blocks to close Marlin's large-batch prefill gap.

## Model-weight memory

The values below are the model-weight memory reported during loading, not the
configured server state pool or total process peak.

| Mode | Reported model memory | Change from dense |
| --- | ---: | ---: |
| Dense FP16 | 3.03 GB | reference |
| W8 accuracy | 2.69 GB | -11.2% |
| W4 hybrid accuracy | 2.44 GB | -19.5% |

## FP16-state performance lane

This lane uses `--mamba-ssm-dtype float16` and the native packed-varlen CUDA WKV
kernel. It is the fastest measured configuration, but FP16 recurrent state is
an explicit performance/precision choice rather than the strict default.

| Prompt | Decode | Mode | Prefill tok/s | Decode tok/s | E2E tok/s | Prefill / dense | Decode / dense | E2E / dense |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | dense | 23,545.4 | 1,105.6 | 1,063.9 | 1.000x | 1.000x | 1.000x |
| 128 | 128 | W8 accuracy | 26,849.0 | 1,181.9 | 1,140.7 | 1.140x | 1.069x | 1.072x |
| 128 | 128 | W4 hybrid accuracy | 25,690.6 | 1,258.4 | 1,208.1 | 1.091x | 1.138x | 1.136x |
| 128 | 512 | dense | 23,875.8 | 1,114.3 | 1,103.5 | 1.000x | 1.000x | 1.000x |
| 128 | 512 | W8 accuracy | 28,088.8 | 1,202.3 | 1,191.9 | 1.176x | 1.079x | 1.080x |
| 128 | 512 | W4 hybrid accuracy | 25,689.8 | 1,259.0 | 1,246.2 | 1.076x | 1.130x | 1.129x |
| 512 | 128 | dense | 26,499.1 | 1,112.4 | 958.9 | 1.000x | 1.000x | 1.000x |
| 512 | 128 | W8 accuracy | 31,075.3 | 1,198.3 | 1,045.1 | 1.173x | 1.077x | 1.090x |
| 512 | 128 | W4 hybrid accuracy | 28,961.6 | 1,269.9 | 1,087.6 | 1.093x | 1.142x | 1.134x |
| 512 | 512 | dense | 26,415.4 | 1,113.1 | 1,070.0 | 1.000x | 1.000x | 1.000x |
| 512 | 512 | W8 accuracy | 31,013.5 | 1,197.5 | 1,154.9 | 1.174x | 1.076x | 1.079x |
| 512 | 512 | W4 hybrid accuracy | 29,278.8 | 1,275.2 | 1,223.9 | 1.108x | 1.146x | 1.144x |
| 2048 | 128 | dense | 25,106.3 | 1,112.8 | 654.0 | 1.000x | 1.000x | 1.000x |
| 2048 | 128 | W8 accuracy | 28,237.3 | 1,199.3 | 716.4 | 1.125x | 1.078x | 1.095x |
| 2048 | 128 | W4 hybrid accuracy | 27,158.9 | 1,283.3 | 733.8 | 1.082x | 1.153x | 1.122x |
| 2048 | 512 | dense | 25,066.0 | 1,113.2 | 946.8 | 1.000x | 1.000x | 1.000x |
| 2048 | 512 | W8 accuracy | 28,231.6 | 1,199.0 | 1,026.6 | 1.126x | 1.077x | 1.084x |
| 2048 | 512 | W4 hybrid accuracy | 27,155.4 | 1,280.1 | 1,078.4 | 1.083x | 1.150x | 1.139x |

Current result:

- W8 accuracy is faster than dense in all 18 measured prefill/decode/E2E gates.
- W4 hybrid accuracy is also faster in all 18 gates. Its minimum gains are
  7.6% prefill, 13.0% decode, and 12.2% end to end.

## Prior strict FP32-state W8 lane

The following commit-`a590b2a` lane is retained from the 2026-07-30 evidence.
FP32 recurrent state passes the chunk-boundary state-cache equality gate. It
retains the W8 prefill, decode, and end-to-end gain while making the recurrent
state-precision cost explicit rather than hiding it in the quantized ratio.

| Prompt | Decode | Dense prefill | W8 prefill | Ratio | Dense decode | W8 decode | Ratio | Dense E2E | W8 E2E | Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 19,542.5 | 22,277.9 | 1.140x | 1,075.2 | 1,151.9 | 1.071x | 1,026.7 | 1,103.0 | 1.074x |
| 128 | 512 | 20,344.2 | 23,058.0 | 1.133x | 1,087.3 | 1,171.6 | 1.077x | 1,075.1 | 1,159.1 | 1.078x |
| 512 | 128 | 21,878.2 | 24,814.0 | 1.134x | 1,086.9 | 1,168.7 | 1.075x | 912.6 | 989.9 | 1.085x |
| 512 | 512 | 21,873.9 | 24,791.6 | 1.133x | 1,087.0 | 1,166.7 | 1.073x | 1,037.4 | 1,116.3 | 1.076x |
| 2048 | 128 | 20,852.0 | 23,029.1 | 1.104x | 1,087.1 | 1,168.8 | 1.075x | 595.2 | 647.5 | 1.088x |
| 2048 | 512 | 20,805.0 | 23,022.3 | 1.107x | 1,087.2 | 1,167.4 | 1.074x | 900.7 | 971.9 | 1.079x |

The strict lane passed deterministic replay, duplicate-request state isolation,
cold/warm chunked-prefill equality, and an exact 128-token recurrent cache hit.
Against the same FP32-state dense reference, W8 is faster in all six prefill,
decode, and end-to-end cells. FP32-state prefill remains slower than the
separate FP16-state performance lane, as expected.

## Quantized alignment

The alignment harness generates a dense Hugging Face continuation and then
teacher-forces the same continuation through quantized SGLang. This avoids
misreporting every token after one early free-running branch as an independent
quantization error.

| Mode | Max chosen-token logprob error | Mean error | Mean top-10 overlap | Teacher-forced top-1 agreement | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| W8 accuracy | 0.1025 | 0.0156 | 0.9664 | 1.0000 | pass (`0.25 / 0.80 / 0.90`) |
| W4 hybrid accuracy | 0.4668 | 0.0518 | 0.9000 | 0.9922 | pass (`1.00 / 0.80 / 0.90`) |

W8 reproduced all four 32-token natural-prompt continuations in the measured
FP16-state run. W4 hybrid reproduced three of four. Long synthetic free-running
sequences can still diverge, so the W4 result is not presented as final
llama.cpp-quality quantization parity.

## Same-model and cross-model baselines

- The isolated native FP16-state WKV kernel measured 11-15% faster than the
  matched Albatross WKV microkernel for sequence lengths 128/512/2048.
- The prior end-to-end dense run was approximately 0.827-0.875x Albatross
  prefill and 0.941-0.951x Albatross decode. The new fused ReLU2 path improves
  dense prefill, but the matched Albatross rerun is still pending; no updated
  end-to-end Albatross ratio is inferred from unmatched measurements.
- The existing HF-derived Qwen3.5 batch-8 speed gate was green in these six
  dense cells. This SGLang checkout does not yet implement
  `Qwen3_5ForConditionalGeneration`, so a same-runtime SGLang Qwen rerun is
  still missing and no final cross-runtime claim is made here.

## Serving and graph validation

- fixed-shape full-prefill CUDA Graph capture and replay: implemented;
- full-graph chunked-prefill metadata handoff: corrected and regression-tested;
- zero-length graph padding for token shift and WKV state: tested;
- dynamic-batch duplicate isolation: passed;
- recurrent cache select/restore and 128-token hit: passed in strict FP32 state;
- W4 hybrid accuracy feature harness: deterministic replay, duplicate-request
  isolation, cold/warm chunked-prefill equality, and a 128-token state-cache
  hit all passed in the measured FP16-state setup;
- W8 FP16-state cache-boundary exactness: not passed; use FP32 state for the
  strict serving lane.

## Remaining Ada work

1. Narrow the FP32-state versus FP16-state prefill cost while retaining exact
   cold/warm cache continuation.
2. Close the Albatross end-to-end projection/norm/fusion gap.
3. Run batch sizes 1/2/4 and the 2.9B/7.2B models.
4. Rerun matched Qwen3.5 and Albatross baselines under the final acceptance
   environment.
