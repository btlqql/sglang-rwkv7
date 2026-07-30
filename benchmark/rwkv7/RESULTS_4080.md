# RTX 4080 RWKV-7 engineering snapshot

Date: 2026-07-30

This page records the current Ada optimization slice. It is intentionally
narrower than the full acceptance contract in `RWKV7_HF_PARITY.md`: one
RWKV-7 1.5B checkpoint, batch size 8, and one RTX 4080. It is an engineering
snapshot, not a claim that the 216-cell, multi-model, or multi-hardware matrix
is complete.

## Workload

- model: RWKV-7 G1 1.5B;
- device: NVIDIA GeForce RTX 4080, 16 GB;
- activation dtype: FP16;
- batch size: 8;
- prompt lengths per request: 128, 512, and 2048 tokens;
- decode lengths per request: 128 and 512 tokens;
- greedy decoding with EOS ignored;
- recurrent radix cache flushed before each sample;
- one warm-up and the median of three measured samples;
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
| W4 accuracy | middle-layer FFN key/value projections | attention, first/last four FFN blocks, low-rank controls, lm_head |

`balanced` and `speed` are explicit opt-in policies. They increase compression
and throughput but did not pass the same alignment gate in this snapshot.

## Model-weight memory

The values below are the model-weight memory reported during loading, not the
configured server state pool or total process peak.

| Mode | Reported model memory | Change from dense |
| --- | ---: | ---: |
| Dense FP16 | 3.03 GB | reference |
| W8 accuracy | 2.69 GB | -11.2% |
| W4 accuracy | 2.35 GB | -22.4% |

## FP16-state performance lane

This lane uses `--mamba-ssm-dtype float16` and the native packed-varlen CUDA WKV
kernel. It is the fastest measured configuration, but FP16 recurrent state is
an explicit performance/precision choice rather than the strict default.

| Prompt | Decode | Mode | Prefill tok/s | Decode tok/s | E2E tok/s | Prefill / dense | Decode / dense | E2E / dense |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | dense | 23,033.3 | 1,099.2 | 1,057.2 | 1.000x | 1.000x | 1.000x |
| 128 | 128 | W8 accuracy | 26,431.2 | 1,169.3 | 1,128.4 | 1.148x | 1.064x | 1.067x |
| 128 | 128 | W4 accuracy | 22,668.9 | 1,293.1 | 1,232.9 | **0.984x** | 1.176x | 1.166x |
| 128 | 512 | dense | 23,557.6 | 1,107.6 | 1,096.9 | 1.000x | 1.000x | 1.000x |
| 128 | 512 | W8 accuracy | 27,302.2 | 1,188.2 | 1,177.7 | 1.159x | 1.073x | 1.074x |
| 128 | 512 | W4 accuracy | 22,975.4 | 1,292.5 | 1,277.0 | **0.975x** | 1.167x | 1.164x |
| 512 | 128 | dense | 25,501.5 | 1,106.4 | 949.0 | 1.000x | 1.000x | 1.000x |
| 512 | 128 | W8 accuracy | 29,768.0 | 1,192.7 | 1,035.3 | 1.167x | 1.078x | 1.091x |
| 512 | 128 | W4 accuracy | 25,572.3 | 1,311.2 | 1,095.1 | 1.003x | 1.185x | 1.154x |
| 512 | 512 | dense | 25,582.8 | 1,106.9 | 1,063.0 | 1.000x | 1.000x | 1.000x |
| 512 | 512 | W8 accuracy | 29,874.2 | 1,193.4 | 1,149.7 | 1.168x | 1.078x | 1.082x |
| 512 | 512 | W4 accuracy | 25,879.4 | 1,317.6 | 1,256.4 | 1.012x | 1.190x | 1.182x |
| 2048 | 128 | dense | 24,203.1 | 1,106.4 | 641.9 | 1.000x | 1.000x | 1.000x |
| 2048 | 128 | W8 accuracy | 27,160.0 | 1,193.0 | 703.8 | 1.122x | 1.078x | 1.096x |
| 2048 | 128 | W4 accuracy | 24,091.2 | 1,317.7 | 704.7 | **0.995x** | 1.191x | 1.098x |
| 2048 | 512 | dense | 24,252.3 | 1,106.4 | 937.3 | 1.000x | 1.000x | 1.000x |
| 2048 | 512 | W8 accuracy | 27,111.8 | 1,193.5 | 1,016.5 | 1.118x | 1.079x | 1.085x |
| 2048 | 512 | W4 accuracy | 24,007.7 | 1,318.0 | 1,082.4 | **0.990x** | 1.191x | 1.155x |

Current result:

- W8 accuracy is faster than dense in prefill, decode, and end-to-end throughput
  in all six measured cells.
- W4 accuracy is faster in every decode and end-to-end cell. Four prefill
  cells remain below dense; the worst deficit is 2.5%.

## Strict FP32-state W8 lane

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
| W8 accuracy | 0.1055 | 0.0156 | 0.9656 | 1.0000 | pass (`0.25 / 0.80 / 0.90`) |
| W4 accuracy | 0.5210 | 0.0760 | 0.8586 | 0.9766 | preliminary pass (`1.00 / 0.80 / 0.90`) |

W8 reproduced all four 32-token natural-prompt continuations in the measured
FP16-state run. W4 reproduced three of four. Long synthetic free-running
sequences can still diverge, so the W4 result is not presented as final
llama.cpp-quality quantization parity.

## Same-model and cross-model baselines

- The isolated native FP16-state WKV kernel measured 11-15% faster than the
  matched Albatross WKV microkernel for sequence lengths 128/512/2048.
- End to end, the current dense path is still approximately 0.827-0.875x
  Albatross prefill and 0.941-0.951x Albatross decode. Projection, norm, graph,
  and serving overhead therefore remain the main same-model gap.
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
- W4 accuracy feature harness: passed in the measured FP16-state setup;
- W8 FP16-state cache-boundary exactness: not passed; use FP32 state for the
  strict serving lane.

## Remaining Ada work

1. Close the four W4 prefill red cells without weakening its accuracy lane.
2. Narrow the FP32-state versus FP16-state prefill cost while retaining exact
   cold/warm cache continuation.
3. Close the Albatross end-to-end projection/norm/fusion gap.
4. Run batch sizes 1/2/4 and the 2.9B/7.2B models.
5. Produce raw JSONL with committed repository SHAs and rerun matched Qwen3.5
   and Albatross baselines under the final acceptance environment.
