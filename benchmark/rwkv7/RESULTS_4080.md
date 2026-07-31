# RTX 4080 RWKV-7 engineering snapshot

Date: 2026-07-31

This page records the current Ada optimization slice. It is intentionally
narrower than the full acceptance contract in `RWKV7_HF_PARITY.md`: complete
1.5B and 2.9B dense/W8/W4 batch matrices, a partial 7.2B capacity lane, and one
RTX 4080. It is an engineering snapshot, not a claim that the multi-hardware
matrix is complete.

The matched dense/W8/W4 raw JSONL, alignment output, serving-feature output,
and public environment manifest are stored under
[`results/2026-07-31/rtx-4080`](results/2026-07-31/rtx-4080/). The earlier
strict FP32-state evidence remains under `results/2026-07-30/rtx-4080`.

## Workload

- models: RWKV-7 G1 1.5B and 2.9B; 7.2B quantized capacity work is in progress;
- device: NVIDIA GeForce RTX 4080, 16 GB;
- activation dtype: FP16;
- batch sizes: 1, 2, 4, and 8;
- prompt lengths per request: 128, 512, and 2048 tokens;
- decode lengths per request: 128 and 512 tokens;
- greedy decoding with EOS ignored;
- recurrent radix cache flushed before each sample;
- two warm-ups and the median of five measured samples;
- full CUDA Graph decode at batch sizes 1/2/4/8;
- fixed-shape full-prefill CUDA Graph buckets at
  128/256/512/1024/2048/4096/8192/16384 packed tokens.

`prefill tok/s` counts all input tokens until every request has emitted its
first token. `decode tok/s` counts aggregate output tokens after that first
batch-wide token. `E2E tok/s` counts all generated output tokens over complete
request wall time.

## Quantization policies

The default policy is `accuracy`.

| Mode | Quantized projections | Dense protections |
| --- | --- | --- |
| W8 accuracy | middle FFN value projections | attention, FFN key, first/last FFN value, low-rank controls, lm_head |
| W4 hybrid accuracy | alternating middle FFN key projections in W4; remaining middle key/value projections in W8 | attention, first/last four FFN blocks, low-rank controls, lm_head |

`balanced` and `speed` are explicit pure-W4 opt-in policies. The default hybrid
accuracy policy keeps meaningful W4 coverage for large prefills. For small
token batches it uses an FP16 shard of the W4 key projection, and W8 uses exact
INT32 accumulation, because recurrent state can amplify row-layout rounding
into different long greedy continuations.

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

The exact aggregate-token graph buckets close the padding loss in the smaller
active batches. Across all 24 cells, the minimum ratios against the matched
dense lane are:

| Mode | Prefill / dense | Decode / dense | E2E / dense |
| --- | ---: | ---: | ---: |
| W8 accuracy | 1.048x | 1.053x | 1.054x |
| W4 hybrid accuracy | 1.023x | 1.139x | 1.127x |

The table below retains the batch-8 absolute values; raw batch-1/2/4 rows are
published beside it under the result directory.

| Prompt | Decode | Mode | Prefill tok/s | Decode tok/s | E2E tok/s | Prefill / dense | Decode / dense | E2E / dense |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | dense | 22,913.6 | 1,140.6 | 1,094.8 | 1.000x | 1.000x | 1.000x |
| 128 | 128 | W8 accuracy | 27,005.4 | 1,234.3 | 1,189.3 | 1.179x | 1.082x | 1.086x |
| 128 | 128 | W4 hybrid accuracy | 25,160.3 | 1,323.9 | 1,267.1 | 1.098x | 1.161x | 1.157x |
| 128 | 512 | dense | 24,080.0 | 1,157.9 | 1,146.3 | 1.000x | 1.000x | 1.000x |
| 128 | 512 | W8 accuracy | 27,174.5 | 1,236.3 | 1,224.8 | 1.129x | 1.068x | 1.068x |
| 128 | 512 | W4 hybrid accuracy | 25,112.4 | 1,324.2 | 1,309.2 | 1.043x | 1.144x | 1.142x |
| 512 | 128 | dense | 27,093.6 | 1,157.5 | 995.1 | 1.000x | 1.000x | 1.000x |
| 512 | 128 | W8 accuracy | 31,065.4 | 1,245.4 | 1,080.6 | 1.147x | 1.076x | 1.086x |
| 512 | 128 | W4 hybrid accuracy | 27,937.9 | 1,327.7 | 1,124.0 | 1.031x | 1.147x | 1.129x |
| 512 | 512 | dense | 27,166.6 | 1,160.6 | 1,114.8 | 1.000x | 1.000x | 1.000x |
| 512 | 512 | W8 accuracy | 31,452.7 | 1,252.4 | 1,206.8 | 1.158x | 1.079x | 1.083x |
| 512 | 512 | W4 hybrid accuracy | 27,793.2 | 1,322.9 | 1,265.1 | 1.023x | 1.140x | 1.135x |
| 2048 | 128 | dense | 25,960.1 | 1,159.9 | 679.5 | 1.000x | 1.000x | 1.000x |
| 2048 | 128 | W8 accuracy | 29,439.0 | 1,257.2 | 749.9 | 1.134x | 1.084x | 1.104x |
| 2048 | 128 | W4 hybrid accuracy | 28,125.2 | 1,346.2 | 765.7 | 1.083x | 1.161x | 1.127x |
| 2048 | 512 | dense | 25,968.8 | 1,159.9 | 985.7 | 1.000x | 1.000x | 1.000x |
| 2048 | 512 | W8 accuracy | 29,462.0 | 1,257.4 | 1,075.2 | 1.135x | 1.084x | 1.091x |
| 2048 | 512 | W4 hybrid accuracy | 27,452.1 | 1,335.3 | 1,119.4 | 1.057x | 1.151x | 1.136x |

Batch-8 result:

- W8 accuracy is faster than dense in all 18 measured prefill/decode/E2E gates.
  Its minimum gains are 12.9% prefill, 6.8% decode, and 6.8% end to end.
- W4 hybrid accuracy is also faster in all 18 gates. Its minimum gains are
  2.3% prefill, 14.0% decode, and 12.7% end to end.

## 2.9B dense/W8/W4 matrix

The 2.9B run uses the same 24 workload cells and exact aggregate-token graph
buckets as 1.5B. All 72 dense/W8/W4 rows completed. Against its matched dense
lane, every quantized prefill, decode, and end-to-end cell passes:

| Mode | Prefill / dense | Decode / dense | E2E / dense |
| --- | ---: | ---: | ---: |
| W8 accuracy | 1.091x | 1.094x | 1.094x |
| W4 hybrid accuracy | 1.087x | 1.231x | 1.179x |

The reported model-weight memory is 5.68 GB dense, 4.98 GB W8, and 4.28 GB W4
hybrid. This is a 12.3% W8 and 24.6% W4 reduction. The independent quantized
alignment gate also passes:

| Mode | Max chosen-token logprob error | Mean error | Mean top-10 overlap | Teacher-forced top-1 agreement | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| W8 accuracy | 0.1322 | 0.0180 | 0.9625 | 0.9844 | pass (`0.25 / 0.80 / 0.90`) |
| W4 hybrid accuracy | 0.6088 | 0.0632 | 0.9203 | 0.9531 | pass (`1.00 / 0.80 / 0.90`) |

### 2.9B production-safe W4 batch-8 slice (`d17883f3`)

The earlier 4.28 GB W4 matrix above is retained as a high-compression
performance experiment, but its small-M Marlin path did not pass repeated and
mixed-length serving checks. Commit `d17883f3f671adb8d0998034072649a88931e95d`
therefore adds a production-safe hybrid path:

- W4 Marlin remains active for large prefills;
- small W4 key projections use their original FP16 shard;
- W8 projections use exact INT32 accumulation up to 1,024 packed tokens;
- fresh/reused recurrent slots are cleared or restored before full-graph replay.

The resulting model-weight load is 5.00 GB versus 5.68 GB dense, a 12.0%
reduction. The committed batch-8 matrix uses two warm-ups and five measured
samples per cell:

| Prompt | Decode | Prefill tok/s | Decode tok/s | E2E tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 14,261.0 | 824.9 | 785.5 |
| 128 | 512 | 14,268.8 | 825.0 | 814.8 |
| 512 | 128 | 16,670.2 | 824.2 | 692.5 |
| 512 | 512 | 16,652.5 | 824.5 | 787.1 |
| 2048 | 128 | 16,045.0 | 824.0 | 454.3 |
| 2048 | 512 | 16,041.5 | 824.3 | 684.7 |

Across these six cells, the minimum ratios are 1.118x prefill, 1.099x decode,
and 1.101x end to end against the matched dense 2.9B lane. Against the fresh
Albatross 2.9B batch-8 reference, the minima are 1.039x prefill and 1.177x
decode. Against the same-runtime Qwen3.5-4B batch-8 baseline, the minima are
1.417x prefill, 1.650x decode, and 1.615x end to end, exceeding the simple
4.0B/2.9B active-parameter proportional decode target.

The strict quantized alignment gate passes at 0.2357 maximum chosen-token
logprob error, 0.9758 mean top-10 overlap, and 0.9766 teacher-forced top-1
agreement (`0.25 / 0.80 / 0.90`). The production lifecycle harness also passes
mixed-length compaction, explicit abort, post-abort slot reuse, full cold/warm
chunked-prefill equality, and a 128-token state-cache hit. Long synthetic
quantized continuations are not claimed bit-exact: the serving artifact records
a two-token repeat/duplicate prefix gate, while natural-prompt quality is
measured by the teacher-forced alignment report.

Raw evidence:

- `rwkv7-g1-2.9b/w4-hybrid-safe-d17883f3.jsonl`
- `rwkv7-g1-2.9b/w4-hybrid-safe-d17883f3-alignment.json`
- `rwkv7-g1-2.9b/w4-hybrid-safe-d17883f3-serving.json`

## 7.2B quantized capacity lane

The 16 GB card cannot host the dense 7.2B production graph matrix, so dense
speed parity remains a 24 GB card requirement. W4 hybrid loads at 9.71 GB and
passes the split-reference alignment gate with 0.3455 maximum chosen-token
logprob error, 0.9203 mean top-10 overlap, and 0.9766 teacher-forced top-1
agreement. W8 loads at 11.56 GB; its corrected state-pool configuration and
split-graph performance run remain in progress.

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
| W8 accuracy | 0.1012 | 0.0166 | 0.9625 | 1.0000 | pass (`0.25 / 0.80 / 0.90`) |
| W4 hybrid accuracy | 0.4618 | 0.0537 | 0.8953 | 0.9922 | pass (`1.00 / 0.80 / 0.90`) |

W8 reproduced all four 32-token natural-prompt continuations in the measured
FP16-state run. W4 hybrid reproduced three of four. Long synthetic free-running
sequences can still diverge, so the W4 result is not presented as final
llama.cpp-quality quantization parity.

## Same-model and cross-model baselines

- The isolated native FP16-state WKV kernel measured 11-15% faster than the
  matched Albatross WKV microkernel for sequence lengths 128/512/2048.
- The matched Albatross commit `343147a` measured B8 throughput of 1,173.1
  tok/s at T=1, 25,705.0 at T=128, 29,076.1 at T=512, and 29,309.1 at T=2048.
- W8 accuracy exceeds those fixed-shape baselines in every matched B8 cell:
  decode is 1.052-1.072x, T=128 prefill is 1.051-1.057x, T=512 is
  1.068-1.082x, and T=2048 is 1.004-1.005x. SGLang serving and Albatross's
  fixed-forward harness have different outer timing semantics, so these ratios
  are recorded as the requested engineering acceptance comparison rather than
  a claim of identical API overhead.
- The same-runtime SGLang Qwen3.5-2B BF16 baseline now contains all 24 cells.
  At batch 8, dense RWKV has minimum prefill/decode/E2E ratios of
  0.992x/1.077x/1.075x. The active-work decode score is currently at least
  1.435x against the 1.75x target. Qwen's hybrid attention path does not support
  the RWKV-specific full-prefill graph, so this baseline uses eager prefill and
  full decode graphs.
- A fresh full Albatross 1.5B rerun contains all 16 fixed-forward shapes. W8
  remains green at batch 8 (minimum 1.006x prefill and 1.012x decode), while
  smaller-batch decode and dense/W4 prefill expose the remaining host-loop and
  short-prefill gaps. These red cells remain visible rather than being replaced
  by the earlier, narrower batch-8 reference.

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
2. Amortize the decode host loop with matched continuous-step runs and close
   the remaining Albatross B1/2/4 and Qwen3.5 active-work gaps.
3. Close short-prefill red cells without weakening chunked-prefill handoff.
4. Complete the running 2.9B/7.2B Albatross, Qwen3.5, and 7.2B split-graph
   matrices.
