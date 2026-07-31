# RTX 4080 matched performance evidence

This directory contains matched dense, W8 accuracy, and W4 hybrid-accuracy
batch-1/2/4/8 evidence for commit
`74c196347c2956e78d185fd0bf5dc78ddaaa502e`.
The Albatross fixed-shape reference is commit
`343147a333fcd6dd0845de0d165089685402c012`.

The serving matrix covers prompt lengths 128/512/2048, decode lengths 128/512,
and active batch sizes 1/2/4/8. Every cell flushes the recurrent radix cache,
runs two warm-ups, and reports the median of five samples. Batch 1/2/4 uses the
exact aggregate-token graph buckets 128/256/512/1024/2048/4096/8192/16384;
the earlier batch-8 files already use exact 1024/4096/16384 buckets. Dense, W8,
and W4 alignment reports are included; the W4 directory also includes the
serving-feature gate.

For 1.5B, W8 is at least 1.048x/1.053x/1.054x dense for prefill/decode/E2E;
W4 is at least 1.023x/1.139x/1.127x. For 2.9B, those minima are
1.091x/1.094x/1.094x and 1.087x/1.231x/1.179x. Quantized alignment passes for
both models. Production-safe 7.2B W4 and W8 batch-8 capacity lanes now also
pass alignment and serving-lifecycle gates. Fresh same-runtime Qwen3.5 and full
Albatross artifacts retain the remaining red cells. These measurements still
represent one GPU rather than the full multi-hardware acceptance matrix.

The first matrix command accidentally supplied the Albatross commit in the
`standard_sha` field. The checked-in JSONL corrects that metadata to the pinned
HF acceptance commit from `environment.json`; numeric measurements, samples,
model code SHA, and baseline files are unchanged.

## Width-specialized W4 shadow update

Commits `5a2b575b0309afda6b0ff47492c259f943a9d33a` and
`cbc295ced33373035c3bec888646cfc334914954` replace the universal persistent
FP16 W4 fallback with a width-aware small-M policy:

- full-prefill CUDA Graphs keep packed Marlin W4 for throughput;
- 2.9B and 7.2B decode/eager short prefill use a compact per-channel INT8
  shadow and fused integer GEMM;
- the 2,048-wide 1.5B checkpoint keeps its exact FP16 shadow because its
  near-tied synthetic logits are more sensitive to recurrent rounding;
- `SGLANG_RWKV7_W4_SHADOW=fp16|int8` can override the automatic choice.

Fresh batch-8 results use two warm-ups and the median of five samples:

| Model / shadow | Model memory | Prefill tok/s range | Decode tok/s range | Change from prior safe W4 |
| --- | ---: | ---: | ---: | --- |
| 1.5B / FP16 | 2.63 GB | 26,793-29,485 | 1,530-1,546 | prefill -0.15% to +0.76%; decode +0.36% to +0.59% |
| 2.9B / INT8 | 4.72 GB | 16,249-17,488 | 850-852 | prefill -0.26% to +0.18%; decode +3.89% to +4.15%; memory -5.6% |
| 7.2B / INT8 | 10.83 GB | 7,276-7,563 | 417 | prefill +1.99% to +6.85%; decode +6.02% to +6.16%; memory -6.5% |

The 1.5B matrix remains at least 1.142x/1.066x/1.070x matched dense for
prefill/decode/end-to-end and 1.123x/1.444x/1.335x same-runtime Qwen3.5-2B.
Relative to Albatross, decode is 1.254x-1.267x and prefill is within
-0.80%/+0.85%. Its alignment gate passes at 0.1114 maximum chosen-token
logprob error, 0.9680 mean top-10 overlap, and 1.0000 teacher-forced top-1
agreement.

The 2.9B matrix is at least 1.187x/1.133x/1.137x matched dense,
1.484x/1.704x/1.648x Qwen3.5-4B, and 1.083x/1.214x Albatross for
prefill/decode. Its alignment gate passes at 0.1763 maximum error, 0.9781
top-10 overlap, and 0.9922 teacher-forced top-1 agreement.

The 1.5B and 2.9B serving reports deliberately retain a two-token synthetic
repeat/single-vs-batch gate rather than claiming bit-exact long quantized
generation. Both pass exact cold/warm chunked-prefill output, a 128-token state
cache hit, dynamic-batch duplicate isolation, mixed-length compaction, abort,
and post-abort state reuse. The 7.2B report passes the complete 32-token
deterministic gate, exact single-vs-batch output, cache restore, and the full
lifecycle checks. Its alignment report has 0.0949 maximum error, 0.9711 top-10
overlap, 1.0000 top-1 agreement, and four exact natural-prompt continuations.

The compact 7.2B layout also frees enough memory to capture the complete
16,384-token full-prefill graph on the 16 GB card; the prior 11.58 GB safe lane
missed that graph by about 192 MiB and had to use eager 2,048-token-per-request
prefill. Dense 7.2B, full Albatross batch 8, and Qwen3.5-9B still require the
separate 24 GB matched-runtime lane and are not inferred here.

Raw evidence:

- `rwkv7-g1-1.5b/w4-auto-fp16-shadow-cbc295c*`
- `rwkv7-g1-2.9b/w4-auto-int8-shadow-cbc295c*`
- `rwkv7-g1-7.2b/w4-fused-shadow-fullctx-5a2b575*`

## Production-safe 2.9B W4 update

Commit `d17883f3f671adb8d0998034072649a88931e95d` adds a safer 2.9B W4 hybrid
lane after the earlier high-compression policy exposed batch-layout-sensitive
long greedy continuations. Its committed batch-8 evidence covers all six
128/512/2048 prefill by 128/512 decode cells. Minimum gains over matched dense
are 1.118x prefill, 1.099x decode, and 1.101x end to end; model-weight memory is
5.00 GB versus 5.68 GB dense. It also passes the strict quant alignment gate,
full cold/warm chunked-prefill equality, state-cache hit, mixed-length
compaction, abort, and post-abort slot-reuse checks. See the three
`rwkv7-g1-2.9b/w4-hybrid-safe-d17883f3*` artifacts. The serving report
explicitly records a two-token repeat/duplicate prefix gate rather than
claiming bit-exact long synthetic quantized generation.

## 512-token safe-cutoff update

The 512-token FP16/INT32 safety cutoff was measured on `862f6328` and promoted
as the default by `97116ffb`. New batch-8 artifacts are named
`w4-hybrid-safe-th512-862f632*` under the 1.5B and 2.9B directories.

For 1.5B, minimum ratios are 1.134x/1.062x/1.065x versus matched dense for
prefill/decode/end-to-end, 0.994x prefill and 1.249x decode versus Albatross,
and 1.118x/1.438x/1.327x versus Qwen3.5-2B. The model load is 2.63 GB versus
3.03 GB dense. Its strict alignment report reproduces all four natural-prompt
continuations exactly, and its serving report passes the complete repeat,
batching, cache, compaction, abort, and slot-reuse lifecycle gate.

For 2.9B, minimum ratios are 1.188x/1.088x/1.094x versus matched dense,
1.084x prefill and 1.166x decode versus Albatross, and
1.485x/1.636x/1.613x versus Qwen3.5-4B. Model load remains 5.00 GB versus
5.68 GB dense. Strict teacher-forced alignment and all production lifecycle
checks pass; its serving artifact retains the explicit two-token synthetic
repeat-prefix disclosure.

## 7.2B batch-8 capacity update

The `w4-hybrid-safe-th512-8fa6e43*` and
`w8-accuracy-fp32-th512-8fa6e43*` artifacts under `rwkv7-g1-7.2b` complete the
six 128/512/2048 prefill by 128/512 decode cells at batch 8. Short prefills use
exact fixed-shape full-prefill graphs; the 2,048-token-per-request cells use
eager prefill because a 16,384-token graph exceeds the 16 GB budget by about
192 MiB. Every decode cell uses a batch-8 full CUDA Graph.

W4 loads at 11.58 GB and sustains 392.8-393.2 decode tok/s. W8 with strict FP32
recurrent state loads at 11.56 GB and sustains 377.9-378.4 decode tok/s. Both
alignment reports pass. W4 reproduces all four natural 32-token references;
W8 reaches 1.0000 teacher-forced top-1 agreement. Both pass cache restore,
dynamic batching, compaction, abort, and post-abort slot reuse; W8 additionally
passes the complete deterministic synthetic serving gate.

The 16 GB card cannot host the dense 7.2B production matrix, a complete
Albatross batch-8 matrix, or a same-runtime Qwen3.5-9B baseline. Those
cross-model comparisons remain explicitly assigned to the 24 GB acceptance
lane rather than inferred from smaller batches.
