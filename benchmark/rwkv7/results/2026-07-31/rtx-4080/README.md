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
both models, and 7.2B W4 alignment also passes. Fresh same-runtime Qwen3.5 and
full Albatross artifacts retain the remaining red cells. These measurements
still represent one GPU rather than the full multi-hardware acceptance matrix.

The first matrix command accidentally supplied the Albatross commit in the
`standard_sha` field. The checked-in JSONL corrects that metadata to the pinned
HF acceptance commit from `environment.json`; numeric measurements, samples,
model code SHA, and baseline files are unchanged.

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
