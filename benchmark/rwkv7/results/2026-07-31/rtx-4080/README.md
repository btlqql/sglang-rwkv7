# RTX 4080 matched performance evidence

This directory contains the matched dense, W8 accuracy, and W4 hybrid-accuracy
batch-1/2/4/8 rerun for commit
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

Across all 24 cells, W8 is at least 1.048x/1.053x/1.054x dense for
prefill/decode/E2E. W4 is at least 1.023x/1.139x/1.127x dense. The W8 lane also
exceeds the matched Albatross B8 fixed-shape throughput at T=1, 128, 512, and
2048. These measurements still represent one model and one GPU rather than the
full multi-model, multi-hardware acceptance matrix.
