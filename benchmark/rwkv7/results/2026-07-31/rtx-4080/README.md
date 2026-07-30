# RTX 4080 matched performance evidence

This directory contains the matched dense, W8 accuracy, and W4 hybrid-accuracy
batch-8 rerun for commit `74c196347c2956e78d185fd0bf5dc78ddaaa502e`.
The Albatross fixed-shape reference is commit
`343147a333fcd6dd0845de0d165089685402c012`.

The serving matrix covers prompt lengths 128/512/2048 and decode lengths
128/512. Every cell flushes the recurrent radix cache, runs two warm-ups, and
reports the median of five samples. Dense, W8, and W4 alignment reports are
included; the W4 directory also includes the serving-feature gate.

The W8 lane exceeds the matched Albatross B8 fixed-shape throughput at T=1,
128, 512, and 2048. These measurements still represent one model, one batch
size, and one GPU rather than the full multi-model, multi-batch, multi-hardware
acceptance matrix.
