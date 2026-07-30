# RTX 4080 matched quantization evidence

This directory contains the matched dense, W8 accuracy, and W4 hybrid-accuracy
batch-8 rerun for commit `a51d446a15bee9d8dd65c7e9d146e1afbf22c022`.

The matrix covers prompt lengths 128/512/2048 and decode lengths 128/512.
Every cell flushes the recurrent radix cache, runs two warm-ups, and reports the
median of five samples. The W4 directory also includes quantized alignment and
serving-feature output.

This remains a one-model, one-batch, one-GPU engineering slice rather than the
full multi-model and multi-hardware acceptance matrix.
