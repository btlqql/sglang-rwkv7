# RTX 4080 raw evidence

This directory contains the raw six-cell JSONL rows behind
[`benchmark/rwkv7/RESULTS_4080.md`](../../../RESULTS_4080.md).

The files cover RWKV-7 G1 1.5B at batch size 8, prompt lengths
128/512/2048, and decode lengths 128/512. Each row contains all three samples
plus the median. The result is a scoped engineering snapshot, not completion
of the full 216-cell acceptance matrix.
