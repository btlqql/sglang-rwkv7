# RTX 4080 native W8/W4 evidence

This directory contains the matched batch-8 acceptance slice for the portable
RWKV-7 native quantization kernels.

## Files

- `*-b8.jsonl`: two warm-ups and median-of-five serving measurements for
  128/512/2048 prompt tokens per request and 128 generated tokens;
- `*-consistency.json`: independent dense reference teacher-forced quantization gate;
- `*-serving.json`: dynamic batching, chunked prefill, recurrent cache and
  request-lifecycle checks;
- `model-memory.jsonl`: model-weight memory reported by the server loader;
- `environment.json`: public hardware and software manifest.

The cache was flushed before every performance sample. The server used full
decode graphs at batch sizes 1/2/4/8 and fixed full-prefill graph buckets at
128/256/512/1024/2048/4096/8192/16384 aggregate tokens.

See [`../../../RESULTS_4080.md`](../../../RESULTS_4080.md) for the matched dense
and Albatross comparison.
