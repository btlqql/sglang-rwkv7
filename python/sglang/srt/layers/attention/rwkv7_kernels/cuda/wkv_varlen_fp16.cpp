// Copyright 2025-2026 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");

#include <torch/extension.h>

void rwkv7_wkv_varlen_fp16_cuda(
    torch::Tensor state_pool,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor kk,
    torch::Tensor a,
    torch::Tensor output,
    torch::Tensor cu_seqlens,
    torch::Tensor cache_indices,
    double scale);

TORCH_LIBRARY(sglang_rwkv7_cuda, m) {
  m.def(
      "wkv_varlen_fp16(Tensor(a!) state_pool, Tensor r, Tensor w, Tensor k, "
      "Tensor v, Tensor kk, Tensor a, Tensor(b!) output, Tensor cu_seqlens, "
      "Tensor cache_indices, float scale) -> ()");
}

TORCH_LIBRARY_IMPL(sglang_rwkv7_cuda, CUDA, m) {
  m.impl("wkv_varlen_fp16", &rwkv7_wkv_varlen_fp16_cuda);
}
