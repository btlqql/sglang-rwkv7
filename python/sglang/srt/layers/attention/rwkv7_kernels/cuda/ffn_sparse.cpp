// Copyright 2025-2026 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");

#include <torch/extension.h>

torch::Tensor rwkv7_sparse_sqrelu_down_cuda(torch::Tensor preact,
                                             torch::Tensor value_weight_t);

TORCH_LIBRARY(sglang_rwkv7_sparse_ffn, m) {
  m.def("sqrelu_down_fp16(Tensor preact, Tensor value_weight_t) -> Tensor");
}

TORCH_LIBRARY_IMPL(sglang_rwkv7_sparse_ffn, CUDA, m) {
  m.impl("sqrelu_down_fp16", &rwkv7_sparse_sqrelu_down_cuda);
}
