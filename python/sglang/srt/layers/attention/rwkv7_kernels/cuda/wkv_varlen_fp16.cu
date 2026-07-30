// Copyright 2025-2026 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");
//
// The warp/head execution layout is informed by Albatross's Apache-2.0
// rwkv7_wkv_fp16_v2 kernel (commit 343147a333fcd6dd0845de0d165089685402c012),
// but this implementation has a distinct SGLang contract: packed-varlen token
// offsets, indexed paged state, zero-length graph sentinels, and precomputed
// RWKV-7 log decay.

#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <type_traits>

namespace {

constexpr int kHead = 64;
constexpr int kPairs = kHead / 2;

template <bool kApplyScale>
__global__ __launch_bounds__(kHead, 2) void rwkv7_wkv_varlen_fp16_kernel(
    half *__restrict__ state_pool, const half *__restrict__ r,
    const half *__restrict__ w, const half *__restrict__ k,
    const half *__restrict__ v, const half *__restrict__ kk,
    const half *__restrict__ a, half *__restrict__ output,
    const int *__restrict__ cu_seqlens, const int *__restrict__ cache_indices,
    int num_sequences, int num_heads, int state_slots, int total_tokens,
    float scale) {
  const int sequence = blockIdx.x / num_heads;
  const int head = blockIdx.x % num_heads;
  const int column = threadIdx.x;
  const int slot = cache_indices[sequence];
  const int begin = cu_seqlens[sequence];
  const int end = cu_seqlens[sequence + 1];
  if (slot < 0 || slot >= state_slots || begin < 0 || end <= begin ||
      end > total_tokens) {
    return;
  }

  // Stage a coalesced [K,V] state tile, then give each thread one V column.
  // The +1 padding removes shared-memory bank conflicts on the transpose.
  __shared__ __align__(128) half state_shared[kHead][kHead + 1];
  half *state = state_pool +
                (static_cast<int64_t>(slot) * num_heads + head) * kHead * kHead;
#pragma unroll
  for (int row = 0; row < kHead; ++row) {
    state_shared[row][column] = state[row * kHead + column];
  }
  __syncthreads();

  half2 state_registers[kPairs];
#pragma unroll
  for (int pair = 0; pair < kPairs; ++pair) {
    state_registers[pair] = __halves2half2(state_shared[pair * 2][column],
                                           state_shared[pair * 2 + 1][column]);
  }

  __shared__ __align__(128) half2 r_shared[kPairs];
  __shared__ __align__(128) half2 decay_shared[kPairs];
  __shared__ __align__(128) half2 k_shared[kPairs];
  __shared__ __align__(128) half2 neg_kk_shared[kPairs];
  __shared__ __align__(128) half2 b_shared[kPairs];

  for (int token = begin; token < end; ++token) {
    const int64_t base =
        (static_cast<int64_t>(token) * num_heads + head) * kHead;

    if (column < kPairs) {
      // Both warps cooperatively publish one K vector. Pairing adjacent lanes
      // keeps the recurrent inner loop entirely in half2 registers.
      const int even = column * 2;
      const int odd = even + 1;
      const int64_t pair_base = base + even;
      half2 r_pair = __halves2half2(r[pair_base], r[pair_base + 1]);
      if constexpr (kApplyScale) {
        r_pair = __hmul2(r_pair, __float2half2_rn(scale));
      }
      r_shared[column] = r_pair;
      decay_shared[column] =
          __halves2half2(__float2half_rn(expf(__half2float(w[pair_base]))),
                         __float2half_rn(expf(__half2float(w[pair_base + 1]))));
      k_shared[column] = __halves2half2(k[pair_base], k[pair_base + 1]);
      const half kk_even = kk[pair_base];
      const half kk_odd = kk[pair_base + 1];
      neg_kk_shared[column] = __halves2half2(__hneg(kk_even), __hneg(kk_odd));
      b_shared[column] = __halves2half2(__hmul(kk_even, a[pair_base]),
                                        __hmul(kk_odd, a[pair_base + 1]));
    }
    // Keep all 64 threads on the same barrier cadence while the first warp
    // publishes the per-token vectors for both warps to consume.
    __syncthreads();

    half2 sa_pair = __float2half2_rn(0.0f);
#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
      sa_pair = __hfma2(neg_kk_shared[pair], state_registers[pair], sa_pair);
    }
    const half sa = __hadd(__low2half(sa_pair), __high2half(sa_pair));
    const half2 sa2 = __halves2half2(sa, sa);
    const half value = v[base + column];
    const half2 value2 = __halves2half2(value, value);
    half2 y_pair = __float2half2_rn(0.0f);
#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
      half2 updated = __hfma2(
          state_registers[pair], decay_shared[pair],
          __hfma2(b_shared[pair], sa2, __hmul2(k_shared[pair], value2)));
      state_registers[pair] = updated;
      y_pair = __hfma2(updated, r_shared[pair], y_pair);
    }
    output[base + column] = __hadd(__low2half(y_pair), __high2half(y_pair));
    __syncthreads();
  }

#pragma unroll
  for (int pair = 0; pair < kPairs; ++pair) {
    state_shared[pair * 2][column] = __low2half(state_registers[pair]);
    state_shared[pair * 2 + 1][column] = __high2half(state_registers[pair]);
  }
  __syncthreads();
#pragma unroll
  for (int row = 0; row < kHead; ++row) {
    state[row * kHead + column] = state_shared[row][column];
  }
}

} // namespace

void rwkv7_wkv_varlen_fp16_cuda(torch::Tensor state_pool, torch::Tensor r,
                                torch::Tensor w, torch::Tensor k,
                                torch::Tensor v, torch::Tensor kk,
                                torch::Tensor a, torch::Tensor output,
                                torch::Tensor cu_seqlens,
                                torch::Tensor cache_indices, double scale) {
  TORCH_CHECK(state_pool.is_cuda(), "state_pool must be a CUDA tensor");
  TORCH_CHECK(state_pool.scalar_type() == at::kHalf, "state_pool must be fp16");
  TORCH_CHECK(r.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf &&
                  k.scalar_type() == at::kHalf &&
                  v.scalar_type() == at::kHalf &&
                  kk.scalar_type() == at::kHalf && a.scalar_type() == at::kHalf,
              "RWKV inputs must be fp16");
  TORCH_CHECK(state_pool.is_contiguous(), "state_pool must be contiguous");
  TORCH_CHECK(r.is_contiguous() && w.is_contiguous() && k.is_contiguous() &&
                  v.is_contiguous() && kk.is_contiguous() &&
                  a.is_contiguous() && output.is_contiguous(),
              "RWKV inputs and output must be contiguous");
  TORCH_CHECK(state_pool.dim() == 4 && state_pool.size(2) == kHead &&
                  state_pool.size(3) == kHead,
              "state_pool must have shape [S,H,64,64]");
  TORCH_CHECK(r.dim() == 4 && r.size(0) == 1 && r.size(3) == kHead,
              "r must have shape [1,T,H,64]");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt &&
                  cache_indices.scalar_type() == at::kInt,
              "cu_seqlens and cache_indices must be int32");
  TORCH_CHECK(cu_seqlens.is_cuda() && cache_indices.is_cuda(),
              "metadata tensors must be CUDA tensors");
  TORCH_CHECK(cu_seqlens.is_contiguous() && cache_indices.is_contiguous(),
              "metadata tensors must be contiguous");
  TORCH_CHECK(r.device() == state_pool.device() &&
                  w.device() == state_pool.device() &&
                  k.device() == state_pool.device() &&
                  v.device() == state_pool.device() &&
                  kk.device() == state_pool.device() &&
                  a.device() == state_pool.device() &&
                  output.device() == state_pool.device() &&
                  cu_seqlens.device() == state_pool.device() &&
                  cache_indices.device() == state_pool.device(),
              "all RWKV tensors must be on the state-pool device");
  TORCH_CHECK(w.sizes() == r.sizes() && k.sizes() == r.sizes() &&
                  v.sizes() == r.sizes() && kk.sizes() == r.sizes() &&
                  a.sizes() == r.sizes() && output.sizes() == r.sizes(),
              "RWKV inputs and output must have identical shapes");
  TORCH_CHECK(scale > 0.0 && std::isfinite(scale),
              "RWKV readout scale must be finite and positive");
  const int num_sequences = cache_indices.numel();
  TORCH_CHECK(cu_seqlens.numel() == num_sequences + 1,
              "cu_seqlens length must be N+1");
  const int num_heads = r.size(2);
  TORCH_CHECK(state_pool.size(1) == num_heads, "state head count mismatch");

  const c10::cuda::CUDAGuard device_guard(state_pool.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto apply_scale) {
    constexpr bool kApplyScale = decltype(apply_scale)::value;
    rwkv7_wkv_varlen_fp16_kernel<kApplyScale>
        <<<num_sequences * num_heads, kHead, 0, stream>>>(
            reinterpret_cast<half *>(state_pool.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(r.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(w.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(v.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(kk.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(a.data_ptr<at::Half>()),
            reinterpret_cast<half *>(output.data_ptr<at::Half>()),
            cu_seqlens.data_ptr<int>(), cache_indices.data_ptr<int>(),
            num_sequences, num_heads, state_pool.size(0), r.size(1),
            static_cast<float>(scale));
  };
  if (scale == 1.0) {
    launch(std::false_type{});
  } else {
    launch(std::true_type{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
