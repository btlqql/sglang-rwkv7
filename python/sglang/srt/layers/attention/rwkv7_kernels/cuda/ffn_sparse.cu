// Copyright 2025-2026 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");
//
// Small-row RWKV-7 FFNs are exactly sparse after SqReLU: negative
// preactivations become zero. This kernel compacts each 128-wide activation
// tile and skips the corresponding rows of the transposed down-projection
// weight.
//
// The dynamic zero-compaction strategy is informed by Albatross's Apache-2.0
// cmix sparse-down kernels (commit
// 343147a333fcd6dd0845de0d165089685402c012). This implementation exposes a
// narrower SGLang tensor contract and retains the standard dense projection as
// the universal fallback.

#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

#if defined(__HIP_PLATFORM_AMD__)
constexpr int kHardwareWarp = 64;
using ballot_type = unsigned long long;
__device__ __forceinline__ ballot_type rwkv7_ballot(bool predicate) {
  return __ballot(predicate);
}
__device__ __forceinline__ int rwkv7_popcount(ballot_type value) {
  return __popcll(value);
}
#else
constexpr int kHardwareWarp = 32;
using ballot_type = unsigned;
__device__ __forceinline__ ballot_type rwkv7_ballot(bool predicate) {
  return __ballot_sync(0xffffffffu, predicate);
}
__device__ __forceinline__ int rwkv7_popcount(ballot_type value) {
  return __popc(value);
}
#endif

__device__ __forceinline__ ballot_type rwkv7_lane_prefix_mask(int lane) {
  return lane == 0 ? ballot_type{0}
                   : (~ballot_type{0} >> (kHardwareWarp - lane));
}

constexpr int kThreads = 128;
constexpr int kFfnTile = 128;
constexpr int kOutputTile = 2 * kThreads;

__global__ __launch_bounds__(kThreads, 4) void rwkv7_sparse_sqrelu_down_kernel(
    int hidden_size, int intermediate_size, const half *__restrict__ preact,
    const half *__restrict__ value_weight_t, half *__restrict__ output) {
  __shared__ __align__(256) half activation[kFfnTile];
  __shared__ __align__(256) int nonzero_ids[kFfnTile];
  __shared__ int warp_counts[kFfnTile / kHardwareWarp];
  __shared__ int warp_prefix[kFfnTile / kHardwareWarp];
  __shared__ int nonzero_count;

  const int f_tile = blockIdx.x;
  const int output_tile = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid % kHardwareWarp;
  const int warp = tid / kHardwareWarp;
  const int f_begin = f_tile * kFfnTile;
  const int output_begin = output_tile * kOutputTile;

  const float x = __half2float(
      preact[static_cast<int64_t>(row) * intermediate_size + f_begin + tid]);
  const float relu = fmaxf(x, 0.0f);
  activation[tid] = __float2half_rn(relu * relu);
  __syncthreads();

  const bool nonzero = (__half_as_ushort(activation[tid]) << 1) != 0;
  const ballot_type mask = rwkv7_ballot(nonzero);
  const int local_position =
      rwkv7_popcount(mask & rwkv7_lane_prefix_mask(lane));
  if (lane == 0) {
    warp_counts[warp] = rwkv7_popcount(mask);
  }
  __syncthreads();

  if (tid == 0) {
    int sum = 0;
#pragma unroll
    for (int w = 0; w < kFfnTile / kHardwareWarp; ++w) {
      warp_prefix[w] = sum;
      sum += warp_counts[w];
    }
    nonzero_count = sum;
  }
  __syncthreads();

  if (nonzero) {
    nonzero_ids[warp_prefix[warp] + local_position] = tid;
  }
  __syncthreads();

  half2 accumulator = __float2half2_rn(0.0f);
  for (int i = 0; i < nonzero_count; ++i) {
    const int local_f = nonzero_ids[i];
    const int f = f_begin + local_f;
    const half2 weight = *reinterpret_cast<const half2 *>(
        value_weight_t + static_cast<int64_t>(f) * hidden_size + output_begin +
        tid * 2);
    accumulator =
        __hfma2(__half2half2(activation[local_f]), weight, accumulator);
  }

  atomicAdd(reinterpret_cast<half2 *>(output +
                                      static_cast<int64_t>(row) * hidden_size +
                                      output_begin + tid * 2),
            accumulator);
}

__global__ __launch_bounds__(256, 2) void rwkv7_sparse_sqrelu_down_t512_kernel(
    int hidden_size, int intermediate_size, const half *__restrict__ preact,
    const half *__restrict__ value_weight_t, half *__restrict__ output) {
  constexpr int kTile = 512;
  constexpr int kTileThreads = 256;
  __shared__ __align__(256) half activation[kTile];
  __shared__ __align__(256) int nonzero_ids[kTile];
  __shared__ int warp_counts[kTile / kHardwareWarp];
  __shared__ int warp_prefix[kTile / kHardwareWarp];
  __shared__ int nonzero_count;

  const int f_tile = blockIdx.x;
  const int output_tile = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid % kHardwareWarp;
  const int warp = tid / kHardwareWarp;
  const int f_begin = f_tile * kTile;
  const int output_begin = output_tile * (2 * kTileThreads);
  const int64_t row_offset = static_cast<int64_t>(row) * intermediate_size;

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * kTileThreads;
    const float x = __half2float(preact[row_offset + f_begin + local_f]);
    const float relu = fmaxf(x, 0.0f);
    activation[local_f] = __float2half_rn(relu * relu);
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * kTileThreads;
    const bool nonzero = (__half_as_ushort(activation[local_f]) << 1) != 0;
    const ballot_type mask = rwkv7_ballot(nonzero);
    if (lane == 0) {
      warp_counts[warp + u * (kTileThreads / kHardwareWarp)] =
          rwkv7_popcount(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int sum = 0;
#pragma unroll
    for (int w = 0; w < kTile / kHardwareWarp; ++w) {
      warp_prefix[w] = sum;
      sum += warp_counts[w];
    }
    nonzero_count = sum;
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * kTileThreads;
    const bool nonzero = (__half_as_ushort(activation[local_f]) << 1) != 0;
    const ballot_type mask = rwkv7_ballot(nonzero);
    const int local_position =
        rwkv7_popcount(mask & rwkv7_lane_prefix_mask(lane));
    const int group = warp + u * (kTileThreads / kHardwareWarp);
    if (nonzero) {
      nonzero_ids[warp_prefix[group] + local_position] = local_f;
    }
  }
  __syncthreads();

  half2 accumulator = __float2half2_rn(0.0f);
  for (int i = 0; i < nonzero_count; ++i) {
    const int local_f = nonzero_ids[i];
    const int f = f_begin + local_f;
    const half2 weight = *reinterpret_cast<const half2 *>(
        value_weight_t + static_cast<int64_t>(f) * hidden_size + output_begin +
        tid * 2);
    accumulator =
        __hfma2(__half2half2(activation[local_f]), weight, accumulator);
  }

  atomicAdd(reinterpret_cast<half2 *>(output +
                                      static_cast<int64_t>(row) * hidden_size +
                                      output_begin + tid * 2),
            accumulator);
}

} // namespace

torch::Tensor rwkv7_sparse_sqrelu_down_cuda(torch::Tensor preact,
                                            torch::Tensor value_weight_t) {
  TORCH_CHECK(preact.is_cuda() && value_weight_t.is_cuda(),
              "preact and value_weight_t must be CUDA tensors");
  TORCH_CHECK(preact.scalar_type() == at::kHalf &&
                  value_weight_t.scalar_type() == at::kHalf,
              "preact and value_weight_t must be fp16");
  TORCH_CHECK(preact.dim() == 2 && value_weight_t.dim() == 2,
              "preact and value_weight_t must be rank-2");
  TORCH_CHECK(preact.is_contiguous() && value_weight_t.is_contiguous(),
              "preact and value_weight_t must be contiguous");
  TORCH_CHECK(preact.device() == value_weight_t.device(),
              "preact and value_weight_t must be on the same device");

  const int rows = preact.size(0);
  const int intermediate_size = preact.size(1);
  const int hidden_size = value_weight_t.size(1);
  TORCH_CHECK(value_weight_t.size(0) == intermediate_size,
              "FFN intermediate dimensions do not match");
  TORCH_CHECK(rows > 0 && rows <= 32, "sparse FFN supports 1..32 rows");
  TORCH_CHECK(intermediate_size % kFfnTile == 0,
              "FFN intermediate size must be divisible by 128");
  TORCH_CHECK(hidden_size % kOutputTile == 0,
              "hidden size must be divisible by 256");

  const c10::cuda::CUDAGuard device_guard(preact.device());
  auto output = at::zeros({rows, hidden_size}, preact.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (rows >= 8 && intermediate_size % 512 == 0 && hidden_size % 512 == 0) {
    const dim3 grid(intermediate_size / 512, hidden_size / 512, rows);
    rwkv7_sparse_sqrelu_down_t512_kernel<<<grid, 256, 0, stream>>>(
        hidden_size, intermediate_size,
        reinterpret_cast<const half *>(preact.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(value_weight_t.data_ptr<at::Half>()),
        reinterpret_cast<half *>(output.data_ptr<at::Half>()));
  } else {
    const dim3 grid(intermediate_size / kFfnTile, hidden_size / kOutputTile,
                    rows);
    rwkv7_sparse_sqrelu_down_kernel<<<grid, kThreads, 0, stream>>>(
        hidden_size, intermediate_size,
        reinterpret_cast<const half *>(preact.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(value_weight_t.data_ptr<at::Half>()),
        reinterpret_cast<half *>(output.data_ptr<at::Half>()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
