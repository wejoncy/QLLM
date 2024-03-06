// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cmath>
#include <math_constants.h>
#include <ATen/cuda/CUDAContext.h>

#include <ATen/cuda/CUDAContext.h>

using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct GridDim {
#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif
enum : CUDA_LONG {
    maxThreadsPerBlock = 256, // max threads per block
    maxElementsPerThread = 4, // max element processed per thread
  };
};

template <class INT, class INT2>
inline __host__ __device__ INT CeilDiv(INT a, INT2 b) // ceil(a/b)
{
return (INT)(((size_t)a + (size_t)b - 1) /
             (size_t)b); // these size_t casts are necessary since b may be
                         // INT_MAX (for maxGridSize[])
}

#define half2 __half2

__device__ __forceinline__ void DequantizeEightElements(uint32_t values_quant, half scale, half zp, half* output) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = __hmul(__float2half (-1.0f),__hmul(scale , (zp)));
  half2 zp_adjust2 = {zp_adjust, zp_adjust};

  alignas(16) half2 results[4];
  half v0 = __uint2half_rn(values_quant & 0xF);
  half v1 = __uint2half_rn((values_quant >> 4) & 0xF);
  results[0] = __hadd2(__hmul2(__halves2half2(v0, v1), scale_half2), zp_adjust2);

  half v2 = __uint2half_rn((values_quant >> 8) & 0xF);
  half v3 = __uint2half_rn((values_quant >> 12) & 0xF);
  results[1] =
      __hadd2(__hmul2(__halves2half2(v2, v3), scale_half2), zp_adjust2);

  half v4 = __uint2half_rn((values_quant >> 16) & 0xF);
  half v5 = __uint2half_rn((values_quant >> 20) & 0xF);
  results[2] =
      __hadd2(__hmul2(__halves2half2(v4, v5), scale_half2), zp_adjust2);

  half v6 = __uint2half_rn((values_quant >> 24) & 0xF);
  half v7 = __uint2half_rn((values_quant >> 28) & 0xF);
  results[3] =
      __hadd2(__hmul2(__halves2half2(v6, v7), scale_half2), zp_adjust2);
  *(reinterpret_cast<float4*>(output)) = *(reinterpret_cast<float4*>(results));
}

__device__ __forceinline__ void DequantizeEightElements(uint32_t values_quant, float scale, float zp, float* output) {
  float zp_adjust = -scale * zp;
  output[0] = float(values_quant & 0xF) * scale + zp_adjust;
  output[1] = float((values_quant >> 4) & 0xF) * scale + zp_adjust;
  output[2] = float((values_quant >> 8) & 0xF) * scale + zp_adjust;
  output[3] = float((values_quant >> 12) & 0xF) * scale + zp_adjust;
  output[4] = float((values_quant >> 16) & 0xF) * scale + zp_adjust;
  output[5] = float((values_quant >> 20) & 0xF) * scale + zp_adjust;
  output[6] = float((values_quant >> 24) & 0xF) * scale + zp_adjust;
  output[7] = float((values_quant >> 28) & 0xF) * scale + zp_adjust;
}
template <class T>
__global__ void Dequantize4BitsKernelReOrder(
    T *output, const uint8_t *quant_data, const T *scale_data,
    const uint8_t *zero_points, const int32_t *reorder_idx, int block_size,
    int groups_per_K, int groups_per_threadblock, int total_groups) {
  int group_id =
      blockIdx.x * groups_per_threadblock + ((threadIdx.x * 8) / block_size);
  if (group_id >= total_groups) {
    return;
  }
  // T __shared__ zero_points_after_reorder[];//K
  // T __shared__ scales_after_reorder[];     // K
  // const int num_r_per_thread = k / 256;

  const int zero_point_shape_x = (groups_per_K + 1) / 2;
  const int scales_shape_x = groups_per_K;
  int n_idx = group_id / scales_shape_x;
  int kb_idx = group_id % scales_shape_x;
  int element_offset =
      group_id * block_size + ((threadIdx.x * 8) & (block_size - 1));
  T *output_i = output + element_offset;
  uint32_t quant_value =
      *(reinterpret_cast<const uint32_t *>(quant_data + element_offset / 2));
  const int32_t* reorder_idx_with_off = reorder_idx+kb_idx * block_size + ((threadIdx.x * 8) &
                              (block_size - 1));
  for (int i = 0; i < 8; i++) {
    int32_t rid = reorder_idx_with_off[i];
    T scale = *(scale_data + n_idx * scales_shape_x + rid);
    uint8_t zp = 8;
    if (zero_points) {
      zp = zero_points[n_idx * zero_point_shape_x + rid / 2];
      zp = (rid & 0x01) ? (zp >> 4) : (zp & 0x0f);
    }

    if constexpr (std::is_same_v<T, half>) {
      T zp_adjust = -scale * __short2half_rn(zp);
      output_i[i] =
          __uint2half_rn((quant_value >> (4 * i)) & 0xF) * scale + zp_adjust;
    } else {
      T zp_adjust = -scale * T(zp);
      output_i[i] = T((quant_value >> (4 * i)) & 0xF) * scale + zp_adjust;
    }
  }
}

template <class T, typename ZeroT = uint8_t>
__global__ void
Dequantize4BitsKernel(T *output, const uint8_t *quant_data, const T *scale_data,
                      const ZeroT *zero_points, int block_size,
                      int groups_per_K, int groups_per_threadblock,
                      int total_groups) {
  int block_id =
      blockIdx.x * groups_per_threadblock + ((threadIdx.x * 8) / block_size);
  if (block_id >= total_groups) {
    return;
  }
  int element_offset =
      block_id * block_size + ((threadIdx.x * 8) & (block_size - 1));
  uint32_t quant_value =
      *(reinterpret_cast<const uint32_t *>(quant_data + element_offset / 2));
  T scale = *(scale_data + block_id);
  T zero_point_value;
  if constexpr (std::is_same_v<ZeroT, uint8_t>) {
    const int scales_shape_x = groups_per_K;
    const int zero_point_shape_x = (groups_per_K + 1) / 2;
    int kb_idx = block_id % scales_shape_x;
    int n_idx = block_id / scales_shape_x;
    uint8_t zp = 8;
    if (zero_points) {
      zp = zero_points[n_idx * zero_point_shape_x + kb_idx / 2];
      zp = (kb_idx & 0x01) ? (zp >> 4) : (zp & 0x0f);
    }
    zero_point_value = static_cast<T>(zp);
  } else {
    zero_point_value =
        zero_points ? *(zero_points + block_id) : static_cast<T>(8);
  }

  output = output + element_offset;
  DequantizeEightElements(quant_value, scale, zero_point_value, output);
}

template <class T, typename ZeroT>
int Dequantize4Bits(T *output, const uint8_t *quant_data, const T *scales_data,
                    const ZeroT *zero_points, // shape: [N, (block_per_K + 1)/2]
                    const int32_t *reorder_idx, int k, int n, int block_size,
                    cudaStream_t stream) {
  // k is padded and equal to block_per_K * block_size
  //ORT_ENFORCE(k % block_size == 0, "k must be a multiplier of block_size");
  constexpr int element_per_thread = 8;
  int groups_per_threadblock =
      GridDim::maxThreadsPerBlock * element_per_thread / block_size;
  int groups_per_K = k / block_size;
  int total_groups = n * groups_per_K; // total elemenets in quant_data
  int groups_per_grid =
      static_cast<int>(CeilDiv(total_groups, groups_per_threadblock));
  if (!reorder_idx) {
    Dequantize4BitsKernel<T, ZeroT>
        <<<groups_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            output, quant_data, scales_data, zero_points, block_size,
            groups_per_K, groups_per_threadblock, total_groups);
  } else {
    // static_assert(std::is_same_v<ZeroT, uint8_t>, "ZeroT must be uint8_t");
    Dequantize4BitsKernelReOrder<<<groups_per_grid, GridDim::maxThreadsPerBlock,
                                   0, stream>>>(
        output, quant_data, scales_data, (const uint8_t *)zero_points,
        reorder_idx, block_size, groups_per_K, groups_per_threadblock,
        total_groups);
  }

  return 0;
}

template <>
int Dequantize4Bits<c10::Half, uint8_t>(c10::Half *output, const uint8_t *quant_data,
                               const c10::Half *scales_data,
                               const uint8_t *zero_points,
                               const int32_t *reorder_idx,
                               int k, int n,
                               int block_size, cudaStream_t stream) {
  return Dequantize4Bits<half, uint8_t>(
      reinterpret_cast<half *>(output), quant_data,
      reinterpret_cast<const half *>(scales_data), zero_points, reorder_idx, k,
      n, block_size, stream);
}

template <>
int Dequantize4Bits<c10::Half, c10::Half>(
    c10::Half *output, const uint8_t *quant_data, const c10::Half *scales_data,
    const c10::Half *zero_points, const int32_t *reorder_idx, int k, int n,
    int block_size, cudaStream_t stream) {
  return Dequantize4Bits<half, half>(
      reinterpret_cast<half *>(output), quant_data,
      reinterpret_cast<const half *>(scales_data), reinterpret_cast<const half *>(zero_points), reorder_idx, k,
      n, block_size, stream);
}

template int
Dequantize4Bits<half, half>(half *output, const uint8_t *quant_data,
                            const half *scales_data, const half *zero_points,
                            const int32_t *reorder_idx, int k, int n,
                            int block_size, cudaStream_t stream);

template int Dequantize4Bits<half, uint8_t>(
    half *output, const uint8_t *quant_data, const half *scales_data,
    const uint8_t *zero_points, const int32_t *reorder_idx, int k, int n,
    int block_size, cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
