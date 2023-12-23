#include <stdio.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>

#include "common.cuh"

#include <torch/extension.h>
#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#define half at::Half

namespace onnxruntime_gptq {
const int width_element_per_block = 32 * 2;
template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
  if (WarpSize >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (WarpSize >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (WarpSize >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (WarpSize >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (WarpSize >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}

template <typename scalar_t>
__global__ void gemv(scalar_t* out, const scalar_t* inA, const uint32_t* inB, const scalar_t* scales, const uint32_t* qzeros, int32_t groupsize, int32_t size_k, int32_t size_n, uint8_t add_zero_bias) {
  int bid = blockIdx.x;
  //__shared__ scalar_t vecA[size_k];
  __shared__ float bsum[2][32][32 + 1];
  float sum[2] = {0, 0};
  const int block_k = ((size_k + 31) / 32 + 7) / 8 * 8;
  int y_start = threadIdx.y * block_k;
  using VEC2 = typename cuda_quant::TYPE_VEC2<scalar_t>::Type;

  VEC2 res2 = {};
  VEC2 res2_1 = {};

  const VEC2* inA_start = (const VEC2*)(inA + blockIdx.y * size_k + y_start);

  int n_offset_x = bid * width_element_per_block + threadIdx.x * 2;

  int start_group_id = (y_start / groupsize);
  int compressed_idx = threadIdx.x % 4;
  VEC2 scale = ((VEC2*)(scales + start_group_id * size_n + n_offset_x))[0];
  int32_t qzero_p = qzeros==nullptr?0x88888888:((int32_t*)(qzeros + n_offset_x / 8 +
                                start_group_id * ((size_n + 7) / 8)))[0];
  uint8_t zero_1 =(qzero_p >> (8 * (compressed_idx))) & 0xf;
  uint8_t zero_2 = ((qzero_p) >> (8 * (compressed_idx) + 4)) & 0xf;
  VEC2 hzero = __halves2half2(__int2half_rn(zero_1+add_zero_bias),
                               __int2half_rn(zero_2+add_zero_bias));
  VEC2 scale_h0 = __half2half2(scale.x);
  VEC2 scale_h1 = __half2half2(scale.y);
  VEC2 hzero_scale_0 = __half2half2(hzero.x * scale.x);
  VEC2 hzero_scale_1 = __half2half2(hzero.y * scale.y);

#pragma unroll
  for (int i = 0; i < block_k / 2; i += 4) {  // read VEC2 * 4
    res2 = {};
    res2_1 = {};
    int k_offset = y_start + i * 2;
    int g_id = k_offset / groupsize;

    const uint32_t* hinB = inB + n_offset_x + k_offset / 8 * size_n;
    uint32_t vbInt1 =
        (n_offset_x < size_n && (k_offset < size_k)) ? hinB[0] : int32_t(0);
    uint32_t vbInt2 = (n_offset_x + 1 < size_n && (k_offset < size_k))
                          ? (hinB)[1]
                          : int32_t(0);
    VEC2 vb[8];
    uint8_t* qweight_p1 = (uint8_t*)&vbInt1;
    uint8_t* qweight_p2 = (uint8_t*)&vbInt2;

#pragma unroll
    for (int j = 0; j < 4; j++) {
      // vb[j] = __halves2half2(__int2half_rn(((vbInt1 >> (j * 8))) & 0xF),
      //                        __int2half_rn(((vbInt1) >> (j*8+4)) & 0xF));
      // vb[j + 4] = __halves2half2(__int2half_rn(((vbInt2)>>(j*8)) & 0xF),
      //                            __int2half_rn((((vbInt2) >> (j*8+4))) &
      //                            0xF));
      vb[j] = __halves2half2(__int2half_rn(((*(qweight_p1 + j))) & 0xF),
                             __int2half_rn(((*(qweight_p1 + j)) >> 4) & 0xF));
      vb[j + 4] =
          __halves2half2(__int2half_rn(((*(qweight_p2 + j))) & 0xF),
                         __int2half_rn((((*(qweight_p2 + j)) >> 4)) & 0xF));
    }

    if (g_id > start_group_id) {
      scale = ((const VEC2*)(scales + g_id * size_n + n_offset_x))[0];
      qzero_p = ((const int32_t*)(qzeros + n_offset_x / 8 + g_id * ((size_n + 7) / 8)))[0];
      hzero = __halves2half2(__int2half_rn((qzero_p >> (8 * (compressed_idx))) & 0xf),
                             __int2half_rn(((qzero_p) >> (8 * (compressed_idx) + 4)) & 0xf));
      scale_h0 = __half2half2(scale.x);
      scale_h1 = __half2half2(scale.y);
      hzero_scale_0 = __half2half2(hzero.x * scale.x);
      hzero_scale_1 = __half2half2(hzero.y * scale.y);
      start_group_id++;
    }

    VEC2 va[4];
    va[0] = (k_offset < size_k) ? ((inA_start))[i] : res2;
    va[1] = (k_offset + 1 < size_k) ? ((inA_start))[i + 1] : res2;
    va[2] = (k_offset + 2 < size_k) ? ((inA_start))[i + 2] : res2;
    va[3] = (k_offset + 3 < size_k) ? ((inA_start))[i + 3] : res2;

#pragma unroll
    for (int j = 0; j < 4; j++) {
      vb[j] = __hfma2(scale_h0, vb[j], -hzero_scale_0);
      res2 = __hfma2(va[j], vb[j], res2);
      vb[4 + j] = __hfma2(scale_h1, vb[4 + j], -hzero_scale_1);
      res2_1 = __hfma2(va[j], vb[4 + j], res2_1);
    }

    sum[0] += __half2float(res2.x) + __half2float(res2.y);
    sum[1] += __half2float(res2_1.x) + __half2float(res2_1.y);
  }
  // sum[0] += __half2float(res2.x);
  // sum[1] +=  __half2float(res2.y);
  bsum[0][threadIdx.x][threadIdx.y] = sum[0];
  bsum[1][threadIdx.x][threadIdx.y] = sum[1];

  __syncthreads();
  sum[0] = 0;
  sum[1] = 0;

#pragma unroll
  for (int i = 0; i < 2; i++) {
    sum[i] = bsum[i][threadIdx.y][threadIdx.x];
    __syncthreads();
    sum[i] = warpReduceSum<32>(sum[i]);
    if (threadIdx.x == 0) {
      out[+blockIdx.y * size_n + bid * width_element_per_block +
          threadIdx.y * 2 + i] = __float2half_rn(sum[i]);
    }
  }
}


void lauch_Gemv_kernel(torch::Tensor& out_fp16, const torch::Tensor& a_fp16, const torch::Tensor& qweight_i32,
                       const torch::Tensor& scale_fp16, const torch::Tensor& qzeros_i32,
                       int bits, int groupsize, uint32_t mat_m, uint32_t mat_k, uint32_t mat_n, uint8_t add_zero_bias) {
  if (bits != 4 || groupsize != 128) {
    printf("only support 4bit quantization, and groupsize must be 128\n");
    abort();
  }
  const int block_k = ((mat_k + 31) / 32 + 7) / 8 * 8;

  dim3 gridDim = {(mat_n + width_element_per_block - 1) / width_element_per_block, mat_m};
  dim3 blockDim = {32, (mat_k + block_k - 1) / block_k};
  using scalar_t = half;

  gemv<scalar_t><<<gridDim, blockDim>>>(out_fp16.data_ptr<scalar_t>(),
                                        a_fp16.data_ptr<scalar_t>(),
                                        (const uint32_t*)(qweight_i32.data_ptr<int32_t>()),
                                        scale_fp16.data_ptr<scalar_t>(),
                                        (const uint32_t*)(qzeros_i32.data_ptr<int32_t>()),
                                        groupsize, mat_k, mat_n, add_zero_bias);
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    abort();
  }
}


constexpr int kBlockSize = 256;
//constexpr int kNumWaves = 32;

namespace cuda_quant {

#define FETCH_UINT2(pointer) (reinterpret_cast<uint2*>(&(pointer))[0])
#define FETCH_HALF2(pointer) (reinterpret_cast<half2*>(&(pointer))[0])
#define FETCH_VEC2(pointer) (reinterpret_cast<VEC2*>(&(pointer))[0])

template <typename scalar_t, int WBITS>
__global__ void DequantizeAndUnpackWeight357_g(
    scalar_t *out, uint32_t *qweight, scalar_t *scale, uint32_t *qzeros,
    int32_t *g_idx, int group_size, const int in_features, const int n,
    uint8_t add_zero_bias) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  int out_x = tid % n;
  int out_y = tid / n;
  int scale_row = g_idx[out_y];

  const int max_num_in_bits = (1 << WBITS) - 1;

  const int qzero_width = (n * WBITS + 32 - 1) / 32;
  scalar_t scale_v = scale[scale_row * n + out_x];
  uint32_t zero_v1 =0x88888888;
  uint8_t zv1 = 0;
  if (qzeros != nullptr){
    int start_bits = out_x * WBITS;
    int first = start_bits / 32;
    int end_bits = (start_bits + WBITS);
    int second = end_bits / 32;
    start_bits = start_bits % 32;
    end_bits = end_bits % 32;

    zero_v1 = qzeros[scale_row * qzero_width + first];
    zv1 = (zero_v1 >> start_bits) & max_num_in_bits;
    if (first != second) {
      zero_v1 = qzeros[scale_row * qzero_width + second];
      zv1 |= (zero_v1 & ((1 << end_bits) - 1)) << (32 - start_bits);
    }
  }

  scalar_t scale_zeros = __hmul(scale_v, __ushort2half_rn(zv1 + add_zero_bias));

  uint32_t weight_int = 0;
  uint8_t wv1 = 0;
  {
    int start_bits = out_y * WBITS;
    int first = start_bits / 32;
    int end_bits = (start_bits + WBITS);
    int second = end_bits / 32;
    start_bits = start_bits % 32;
    end_bits = end_bits % 32;

    weight_int = qweight[first * n + out_x];
    wv1 = (weight_int >> start_bits) & ((1<<start_bits) - 1);
    if (first != second) {
      weight_int = qweight[second * n + out_x];
      wv1 |= (weight_int & ((1 << end_bits) - 1)) << (32 - start_bits);
    }
  }

  scalar_t wv = __ushort2half_rn(wv1);
  out[tid] = __hfma(wv, scale_v, -scale_zeros);
}

template <typename scalar_t, int WBITS>
__global__ void DequantizeAndUnpackWeight248_g(scalar_t* out, uint32_t* qweight, scalar_t* scale, uint32_t* qzeros, int32_t* g_idx, 
int group_size, const int in_features, const int n, uint8_t add_zero_bias) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  int out_x = tid % n;
  int out_y = tid / n;
  int scale_row = g_idx[out_y];

  const int compress_group_size = 32 / WBITS;
  const int max_num_in_bits = (1 << WBITS) - 1;

  scalar_t scale_v = scale[scale_row * n + out_x];
  uint32_t zero_v = qzeros == nullptr
                        ? 0x88888888
                        : qzeros[scale_row * (n / compress_group_size) +
                                 (out_x / compress_group_size)];
  int zero_ind = out_x % compress_group_size;
  uint8_t zv1 = (zero_v >> (zero_ind * WBITS)) & max_num_in_bits;

  scalar_t scale_zeros = __hmul(scale_v, __ushort2half_rn(zv1 + add_zero_bias));

  int weight_int = qweight[(out_y / compress_group_size) * n + out_x];
  int weight_ind = (out_y) % compress_group_size;
  uint8_t wv1 = (weight_int >> (weight_ind * WBITS)) & max_num_in_bits;
  scalar_t wv = __ushort2half_rn(wv1);
  out[tid] = __hfma(wv, scale_v, -scale_zeros);
}

template <typename scalar_t, int WBITS>
__global__ void DequantizeAndUnpackWeight248(scalar_t* out, uint32_t* qweight, scalar_t* scale, uint32_t* qzeros, 
int group_size, const int in_features, const int n, uint8_t add_zero_bias) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  const int half_n = n/2;

  using VEC2 = typename TYPE_VEC2<scalar_t>::Type;
  const int compress_group_size = 32 / WBITS;
  const int max_num_in_bits = (1 << WBITS) - 1;
  int col_ind = (tid % half_n)*2;
  int weight_in_row = tid / half_n * compress_group_size;
  VEC2 scale_v = FETCH_VEC2(scale[weight_in_row / group_size * n + col_ind]);
  uint32_t zero_v = qzeros==nullptr?0x88888888:qzeros[weight_in_row / group_size * (n / compress_group_size) + (col_ind) / compress_group_size];
  int zero_ind = col_ind % compress_group_size;
  uint8_t zv1 = (zero_v >> (zero_ind * WBITS)) & max_num_in_bits;
  uint8_t zv2 = (zero_v >> (zero_ind * WBITS + WBITS)) & max_num_in_bits;
  VEC2 scale_zeros = __hmul2((Short22Vec2<scalar_t>(zv1 + add_zero_bias, zv2 + add_zero_bias)),
                             scale_v);
  VEC2* out_h2 = reinterpret_cast<VEC2*>(out);

  uint2 weight_int2 = FETCH_UINT2(qweight[tid * 2]);
  uint32_t weight_v1 = weight_int2.x;
  uint32_t weight_v2 = weight_int2.y;
  // decompress weights
  int remains = in_features - weight_in_row;
  if (remains >= compress_group_size) {
#pragma unroll
    for (int i = 0; i < compress_group_size; i++) {
      uint8_t wv1 = (weight_v1 >> (i * WBITS)) & max_num_in_bits;
      uint8_t wv2 = (weight_v2 >> (i * WBITS)) & max_num_in_bits;
      VEC2 wv = Short22Vec2<scalar_t>(wv1, wv2);
      out_h2[((weight_in_row + i) * n + col_ind) / 2] = __hfma2(wv, scale_v, -scale_zeros);
    }
  } else {
    for (int i = 0; i < remains; i++) {
      uint8_t wv1 = (weight_v1 >> (i * WBITS)) & max_num_in_bits;
      uint8_t wv2 = (weight_v2 >> (i * WBITS)) & max_num_in_bits;
      VEC2 wv = Short22Vec2<scalar_t>(wv1, wv2);
      out_h2[((weight_in_row + i) * n + col_ind) / 2] = __hfma2(wv, scale_v, -scale_zeros);
    }
  }
}


template <typename scalar_t, int WBITS>
__device__ __forceinline__ uchar2 iterator_qweight_v2(const scalar_t* ptr, int idx) {
  int start_bits = idx * WBITS;
  int first = start_bits / 32;
  int end_bits = (start_bits + WBITS);
  int second = end_bits / 32;
  start_bits = start_bits % 32;
  end_bits = end_bits % 32;
  uchar2 res;
  if (first == second) {
    res.x = (ptr[first].x >> (start_bits)) & ((1 << WBITS) - 1);
    res.y = (ptr[first].y >> (start_bits)) & ((1 << WBITS) - 1);
    return res;
  } else {
    res.x = (ptr[first].x >> (start_bits));
    res.y = (ptr[first].y >> (start_bits));

    res.x |= ((ptr[second].x) & ((1 << (end_bits)) - 1))<< (32-start_bits);
    res.y |= ((ptr[second].y) & ((1 << (end_bits)) - 1))<< (32-start_bits);
    return res;
  }
}

template <typename scalar_t, int WBITS>
__global__ void DequantizeAndUnpackWeight3567_v2(scalar_t* out, const uint32_t* qweight, const scalar_t* scale, const uint32_t* qzeros, int group_size, const int in_features, const int row_n, uint8_t add_zero_bias) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  const int qweight_rows = (in_features * WBITS + 31) / 32;
  __shared__ uint2 qweight_shared[WBITS * kBlockSize];
  const int half_n = row_n / 2;
  using VEC2 = typename TYPE_VEC2<scalar_t>::Type;

  const int group_row_n = half_n * (WBITS==6?3:WBITS);
  int total_qw = qweight_rows * half_n;

  uint2* qweight_thread = qweight_shared + WBITS * threadIdx.x;

  int qweight_start = tid / half_n * group_row_n + tid % half_n;
  const uint2* qweigh2 = (const uint2*)qweight;
#pragma unroll
  for (int j = 0; j < WBITS; j++) {
    int ind = qweight_start + half_n * j;
    qweight_thread[j] = ind < total_qw ? (qweigh2[ind]) : uint2();
  }
  
  const int max_num_in_bits = (1 << WBITS) - 1;
  const int col_ind = (tid % half_n);
  const int compress_group_size = 32;
  const int fp16_weight_in_row = tid / half_n * compress_group_size;
  VEC2 scale_v[4];
  const int scale_zero_from = fp16_weight_in_row / group_size;
  const int scale_zero_to = (fp16_weight_in_row + compress_group_size) / group_size;

  // decompress scales
  const VEC2 *scale2 = reinterpret_cast<const VEC2*>(scale);
  for (int i = 0, scale_zero_from_i = scale_zero_from; scale_zero_from_i <= scale_zero_to; scale_zero_from_i++, i++) {
    scale_v[i] = (scale2[scale_zero_from_i * half_n + col_ind]);
  }

  // decompress qzeros
  uchar2 zv1[4];
  int half_col_ind = col_ind * 2;
  const int zero_col_from = half_col_ind * WBITS / 32;
  const int zero_col_to = ((half_col_ind + 1) * WBITS - 1) / 32;
  const int zero_col_to_2 = ((half_col_ind + 2) * WBITS - 1) / 32;
  const int qzero_width = (row_n * WBITS + 32 - 1) / 32;
  for (int i = 0, scale_zero_from_i = scale_zero_from; scale_zero_from_i <= scale_zero_to; scale_zero_from_i++, i++) {
    uint32_t zero_v = qzeros==nullptr?0x88888888:qzeros[scale_zero_from_i * qzero_width + zero_col_from];
    const int zero_bits_last = (((half_col_ind)*WBITS) % 32);
    zv1[i].x = (zero_v >> zero_bits_last) & max_num_in_bits;
    if (zero_col_from != zero_col_to) {
      const int zero_bits_first = ((half_col_ind + 1) * WBITS) % 32;
      uint32_t zero_v1 = qzeros==nullptr?0x88888888:qzeros[scale_zero_from * qzero_width + zero_col_to];
      zv1[i].x |= (zero_v1 & ((1 << zero_bits_first) - 1)) << (32-zero_bits_last);

      zv1[i].y = (zero_v1 >> zero_bits_first) & max_num_in_bits;
    } else {
      zv1[i].y = (zero_v >> (zero_bits_last+WBITS)) & max_num_in_bits;
    }

    if (zero_col_to != zero_col_to_2) {
      const int zero_bits_first = ((half_col_ind + 2) * WBITS) % 32;
      uint32_t zero_v1 = qzeros==nullptr?0x88888888:qzeros[scale_zero_from * qzero_width + zero_col_to_2];
      zv1[i].y |= (zero_v1 & ((1 << zero_bits_first) - 1)) << (32 - zero_bits_last - WBITS);
    }
  }

  VEC2 scale_zeros[4];
  for (int i = 0; i <= scale_zero_to - scale_zero_from; i++) {
    scale_zeros[i] = __hmul2(__halves2half2(__ushort2half_rn(zv1[i].x+add_zero_bias), __ushort2half_rn(zv1[i].y+add_zero_bias)), scale_v[i]);
  }
  VEC2 scale_2 =  scale_v[0];
  VEC2 scale_zeros_2 = scale_zeros[0];

  const int out_offset = ((fp16_weight_in_row)*half_n + col_ind);
  VEC2* out_h2 = reinterpret_cast<VEC2*>(out);
  // decompress weights
  int remains = in_features - fp16_weight_in_row;
  if (remains >= compress_group_size) {
#pragma unroll
  for (int i = 0; i < compress_group_size / 2; i++) {
    uchar2 wv1= iterator_qweight_v2<uint2, WBITS>(qweight_thread, i);
    uchar2 wv2 = iterator_qweight_v2<uint2, WBITS>(qweight_thread, 16 + i);

    VEC2 wv = __halves2half2(__ushort2half_rn(wv1.x), __ushort2half_rn(wv1.y));
    if (group_size < 32) {
      VEC2 scale_2 = scale_v[i / group_size];
      VEC2 scale_zeros_2 = scale_zeros[i / group_size];
    }
    VEC2 res = __hfma2(wv, scale_2, -scale_zeros_2);
    out_h2[out_offset + i * half_n] = res;

    wv = __halves2half2(__ushort2half_rn(wv2.x), __ushort2half_rn(wv2.y));
    if (group_size < 32) {
      VEC2 scale_2 = scale_v[(i + 16) / group_size];
      VEC2 scale_zeros_2 = scale_zeros[(i + 16) / group_size];
    }
    res = __hfma2(wv, scale_2, -scale_zeros_2);
    out_h2[(out_offset + (i + 16) * half_n)] = res;
  }
  } else {
    // decompress weights
    for (int i = 0; i < remains; i++) {
      uchar2 wv1 = iterator_qweight_v2<uint2, WBITS>(qweight_thread, i);

      VEC2 wv = __halves2half2(__ushort2half_rn(wv1.x), __ushort2half_rn(wv1.y));
      if (group_size < 32) {
        scale_2 = scale_v[i / group_size];
        scale_zeros_2 = scale_zeros[i / group_size];
      }
      VEC2 res = __hfma2(wv, scale_2, -scale_zeros_2);
      out_h2[out_offset + i * half_n] = res;
    }
  }
}

}  // namespace cuda_quant
#ifndef assert
#define assert(x) \
  if (!(x)) {     \
    abort();      \
  }
#endif

template <typename scalar_t>
void lauch_dq_general_g(scalar_t* b_fp16, int32_t* qweight_i32_i, scalar_t* scale_fp16, 
                  int32_t* qzeros_i32_i, int32_t *g_dix, int bits,
                  int groupsize, uint32_t mat_k, uint32_t mat_n, uint8_t add_zero_bias=0) {
  if constexpr (std::is_same<scalar_t, double>::value) {
    return;
  } else if constexpr (std::is_same<scalar_t, float>::value) {
    return;
  }
  const uint32_t conpress_ratio = 32 / bits;
  dim3 gridDim = {mat_k*mat_n / kBlockSize};
  dim3 blockDim = {kBlockSize};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  uint32_t *qweight_i32 = reinterpret_cast<uint32_t *>(qweight_i32_i);
  uint32_t *qzeros_i32 = reinterpret_cast<uint32_t *>(qzeros_i32_i);
  using cuda_quant::DequantizeAndUnpackWeight248_g;
  using cuda_quant::DequantizeAndUnpackWeight357_g;
  #define CASE_EVEN(WBITS) \
    case WBITS:       \
      DequantizeAndUnpackWeight248_g<scalar_t, WBITS> \
          <<<gridDim, blockDim, 0, stream>>>( \
              (scalar_t *)b_fp16, qweight_i32, (scalar_t *)scale_fp16, qzeros_i32, g_dix, groupsize, mat_k, mat_n, add_zero_bias); \
      break;
  #define CASE_ODD(WBITS) \
    case WBITS:       \
      DequantizeAndUnpackWeight357_g<scalar_t, WBITS> \
          <<<gridDim, blockDim, 0, stream>>>( \
              (scalar_t *)b_fp16, qweight_i32, (scalar_t *)scale_fp16, qzeros_i32, g_dix, groupsize, mat_k, mat_n, add_zero_bias); \
      break;
  switch (bits) {
    CASE_EVEN(2);
    CASE_EVEN(4);
    CASE_EVEN(8);
    CASE_ODD(3);
    CASE_ODD(5);
    CASE_ODD(6);
    CASE_ODD(7);
  default:
    printf("error bits\n");
    assert(false);
  }
}

template <typename scalar_t>
void lauch_dq_248(scalar_t* b_fp16, int32_t* qweight_i32_i, scalar_t* scale_fp16, 
                  int32_t* qzeros_i32_i, int bits,
                  int groupsize, uint32_t mat_k, uint32_t mat_n, uint8_t add_zero_bias=0) {
  if constexpr (std::is_same<scalar_t, double>::value) {
    return;
  } else if constexpr (std::is_same<scalar_t, float>::value) {
    return;
  } 
  const uint32_t conpress_ratio = 32 / bits;
  dim3 gridDim = {(mat_n / 2 * ((mat_k + conpress_ratio-1) / conpress_ratio) + kBlockSize - 1) / kBlockSize};
  dim3 blockDim = {kBlockSize};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  uint32_t* qweight_i32 = reinterpret_cast<uint32_t*>(qweight_i32_i);
  uint32_t* qzeros_i32 = reinterpret_cast<uint32_t*>(qzeros_i32_i);
  using cuda_quant::DequantizeAndUnpackWeight248;
  switch(bits) {
    case 2:
      DequantizeAndUnpackWeight248<scalar_t, 2><<<gridDim, blockDim, 0, stream>>>((scalar_t*)b_fp16, qweight_i32, (scalar_t*)scale_fp16, qzeros_i32, groupsize, mat_k, mat_n, add_zero_bias);
      break;
  case 4:
      DequantizeAndUnpackWeight248<scalar_t, 4><<<gridDim, blockDim, 0, stream>>>((scalar_t*)b_fp16, qweight_i32, (scalar_t*)scale_fp16, qzeros_i32, groupsize, mat_k, mat_n, add_zero_bias);
      break;
  case 8:
      DequantizeAndUnpackWeight248<scalar_t, 8><<<gridDim, blockDim, 0, stream>>>((scalar_t*)b_fp16, qweight_i32, (scalar_t*)scale_fp16, qzeros_i32, groupsize, mat_k, mat_n, add_zero_bias);
      break;
  default:
  printf("error bits\n");
      assert(false);
  }
}

template <typename scalar_t>
void lauch_dq_3567(scalar_t* b_fp16, int32_t* qweight_i32_i, scalar_t* scale_fp16, int32_t* qzeros_i32_i, int bits, int groupsize,
                   uint32_t mat_k, uint32_t mat_n, uint8_t add_zero_bias=0) {
  if constexpr (std::is_same<scalar_t, double>::value) {
  return;
  } else if constexpr (std::is_same<scalar_t, float>::value) {
  return;
  }
  const uint32_t conpress_ratio = 32;
  dim3 gridDim = {static_cast<unsigned int>((mat_n / 2 * (mat_k + conpress_ratio - 1) / conpress_ratio + kBlockSize - 1) / kBlockSize)};
  dim3 blockDim = {kBlockSize};
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  uint32_t* qweight_i32 = reinterpret_cast<uint32_t*>(qweight_i32_i);
  uint32_t* qzeros_i32 = reinterpret_cast<uint32_t*>(qzeros_i32_i);
  using cuda_quant::DequantizeAndUnpackWeight3567_v2;
  switch (bits) {
  case 3:
      DequantizeAndUnpackWeight3567_v2<scalar_t, 3><<<gridDim, blockDim, 0, stream>>>((scalar_t*)b_fp16, qweight_i32, (scalar_t*)scale_fp16, qzeros_i32, groupsize, mat_k, mat_n, add_zero_bias);
      break;
  case 5:
      DequantizeAndUnpackWeight3567_v2<scalar_t, 5><<<gridDim, blockDim, 0, stream>>>((scalar_t*)b_fp16, qweight_i32, (scalar_t*)scale_fp16, qzeros_i32, groupsize, mat_k, mat_n, add_zero_bias);
      break;
  case 6:
      DequantizeAndUnpackWeight3567_v2<scalar_t, 6><<<gridDim, blockDim, 0, stream>>>((scalar_t*)b_fp16, qweight_i32, (scalar_t*)scale_fp16, qzeros_i32, groupsize, mat_k, mat_n, add_zero_bias);
      break;
  case 7:
      DequantizeAndUnpackWeight3567_v2<scalar_t, 7><<<gridDim, blockDim, 0, stream>>>((scalar_t*)b_fp16, qweight_i32, (scalar_t*)scale_fp16, qzeros_i32, groupsize, mat_k, mat_n, add_zero_bias);
      break;
  default:
  printf("error bits\n");
      assert(false);
  }
}


void lauch_deqantize_cuda_pt_kernel(torch::Tensor& b_fp16, const torch::Tensor& qweight_i32, const torch::Tensor& scale_fp16, 
                                    const torch::Tensor& qzeros_i32, c10::optional<torch::Tensor> g_idx, 
                                    int bits, int groupsize, uint32_t mat_k, uint32_t mat_n, uint8_t add_zero_bias) {

  using scalar_t_map = half;
  if (g_idx.has_value()) {
    lauch_dq_general_g<scalar_t_map>(
        b_fp16.data_ptr<scalar_t_map>(), qweight_i32.data_ptr<int32_t>(),
        scale_fp16.data_ptr<scalar_t_map>(), qzeros_i32.data_ptr<int32_t>(),g_idx.value().data_ptr<int32_t>(), 
        bits, groupsize, mat_k, mat_n, add_zero_bias);

  } else if (bits == 2 || bits == 4 || bits == 8) {
    lauch_dq_248<scalar_t_map>(
        b_fp16.data_ptr<scalar_t_map>(), qweight_i32.data_ptr<int32_t>(),
        scale_fp16.data_ptr<scalar_t_map>(), qzeros_i32.data_ptr<int32_t>(),
        bits, groupsize, mat_k, mat_n, add_zero_bias);
  } else {
    lauch_dq_3567<scalar_t_map>(
        b_fp16.data_ptr<scalar_t_map>(), qweight_i32.data_ptr<int32_t>(),
        scale_fp16.data_ptr<scalar_t_map>(), qzeros_i32.data_ptr<int32_t>(),
        bits, groupsize, mat_k, mat_n, add_zero_bias);
  }
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
}