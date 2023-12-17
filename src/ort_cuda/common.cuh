#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <torch/extension.h>

//  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)
#define XBITOPS_DISPATCH_CASE_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define XBITOPS_DISPATCH_CASE_FLOATING_TYPES_HALF(...)      \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define XBITOPS_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                             \
      TYPE, NAME, XBITOPS_DISPATCH_CASE_FLOATING_TYPES_HALF(__VA_ARGS__))

#define XBITOPS_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                             \
      TYPE, NAME, XBITOPS_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))


#if __CUDA_ARCH__ >= 800
#define XBITOPS_DISPATCH_TYPES XBITOPS_DISPATCH_FLOATING_TYPES
#else
#define XBITOPS_DISPATCH_TYPES XBITOPS_DISPATCH_FLOATING_TYPES_AND_HALF
#endif


#if __CUDA_ARCH__ >= 800
#define __has_bfloat16 1
#else
#define __has_bfloat16 0
#endif

namespace onnxruntime_gptq {
namespace cuda_quant {
template <typename T>
struct TYPE_VEC2 {
};

template <>
struct TYPE_VEC2<half> {
  using Type = half2;
};

template <>
struct TYPE_VEC2<at::Half> {
  using Type = __half2;
};

template <>
struct TYPE_VEC2<float> {
  using Type = float2;
};

template <>
struct TYPE_VEC2<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

template <>
struct TYPE_VEC2<c10::BFloat16> {
  using Type = __nv_bfloat162;
};

template <typename toT>
__device__ toT ConvertFromShort(const short a, toT v={}) {
  if constexpr (std::is_same<toT, half>::value) {
    return __short2half_rn(a);
  } else if constexpr (std::is_same<toT, at::Half>::value) {
    return __short2half_rn(a);
  }
#if __has_bfloat16
  if constexpr (std::is_same<toT, __nv_bfloat16>::value) {
    return __short2bfloat16_rn(a);
  }
#endif
  else {
    //static_assert(false, "Not supported type");
    return __short2half_rn(a);
  }
}

template <typename toT>
__device__ toT ConvertFromInt(const int a) {
  if constexpr (std::is_same<toT, half>::value ||
                std::is_same<toT, at::Half>::value ||
                std::is_same<toT, c10::Half>::value) {
    return __int2half_rn(a);
  } else if constexpr (std::is_same<toT, at::Half>::value) {
    return __int2half_rn(a);
  }
#if __has_bfloat16
  if constexpr (std::is_same<toT, __nv_bfloat16>::value) {
    return __int2bfloat16_rn(a);
  }
#endif
  else {
    //static_assert(false, "Not supported type");
    return __int2half_rn(a);
  }
}

template <typename toT, int type_rn=1>
__device__ toT ConvertFromFloat(const float a) {
  if constexpr (std::is_same<toT, half>::value) {
    return __float2half_rn(a);
  } else if constexpr (std::is_same<toT, at::Half>::value) {
    return __float2half_rn(a);
  }
#if __has_bfloat16
  if constexpr (std::is_same<toT, __nv_bfloat16>::value) {
    return __float2bfloat16_rn(a);
  }
#endif
  else {
    //static_assert(false, "Not supported type");
    return __float2half_rn(a);
  }
}

template <typename fromT>
__device__ fromT ConvertToFloat(const float a) {
  if constexpr (std::is_same<fromT, half>::value) {
    return __half2float(a);
  } else if constexpr (std::is_same<fromT, at::Half>::value) {
    return __half2float(a);
  }
#if __has_bfloat16
  if constexpr (std::is_same<fromT, __nv_bfloat16>::value) {
    return __bfloat162float(a);
  }
#endif
  else {
    return __half2float(a);
    //static_assert(false, "Not supported type");
  }
}

template <typename T>
__device__ auto MakeVec2(T a, T b) {
  if constexpr (std::is_same<T, half>::value || 
                std::is_same<T, __half>::value || 
                std::is_same<T, at::Half>::value || 
                std::is_same<T, c10::Half>::value) {
    return __halves2half2(a, b);
  }
#if __has_bfloat16
  if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    return __halves2bfloat162(a, b);
  }
#endif
  else {
    return __halves2half2(a, b);
    //static_assert(false, "Not supported type");
  }
}

template <typename T>
__device__ auto Short22Vec2(const short a, const short b) {
  return MakeVec2<T>(ConvertFromShort<T>(a), ConvertFromShort<T>(b));
}

template <typename T>
__device__ auto Int22Vec2(const int a, const int b) {
  return MakeVec2<T>(ConvertFromInt<T>(a), ConvertFromInt<T>(b));
}

template <typename T>
__device__ auto Element2Vec2(const T a) {
  if constexpr (std::is_same<T, half>::value ||
                std::is_same<T, at::Half>::value ||
                std::is_same<T, c10::Half>::value) {
    return __half2half2(a);
  }
#if __has_bfloat16
  if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    return __bfloat162bfloat162(a);
  }
#endif
  else {
    // static_assert(false, "Not supported type");
  }
}
}  // namespace cuda_quant
}

template <typename scalar_t>
struct C10Type2Type {
};

template <>
struct C10Type2Type<at::Half> {
  using Type = half;
};

template <>
struct C10Type2Type<at::BFloat16> {
  using Type = __nv_bfloat16;
};