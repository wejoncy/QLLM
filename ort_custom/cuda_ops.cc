// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include <cstdio>
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "cuda_context.h"
#include "onnxruntime_lite_custom_op.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
template <typename T1, typename T2, typename T3>
void cuda_add(int64_t, T3*, const T1*, const T2*, cudaStream_t compute_stream);

using namespace Ort::Custom;

#define CUSTOM_ENFORCE(cond, msg)  \
  if (!(cond)) {                   \
    throw std::runtime_error(msg); \
  }

namespace Cuda {

void KernelOne(const Ort::Custom::CudaContext &cuda_ctx,
               const Ort::Custom::Tensor<Ort::Float16_t> &X,
               const Ort::Custom::Tensor<int> &Y,
               const Ort::Custom::Tensor<Ort::Float16_t> &Z,
               Ort::Custom::Tensor<Ort::Float16_t> &O) {
  using T = Ort::Float16_t;

  auto input_shape = X.Shape();
  int device_id = 0;
  CUSTOM_ENFORCE(cuda_ctx.cuda_stream,
                               "failed to fetch cuda stream");
  CUSTOM_ENFORCE(cuda_ctx.cudnn_handle, "failed to fetch cudnn handle");
  CUSTOM_ENFORCE(cuda_ctx.cublas_handle, "failed to fetch cublas handle");
  auto O_raw = O.Allocate(input_shape);
  auto input_x = torch::from_blob(const_cast<T *>(X.Data()), input_shape,
                                  torch::TensorOptions()
                                      .device(torch::kCUDA, device_id)
                                      .dtype(torch::kFloat16));
  auto input_y = torch::from_blob(
      const_cast<int *>(Y.Data()), Y.Shape(),
      torch::TensorOptions().device(torch::kCUDA, device_id).dtype(torch::kInt32));
  auto input_z = torch::from_blob(const_cast<T *>(Z.Data()), Z.Shape(),
                                  torch::TensorOptions()
                                      .device(torch::kCUDA, device_id)
                                      .dtype(torch::kFloat16));
  auto output = torch::from_blob(const_cast<T *>(O_raw), input_shape,
                                 torch::TensorOptions()
                                     .device(torch::kCUDA, device_id)
                                     .dtype(torch::kFloat16));
  input_x.index_add_(0, input_y, input_z);
  output.copy_(input_x);
}

void RegisterOps1(Ort::CustomOpDomain &domain) {
  static const std::unique_ptr<OrtLiteCustomOp> c_CustomOpOne{
      Ort::Custom::CreateLiteCustomOp("IndexAdd1", "CUDAExecutionProvider",
                                      KernelOne)};
  domain.Add(c_CustomOpOne.get());
}

}  // namespace Cuda

torch::Tensor lltm_forward(torch::Tensor input) {
  auto X = input;

  return X;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
}