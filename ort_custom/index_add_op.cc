// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cstdio>
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "cuda_context.h"
#include "onnxruntime_lite_custom_op.h"

#include "cuda_context.h"
#include "index_add_op.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>


namespace Cuda {


TorchExtensionKernel::TorchExtensionKernel(const OrtApi &api,
                                           const OrtKernelInfo *info)
    : _api(api), _info(info) {
 
};

void TorchExtensionKernel::Compute(OrtKernelContext *context) {
  Ort::Custom::CudaContext cuda_ctx;
  cuda_ctx.Init(*context);
  Ort::KernelContext ctx(context);

  using T = Ort::Float16_t;
  const char *LOCAL_RANK = getenv("LOCAL_RANK");
  int device_id = LOCAL_RANK ? std::stoi(LOCAL_RANK) : 0;
  at::cuda::CUDAStream myStream =
      at::cuda::getStreamFromExternal(cuda_ctx.cuda_stream, device_id);
  at::cuda::setCurrentCUDAStream(myStream);

  // get input tensors
  std::vector<Ort::ConstValue> input_ort_tensors(3);
  std::vector<torch::Tensor> input_torch_tensors(3);
  for (int i = 0; i < 3; ++i) {
    input_ort_tensors[i] = ctx.GetInput(i);
    auto itype =
        input_ort_tensors[i].GetTensorTypeAndShapeInfo().GetElementType();
    auto pt_options = torch::TensorOptions().device(torch::kCUDA, device_id);
    if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      pt_options = pt_options.dtype(torch::kFloat32);
    } else if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      pt_options = pt_options.dtype(torch::kFloat16);
    } else if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
      pt_options = pt_options.dtype(torch::kFloat64);
    } else if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      pt_options = pt_options.dtype(torch::kInt32);
    } else if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      pt_options = pt_options.dtype(torch::kInt64);
    } else if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      pt_options = pt_options.dtype(torch::kUInt8);
    } else if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
      pt_options = pt_options.dtype(torch::kInt8);
    } else if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
      pt_options = pt_options.dtype(torch::kBool);
    } else {
      //CUSTOM_ENFORCE(false, "Unsupported data type: " + std::to_string(itype));
    }
    input_torch_tensors[i] = torch::from_blob(
        const_cast<void *>(input_ort_tensors[i].GetTensorRawData()),
        input_ort_tensors[i].GetTensorTypeAndShapeInfo().GetShape(),
        pt_options);
  }
  auto output_shape = input_torch_tensors[0].sizes().vec();
  auto output_tensor = ctx.GetOutput(0, output_shape);
  auto output_torch =
      torch::from_blob((output_tensor.GetTensorMutableData<T>()),
                       output_tensor.GetTensorTypeAndShapeInfo().GetShape(),
                       torch::TensorOptions()
                           .device(torch::kCUDA, device_id)
                           .dtype(torch::kFloat16));
  //std::cerr << input_torch_tensors[0][0];
  std::cerr << input_torch_tensors[0].sizes()
            << " " <<input_torch_tensors[1].sizes()
            << " " <<input_torch_tensors[2].sizes() << " " << std::endl;
  //input_torch_tensors[0].index_add_(0, input_torch_tensors[1],
  //                                  input_torch_tensors[2]);
  //output_torch.copy_(input_torch_tensors[0]);

  return;
}

void RegisterOps(Ort::CustomOpDomain &domain) {
  static const TorchExtensionOp ptext;
  domain.Add(&ptext);
}

} // namespace Cuda

namespace torch_ext {}