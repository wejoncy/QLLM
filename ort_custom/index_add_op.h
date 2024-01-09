#pragma once
#include "cuda_context.h"
#include "index_add_op.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

using namespace Ort::Custom;

namespace Cuda {
struct PTExtensionParam {
  int64_t num_inputs;
  int64_t num_outputs;
  std::string func_name;
  std::unordered_map<std::string, std::string> attr_dict_;
};

struct TorchExtensionKernel {
  const OrtApi &_api;
  const OrtKernelInfo *_info;
  PTExtensionParam ext_param_;

  TorchExtensionKernel(const OrtApi &api, const OrtKernelInfo *info);

  void Compute(OrtKernelContext *context);
};

// legacy custom op registration
struct TorchExtensionOp
    : Ort::CustomOpBase<TorchExtensionOp, TorchExtensionKernel> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
    return std::make_unique<TorchExtensionKernel>(api, info).release();
  };
  const char *GetName() const { return "IndexAdd"; };
  const char *GetExecutionProviderType() const {
    return "CUDAExecutionProvider";
  };
  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };
  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  }
  constexpr bool GetVariadicInputHomogeneity() const noexcept {
    return false; // heterogenous
  }

  //constexpr bool GetVariadicOutputHomogeneity() const noexcept {
  //  return false; // heterogeneous
  //}
  constexpr OrtCustomOpInputOutputCharacteristic
  GetInputCharacteristic(size_t /* index */) const noexcept {
    return INPUT_OUTPUT_VARIADIC;
  }

  constexpr OrtCustomOpInputOutputCharacteristic
  GetOutputCharacteristic(size_t /* index */) const noexcept {
    return INPUT_OUTPUT_REQUIRED;
  }
};

} // namespace Cuda
