#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {
template <class T>
int Dequantize4Bits(T *output, const uint8_t *quant_data,
                       const T *scales_data, const uint8_t *zero_points, int k,
                       int n, int block_size, cudaStream_t stream);

} // namespace cuda
} // namespace contrib
} // namespace onnxruntime

#define QLLM_DISPATCH_CASE_FLOATING_TYPES(...)                                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                         \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

#define QLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  AT_DISPATCH_SWITCH(TYPE, NAME, QLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

torch::Tensor Dequantize4Bits(torch::Tensor &qweight, torch::Tensor &qzeros,
                              torch::Tensor &scales, int block_size,
                              int in_features, int out_features) {
  TORCH_CHECK(qweight.device().type() == torch::kCUDA, "qweight must be a CUDA tensor");
  TORCH_CHECK(qzeros.device().type() == torch::kCUDA, "qzeros must be a CUDA tensor");
  TORCH_CHECK(scales.device().type() == torch::kCUDA, "scales must be a CUDA tensor");

  torch::Tensor out = torch::empty({out_features, in_features}, scales.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  QLLM_DISPATCH_FLOATING_TYPES(out.scalar_type(), "Dequantize4Bits", [&] {
    onnxruntime::contrib::cuda::Dequantize4Bits<scalar_t>(
        out.data_ptr<scalar_t>(), qweight.data_ptr<uint8_t>(),scales.data_ptr<scalar_t>(),
        qzeros.data_ptr<uint8_t>(), in_features, out_features, block_size,
        stream);
  });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("Dequantize4Bits", &Dequantize4Bits, "Dequantize4Bits.");
}
