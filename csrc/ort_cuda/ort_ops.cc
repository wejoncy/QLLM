#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace onnxruntime_gptq {
void lauch_deqantize_cuda_pt_kernel(torch::Tensor& b_fp16, const torch::Tensor& qweight_i32,
                                    const torch::Tensor& scale_fp16, const torch::Tensor& qzeros_i32,c10::optional<torch::Tensor> g_idx,
                                    int bits, int groupsize, uint32_t mat_k, uint32_t mat_n, uint8_t add_zero_bias);


void lauch_Gemv_kernel(torch::Tensor &out_fp16, const torch::Tensor &a_fp16,
                       const torch::Tensor &qweight_i32,
                       const torch::Tensor &scale_fp16,
                       const torch::Tensor &qzeros_i32, int bits, int groupsize,
                       uint32_t mat_m, uint32_t mat_k, uint32_t mat_n,
                       uint8_t add_zero_bias);
void Launch_gemv_g(const torch::Tensor &input, const torch::Tensor &qweight,
                   torch::Tensor &output, const torch::Tensor &scales,
                   const torch::Tensor &qzeros, const torch::Tensor &g_idx,
                   int bits);
} // namespace onnxruntime_gptq

torch::Tensor dequant_any_bit(const torch::Tensor &qweight,
                              const torch::Tensor &scales,
                              const torch::Tensor &qzeros,
                              c10::optional<torch::Tensor> g_idx, int groupsize,
                              int bits, int in_features,
                              uint8_t add_zero_bias) {
  CHECK_INPUT(qweight);
  CHECK_INPUT(scales);
  CHECK_INPUT(qzeros);
  TORCH_CHECK(qweight.dim() == 2, "qweight must be 2-dimensional");
  TORCH_CHECK(groupsize >= 16, "groupsize must be >= 16");
  TORCH_CHECK(bits >= 1 && bits <= 8, "bits must be >= 1 and <= 8");
  TORCH_CHECK((in_features * bits + 31) / 32 == qweight.size(0), "in_features must be >= 1");
  TORCH_CHECK(qweight.device().index() == scales.device().index() ||
                  qweight.device().index() == qzeros.device().index(),
              "input and weight/qzeros must be on the same device");

  at::cuda::OptionalCUDAGuard guard(qweight.device());
  auto f16_scale = scales;
  auto ori_dtype = scales.scalar_type();
  if (ori_dtype == torch::kBFloat16) {
    f16_scale = scales.to(torch::kFloat16);
  }
  at::Tensor output = at::zeros({in_features, qweight.size(1)}, f16_scale.options());
  onnxruntime_gptq::lauch_deqantize_cuda_pt_kernel(
      output, qweight, f16_scale, qzeros, g_idx, bits, groupsize, in_features,
      qweight.size(1), add_zero_bias);
  if (ori_dtype == torch::kBFloat16) {
    output = output.to(torch::kBFloat16);
  }
  return output;
}

torch::Tensor op_gemv(const torch::Tensor &input_a,
                      const torch::Tensor &qweight, const torch::Tensor &scales,
                      const torch::Tensor &qzeros,
                      const c10::optional<torch::Tensor> g_idx, int groupsize,
                      int bits, int in_features, uint8_t add_zero_bias) {
  CHECK_INPUT(input_a);
  CHECK_INPUT(qweight);
  CHECK_INPUT(scales);
  CHECK_INPUT(qzeros);
  TORCH_CHECK(qweight.dim() == 2, "qweight must be 2-dimensional");
  TORCH_CHECK(groupsize >= 16, "groupsize must be >= 16");
  TORCH_CHECK(bits >= 1 && bits <= 8, "bits must be >= 1 and <= 8");
  TORCH_CHECK((in_features * bits + 31) / 32 == qweight.size(0), "in_features must be >= 1");
  TORCH_CHECK(qweight.device().index() == input_a.device().index(), "input and weight must be on the same device");
  at::cuda::OptionalCUDAGuard guard(qweight.device());
  std::vector<int64_t> outputshape ={input_a.size(0), qweight.size(1)};
  uint32_t mat_m = input_a.size(0);
  if (input_a.dim() > 2) {
    outputshape.insert(outputshape.begin()+1, input_a.size(1));
    mat_m *= input_a.size(1);
  }
  auto f16_scale = scales;
  auto ori_dtype = scales.scalar_type();
  if (ori_dtype == torch::kBFloat16) {
    f16_scale = scales.to(torch::kFloat16);
  }

  at::Tensor output = at::zeros(outputshape, f16_scale.options());
  if (g_idx.has_value()) {
    onnxruntime_gptq::Launch_gemv_g(input_a, qweight, output, f16_scale, qzeros,
                                    g_idx.value(), bits);

  }else{
    onnxruntime_gptq::lauch_Gemv_kernel(output, input_a, qweight, f16_scale, qzeros, bits, groupsize, mat_m, in_features, qweight.size(1), add_zero_bias);
  }

  if (ori_dtype == torch::kBFloat16) {
    output = output.to(torch::kBFloat16);
  }
  return output;
}

namespace onnxruntime {
namespace contrib {
namespace cuda {
template <class T, class ZeroT>
int Dequantize4Bits(T *output, const uint8_t *quant_data, const T *scales_data,
                    const ZeroT *zero_points, const int32_t *reorder_idx, int k,
                    int n, int block_size, cudaStream_t stream);

} // namespace cuda
} // namespace contrib
} // namespace onnxruntime

#define QLLM_DISPATCH_CASE_FLOATING_TYPES(...)                                 \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)
  //  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \

#define QLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  AT_DISPATCH_SWITCH(TYPE, NAME, QLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

torch::Tensor Dequantize4Bits(const torch::Tensor &qweight,
                              const torch::Tensor &scales,
                              const torch::Tensor &qzeros,
                              const c10::optional<torch::Tensor> &g_idx,
                              int block_size, int in_features,
                              int out_features) {
  CHECK_INPUT(qweight);
  CHECK_INPUT(scales);
  CHECK_INPUT(qzeros);
  if (g_idx.has_value()) {
    CHECK_INPUT(g_idx.value());
  }
  at::cuda::OptionalCUDAGuard guard(qweight.device());

  torch::Tensor out = torch::empty({out_features, in_features}, scales.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  QLLM_DISPATCH_FLOATING_TYPES(out.scalar_type(), "Dequantize4Bits", [&] {
    if (qzeros.scalar_type() == torch::kFloat16){
      onnxruntime::contrib::cuda::Dequantize4Bits<scalar_t, scalar_t>(
          out.data_ptr<scalar_t>(), qweight.data_ptr<uint8_t>(),
          scales.data_ptr<scalar_t>(), qzeros.data_ptr<scalar_t>(),
          nullptr, in_features, out_features, block_size,
          stream);
    } else {
      onnxruntime::contrib::cuda::Dequantize4Bits<scalar_t, uint8_t>(
          out.data_ptr<scalar_t>(), qweight.data_ptr<uint8_t>(),
          scales.data_ptr<scalar_t>(), qzeros.data_ptr<uint8_t>(),
          g_idx.has_value()?g_idx->data_ptr<int32_t>():nullptr, in_features, out_features, block_size,
          stream);
    }
  });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("Dequantize4Bits", &Dequantize4Bits, "Dequantize4Bits.");
  m.def("dequant", &dequant_any_bit, "dequantize qweight to fp16, \nfunction type: const torch::Tensor& qweight, "
        "const torch::Tensor& scales, const torch::Tensor& qzeros, tensor g_idx, int groupsize, int bits, int in_features");
  m.def("gemv", &op_gemv, "gemv, \nfunction type: const torch::Tensor& input_a, const torch::Tensor& qweight, "
        "const torch::Tensor& scales, const torch::Tensor& qzeros, int groupsize, int bits, int in_features");
}
