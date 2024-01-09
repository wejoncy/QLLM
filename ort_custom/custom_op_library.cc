#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>
#include <system_error>

//#include "core/common/common.h"
//#include "core/framework/ortdevice.h"
//#include "core/framework/ortmemoryinfo.h"
#include "cuda_ops.h"
#include "onnxruntime_lite_custom_op.h"

#define ORT_TRY try
#define ORT_CATCH(x) catch (x)
#define ORT_RETHROW throw;
#define ORT_HANDLE_EXCEPTION(func) func()

static const char* c_OpDomain = "test.customop";

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain &&domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);
  OrtStatus* result = nullptr;
  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};
    Cuda::RegisterOps(domain);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);

    AddOrtCustomOpDomainToContainer(std::move(domain));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      Ort::Status status{e};
      result = status.release();
    });
  }
  return result;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
  return RegisterCustomOps(options, api);
}
