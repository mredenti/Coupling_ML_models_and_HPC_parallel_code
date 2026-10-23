#include "mlhpc/inference_factory.hpp"

#if MLHPC_BACKEND_ONNX
#include "mlhpc/inference_onnx.hpp"
#endif

#include <stdexcept>

namespace mlhpc
{

std::unique_ptr<InferenceBackend>
make_backend (const std::string &model_path)
{
#if MLHPC_BACKEND_ONNX
  if (model_path.empty ())
    throw std::invalid_argument ("ONNX backend requires --model=PATH");

  return make_onnx_backend (model_path);
#else
  if (!model_path.empty ())
    throw std::invalid_argument ("DUMMY backend does not use --model");

  return std::make_unique<DummyBackend> ();
#endif
}

const char *
compiled_backend_label () noexcept
{
#if MLHPC_BACKEND_ONNX
  return "ONNX";
#else
  return "DUMMY";
#endif
}

}