#ifndef MLHPC_INFERENCE_ONNX_HPP
#define MLHPC_INFERENCE_ONNX_HPP

#include "mlhpc/inference.hpp"

#include <memory>
#include <string>

namespace mlhpc
{
std::unique_ptr<InferenceBackend>
make_onnx_backend (const std::string &model_path);
}

#endif
