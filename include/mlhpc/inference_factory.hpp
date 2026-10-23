#ifndef MLHPC_INFERENCE_FACTORY_HPP
#define MLHPC_INFERENCE_FACTORY_HPP

#include "mlhpc/inference.hpp"

#include <memory>
#include <string>

namespace mlhpc
{

std::unique_ptr<InferenceBackend>
make_backend (const std::string &model_path);

const char *
compiled_backend_label () noexcept;

}

#endif