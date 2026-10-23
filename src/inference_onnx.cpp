#include "mlhpc/inference_onnx.hpp"
#include <onnxruntime_cxx_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlhpc
{
namespace
{
class OnnxBackend final : public InferenceBackend
{
public:
  explicit OnnxBackend (const std::string &path)
    : m_env (ORT_LOGGING_LEVEL_WARNING, "mlhpc"),
      m_session (nullptr)
  {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads (1);
    m_session = Ort::Session (m_env, path.c_str (), options);

    if (m_session.GetInputCount () != 1 || m_session.GetOutputCount () != 1)
      throw std::runtime_error ("Expected one model input and one output");

    check_tensor (true, "physical_inputs", 2);
    check_tensor (false, "cooling_rate", 1);
  }

  void batch_inference (const ParticleBatch &batch,
                        Real *__restrict output) const override
  {
    if (batch.rho.size () != batch.n || batch.u.size () != batch.n)
      throw std::invalid_argument ("rho and u must both have batch.n elements");
    if (batch.n == 0)
      return;
    if (output == nullptr)
      throw std::invalid_argument ("Non-empty batch has a null output buffer");
    if (batch.n > std::numeric_limits<std::size_t>::max () / 2
        || batch.n > static_cast<std::size_t> (
             std::numeric_limits<std::int64_t>::max ()))
      throw std::length_error ("Batch exceeds tensor size limits");

    std::vector<float> inputs (2 * batch.n);
    constexpr double limit =
        static_cast<double> (std::numeric_limits<float>::max ());

    for (std::size_t i = 0; i < batch.n; ++i)
      {
        const double rho = static_cast<double> (batch.rho[i]);
        const double u = static_cast<double> (batch.u[i]);
        if (!std::isfinite (rho) || !std::isfinite (u)
            || std::abs (rho) > limit || std::abs (u) > limit)
          throw std::range_error ("rho/u input is not representable as finite float32");
        inputs[2 * i] = static_cast<float> (rho);
        inputs[2 * i + 1] = static_cast<float> (u);
      }

    const auto rows = static_cast<std::int64_t> (batch.n);
    const std::array<std::int64_t, 2> shape { rows, 2 };
    auto memory = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float> (
        memory, inputs.data (), inputs.size (), shape.data (), shape.size ());
    const char *input_names[] = { "physical_inputs" };
    const char *output_names[] = { "cooling_rate" };
    auto result = m_session.Run (Ort::RunOptions { nullptr },
                                 input_names, &tensor, 1, output_names, 1);
    if (!result[0].IsTensor ())
      throw std::runtime_error ("Expected a tensor output");
    const auto info = result[0].GetTensorTypeAndShapeInfo ();
    if (info.GetElementType () != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || info.GetShape () != std::vector<std::int64_t> { rows, 1 })
      throw std::runtime_error ("Expected float32 output with shape [N, 1]");

    const float *values = result[0].GetTensorData<float> ();
    for (std::size_t i = 0; i < batch.n; ++i)
      {
        if (!std::isfinite (values[i]))
          throw std::runtime_error ("ONNX returned a non-finite prediction");
        output[i] = static_cast<Real> (values[i]);
      }
  }

private:
  void check_tensor (bool input, const char *name, std::int64_t width) const
  {
    Ort::AllocatorWithDefaultOptions allocator;
    auto actual_name = input
        ? m_session.GetInputNameAllocated (0, allocator)
        : m_session.GetOutputNameAllocated (0, allocator);
    auto type = input
        ? m_session.GetInputTypeInfo (0)
        : m_session.GetOutputTypeInfo (0);
    if (type.GetONNXType () != ONNX_TYPE_TENSOR)
      throw std::runtime_error ("Expected model tensor");
    const auto info = type.GetTensorTypeAndShapeInfo ();
    if (std::string (actual_name.get ()) != name
        || info.GetElementType () != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || info.GetShape () != std::vector<std::int64_t> { -1, width })
      throw std::runtime_error (
          std::string ("Expected ") + (input ? "input " : "output ")
          + name + " float32 [N, " + std::to_string (width) + "]");
  }

  Ort::Env m_env;
  // Run updates runtime bookkeeping, not the logical model or particle inputs.
  mutable Ort::Session m_session;
};
}

std::unique_ptr<InferenceBackend>
make_onnx_backend (const std::string &model_path)
{
  return std::make_unique<OnnxBackend> (model_path);
}
}
