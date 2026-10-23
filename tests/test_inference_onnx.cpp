#include "mlhpc/inference_onnx.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main (int argc, char **argv)
{
  if (argc != 2)
    {
      std::cerr << "Usage: test_inference_onnx rho_u_model.onnx\n";
      return 1;
    }

  try
    {
      const mlhpc::DummyBackend reference;
      auto backend = mlhpc::make_onnx_backend (argv[1]);
      mlhpc::ParticleBatch batch {};
      backend->batch_inference (batch, nullptr);

      for (std::size_t n : { std::size_t { 1 }, std::size_t { 4 },
                              std::size_t { 17 } })
        {
          batch.resize (n, 0);
          for (std::size_t i = 0; i < n; ++i)
            {
              batch.rho[i] = static_cast<mlhpc::Real> (i % 5);
              batch.u[i] = static_cast<mlhpc::Real> ((3 * i + 1) % 7);
            }

          std::vector<mlhpc::Real> expected (n), actual (n);
          reference.batch_inference (batch, expected.data ());
          backend->batch_inference (batch, actual.data ());

          for (std::size_t i = 0; i < n; ++i)
            {
              const double value = static_cast<double> (actual[i]);
              const double want = static_cast<double> (expected[i]);
              const double tolerance =
                  8.0 * static_cast<double> (std::numeric_limits<float>::epsilon ())
                  * std::abs (want);
              if (!std::isfinite (value) || std::abs (value - want) > tolerance)
                throw std::runtime_error ("ONNX output differs from DummyBackend");
            }
        }

      batch.rho.pop_back ();
      std::vector<mlhpc::Real> output (batch.n);
      bool rejected = false;
      try
        {
          backend->batch_inference (batch, output.data ());
        }
      catch (const std::invalid_argument &)
        {
          rejected = true;
        }
      if (!rejected)
        throw std::runtime_error ("Malformed batch was accepted");

      std::cout << "ONNX backend matches DummyBackend; "
                   "batch sizes 0, 1, 4, 17 passed\n";
    }
  catch (const std::exception &error)
    {
      std::cerr << error.what () << '\n';
      return 1;
    }
}
