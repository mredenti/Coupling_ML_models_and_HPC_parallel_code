// ---------------------------------------------------------------------
//  tests/test_coupling_A.cpp -- MPI test for the case-A Coupling.
//
//  Runs under `mpirun -np 4`.  Verifies, on every rank:
//
//    1. coupling->physics_comm() has the same size as MPI_COMM_WORLD
//       (case A: every rank is a physics rank).
//    2. is_physics() == true and is_inference() == true everywhere.
//    3. The selected backend produces values bit-equal to the native
//       DummyBackend on the deterministic compatibility-model inputs.
//    4. After one evaluate() call, timings().n_calls == 1 and
//       timings().total >= 0.  send and wait are zero (no MPI traffic
//       inside evaluate() for case A).
//    5. serve() returns immediately (no-op for case A).
//    6. shutdown() is collective and idempotent.
//
//  This test only makes sense when the binary was built with
//  MLHPC_CASE_A=1.  For other case builds the body is skipped with
//  an "OK (skipped)" line so the Makefile can still run the test
//  binary as part of `make test` without conditional gymnastics.
// ---------------------------------------------------------------------

#include "mlhpc/coupling.hpp"
#include "mlhpc/inference.hpp"
#include "mlhpc/inference_factory.hpp"
#include "mlhpc/particle_batch.hpp"
#include "mlhpc/types.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

int errors = 0;
int rank   = 0;
int size   = 1;

void
fail (const char *what)
{
  std::fprintf (stderr, "[rank %d] FAIL %s\n", rank, what);
  ++errors;
}

// [[maybe_unused]] because under CASE != A the test body is excised
// by the preprocessor and the helpers below are never called.
[[maybe_unused]] void
expect (bool cond, const char *what)
{
  if (!cond) fail (what);
}

[[maybe_unused]] void
expect_eq_real (const char     *what,
                mlhpc::Real     got,
                mlhpc::Real     want)
{
  if (!(got == want))
    {
      std::fprintf (stderr,
                    "[rank %d] FAIL %s: got %.9g, want %.9g\n",
                    rank, what,
                    static_cast<double> (got),
                    static_cast<double> (want));
      ++errors;
    }
}

}  // namespace

int
main (int argc, char **argv)
{
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

#if defined(MLHPC_CASE_A) && MLHPC_CASE_A
  std::unique_ptr<mlhpc::Coupling> coupling;
  std::unique_ptr<mlhpc::InferenceBackend> backend;
  try
  {
    if (argc > 2)
      {
        throw std::invalid_argument ("Usage: test_coupling_A [model.onnx]");
      }

    mlhpc::CouplingConfig  cfg;
    coupling = mlhpc::make_coupling (MPI_COMM_WORLD, cfg);
    backend = mlhpc::make_backend (argc == 2 ? argv[1] : "");

    const int local_backend = backend ? 1 : 0;
    int backend_count = 0;
    MPI_Allreduce (&local_backend, &backend_count, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD);
    expect (backend_count == size, "every Case A rank owns a backend");
    if (rank == 0)
      {
        std::printf ("test_coupling_A: backend=%s, instances=%d\n",
                     mlhpc::compiled_backend_label (), backend_count);
      }

    // ---------------- (1) physics_comm size ---------------------------
    int pcomm_size = 0;
    MPI_Comm_size (coupling->physics_comm (), &pcomm_size);
    expect (pcomm_size == size, "physics_comm size == world size");

    // ---------------- (2) roles ---------------------------------------
    expect (coupling->is_physics (),   "is_physics() on case-A rank");
    expect (coupling->is_inference (), "is_inference() on case-A rank");

    // ---------------- (3) bit-equality of evaluate vs direct ----------
    // Deterministic per-rank batch so a multi-rank failure pinpoints
    // the offending rank.  Values are exactly representable in IEEE
    // floating-point so equality is bit-exact regardless of Real width.
    const std::size_t  n = 8;
    mlhpc::ParticleBatch  batch {};
    batch.resize (n, 0);
    batch.ne.clear ();
    batch.nh.clear ();
    batch.Zs.clear ();
    for (std::size_t i = 0; i < n; ++i)
      {
        // i ranges 0..7, rank shifts to make per-rank slices unique.
        batch.rho [i] = static_cast<mlhpc::Real> (1 + rank * 16 + i);
        batch.u   [i] = static_cast<mlhpc::Real>
            (4 + (i * 7 + static_cast<std::size_t> (rank) * 3) % 17);
      }

    std::vector<mlhpc::Real>  out_direct (n);
    std::vector<mlhpc::Real>  out_via    (n);

    const mlhpc::DummyBackend reference;
    reference.batch_inference (batch, out_direct.data ());

    bool rejected_missing_backend = false;
    try
      {
        coupling->evaluate (batch, out_via.data ());
      }
    catch (const std::logic_error &)
      {
        rejected_missing_backend = true;
      }
    expect (rejected_missing_backend, "Case A rejects a missing local backend");

    const mlhpc::CouplingRequest  request =
        coupling->evaluate_async (batch, out_via.data (), *backend);
    expect (coupling->test (request),
            "case-A async request completes immediately");
    coupling->wait (request);

    for (std::size_t i = 0; i < n; ++i)
      {
        char buf [64];
        std::snprintf (buf, sizeof (buf),
                       "out[%zu] coupling == direct", i);
        expect_eq_real (buf, out_via [i], out_direct [i]);
      }

    // ---------------- (4) timing accumulators -------------------------
    const mlhpc::Timings  t = coupling->timings ();
    expect (t.n_calls == 1, "timings.n_calls == 1 after one evaluate()");
    expect (t.total  >= 0.0, "timings.total >= 0");
    expect (t.kernel >= 0.0, "timings.kernel >= 0");
    expect (t.send   == 0.0, "timings.send == 0 for case A");
    expect (t.wait   == 0.0, "timings.wait == 0 for case A");

    out_via.assign (n, static_cast<mlhpc::Real> (-1));
    coupling->evaluate (batch, out_via.data (), backend.get ());
    for (std::size_t i = 0; i < n; ++i)
      {
        expect_eq_real ("blocking Case A output", out_via[i], out_direct[i]);
      }
    expect (coupling->timings ().n_calls == 2,
            "blocking evaluate processes a second local batch");

    // ---------------- (5) serve() is a no-op --------------------------
    // If serve() blocked or aborted, this test would never return.
    coupling->serve (*backend);

    // ---------------- (6) shutdown() is idempotent --------------------
    coupling->shutdown ();
    coupling->shutdown ();

    // ---------------- label sanity ------------------------------------
    expect (std::strcmp (mlhpc::compiled_case_label (), "A") == 0,
            "compiled_case_label() == \"A\"");
  }
  catch (const std::exception &error)
    {
      std::fprintf (stderr, "[rank %d] test_coupling_A: %s\n",
                    rank, error.what ());
      MPI_Abort (MPI_COMM_WORLD, 1);
      return EXIT_FAILURE;
    }
  backend.reset ();
  coupling.reset ();
#else
  if (rank == 0)
    {
      std::printf ("test_coupling_A: OK (skipped -- not MLHPC_CASE_A)\n");
      std::fflush (stdout);
    }
#endif

  // Gather error count to rank 0 so any single-rank failure fails the
  // whole job.  MPI_Reduce on int is portable and avoids depending on
  // every rank printing.
  int total_errors = 0;
  MPI_Reduce (&errors, &total_errors, 1, MPI_INT, MPI_SUM, 0,
              MPI_COMM_WORLD);
  int exit_code = EXIT_SUCCESS;
  if (rank == 0)
    {
      if (total_errors == 0)
        {
          std::printf ("test_coupling_A: OK\n");
        }
      else
        {
          std::fprintf (stderr,
                        "test_coupling_A: %d check(s) failed across all ranks\n",
                        total_errors);
          exit_code = EXIT_FAILURE;
        }
    }
  MPI_Bcast (&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Finalize ();
  return exit_code;
}
