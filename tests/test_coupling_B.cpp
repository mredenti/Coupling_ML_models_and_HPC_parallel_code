// ---------------------------------------------------------------------
//  tests/test_coupling_B.cpp -- MPI test for the case-B Coupling.
//
//  Runs under `mpirun -np 4` (script in Makefile).  Case B is the
//  service topology: server placement is governed by
//  CouplingConfig::ml_server_nodes at runtime.  The test uses the
//  default value (0), which places one ML server rank per shared-memory
//  node (the last rank in that node's MPI_COMM_TYPE_SHARED domain).
//
//  On a single-node workstation all 4 ranks share one node, so the
//  test sees 3 physics ranks and 1 inference rank.  On a multi-node
//  cluster the test self-adapts: each inference rank's expected query
//  count is derived from a test-local MPI_Comm_split_type so we don't
//  hardcode a topology.
//
//  Verifies, on every rank:
//
//    1. is_physics() XOR is_inference() is true.
//    2. is_physics() / is_inference() agree with the test-derived
//       shared-memory topology.
//    3. On physics ranks: coupling->evaluate_async() round-trips a
//       deterministic batch and produces values bit-identical to a
//       direct DummyBackend::batch_inference() call on the same batch.
//       Only inference ranks construct the selected backend.
//    4. On inference ranks: serve() processes exactly
//       (own_node_size - 1) * 6 chunks (two batches, three chunks each because
//       max_batch_particles=3 splits the 8-particle batch into three
//       chunks of size 3, 3, 2).
//    5. shutdown() is idempotent.
//    6. compiled_case_label() == "B".
//
//  Under non-B builds the body is skipped with an "OK (skipped)" line.
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

int  errors = 0;
int  rank   = 0;
int  size   = 1;

void
fail (const char *what)
{
  std::fprintf (stderr, "[rank %d] FAIL %s\n", rank, what);
  ++errors;
}

// [[maybe_unused]] because under CASE != B the test body is excised
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

#if defined(MLHPC_CASE_B) && MLHPC_CASE_B
  std::unique_ptr<mlhpc::Coupling> coupling;
  std::unique_ptr<mlhpc::InferenceBackend> backend;
  try
  {
    if (argc > 2)
      {
        throw std::invalid_argument ("Usage: test_coupling_B [model.onnx]");
      }

    if (size < 2)
      {
        if (rank == 0)
          {
            std::fprintf (stderr,
                          "test_coupling_B: needs at least 2 ranks, got %d\n",
                          size);
          }
        MPI_Finalize ();
        return EXIT_FAILURE;
      }

    // Derive own-node topology independently of the Coupling internals
    // so the test does not rely on any Coupling-private accessor.
    MPI_Comm  test_node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type (MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                         MPI_INFO_NULL, &test_node_comm);
    int  test_node_rank = 0;
    int  test_node_size = 0;
    MPI_Comm_rank (test_node_comm, &test_node_rank);
    MPI_Comm_size (test_node_comm, &test_node_size);
    MPI_Comm_free (&test_node_comm);

    // With ml_server_nodes == 0 (default), the last rank on each node
    // becomes the inference rank for that node.
    const bool  i_am_inference  = (test_node_rank == test_node_size - 1);
    const int   expected        = test_node_size - 1;

    mlhpc::CouplingConfig  cfg;
    cfg.max_batch_particles   = 3;
    cfg.max_inflight_requests = 2;
    coupling = mlhpc::make_coupling (MPI_COMM_WORLD, cfg);

    // ---------------- (1)/(2) role checks ----------------------------
    expect (coupling->is_physics ()   == !i_am_inference,
            "is_physics matches expected role");
    expect (coupling->is_inference () ==  i_am_inference,
            "is_inference matches expected role");

    if (coupling->is_inference ())
      {
        backend = mlhpc::make_backend (argc == 2 ? argv[1] : "");
      }
    expect ((backend != nullptr) == i_am_inference,
            "only inference ranks own a selected backend");

    const int local_counts[2] = { backend ? 1 : 0, i_am_inference ? 1 : 0 };
    int counts[2] = {};
    MPI_Allreduce (local_counts, counts, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    expect (counts[0] == counts[1], "backend count matches inference rank count");
    if (rank == 0)
      {
        std::printf ("test_coupling_B: backend=%s, instances=%d\n",
                     mlhpc::compiled_backend_label (), counts[0]);
      }

    if (i_am_inference)
      {
        // ---------------- (4) inference-side ------------------------
        expect (coupling->physics_comm () == MPI_COMM_NULL,
                "inference rank: physics_comm is MPI_COMM_NULL");

        coupling->serve (*backend);

        const mlhpc::Timings  t = coupling->timings ();
        expect (t.n_calls == static_cast<std::size_t> (expected * 6),
                "inference rank: n_calls == expected physics chunks");
        expect (t.kernel >= 0.0, "inference rank: kernel >= 0");
        expect (t.send   >= 0.0, "inference rank: send   >= 0");
        expect (t.wait   >= 0.0, "inference rank: wait   >= 0");
      }
    else
      {
        // ---------------- physics-side ------------------------------
        int  phys_rank = 0;
        MPI_Comm_rank (coupling->physics_comm (), &phys_rank);

        // Deterministic per-physics-rank batch; values exactly
        // representable in IEEE binary fp so equality is bit-exact.
        const std::size_t  n = 8;
        mlhpc::ParticleBatch  batch {};
        batch.resize (n, 0);
        batch.ne.clear ();
        batch.nh.clear ();
        batch.Zs.clear ();
        for (std::size_t i = 0; i < n; ++i)
          {
            batch.rho [i] = static_cast<mlhpc::Real> (1 + phys_rank * 16 + i);
            batch.u   [i] = static_cast<mlhpc::Real>
                (4 + (i * 7 + static_cast<std::size_t> (phys_rank) * 3) % 17);
          }

        // ---------------- (3) reference (direct) vs B (wire) --------
        std::vector<mlhpc::Real>  out_direct (n);
        const mlhpc::DummyBackend reference;
        reference.batch_inference (batch, out_direct.data ());

        std::vector<mlhpc::Real>  out_via    (n);
        const mlhpc::CouplingRequest  request =
            coupling->evaluate_async (batch, out_via.data ());
        while (!coupling->test (request))
          {
            coupling->progress ();
          }
        coupling->wait (request);

        for (std::size_t i = 0; i < n; ++i)
          {
            char  buf [64];
            std::snprintf (buf, sizeof (buf),
                           "out[%zu] B coupling == direct", i);
            expect_eq_real (buf, out_via [i], out_direct [i]);
          }

        const mlhpc::Timings  t = coupling->timings ();
        expect (t.n_calls == 3,   "physics rank: n_calls == 3 chunks");
        expect (t.send   >= 0.0,  "physics rank: send   >= 0");
        expect (t.wait   >= 0.0,  "physics rank: wait   >= 0");
        expect (t.kernel == 0.0,
                "physics rank: kernel == 0 (no local kernel in case B)");

        out_via.assign (n, static_cast<mlhpc::Real> (-1));
        coupling->evaluate (batch, out_via.data ());
        for (std::size_t i = 0; i < n; ++i)
          {
            expect_eq_real ("blocking Case B output", out_via[i], out_direct[i]);
          }
        expect (coupling->timings ().n_calls == 6,
                "blocking evaluate completes another three remote chunks");
      }

    // ---------------- (5) shutdown is idempotent ---------------------
    coupling->shutdown ();
    coupling->shutdown ();

    // ---------------- (6) compiled case label ------------------------
    expect (std::strcmp (mlhpc::compiled_case_label (), "B") == 0,
            "compiled_case_label() == \"B\"");
  }
  catch (const std::exception &error)
    {
      std::fprintf (stderr, "[rank %d] test_coupling_B: %s\n",
                    rank, error.what ());
      MPI_Abort (MPI_COMM_WORLD, 1);
      return EXIT_FAILURE;
    }
  backend.reset ();
  coupling.reset ();
#else
  if (rank == 0)
    {
      std::printf ("test_coupling_B: OK (skipped -- not MLHPC_CASE_B)\n");
      std::fflush (stdout);
    }
#endif

  int  total_errors = 0;
  MPI_Reduce (&errors, &total_errors, 1, MPI_INT, MPI_SUM, 0,
              MPI_COMM_WORLD);
  int  exit_code = EXIT_SUCCESS;
  if (rank == 0)
    {
      if (total_errors == 0)
        {
          std::printf ("test_coupling_B: OK\n");
        }
      else
        {
          std::fprintf (stderr,
                        "test_coupling_B: %d check(s) failed across all ranks\n",
                        total_errors);
          exit_code = EXIT_FAILURE;
        }
    }
  MPI_Bcast (&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Finalize ();
  return exit_code;
}
