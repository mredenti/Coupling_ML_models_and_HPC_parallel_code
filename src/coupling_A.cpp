// ---------------------------------------------------------------------
//  src/coupling_A.cpp -- Case A coupling implementation.
//
//  Case A is the "no coupling at all" baseline: every MPI rank is
//  *both* a physics rank and an inference rank, and the cooling-rate
//  evaluation happens locally with no inter-rank traffic.  This is
//  the embarrassingly-parallel reference point that B and C will be
//  compared against.
//
//  Concretely:
//      physics_comm   == MPI_COMM_WORLD (duped, so we own the handle)
//      is_physics     == true on every rank
//      is_inference   == true on every rank
//      evaluate()     -> backend.batch_inference (local call)
//      serve()        -> no-op (returns immediately)
//      shutdown()     -> no-op
//
//  The "interesting" work here is just the timing harness: we record
//  per-rank wall-clock for backend.batch_inference so step 10 of the
//  roadmap (benchmarking) has a baseline to compare A/B/C on equal
//  footing.  All other timing fields (send, wait) stay at zero
//  because there is no MPI traffic for the cooling step.
// ---------------------------------------------------------------------

#include "mlhpc/coupling.hpp"

#include "mlhpc/inference.hpp"
#include "mlhpc/particle_batch.hpp"
#include "mlhpc/types.hpp"

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace mlhpc
{

namespace
{

// ---------------------------------------------------------------------
//  CouplingA -- the actual implementation.  File-static, exposed only
//  through the make_coupling factory; outside callers see it as a
//  Coupling *.
// ---------------------------------------------------------------------
class CouplingA final : public Coupling
{
public:

  // -------------------------------------------------------------------
  //  Constructor.  Dupes `world` into our private communicator so
  //  the rest of the harness has a stable handle even if the user's
  //  world is later freed or replaced.  Collective on `world`.
  // -------------------------------------------------------------------
  explicit
  CouplingA (MPI_Comm   world)
    : m_phys_comm (MPI_COMM_NULL)
  {
    if (MPI_Comm_dup (world, &m_phys_comm) != MPI_SUCCESS)
      {
        std::fprintf (stderr,
                      "CouplingA: MPI_Comm_dup failed; aborting.\n");
        MPI_Abort (world, 10);
      }
  }

  ~CouplingA () override
  {
    if (m_phys_comm != MPI_COMM_NULL)
      {
        MPI_Comm_free (&m_phys_comm);
      }
  }

  // -------------------------------------------------------------------
  //  Trivial accessors.
  // -------------------------------------------------------------------
  MPI_Comm
  physics_comm () const noexcept override
  {
    return m_phys_comm;
  }

  bool
  is_physics () const noexcept override
  {
    return true;
  }

  bool
  is_inference () const noexcept override
  {
    return true;
  }

  // -------------------------------------------------------------------
  //  evaluate_async -- local, already-complete request for case A.
  //
  //  Case A remains the no-transport baseline: the calling rank owns
  //  the particle data and invokes the backend directly.  The function
  //  still returns a CouplingRequest so higher-level benchmark code can
  //  use the same submit/progress/wait lifecycle for all cases.  The
  //  output buffer is filled before return in this first implementation;
  //  a future GPU backend can replace the direct call with a local queue
  //  without changing the MPI topology code.
  // -------------------------------------------------------------------
  CouplingRequest
  evaluate_async (const ParticleBatch       &batch,
                  Real *__restrict           out,
                  const InferenceBackend    *backend) const override
  {
    if (backend == nullptr)
      {
        throw std::logic_error ("CouplingA::evaluate_async requires a local backend");
      }

    const double  t0 = MPI_Wtime ();
    backend->batch_inference (batch, out);
    const double  t1 = MPI_Wtime ();

    m_timings.kernel  += (t1 - t0);
    m_timings.total   += (t1 - t0);
    m_timings.n_calls += 1;

    CouplingRequest  request {};
    request.id = ++m_next_request;
    return request;
  }

  // -------------------------------------------------------------------
  //  progress / test / wait / wait_all -- no-op completion API.
  //
  //  Requests are complete when evaluate_async() returns because case A
  //  does not send work to another rank.  Keeping these methods cheap
  //  preserves the zero-transport baseline while satisfying the common
  //  async interface.
  // -------------------------------------------------------------------
  void
  progress () const override
  {
    // Intentionally empty.
  }

  bool
  test (CouplingRequest /*request*/) const override
  {
    return true;
  }

  void
  wait (CouplingRequest /*request*/) const override
  {
    // Intentionally empty.
  }

  void
  wait_all () const override
  {
    // Intentionally empty.
  }

  // -------------------------------------------------------------------
  //  evaluate -- blocking compatibility wrapper.
  //
  //  The public synchronous path is now expressed in terms of the async
  //  lifecycle.  In case A the request is complete immediately, so the
  //  wrapper adds no meaningful overhead beyond returning a small handle.
  // -------------------------------------------------------------------
  void
  evaluate (const ParticleBatch       &batch,
            Real *__restrict           out,
            const InferenceBackend    *backend) const override
  {
    const CouplingRequest  request = evaluate_async (batch, out, backend);
    wait (request);
  }

  // -------------------------------------------------------------------
  //  serve -- no-op.  The inference work for an A-rank already
  //  happens in-line inside evaluate(); a serve loop here would
  //  block waiting for messages that never come.
  // -------------------------------------------------------------------
  void
  serve (const InferenceBackend       &/*backend*/) const override
  {
    // Intentionally empty.
  }

  // -------------------------------------------------------------------
  //  shutdown -- no-op.  There is no inference pool to drain.
  // -------------------------------------------------------------------
  void
  shutdown () const override
  {
    // Intentionally empty.
  }

  // -------------------------------------------------------------------
  //  timings -- snapshot of the accumulated counters.
  // -------------------------------------------------------------------
  Timings
  timings () const noexcept override
  {
    return m_timings;
  }

private:

         MPI_Comm        m_phys_comm;       // duped from `world` at ctor
  mutable std::size_t    m_next_request { 0 };
  mutable Timings        m_timings      { }; // wall-clock accumulators
};

}  // namespace

// ---------------------------------------------------------------------
//  Factory body for case A.  The B and C TUs provide their own
//  alternative definitions; the Makefile picks exactly one.
// ---------------------------------------------------------------------
std::unique_ptr<Coupling>
make_coupling (MPI_Comm                world,
               const CouplingConfig   &/*cfg*/)
{
  return std::unique_ptr<Coupling> (new CouplingA (world));
}

const char *
compiled_case_label () noexcept
{
  return "A";
}

}  // namespace mlhpc
