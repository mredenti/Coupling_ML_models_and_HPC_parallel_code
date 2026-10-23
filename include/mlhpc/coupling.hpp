// -*- C++ -*-
// ---------------------------------------------------------------------
//  mlhpc/coupling.hpp -- abstract MPI-coupling interface (cases A/B/C).
//
//  The single most important abstraction in this harness.  The
//  project's central question is *how* MPI tasks should be arranged
//  around the on-line ML inference of the cooling function; the
//  three competing answers are:
//
//      A : every MPI rank runs its own inference.  No inter-rank
//          query/answer traffic for the cooling step; the only MPI
//          work is the snapshot read and the usual physics
//          communication.  Embarrassingly parallel from the
//          coupling's point of view.
//
//      B : physics ranks send particle batches to designated ML server
//          ranks via a shared non-blocking MPI transport and receive
//          cooling-rate answers back.  Server placement is controlled
//          at runtime by --ml-server-nodes=N:
//
//            N == 0 (default): one server rank per shared-memory node.
//            N  > 0          : N whole ML server nodes.
//
//          N == 0 is topology-local (each physics rank talks to its
//          own node's server); N > 0 defines one server node per group
//          of N contiguous nodes, covering the former "Case C" and
//          "Case B with a global pool" scenarios in a single
//          parameterised implementation.
//
//  Selection is done at build time with -DMLHPC_CASE_{A,B}=1 (see
//  the Makefile's CASE variable).  Exactly one of
//  src/coupling_{A,B}.cpp is compiled and linked into the harness;
//  the other is skipped, so a missing implementation is a build
//  error, not a runtime surprise.
//
//  The orchestration code (`src/main.cpp` and friends) speaks only
//  through this interface.  In particular it never touches MPI tags,
//  sub-communicators or wire protocols on its own -- those are
//  private to the case implementations.  This is what makes A/B/C
//  comparable on equal footing, and what lets us drop in a real ML
//  surrogate (or a GPU backend) without changing a single line of
//  the MPI scaffolding.
//
//  Lifetime contract:
//      auto coupling = mlhpc::make_coupling (MPI_COMM_WORLD, cfg);
//      if (coupling->is_physics ())
//        {
//          // open Format2Distributor on coupling->physics_comm (),
//          // build a ParticleBatch, then:
//          coupling->evaluate (batch, out_buffer, backend);
//        }
//      else  // inference rank
//        {
//          coupling->serve (backend);   // blocks until shutdown
//        }
//      coupling->shutdown ();           // collective; no-op for A
//
//  All collective calls below are collective on `MPI_COMM_WORLD`
//  unless the comment says otherwise.
// ---------------------------------------------------------------------

#ifndef MLHPC_COUPLING_HPP
#define MLHPC_COUPLING_HPP

#include "mlhpc/inference.hpp"
#include "mlhpc/particle_batch.hpp"
#include "mlhpc/types.hpp"

#include <mpi.h>

#include <cstddef>
#include <memory>

namespace mlhpc
{

// ---------------------------------------------------------------------
//  CouplingConfig -- knobs read by the factory.  All fields have safe
//  defaults so a caller that doesn't care can pass {} and get the
//  reasonable behaviour for the compiled case.
// ---------------------------------------------------------------------
struct CouplingConfig
{
                  int            ml_server_nodes       = 0;  // 0: one server rank per node; >0: N whole server nodes.
                  std::size_t    max_batch_particles   = 0;  // 0: do not split physics batches.
                  std::size_t    max_inflight_requests = 8;  // transport back-pressure per physics rank.
                  std::size_t    nsteps                = 1;  // benchmark driver time steps.
                  unsigned int   active_seed           = 1u; // deterministic active-particle selection seed.
};

// ---------------------------------------------------------------------
//  Timings -- per-rank wall-clock accounting filled by the
//  implementation.  Times are in seconds, accumulated across all
//  evaluate/serve calls during the lifetime of the Coupling object.
//
//      total   -- end-to-end time spent inside evaluate() (physics
//                 ranks) or serve() loop body (inference ranks).
//      send    -- time spent in MPI_Send / MPI_Isend (B and C only).
//      wait    -- time spent waiting for MPI_Recv to complete (B/C).
//      kernel  -- time spent inside backend.batch_inference (CPU/GPU).
//      n_calls -- number of batches processed.
//
//  For case A: total == kernel, send == wait == 0.
// ---------------------------------------------------------------------
struct Timings
{
              double         total    = 0.0;
              double         send     = 0.0;
              double         wait     = 0.0;
              double         kernel   = 0.0;
       std::size_t           n_calls  = 0;
};

// ---------------------------------------------------------------------
//  CouplingRequest -- opaque handle returned by evaluate_async().
//
//  The handle names one logical cooling request as seen by the caller.
//  A logical request may be split into several transport chunks under
//  the hood when CouplingConfig::max_batch_particles is non-zero.  The
//  caller treats the handle as a single unit and completes it through
//  test() / wait().
// ---------------------------------------------------------------------
struct CouplingRequest
{
       std::size_t           id = 0;
};

// ---------------------------------------------------------------------
//  Coupling -- abstract base.  See file-level comment for the full
//  contract.  Implementations live in coupling_{A,B,C}.cpp and are
//  selected by the build flag.
//
//  Memory model: the implementation owns its private sub-communicators
//  and frees them in the destructor.  Callers must not free
//  physics_comm() themselves.
//
//  Thread safety: a single Coupling instance is intended to be used by
//  one thread per rank.  MPI_THREAD_FUNNELED is enough; the harness
//  does not require MPI_THREAD_MULTIPLE.
// ---------------------------------------------------------------------
class Coupling
{
public:

  virtual ~Coupling () = default;

  // -------------------------------------------------------------------
  //  physics_comm -- communicator that contains only the ranks that
  //  read the snapshot and own physical particles.  For case A this
  //  equals MPI_COMM_WORLD; for B/C it is world minus the inference
  //  pool.  The Format2Distributor must be constructed on this
  //  communicator, not on world, otherwise the inference ranks would
  //  also try to read particles.
  //
  //  Returns MPI_COMM_NULL on ranks that are NOT physics ranks.
  // -------------------------------------------------------------------
  virtual MPI_Comm
  physics_comm () const noexcept = 0;

  // -------------------------------------------------------------------
  //  Role predicates -- exactly one of is_physics() / is_inference()
  //  is true for each rank in cases B and C.  For case A both return
  //  true (every rank does both jobs).
  // -------------------------------------------------------------------
  virtual bool
  is_physics () const noexcept = 0;

  virtual bool
  is_inference () const noexcept = 0;

  // -------------------------------------------------------------------
  //  evaluate -- physics-rank entry point.
  //
  //  Given the calling rank's local particle batch, fill `out` with
  //  one cooling-rate value per particle.  Implementation depends on
  //  the compiled case:
  //
  //      A : direct call to backend.batch_inference (local, no MPI).
  //      B/C service topology: submit one or more non-blocking MPI
  //          chunks to the assigned ML server rank and complete them
  //          through progress/test/wait.  The blocking evaluate() path
  //          remains submit + wait.
  //
  //  Preconditions:
  //      - `batch.n == out_length` (caller's responsibility);
  //      - is_physics() == true on the calling rank.
  //      - Case A requires a non-null backend; service clients need none.
  //        Omitting the backend in Case A throws std::logic_error.
  //
  //  Postconditions:
  //      - out[0 .. batch.n) is filled;
  //      - timings() reflects the wall-clock cost of this call,
  //        broken down per phase.
  //
  //  The function is `const` because Coupling state is effectively
  //  immutable across calls; the Timings counters live in a mutable
  //  cache that the implementation updates internally.
  // -------------------------------------------------------------------
  virtual CouplingRequest
  evaluate_async (const ParticleBatch       &batch,    // in : copied/packed before return
                  Real *__restrict           out,      // out: must live until wait() succeeds
                  const InferenceBackend    *backend = nullptr // required by A; unused by B clients
                 ) const = 0;

  CouplingRequest
  evaluate_async (const ParticleBatch       &batch,
                  Real *__restrict           out,
                  const InferenceBackend    &backend) const
  {
    return evaluate_async (batch, out, &backend);
  }

  // -------------------------------------------------------------------
  //  progress -- advance any non-blocking transport work owned by this
  //  Coupling instance.
  //
  //  For MPI-backed cases this polls outstanding sends/receives and
  //  marks completed requests.  The harness still requires only
  //  MPI_THREAD_FUNNELED: progress happens when the owner thread calls
  //  this function, test(), or wait().  Case A has no MPI traffic and
  //  progress() is a no-op.
  // -------------------------------------------------------------------
  virtual void
  progress () const = 0;

  // -------------------------------------------------------------------
  //  test / wait -- completion API for evaluate_async().
  //
  //  test() is non-blocking and returns true once all chunks belonging
  //  to the request have filled the caller-owned output buffer.  wait()
  //  drives progress until the same condition is true.
  // -------------------------------------------------------------------
  virtual bool
  test (CouplingRequest request) const = 0;

  virtual void
  wait (CouplingRequest request) const = 0;

  virtual void
  wait_all () const = 0;

  virtual void
  evaluate (const ParticleBatch       &batch,    // in : SoA particles (length batch.n)
            Real *__restrict           out,      // out: cooling rate per particle
            const InferenceBackend    *backend = nullptr // required by A; unused by B clients
           ) const = 0;

  void
  evaluate (const ParticleBatch       &batch,
            Real *__restrict           out,
            const InferenceBackend    &backend) const
  {
    evaluate (batch, out, &backend);
  }

  // -------------------------------------------------------------------
  //  serve -- inference-rank entry point.
  //
  //  Blocks in a receive loop: probes for incoming queries, dispatches
  //  each batch through backend.batch_inference, and sends the answer
  //  back to the originating physics rank.  Returns only when every
  //  expected shutdown sentinel has been received.
  //
  //  For case A this is a no-op (returns immediately): an A-rank is
  //  *both* physics and inference, and the inference work happens
  //  in-line inside evaluate().
  //
  //  Preconditions: is_inference() == true on the calling rank.
  // -------------------------------------------------------------------
  virtual void
  serve (const InferenceBackend       &backend   // in : plug-in model (must outlive this call)
        ) const = 0;

  // -------------------------------------------------------------------
  //  shutdown -- collective on MPI_COMM_WORLD.
  //
  //  Physics ranks call this after their last evaluate() to drain the
  //  inference pool; the implementation emits the shutdown sentinels
  //  that let each serve() loop terminate gracefully.  In case A this
  //  is a no-op.
  //
  //  Calling shutdown() more than once on the same Coupling instance
  //  is allowed but unnecessary; subsequent calls do nothing.
  // -------------------------------------------------------------------
  virtual void
  shutdown () const = 0;

  // -------------------------------------------------------------------
  //  timings -- snapshot of the per-rank wall-clock accumulators.
  //  Values keep growing across multiple evaluate/serve calls.  Use
  //  MPI_Reduce on physics_comm() (or world) to get aggregates.
  // -------------------------------------------------------------------
  virtual Timings
  timings () const noexcept = 0;

protected:

  Coupling () = default;

  // Non-copyable, non-movable: the implementation owns MPI handles.
  Coupling (const Coupling &)             = delete;
  Coupling (Coupling &&)                  = delete;
  Coupling &operator= (const Coupling &)  = delete;
  Coupling &operator= (Coupling &&)       = delete;
};

// ---------------------------------------------------------------------
//  make_coupling -- factory.
//
//  Constructs the case-specific Coupling implementation for the
//  build-time-selected case.  Collective on `world`; every rank
//  must call it.
//
//  Inputs:
//      world  -- communicator the caller wishes to use as the
//                coupling's universe (typically MPI_COMM_WORLD).
//      cfg    -- runtime knobs (see CouplingConfig).  Unused fields
//                are silently ignored by cases that don't need them.
//
//  Returns: owning pointer; the destructor frees the private
//  sub-communicators.  Never returns null -- a fatal configuration is
//  reported via MPI_Abort, on the principle that there is no
//  recoverable state on rank 0 if rank N's communicator construction
//  failed.
// ---------------------------------------------------------------------
std::unique_ptr<Coupling>
make_coupling (MPI_Comm                world,
               const CouplingConfig   &cfg);

// ---------------------------------------------------------------------
//  compiled_case_label -- short human-readable name of the case the
//  current binary was built for ("A", "B", "C").  Useful in log lines
//  so a binary's behaviour is unambiguous in mixed benchmark output.
// ---------------------------------------------------------------------
const char *
compiled_case_label () noexcept;

}  // namespace mlhpc

#endif  // MLHPC_COUPLING_HPP
