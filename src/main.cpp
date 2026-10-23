// ---------------------------------------------------------------------
//  src/main.cpp -- entry point of the mlhpc harness.
//
//  Usage:
//      mlhpc [options] [snapshot_path]
//
//  Two modes, kept from the previous iteration:
//
//    * No positional argument : MPI smoke test.  Every rank prints
//                               its role (physics / inference) under
//                               the compiled coupling case, plus
//                               basic environment info.
//
//    * One positional argument: full read -> units -> evaluate
//                               pipeline.  Physics ranks open the
//                               snapshot through Format2Distributor
//                               on `coupling->physics_comm()`, build
//                               a ParticleBatch, call evaluate(),
//                               and reduce per-field min/max on the
//                               physics communicator.  Inference
//                               ranks block in serve() until shutdown
//                               sentinels arrive.  In case A every
//                               rank is both, so the "inference rank"
//                               branch is empty and the physics path
//                               runs on every rank.
//
//  CLI flags:
//      --inf-pool=N           Legacy Case-B pool-size knob.
//      --ml-server-nodes=N    0: one server rank per node; >0: use N
//                             whole ML server nodes.
//      --max-batch-particles=N
//                             Split logical cooling requests into chunks
//                             of at most N particles (0 disables split).
//      --max-inflight=N       Max transport chunks in flight per physics
//                             rank before evaluate_async() applies
//                             back-pressure.
//      --nsteps=N             Benchmark time steps; values >1 select
//                             pseudo-random active particle subsets.
//      --active-seed=N        Seed for deterministic active-particle
//                             selection in benchmark modes.
//      --model=PATH           Model loaded by inference ranks in ONNX builds.
//
//  The orchestration code below speaks only through the Coupling
//  abstraction.  Anything case-specific (sub-communicators, query/
//  answer protocols, shutdown signalling) is private to
//  coupling_{A,B,C}.cpp -- changing this file should not be needed
//  when wiring B or C.
// ---------------------------------------------------------------------

#include "mlhpc/coupling.hpp"
#include "mlhpc/format2_distributor.hpp"
#include "mlhpc/group_distribution.hpp"
#include "mlhpc/inference.hpp"
#include "mlhpc/inference_factory.hpp"
#include "mlhpc/io_gadget2.hpp"
#include "mlhpc/particle_batch.hpp"
#include "mlhpc/types.hpp"
#include "mlhpc/units.hpp"

#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

// ---------------------------------------------------------------------
//  CLI -- small struct holding the parsed command-line arguments.
// ---------------------------------------------------------------------
struct CLI
{
  bool        want_snapshot  = false;   // true iff a positional path was given
  std::string snapshot_path;
  std::string model_path;
  int         inference_pool      = 1;   // CouplingConfig.inference_pool_size
  int         ml_server_nodes     = 0;   // CouplingConfig.ml_server_nodes
  std::size_t max_batch_particles = 0;   // CouplingConfig.max_batch_particles
  std::size_t max_inflight        = 8;   // CouplingConfig.max_inflight_requests
  std::size_t nsteps              = 1;   // CouplingConfig.nsteps
  unsigned int active_seed        = 1u;  // CouplingConfig.active_seed
};


// ---------------------------------------------------------------------
//  parse_unsigned_long -- strict decimal parser for non-negative CLI
//  integer values.
//
//  std::atoi silently accepts partial strings and overflows, which is a
//  bad fit for benchmark knobs that control communicator layout and
//  message sizes.  This helper accepts only base-10 digits and returns
//  false on empty or malformed input.
// ---------------------------------------------------------------------
bool
parse_unsigned_long (const char           *text,
                     unsigned long long   &out)
{
  if (text == nullptr || text [0] == '\0')
    {
      return false;
    }

  char              *end = nullptr;
  const auto         val = std::strtoull (text, &end, 10);
  if (end == text || *end != '\0')
    {
      return false;
    }

  out = val;
  return true;
}

// ---------------------------------------------------------------------
//  report_bad_flag_value -- rank-0-only CLI diagnostic.
// ---------------------------------------------------------------------
void
report_bad_flag_value (int          rank,
                       const char  *flag,
                       const char  *value,
                       const char  *rule)
{
  if (rank == 0)
    {
      std::fprintf (stderr,
                    "mlhpc: %s requires %s (got '%s')\n",
                    flag, rule, value);
    }
}

// ---------------------------------------------------------------------
//  parse_cli -- ad-hoc parser for the short flag set above.  Returns
//  false (with a stderr message on rank 0) if the syntax is wrong.
//
//  Recognised forms:
//      mlhpc
//      mlhpc <snapshot>
//      mlhpc --inf-pool=<N>
//      mlhpc --inf-pool=<N> <snapshot>
// ---------------------------------------------------------------------
bool
parse_cli (int       argc,
           char    **argv,
           int       rank,
           CLI      &out)
{
  for (int i = 1; i < argc; ++i)
    {
      const char *arg = argv [i];
      if (std::strncmp (arg, "--inf-pool=", 11) == 0)
        {
          const char          *val = arg + 11;
          unsigned long long   n   = 0;
          if (!parse_unsigned_long (val, n) || n < 1)
            {
              report_bad_flag_value (rank, "--inf-pool", val,
                                     "a positive integer");
              return false;
            }
          out.inference_pool = static_cast<int> (n);
        }
      else if (std::strncmp (arg, "--ml-server-nodes=", 18) == 0)
        {
          const char          *val = arg + 18;
          unsigned long long   n   = 0;
          if (!parse_unsigned_long (val, n))
            {
              report_bad_flag_value (rank, "--ml-server-nodes", val,
                                     "a non-negative integer");
              return false;
            }
          out.ml_server_nodes = static_cast<int> (n);
        }
      else if (std::strncmp (arg, "--max-batch-particles=", 22) == 0)
        {
          const char          *val = arg + 22;
          unsigned long long   n   = 0;
          if (!parse_unsigned_long (val, n))
            {
              report_bad_flag_value (rank, "--max-batch-particles", val,
                                     "a non-negative integer");
              return false;
            }
          out.max_batch_particles = static_cast<std::size_t> (n);
        }
      else if (std::strncmp (arg, "--max-inflight=", 15) == 0)
        {
          const char          *val = arg + 15;
          unsigned long long   n   = 0;
          if (!parse_unsigned_long (val, n) || n < 1)
            {
              report_bad_flag_value (rank, "--max-inflight", val,
                                     "a positive integer");
              return false;
            }
          out.max_inflight = static_cast<std::size_t> (n);
        }
      else if (std::strncmp (arg, "--nsteps=", 9) == 0)
        {
          const char          *val = arg + 9;
          unsigned long long   n   = 0;
          if (!parse_unsigned_long (val, n) || n < 1)
            {
              report_bad_flag_value (rank, "--nsteps", val,
                                     "a positive integer");
              return false;
            }
          out.nsteps = static_cast<std::size_t> (n);
        }
      else if (std::strncmp (arg, "--active-seed=", 14) == 0)
        {
          const char          *val = arg + 14;
          unsigned long long   n   = 0;
          if (!parse_unsigned_long (val, n))
            {
              report_bad_flag_value (rank, "--active-seed", val,
                                     "a non-negative integer");
              return false;
            }
          out.active_seed = static_cast<unsigned int> (n);
        }
      else if (std::strncmp (arg, "--model=", 8) == 0)
        {
          out.model_path = arg + 8;
          if (out.model_path.empty ())
            {
              report_bad_flag_value (rank, "--model", arg + 8,
                                    "a non-empty model path");
              return false;
            }
        }
      else if (arg [0] == '-')
        {
          if (rank == 0)
            {
              std::fprintf (stderr, "mlhpc: unknown flag '%s'\n", arg);
            }
          return false;
        }
      else
        {
          if (out.want_snapshot)
            {
              if (rank == 0)
                {
                  std::fprintf (stderr,
                                "mlhpc: at most one positional argument expected (snapshot path)\n");
                }
              return false;
            }
          out.want_snapshot  = true;
          out.snapshot_path  = arg;
        }
    }
  return true;
}

// ---------------------------------------------------------------------
//  print_header_summary -- rank-0-only snapshot summary printer.
// ---------------------------------------------------------------------
void
print_header_summary (const mlhpc::Format2Distributor &dist)
{
  const auto &h = dist.reference_header ();
  std::printf ("mlhpc: snapshot summary\n");
  std::printf ("  num_files                 : %d\n",   dist.num_files ());
  std::printf ("  npart_total[0] (gas)      : %zu\n",  dist.npart_total_gas ());
  for (int t = 1; t < 6; ++t)
    {
      const std::uint64_t lo = h.npartTotal [t];
      const std::uint64_t hi = h.npartTotalHighWord [t];
      const std::size_t   n  = static_cast<std::size_t> ((hi << 32) | lo);
      if (n > 0)
        {
          std::printf ("  npart_total[%d]            : %zu\n", t, n);
        }
    }
  std::printf ("  redshift z                : %.6f\n", h.redshift);
  std::printf ("  scale factor a            : %.6f\n", h.time);
  std::printf ("  BoxSize (code units)      : %.6f\n", h.BoxSize);
  std::printf ("  HubbleParam h             : %.6f\n", h.HubbleParam);
  std::printf ("  Omega0 / OmegaLambda      : %.4f / %.4f\n",
               h.Omega0, h.OmegaLambda);
  std::printf ("  flag_doubleprecision      : %d\n", h.flag_doubleprecision);
  std::printf ("  flag_entropy_instead_u    : %d\n",
               h.flag_entropy_instead_u);
  std::fflush (stdout);
}

// ---------------------------------------------------------------------
//  smoke_test -- minimal MPI sanity print + role announcement.
//
//  Every rank prints whether it is physics / inference under the
//  compiled coupling case.  This is the no-snapshot path; it does
//  not exercise serve() or evaluate(), so it works as a quick
//  topology sanity check for every case.
// ---------------------------------------------------------------------
int
smoke_test (int                     world_rank,
            int                     world_size,
            int                     provided,
            const mlhpc::Coupling  &coupling)
{
  char host [MPI_MAX_PROCESSOR_NAME] = {};
  int  host_len = 0;
  MPI_Get_processor_name (host, &host_len);

  if (world_rank == 0)
    {
      std::printf ("mlhpc smoke test\n");
      std::printf ("  world size       : %d\n", world_size);
      std::printf ("  coupling case    : %s\n",
                   mlhpc::compiled_case_label ());
      std::printf ("  MPI thread level : %d "
                   "(requested MPI_THREAD_FUNNELED=%d)\n",
                   provided, MPI_THREAD_FUNNELED);
      std::printf ("  GadgetHeader2 sz : %zu bytes (must be 256)\n",
                   sizeof (mlhpc::GadgetHeader2));
      std::printf ("  (no snapshot given -- pass a path as the last "
                   "argument to exercise the full pipeline)\n");
      std::fflush (stdout);
    }

  MPI_Barrier (MPI_COMM_WORLD);
  std::printf ("  [rank %4d/%4d] alive on %-32s  role: %s%s\n",
               world_rank, world_size, host,
               coupling.is_physics ()   ? "physics "  : "        ",
               coupling.is_inference () ? "inference" : "         ");
  std::fflush (stdout);
  return EXIT_SUCCESS;
}


// ---------------------------------------------------------------------
//  ActiveSelection -- one time-step's active-particle mini-batch.
//
//  `batch` is packed densely for inference.  `indices[j]` records where
//  active particle j came from in the full local particle array, so the
//  returned cooling values can be scattered back without changing the
//  backend interface.
// ---------------------------------------------------------------------
struct ActiveSelection
{
  mlhpc::ParticleBatch       batch;
  std::vector<std::size_t>   indices;
};

// ---------------------------------------------------------------------
//  lcg_next -- tiny deterministic generator for benchmark selection.
//
//  This is not a statistical simulation RNG.  It is a stable, cheap way
//  to create different active subsets across ranks and time steps without
//  adding library machinery or hidden global state.
// ---------------------------------------------------------------------
std::uint64_t
lcg_next (std::uint64_t &state)
{
  state = state * 6364136223846793005ull + 1442695040888963407ull;
  return state;
}

// ---------------------------------------------------------------------
//  choose_coprime_stride -- pick a stride that visits every local
//  particle index before repeating.
//
//  The active subset is generated as offset + j*stride modulo n.  Making
//  stride coprime with n gives unique indices for j in [0, n), avoiding a
//  large shuffled index array on production-sized particle batches.
// ---------------------------------------------------------------------
std::size_t
choose_coprime_stride (std::size_t     n,
                       std::uint64_t  &rng)
{
  if (n <= 1)
    {
      return 1;
    }

  std::size_t  stride = static_cast<std::size_t> (lcg_next (rng) % n);
  if (stride == 0)
    {
      stride = 1;
    }
  while (std::gcd (stride, n) != 1)
    {
      stride += 1;
      if (stride >= n)
        {
          stride = 1;
        }
    }
  return stride;
}

// ---------------------------------------------------------------------
//  build_active_selection -- copy one pseudo-random active subset into a
//  dense ParticleBatch.
//
//  Inputs:  full  -- full local particle batch in CGS units.
//           rng   -- per-rank deterministic generator state, updated in
//                    place so every step chooses a different subset.
//  Output:  out   -- dense active batch plus source indices.  Existing
//                    buffers are reused across calls when capacity allows.
// ---------------------------------------------------------------------
void
build_active_selection (const mlhpc::ParticleBatch  &full,
                        std::uint64_t               &rng,
                        ActiveSelection             &out)
{
  const std::size_t  n = full.n;
  const std::size_t  active = (n == 0)
                             ? 0
                             : static_cast<std::size_t> (lcg_next (rng)
                                                        % (n + 1));

  out.batch.n      = active;
  out.batch.nmetal = full.nmetal;
  out.batch.rho.resize (active);
  out.batch.u.resize   (active);
  out.indices.resize   (active);

  const bool  has_ne = (full.ne.size () == n);
  const bool  has_nh = (full.nh.size () == n);
  const bool  has_zs = (full.nmetal > 0 && full.Zs.size () == n * full.nmetal);

  if (has_ne) out.batch.ne.resize (active);
  else        out.batch.ne.clear  ();
  if (has_nh) out.batch.nh.resize (active);
  else        out.batch.nh.clear  ();
  if (has_zs) out.batch.Zs.resize (active * full.nmetal);
  else        out.batch.Zs.clear  ();

  if (active == 0)
    {
      return;
    }

  const std::size_t  offset = static_cast<std::size_t> (lcg_next (rng) % n);
  const std::size_t  stride = choose_coprime_stride (n, rng);

  for (std::size_t j = 0; j < active; ++j)
    {
      const std::size_t  src = (offset + j * stride) % n;
      out.indices [j] = src;
      out.batch.rho [j] = full.rho [src];
      out.batch.u   [j] = full.u   [src];
      if (has_ne) out.batch.ne [j] = full.ne [src];
      if (has_nh) out.batch.nh [j] = full.nh [src];
      if (has_zs)
        {
          const std::size_t  src_z = src * full.nmetal;
          const std::size_t  dst_z = j   * full.nmetal;
          for (std::size_t k = 0; k < full.nmetal; ++k)
            {
              out.batch.Zs [dst_z + k] = full.Zs [src_z + k];
            }
        }
    }
}

// ---------------------------------------------------------------------
//  physics_pipeline -- the full read -> units -> evaluate path,
//  running on every rank in coupling.physics_comm().
//
//  All reductions and per-rank gathers are taken on physics_comm so
//  the case B/C case (where inference ranks have no particles)
//  remains correct without code changes here.
// ---------------------------------------------------------------------
int
physics_pipeline (const std::string                 &path,
                  const CLI                         &cli,
                  const mlhpc::Coupling             &coupling,
                  const mlhpc::InferenceBackend     *backend)
{
  const MPI_Comm  pcomm = coupling.physics_comm ();
  int             prank = 0;
  int             psize = 1;
  MPI_Comm_rank (pcomm, &prank);
  MPI_Comm_size (pcomm, &psize);

  try
    {
      mlhpc::Format2Distributor dist (pcomm, path);

      if (prank == 0)
        {
          print_header_summary (dist);
        }

      // -------- precision cross-check (driver-side, see types.hpp) ---
      const bool   disk_is_double = (dist.reference_header ().flag_doubleprecision != 0);
      const bool   mem_is_double  = (sizeof (mlhpc::Real) == 8);
      if (disk_is_double != mem_is_double)
        {
          if (prank == 0)
            {
              std::fprintf (stderr,
                            "mlhpc: precision mismatch -- snapshot is %s "
                            "but mlhpc::Real is %s (sizeof=%zu).  "
                            "Rebuild with %s.\n",
                            disk_is_double ? "double-precision" : "single-precision",
                            mem_is_double  ? "double" : "float",
                            sizeof (mlhpc::Real),
                            disk_is_double ? "FEATURE_DOUBLE=1"
                                           : "FEATURE_DOUBLE=0");
            }
          MPI_Abort (MPI_COMM_WORLD, 2);
          return EXIT_FAILURE;
        }

      const MPI_Datatype  mpi_real = mem_is_double ? MPI_DOUBLE : MPI_FLOAT;

      // -------- read RHO + U into a ParticleBatch (still code units)
      mlhpc::ParticleBatch  batch {};
      batch.resize (dist.local_count (), 0);
      // Convention (see particle_batch.hpp): size 0 means "absent".
      // The driver only reads RHO and U for now; the optional columns
      // remain unread, so clear them here so any downstream consumer
      // (e.g. the case B/C wire protocol) sees them as absent.
      batch.ne.clear ();
      batch.nh.clear ();
      batch.Zs.clear ();

      const bool   have_rho = dist.read_gas_block ("RHO ",
                                                   sizeof (mlhpc::Real),
                                                   batch.rho.data ());
      const bool   have_u   = dist.read_gas_block ("U   ",
                                                   sizeof (mlhpc::Real),
                                                   batch.u.data ());
      if (!have_rho || !have_u)
        {
          if (prank == 0)
            {
              std::fprintf (stderr,
                            "mlhpc: missing required block(s) "
                            "(RHO=%d, U=%d) in snapshot.\n",
                            have_rho, have_u);
            }
          MPI_Abort (MPI_COMM_WORLD, 3);
          return EXIT_FAILURE;
        }

      // -------- code units -> physical CGS --------------------------
      const mlhpc::Cosmology cosmo =
          mlhpc::make_cosmology (dist.reference_header ());

      if (dist.reference_header ().flag_entropy_instead_u != 0)
        {
          mlhpc::entropy_to_u_code (batch.u.data (),
                                    batch.rho.data (),
                                    batch.n);
        }

      mlhpc::rho_code_to_cgs (batch.rho.data (), batch.n, cosmo);
      mlhpc::u_code_to_cgs   (batch.u.data (),   batch.n, cosmo);

      // -------- inference via the Coupling abstraction --------------
      // In case A this is a local backend call; in service cases the
      // blocking wrapper submits non-blocking transport chunks and waits
      // for their answers.  With --nsteps > 1, each step evaluates only a
      // deterministic pseudo-random active subset and scatters returned
      // cooling values back into the full-rank output array.
      std::vector<mlhpc::Real>  cooling_rate (batch.n);
      mlhpc::Real              cr_local_min =
          std::numeric_limits<mlhpc::Real>::max ();
      mlhpc::Real              cr_local_max =
          -std::numeric_limits<mlhpc::Real>::max ();
      std::size_t              active_total = 0;

      if (cli.nsteps <= 1)
        {
          coupling.evaluate (batch, cooling_rate.data (), backend);
          for (std::size_t i = 0; i < batch.n; ++i)
            {
              if (cooling_rate [i] < cr_local_min) cr_local_min = cooling_rate [i];
              if (cooling_rate [i] > cr_local_max) cr_local_max = cooling_rate [i];
            }
          active_total = batch.n;
        }
      else
        {
          ActiveSelection          active;
          std::vector<mlhpc::Real> active_out;
          std::uint64_t            rng =
              static_cast<std::uint64_t> (cli.active_seed)
              ^ (static_cast<std::uint64_t> (prank + 1)
                 * 0x9e3779b97f4a7c15ull);

          for (std::size_t step = 0; step < cli.nsteps; ++step)
            {
              build_active_selection (batch, rng, active);
              active_out.resize (active.batch.n);

              const mlhpc::CouplingRequest request =
                  coupling.evaluate_async (active.batch, active_out.data (), backend);
              coupling.wait (request);

              active_total += active.batch.n;
              for (std::size_t j = 0; j < active.batch.n; ++j)
                {
                  const mlhpc::Real  value = active_out [j];
                  cooling_rate [active.indices [j]] = value;
                  if (value < cr_local_min) cr_local_min = value;
                  if (value > cr_local_max) cr_local_max = value;
                }
            }
        }

      // -------- reductions for sanity reporting on physics rank 0 ---
      auto reduce_min_max = [&] (const mlhpc::Real *data,
                                 std::size_t        n,
                                 mlhpc::Real       &out_min,
                                 mlhpc::Real       &out_max)
        {
          mlhpc::Real  loc_min =  std::numeric_limits<mlhpc::Real>::max ();
          mlhpc::Real  loc_max = -std::numeric_limits<mlhpc::Real>::max ();
          for (std::size_t i = 0; i < n; ++i)
            {
              if (data [i] < loc_min) loc_min = data [i];
              if (data [i] > loc_max) loc_max = data [i];
            }
          MPI_Reduce (&loc_min, &out_min, 1, mpi_real, MPI_MIN, 0, pcomm);
          MPI_Reduce (&loc_max, &out_max, 1, mpi_real, MPI_MAX, 0, pcomm);
        };

      mlhpc::Real  rho_min = 0, rho_max = 0;
      mlhpc::Real  u_min   = 0, u_max   = 0;
      mlhpc::Real  cr_min  = 0, cr_max  = 0;
      reduce_min_max (batch.rho.data (), batch.n, rho_min, rho_max);
      reduce_min_max (batch.u.data (),   batch.n, u_min,   u_max);
      MPI_Reduce (&cr_local_min, &cr_min, 1, mpi_real, MPI_MIN, 0, pcomm);
      MPI_Reduce (&cr_local_max, &cr_max, 1, mpi_real, MPI_MAX, 0, pcomm);

      std::size_t  active_total_all = 0;
      MPI_Reduce (&active_total, &active_total_all, 1, MPI_UNSIGNED_LONG,
                  MPI_SUM, 0, pcomm);

      // -------- per-rank slice / timing gather on physics rank 0 ----
      const std::size_t          my_count = dist.local_count ();
      const int                  my_group = dist.my_group ();
      const int                  my_pos   = dist.my_position ();
      const int                  my_gs    = dist.my_group_size ();
      const int                  my_lead  = dist.my_leader ();
      const mlhpc::Timings       my_t     = coupling.timings ();
      const double               my_total = my_t.total;

      std::vector<std::size_t>   counts    (prank == 0 ? psize : 0);
      std::vector<int>           groups    (prank == 0 ? psize : 0);
      std::vector<int>           positions (prank == 0 ? psize : 0);
      std::vector<int>           gsizes    (prank == 0 ? psize : 0);
      std::vector<int>           leaders   (prank == 0 ? psize : 0);
      std::vector<double>        totals    (prank == 0 ? psize : 0);
      MPI_Gather (&my_count, 1, MPI_UNSIGNED_LONG,
                  counts.data (), 1, MPI_UNSIGNED_LONG, 0, pcomm);
      MPI_Gather (&my_group, 1, MPI_INT,
                  groups.data (), 1, MPI_INT, 0, pcomm);
      MPI_Gather (&my_pos,   1, MPI_INT,
                  positions.data (), 1, MPI_INT, 0, pcomm);
      MPI_Gather (&my_gs,    1, MPI_INT,
                  gsizes.data (), 1, MPI_INT, 0, pcomm);
      MPI_Gather (&my_lead,  1, MPI_INT,
                  leaders.data (), 1, MPI_INT, 0, pcomm);
      MPI_Gather (&my_total, 1, MPI_DOUBLE,
                  totals.data (), 1, MPI_DOUBLE, 0, pcomm);

      // -------- aggregate evaluate() timings across physics ranks ---
      double  t_total_min = 0.0,  t_total_max = 0.0,  t_total_sum = 0.0;
      MPI_Reduce (&my_total, &t_total_min, 1, MPI_DOUBLE, MPI_MIN, 0, pcomm);
      MPI_Reduce (&my_total, &t_total_max, 1, MPI_DOUBLE, MPI_MAX, 0, pcomm);
      MPI_Reduce (&my_total, &t_total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, pcomm);

      if (prank == 0)
        {
          std::printf ("mlhpc: end-to-end pipeline summary "
                       "(case %s)\n",
                       mlhpc::compiled_case_label ());
          std::printf ("  rho   [g/cm^3, physical]  : [%.6g, %.6g]\n",
                       static_cast<double> (rho_min),
                       static_cast<double> (rho_max));
          std::printf ("  u     [erg/g, physical]   : [%.6g, %.6g]\n",
                       static_cast<double> (u_min),
                       static_cast<double> (u_max));
          if (active_total_all > 0)
            {
              std::printf ("  cooling-rate output (%s) : [%.6g, %.6g]\n",
                           mlhpc::compiled_backend_label (),
                           static_cast<double> (cr_min),
                           static_cast<double> (cr_max));
            }
          else
            {
              std::printf ("  cooling-rate output (%s) : no active particles\n",
                           mlhpc::compiled_backend_label ());
            }
          std::printf ("  active cooling evaluations : %zu over %zu step(s)\n",
                       active_total_all, cli.nsteps);
          std::printf ("  evaluate() wall-clock     : "
                       "min %.6g s,  max %.6g s,  mean %.6g s\n",
                       t_total_min, t_total_max,
                       t_total_sum / static_cast<double> (psize));
          std::printf ("  per-rank slice / timing   :\n");
          for (int r = 0; r < psize; ++r)
            {
              std::printf ("    rank %4d : group %3d  pos %3d/%-3d  "
                           "leader %4d  %10zu particles  "
                           "%.6g s\n",
                           r, groups [r], positions [r], gsizes [r],
                           leaders [r], counts [r], totals [r]);
            }
          std::fflush (stdout);
        }
    }
  catch (const std::exception &e)
    {
      std::fprintf (stderr, "[prank %d] mlhpc: %s\n", prank, e.what ());
      MPI_Abort (MPI_COMM_WORLD, 1);
      return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}

}  // namespace

int
main (int argc, char **argv)
{
  int provided = MPI_THREAD_SINGLE;
  if (MPI_Init_thread (&argc, &argv, MPI_THREAD_FUNNELED, &provided)
      != MPI_SUCCESS)
    {
      std::fprintf (stderr, "MPI_Init_thread failed\n");
      return EXIT_FAILURE;
    }

  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);

  CLI cli;
  if (!parse_cli (argc, argv, world_rank, cli))
    {
      MPI_Abort (MPI_COMM_WORLD, 4);
      return EXIT_FAILURE;
    }

  // -------- build the Coupling object collectively ---------------------
  mlhpc::CouplingConfig  cfg;
  cfg.ml_server_nodes       = cli.ml_server_nodes;
  cfg.max_batch_particles   = cli.max_batch_particles;
  cfg.max_inflight_requests = cli.max_inflight;
  cfg.nsteps                = cli.nsteps;
  cfg.active_seed           = cli.active_seed;
  std::unique_ptr<mlhpc::Coupling>  coupling =
      mlhpc::make_coupling (MPI_COMM_WORLD, cfg);

  // -------- dispatch ---------------------------------------------------
  // The order below matters for cases B and C: every inference-only
  // rank must be inside serve() before any physics rank calls
  // shutdown(), otherwise the SHUTDOWN sentinel would land in an
  // unposted-Recv slot.  We run smoke_test / physics_pipeline first
  // (both are no-ops on inference-only ranks), then unconditionally
  // call serve() on inference-only ranks, then shutdown() everywhere.
  int                  rc = EXIT_SUCCESS;
  std::unique_ptr<mlhpc::InferenceBackend> backend;

  try
    {
      if (coupling->is_inference ())
        {
          backend = mlhpc::make_backend (cli.model_path);
        }

      const int local_backend = backend ? 1 : 0;
      int backend_count = 0;
      MPI_Allreduce (&local_backend, &backend_count, 1, MPI_INT, MPI_SUM,
                     MPI_COMM_WORLD);
      if (world_rank == 0)
        {
          std::printf ("mlhpc: inference backend = %s; backend instances = %d\n",
                       mlhpc::compiled_backend_label (), backend_count);
          std::fflush (stdout);
        }

      if (!cli.want_snapshot)
        {
          rc = smoke_test (world_rank, world_size, provided, *coupling);
        }
      else if (coupling->is_physics ())
        {
          rc = physics_pipeline (cli.snapshot_path, cli, *coupling, backend.get ());
        }

      if (coupling->is_inference () && !coupling->is_physics ())
        {
          coupling->serve (*backend);
        }

      coupling->shutdown ();
    }
  catch (const std::exception &error)
    {
      std::fprintf (stderr, "[rank %d] mlhpc: %s\n",
                    world_rank, error.what ());
      MPI_Abort (MPI_COMM_WORLD, 5);
      return EXIT_FAILURE;
    }

  // Drain requests before releasing model resources and MPI communicators.
  backend.reset ();
  coupling.reset ();

  MPI_Finalize ();
  return rc;
}
