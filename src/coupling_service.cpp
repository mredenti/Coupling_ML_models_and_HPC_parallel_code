// ---------------------------------------------------------------------
//  src/coupling_service.cpp -- generalized service-topology Coupling.
// ---------------------------------------------------------------------

#include "mlhpc/detail/coupling_service.hpp"

#include "mlhpc/coupling.hpp"
#include "mlhpc/inference.hpp"
#include "mlhpc/partition.hpp"
#include "mlhpc/particle_batch.hpp"
#include "mlhpc/types.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mlhpc::detail
{

namespace
{

// ---------------------------------------------------------------------
//  dup_world_or_abort -- create the owned communicator used by the
//  service coupling.
//
//  Keeping this in the initializer path lets AsyncMpiTransport borrow
//  the already-duplicated communicator and avoids a second duplicate.
// ---------------------------------------------------------------------
MPI_Comm
dup_world_or_abort (MPI_Comm world)
{
  MPI_Comm  dup = MPI_COMM_NULL;
  if (MPI_Comm_dup (world, &dup) != MPI_SUCCESS)
    {
      std::fprintf (stderr,
                    "ServiceCoupling: MPI_Comm_dup failed; aborting.\n");
      MPI_Abort (world, 31);
    }
  return dup;
}

// ---------------------------------------------------------------------
//  find_group_for_node -- locate the uniform node group containing a
//  compact node index.
// ---------------------------------------------------------------------
int
find_group_for_node (int node_index,
                     int node_count,
                     int ngroups) noexcept
{
  for (int g = 0; g < ngroups; ++g)
    {
      const GlobalSlice  slice =
          partition_uniform (static_cast<std::size_t> (node_count),
                             g, ngroups);
      if (static_cast<std::size_t> (node_index) >= slice.gstart
          && static_cast<std::size_t> (node_index) < slice.gend)
        {
          return g;
        }
    }
  return ngroups - 1;
}

}  // namespace

// ---------------------------------------------------------------------
//  ServiceCoupling constructor.
//
//  The constructor owns all topology discovery: it duplicates world,
//  builds a per-node shared-memory communicator, computes compact node
//  indices ordered by leader world rank, assigns roles, and creates the
//  physics communicator.  Message transport is not configured here
//  beyond passing the final target/expected counts to AsyncMpiTransport
//  during evaluate_async(), serve(), and shutdown().
//
//  cfg.ml_server_nodes controls server placement (0: one server rank
//  per shared-memory node; N>0: N whole server nodes).  The value is
//  set by --ml-server-nodes on the CLI or left at the default of 0.
// ---------------------------------------------------------------------
ServiceCoupling::ServiceCoupling (MPI_Comm                world,
                                  const CouplingConfig   &cfg)
  : m_world              (dup_world_or_abort (world)),
    m_node_comm          (MPI_COMM_NULL),
    m_phys_comm          (MPI_COMM_NULL),
    m_transport          (m_world, cfg),
    m_world_rank         (0),
    m_world_size         (0),
    m_node_rank          (0),
    m_node_size          (0),
    m_node_index         (0),
    m_node_count         (0),
    m_target_server      (-1),
    m_expected_shutdowns (0),
    m_is_inference       (false),
    m_already_shutdown   (false)
{
  MPI_Comm_rank (m_world, &m_world_rank);
  MPI_Comm_size (m_world, &m_world_size);

  build_topology (cfg);
}

ServiceCoupling::~ServiceCoupling ()
{
  if (m_phys_comm != MPI_COMM_NULL)
    {
      MPI_Comm_free (&m_phys_comm);
    }
  if (m_node_comm != MPI_COMM_NULL)
    {
      MPI_Comm_free (&m_node_comm);
    }
  if (m_world != MPI_COMM_NULL)
    {
      MPI_Comm_free (&m_world);
    }
}

// ---------------------------------------------------------------------
//  build_topology -- select service mode and build physics_comm.
//
//  cfg.ml_server_nodes == 0 means one ML server rank per shared-memory
//  node, specifically the last rank in that node.  A positive value asks
//  for that many whole ML server nodes.  If the requested value is not
//  usable (greater than or equal to the number of nodes), the code falls
//  back to the one-rank-per-node mode so some physics ranks remain.
// ---------------------------------------------------------------------
void
ServiceCoupling::build_topology (const CouplingConfig   &cfg)
{
  if (MPI_Comm_split_type (m_world, MPI_COMM_TYPE_SHARED, m_world_rank,
                           MPI_INFO_NULL, &m_node_comm) != MPI_SUCCESS)
    {
      std::fprintf (stderr,
                    "ServiceCoupling: MPI_Comm_split_type failed "
                    "on rank %d.\n", m_world_rank);
      MPI_Abort (m_world, 32);
    }

  MPI_Comm_rank (m_node_comm, &m_node_rank);
  MPI_Comm_size (m_node_comm, &m_node_size);

  int  leader_world_rank = (m_node_rank == 0) ? m_world_rank : -1;
  MPI_Bcast (&leader_world_rank, 1, MPI_INT, 0, m_node_comm);

  std::vector<int>  leader_by_rank (static_cast<std::size_t> (m_world_size));
  MPI_Allgather (&leader_world_rank, 1, MPI_INT,
                 leader_by_rank.data (), 1, MPI_INT, m_world);

  std::vector<int>  node_leaders = leader_by_rank;
  std::sort (node_leaders.begin (), node_leaders.end ());
  node_leaders.erase (std::unique (node_leaders.begin (), node_leaders.end ()),
                      node_leaders.end ());

  m_node_count = static_cast<int> (node_leaders.size ());
  const auto  it = std::lower_bound (node_leaders.begin (),
                                     node_leaders.end (),
                                     leader_world_rank);
  m_node_index = static_cast<int> (it - node_leaders.begin ());

  std::vector<int>  rank_node_index (static_cast<std::size_t> (m_world_size));
  MPI_Allgather (&m_node_index, 1, MPI_INT,
                 rank_node_index.data (), 1, MPI_INT, m_world);

  const int  requested = cfg.ml_server_nodes;
  if (requested <= 0 || requested >= m_node_count)
    {
      build_local_rank_per_node_topology (rank_node_index.data ());
    }
  else
    {
      build_server_node_topology (requested, rank_node_index.data ());
    }

  const int  phys_contribution = m_is_inference ? 0 : 1;
  int        total_physics     = 0;
  MPI_Allreduce (&phys_contribution, &total_physics, 1, MPI_INT,
                 MPI_SUM, m_world);
  if (total_physics == 0)
    {
      if (m_world_rank == 0)
        {
          std::fprintf (stderr,
                        "ServiceCoupling: topology produced no physics "
                        "ranks; aborting.\n");
        }
      MPI_Abort (m_world, 33);
    }

  const int  color = m_is_inference ? MPI_UNDEFINED : 0;
  const int  key   = m_world_rank;
  if (MPI_Comm_split (m_world, color, key, &m_phys_comm) != MPI_SUCCESS)
    {
      std::fprintf (stderr,
                    "ServiceCoupling: MPI_Comm_split failed on rank %d.\n",
                    m_world_rank);
      MPI_Abort (m_world, 34);
    }
}

// ---------------------------------------------------------------------
//  build_local_rank_per_node_topology -- mode 0 placement.
//
//  Each shared-memory node contributes one ML server rank: the last rank
//  in node_comm.  All other ranks on that node are physics ranks and
//  route their requests to the local server.  rank_node_index is unused
//  in this mode but retained in the signature for symmetry with the
//  whole-server-node path.
// ---------------------------------------------------------------------
void
ServiceCoupling::build_local_rank_per_node_topology (const int * /*rank_node_index*/)
{
  m_is_inference = (m_node_rank == m_node_size - 1);

  int  server_world_rank = m_is_inference ? m_world_rank : -1;
  MPI_Bcast (&server_world_rank, 1, MPI_INT,
             m_node_size - 1, m_node_comm);

  m_target_server      = server_world_rank;
  m_expected_shutdowns = m_is_inference ? (m_node_size - 1) : 0;
}

// ---------------------------------------------------------------------
//  build_server_node_topology -- whole-node ML server placement.
//
//  server_nodes gives the number of requested ML server nodes.  The
//  compact node index space is split into that many contiguous groups;
//  the last node in each group is removed from physics work and all of
//  its ranks become ML server ranks.  Physics ranks in the group are
//  assigned round-robin across the ranks on the group's server node.
// ---------------------------------------------------------------------
void
ServiceCoupling::build_server_node_topology (int          server_nodes,
                                             const int   *rank_node_index)
{
  const int          group       = find_group_for_node (m_node_index,
                                                        m_node_count,
                                                        server_nodes);
  const GlobalSlice  my_slice    =
      partition_uniform (static_cast<std::size_t> (m_node_count),
                         group, server_nodes);
  const int          server_node = static_cast<int> (my_slice.gend - 1);

  m_is_inference = (m_node_index == server_node);

  std::vector<int>  server_ranks;
  server_ranks.reserve (static_cast<std::size_t> (m_node_size));
  for (int r = 0; r < m_world_size; ++r)
    {
      if (rank_node_index [r] == server_node)
        {
          server_ranks.push_back (r);
        }
    }

  if (server_ranks.empty ())
    {
      std::fprintf (stderr,
                    "ServiceCoupling: no ranks found on server node %d.\n",
                    server_node);
      MPI_Abort (m_world, 35);
    }

  int  server_slot = -1;
  for (std::size_t i = 0; i < server_ranks.size (); ++i)
    {
      if (server_ranks [i] == m_world_rank)
        {
          server_slot = static_cast<int> (i);
          break;
        }
    }

  int  physics_ordinal = 0;
  int  my_physics_slot = -1;
  for (int r = 0; r < m_world_size; ++r)
    {
      const int  node = rank_node_index [r];
      const bool in_group =
          static_cast<std::size_t> (node) >= my_slice.gstart
          && static_cast<std::size_t> (node) < my_slice.gend;
      const bool is_server_node = (node == server_node);
      if (!in_group || is_server_node)
        {
          continue;
        }

      const int  slot = physics_ordinal
                        % static_cast<int> (server_ranks.size ());
      if (r == m_world_rank)
        {
          my_physics_slot = slot;
        }
      if (m_is_inference && slot == server_slot)
        {
          m_expected_shutdowns += 1;
        }
      physics_ordinal += 1;
    }

  if (!m_is_inference)
    {
      if (my_physics_slot < 0)
        {
          std::fprintf (stderr,
                        "ServiceCoupling: physics rank %d was not "
                        "assigned to an ML server.\n", m_world_rank);
          MPI_Abort (m_world, 36);
        }
      m_target_server = server_ranks [static_cast<std::size_t> (my_physics_slot)];
    }
  else
    {
      m_target_server = -1;
    }
}

MPI_Comm
ServiceCoupling::physics_comm () const noexcept
{
  return m_phys_comm;
}

bool
ServiceCoupling::is_physics () const noexcept
{
  return !m_is_inference;
}

bool
ServiceCoupling::is_inference () const noexcept
{
  return m_is_inference;
}

// ---------------------------------------------------------------------
//  evaluate_async -- submit physics work to the topology-selected ML
//  server rank.
// ---------------------------------------------------------------------
CouplingRequest
ServiceCoupling::evaluate_async (const ParticleBatch       &batch,
                                 Real *__restrict           out,
                                 const InferenceBackend    */*backend*/) const
{
  if (m_is_inference)
    {
      throw std::logic_error ("ServiceCoupling::evaluate_async called "
                              "on an ML server rank");
    }
  return m_transport.evaluate_async (batch, out, m_target_server);
}

void
ServiceCoupling::progress () const
{
  m_transport.progress ();
}

bool
ServiceCoupling::test (CouplingRequest request) const
{
  return m_transport.test (request);
}

void
ServiceCoupling::wait (CouplingRequest request) const
{
  m_transport.wait (request);
}

void
ServiceCoupling::wait_all () const
{
  m_transport.wait_all ();
}

// ---------------------------------------------------------------------
//  evaluate -- blocking compatibility wrapper around the async path.
// ---------------------------------------------------------------------
void
ServiceCoupling::evaluate (const ParticleBatch       &batch,
                           Real *__restrict           out,
                           const InferenceBackend    *backend) const
{
  const CouplingRequest  request = evaluate_async (batch, out, backend);
  wait (request);
}

// ---------------------------------------------------------------------
//  serve -- run the shared ML-server event loop for this rank's quota.
// ---------------------------------------------------------------------
void
ServiceCoupling::serve (const InferenceBackend &backend) const
{
  if (!m_is_inference)
    {
      throw std::logic_error ("ServiceCoupling::serve called on a "
                              "physics rank");
    }
  m_transport.serve (m_expected_shutdowns, backend);
}

// ---------------------------------------------------------------------
//  shutdown -- drain this physics rank's selected ML server.
// ---------------------------------------------------------------------
void
ServiceCoupling::shutdown () const
{
  if (m_already_shutdown)
    {
      return;
    }
  m_already_shutdown = true;

  if (m_is_inference)
    {
      return;
    }

  m_transport.wait_all ();
  m_transport.shutdown (m_target_server);
}

Timings
ServiceCoupling::timings () const noexcept
{
  return m_transport.timings ();
}

}  // namespace mlhpc::detail
