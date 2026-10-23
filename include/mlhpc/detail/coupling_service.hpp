// -*- C++ -*-
// ---------------------------------------------------------------------
//  mlhpc/detail/coupling_service.hpp -- generalized service-topology
//  Coupling implementation shared by the old B/C build labels.
//
//  The class separates topology from transport.  It decides which ranks
//  are physics ranks, which ranks are ML servers, and which server each
//  physics rank targets.  All MPI message exchange is delegated to
//  AsyncMpiTransport.
// ---------------------------------------------------------------------

#ifndef MLHPC_DETAIL_COUPLING_SERVICE_HPP
#define MLHPC_DETAIL_COUPLING_SERVICE_HPP

#include "mlhpc/coupling.hpp"
#include "mlhpc/detail/coupling_async_mpi.hpp"
#include "mlhpc/inference.hpp"
#include "mlhpc/particle_batch.hpp"
#include "mlhpc/types.hpp"

#include <mpi.h>

namespace mlhpc::detail
{

class ServiceCoupling final : public Coupling
{
public:

  // -------------------------------------------------------------------
  //  Constructor.
  //
  //  Inputs:  world -- global communicator for the coupling experiment.
  //           cfg   -- runtime transport/topology knobs.  ml_server_nodes
  //                    controls server placement (0: one server rank per
  //                    shared-memory node; N>0: N whole server nodes).
  // -------------------------------------------------------------------
  ServiceCoupling (MPI_Comm                world,
                   const CouplingConfig   &cfg);

  ~ServiceCoupling () override;

  MPI_Comm
  physics_comm () const noexcept override;

  bool
  is_physics () const noexcept override;

  bool
  is_inference () const noexcept override;

  CouplingRequest
  evaluate_async (const ParticleBatch       &batch,
                  Real *__restrict           out,
                  const InferenceBackend    *backend) const override;

  void
  progress () const override;

  bool
  test (CouplingRequest request) const override;

  void
  wait (CouplingRequest request) const override;

  void
  wait_all () const override;

  void
  evaluate (const ParticleBatch       &batch,
            Real *__restrict           out,
            const InferenceBackend    *backend) const override;

  void
  serve (const InferenceBackend &backend) const override;

  void
  shutdown () const override;

  Timings
  timings () const noexcept override;

private:

  void
  build_topology (const CouplingConfig   &cfg);

  void
  build_local_rank_per_node_topology (const int *rank_node_index);

  void
  build_server_node_topology (int          server_nodes,
                              const int   *rank_node_index);

private:

                  MPI_Comm             m_world;
                  MPI_Comm             m_node_comm;
                  MPI_Comm             m_phys_comm;
  mutable         AsyncMpiTransport    m_transport;

                  int                  m_world_rank;
                  int                  m_world_size;
                  int                  m_node_rank;
                  int                  m_node_size;
                  int                  m_node_index;
                  int                  m_node_count;
                  int                  m_target_server;
                  int                  m_expected_shutdowns;
                  bool                 m_is_inference;

  mutable         bool                 m_already_shutdown;
};

}  // namespace mlhpc::detail

#endif  // MLHPC_DETAIL_COUPLING_SERVICE_HPP
