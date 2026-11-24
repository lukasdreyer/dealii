// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_mg_smoother_hiptmair_h
#define dealii_mg_smoother_hiptmair_h

#include <deal.II/base/config.h>

#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/observer_pointer.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_transfer_matrices.h>

DEAL_II_NAMESPACE_OPEN

/**
 * @addtogroup mg
 * @{
 */

/**
 * Hiptmair's hybrid smoother for H(curl) problems such as
 * curl curl u + beta u = f discretized with edge elements (R. Hiptmair,
 * Multigrid method for Maxwell's equations, SIAM J. Numer. Anal. 36 (1998)).
 * A point smoother on the edge operator A alone does not damp the error
 * components in the kernel of the curl (discrete gradients), which are
 * poorly approximated on coarser meshes; the hybrid smoother therefore adds
 * a smoothing step for the gradients in the nodal space: with the discrete
 * gradient G (see DoFTools::make_discrete_gradient()) and the nodal operator
 * G^T A G (see MGTools::make_hiptmair_nodal_matrix()), one step is
 *
 *   u += R_A (f - A u),                          (edge relaxation)
 *   u += G R_S G^T (f - A u),                    (nodal relaxation)
 *   u += R_A (f - A u),                          (edge relaxation)
 *
 * where R_A and R_S are the relaxation operators of @p PreconditionerType
 * (e.g. PreconditionSSOR, PreconditionJacobi) for A and G^T A G. With
 * symmetric relaxations the step is symmetric, so the same smoother can be
 * used for pre- and post-smoothing in a multigrid preconditioner for CG.
 *
 * The matrices are stored by reference and must be kept alive by the user.
 * @p PreconditionerType must have a function initialize(matrix,
 * AdditionalData) and vmult(dst, src).
 */
template <typename VectorType,
          typename MatrixType,
          typename PreconditionerType>
class MGSmootherHiptmair : public MGSmootherBase<VectorType>
{
public:
  /**
   * Parameters: the number of steps (each one consisting of the three
   * relaxations above) and the parameters of the relaxations of A and of
   * G^T A G.
   */
  struct AdditionalData
  {
    AdditionalData(const unsigned int n_steps = 1,
                   const typename PreconditionerType::AdditionalData
                     &edge_relaxation_data =
                       typename PreconditionerType::AdditionalData(),
                   const typename PreconditionerType::AdditionalData
                     &nodal_relaxation_data =
                       typename PreconditionerType::AdditionalData())
      : n_steps(n_steps)
      , edge_relaxation_data(edge_relaxation_data)
      , nodal_relaxation_data(nodal_relaxation_data)
    {}

    unsigned int                                n_steps;
    typename PreconditionerType::AdditionalData edge_relaxation_data;
    typename PreconditionerType::AdditionalData nodal_relaxation_data;
  };

  /**
   * Store the level matrices A (edge space), G (from the nodal to the edge
   * space) and G^T A G (nodal space), and initialize the relaxations.
   */
  void
  initialize(const MGLevelObject<MatrixType> &edge_matrices,
             const MGLevelObject<MatrixType> &gradients,
             const MGLevelObject<MatrixType> &nodal_matrices,
             const AdditionalData            &data = AdditionalData());

  virtual void
  clear() override;

  /**
   * Apply n_steps steps of the smoother to @p u with right-hand side
   * @p rhs on @p level.
   */
  virtual void
  smooth(const unsigned int level,
         VectorType        &u,
         const VectorType  &rhs) const override;

  /**
   * Set @p u to zero and apply smooth().
   */
  virtual void
  apply(const unsigned int level,
        VectorType        &u,
        const VectorType  &rhs) const override;

private:
  /**
   * One relaxation with A: u += R_A (rhs - A u).
   */
  void
  relax_edges(const unsigned int level,
              VectorType        &u,
              const VectorType  &rhs) const;

  /**
   * Reinit @p v as a vector of the nodal space (the domain of the gradient).
   */
  void
  reinit_nodal_vector(const unsigned int level, VectorType &v) const;

  MGLevelObject<ObserverPointer<const MatrixType>> edge_matrices;
  MGLevelObject<ObserverPointer<const MatrixType>> gradients;
  MGLevelObject<ObserverPointer<const MatrixType>> nodal_matrices;
  MGLevelObject<PreconditionerType>                edge_relaxations;
  MGLevelObject<PreconditionerType>                nodal_relaxations;
  unsigned int                                     n_steps = 1;
};



namespace MGTools
{
  /**
   * The nodal operator G^T A G of the Hiptmair smoother (MGSmootherHiptmair)
   * for the edge operator @p edge_matrix and the discrete gradient
   * @p gradient. The rows of nodal DoFs that do not appear in the gradient
   * (constrained vertices, e.g. on Dirichlet boundaries, whose columns of G
   * are empty) get a unit diagonal, so that point relaxations can be
   * applied; their right-hand side G^T r is zero, so they do not change the
   * result. @p sparsity is rebuilt and must be kept alive as long as
   * @p nodal_matrix.
   */
  template <typename number>
  void
  make_hiptmair_nodal_matrix(const SparseMatrix<number> &edge_matrix,
                             const SparseMatrix<number> &gradient,
                             SparsityPattern            &sparsity,
                             SparseMatrix<number>       &nodal_matrix);
} // namespace MGTools

/** @} */


#ifndef DOXYGEN

template <typename VectorType,
          typename MatrixType,
          typename PreconditionerType>
void
MGSmootherHiptmair<VectorType, MatrixType, PreconditionerType>::initialize(
  const MGLevelObject<MatrixType> &edge_matrices_in,
  const MGLevelObject<MatrixType> &gradients_in,
  const MGLevelObject<MatrixType> &nodal_matrices_in,
  const AdditionalData            &data)
{
  const unsigned int min_level = edge_matrices_in.min_level();
  const unsigned int max_level = edge_matrices_in.max_level();
  AssertDimension(gradients_in.min_level(), min_level);
  AssertDimension(gradients_in.max_level(), max_level);
  AssertDimension(nodal_matrices_in.min_level(), min_level);
  AssertDimension(nodal_matrices_in.max_level(), max_level);

  n_steps = data.n_steps;
  edge_matrices.resize(min_level, max_level);
  gradients.resize(min_level, max_level);
  nodal_matrices.resize(min_level, max_level);
  edge_relaxations.resize(min_level, max_level);
  nodal_relaxations.resize(min_level, max_level);
  for (unsigned int level = min_level; level <= max_level; ++level)
    {
      edge_matrices[level]  = &edge_matrices_in[level];
      gradients[level]      = &gradients_in[level];
      nodal_matrices[level] = &nodal_matrices_in[level];
      edge_relaxations[level].initialize(edge_matrices_in[level],
                                         data.edge_relaxation_data);
      nodal_relaxations[level].initialize(nodal_matrices_in[level],
                                          data.nodal_relaxation_data);
    }
}



template <typename VectorType,
          typename MatrixType,
          typename PreconditionerType>
void
MGSmootherHiptmair<VectorType, MatrixType, PreconditionerType>::clear()
{
  edge_matrices.clear_elements();
  gradients.clear_elements();
  nodal_matrices.clear_elements();
  edge_relaxations.clear_elements();
  nodal_relaxations.clear_elements();
}



template <typename VectorType,
          typename MatrixType,
          typename PreconditionerType>
void
MGSmootherHiptmair<VectorType, MatrixType, PreconditionerType>::relax_edges(
  const unsigned int level,
  VectorType        &u,
  const VectorType  &rhs) const
{
  VectorType residual, correction;
  residual.reinit(u, true);
  correction.reinit(u, true);
  edge_matrices[level]->vmult(residual, u);
  residual.sadd(-1., 1., rhs);
  edge_relaxations[level].vmult(correction, residual);
  u += correction;
}



template <typename VectorType,
          typename MatrixType,
          typename PreconditionerType>
void
MGSmootherHiptmair<VectorType, MatrixType, PreconditionerType>::
  reinit_nodal_vector(const unsigned int level, VectorType &v) const
{
  internal::MGTransferMatrices::reinit_vector(*gradients[level], true, v);
}



template <typename VectorType,
          typename MatrixType,
          typename PreconditionerType>
void
MGSmootherHiptmair<VectorType, MatrixType, PreconditionerType>::smooth(
  const unsigned int level,
  VectorType        &u,
  const VectorType  &rhs) const
{
  VectorType residual;
  VectorType nodal_rhs, nodal_correction;
  residual.reinit(u, true);
  reinit_nodal_vector(level, nodal_rhs);
  reinit_nodal_vector(level, nodal_correction);

  for (unsigned int step = 0; step < n_steps; ++step)
    {
      relax_edges(level, u, rhs);

      edge_matrices[level]->vmult(residual, u);
      residual.sadd(-1., 1., rhs);
      gradients[level]->Tvmult(nodal_rhs, residual);
      nodal_relaxations[level].vmult(nodal_correction, nodal_rhs);
      gradients[level]->vmult_add(u, nodal_correction);

      relax_edges(level, u, rhs);
    }
}



template <typename VectorType,
          typename MatrixType,
          typename PreconditionerType>
void
MGSmootherHiptmair<VectorType, MatrixType, PreconditionerType>::apply(
  const unsigned int level,
  VectorType        &u,
  const VectorType  &rhs) const
{
  u = 0;
  smooth(level, u, rhs);
}



namespace MGTools
{
  template <typename number>
  void
  make_hiptmair_nodal_matrix(const SparseMatrix<number> &edge_matrix,
                             const SparseMatrix<number> &gradient,
                             SparsityPattern            &sparsity,
                             SparseMatrix<number>       &nodal_matrix)
  {
    // A G, then G^T (A G); mmult and Tmmult rebuild the sparsity patterns
    SparsityPattern      sparsity_ag;
    SparseMatrix<number> ag(sparsity_ag);
    edge_matrix.mmult(ag, gradient);
    nodal_matrix.reinit(sparsity);
    gradient.Tmmult(nodal_matrix, ag);

    for (unsigned int i = 0; i < nodal_matrix.m(); ++i)
      if (nodal_matrix.diag_element(i) == number())
        nodal_matrix.diag_element(i) = 1;
  }
} // namespace MGTools

#endif // DOXYGEN

DEAL_II_NAMESPACE_CLOSE

#endif
