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

#ifndef dealii_mg_transfer_matrices_h
#define dealii_mg_transfer_matrices_h

#include <deal.II/base/config.h>

#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/observer_pointer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <deal.II/multigrid/mg_base.h>

#include <type_traits>

DEAL_II_NAMESPACE_OPEN

namespace internal
{
  namespace MGTransferMatrices
  {
    /**
     * Reinit @p v as a vector of the domain (@p domain = true) or the range
     * of @p matrix, for serial matrices (Vector) and distributed ones with
     * locally_owned_domain_indices() and get_mpi_communicator() (Trilinos,
     * PETSc).
     */
    template <typename VectorType, typename MatrixType>
    void
    reinit_vector(const MatrixType &matrix, const bool domain, VectorType &v)
    {
      if constexpr (std::is_same_v<VectorType,
                                   Vector<typename VectorType::value_type>>)
        v.reinit(domain ? matrix.n() : matrix.m());
      else
        v.reinit(domain ? matrix.locally_owned_domain_indices() :
                          matrix.locally_owned_range_indices(),
                 matrix.get_mpi_communicator());
    }
  } // namespace MGTransferMatrices
} // namespace internal



/**
 * @addtogroup mg
 * @{
 */

/**
 * Multigrid transfer between the levels of a sequence of meshes (global
 * coarsening) by given prolongation matrices: prolongations[l] maps the
 * vectors of level l - 1 to those of level l (the entry of the minimal level
 * is not used), and the restriction is its transpose. This works with any
 * element for which the prolongation matrices are available, e.g. the
 * lowest-order edge elements with
 * DoFTools::make_edge_element_prolongation(), whose rows of constrained fine
 * DoFs are zero and whose constrained coarse DoFs are expressed by their
 * constraints, so the vectors of all levels contain only the unknowns
 * (zero entries for constrained DoFs).
 *
 * The level vectors get the partition of the domain and range of the
 * matrices. With copy_to_mg() and copy_from_mg(), the class can be used in
 * PreconditionMG; the vector of the finest level is the global vector.
 *
 * The matrices are stored by reference and must be kept alive by the user.
 */
template <typename VectorType, typename MatrixType>
class MGTransferMatrices : public MGTransferBase<VectorType>
{
public:
  /**
   * Store the prolongation matrices.
   */
  void
  initialize(const MGLevelObject<MatrixType> &prolongations);

  virtual void
  prolongate(const unsigned int to_level,
             VectorType        &dst,
             const VectorType  &src) const override;

  virtual void
  prolongate_and_add(const unsigned int to_level,
                     VectorType        &dst,
                     const VectorType  &src) const override;

  virtual void
  restrict_and_add(const unsigned int from_level,
                   VectorType        &dst,
                   const VectorType  &src) const override;

  /**
   * Initialize the vectors of all levels (zero) and copy @p src to the
   * finest one. The DoFHandler is not used.
   */
  template <int dim, typename OtherVectorType, int spacedim>
  void
  copy_to_mg(const DoFHandler<dim, spacedim> &dof_handler,
             MGLevelObject<VectorType>       &dst,
             const OtherVectorType           &src) const;

  /**
   * Copy the vector of the finest level to @p dst.
   */
  template <int dim, typename OtherVectorType, int spacedim>
  void
  copy_from_mg(const DoFHandler<dim, spacedim> &dof_handler,
               OtherVectorType                 &dst,
               const MGLevelObject<VectorType> &src) const;

  /**
   * Add the vector of the finest level to @p dst.
   */
  template <int dim, typename OtherVectorType, int spacedim>
  void
  copy_from_mg_add(const DoFHandler<dim, spacedim> &dof_handler,
                   OtherVectorType                 &dst,
                   const MGLevelObject<VectorType> &src) const;

private:
  MGLevelObject<ObserverPointer<const MatrixType>> prolongations;
};

/** @} */


#ifndef DOXYGEN

template <typename VectorType, typename MatrixType>
void
MGTransferMatrices<VectorType, MatrixType>::initialize(
  const MGLevelObject<MatrixType> &prolongations_in)
{
  prolongations.resize(prolongations_in.min_level(),
                       prolongations_in.max_level());
  for (unsigned int level = prolongations_in.min_level() + 1;
       level <= prolongations_in.max_level();
       ++level)
    prolongations[level] = &prolongations_in[level];
}



template <typename VectorType, typename MatrixType>
void
MGTransferMatrices<VectorType, MatrixType>::prolongate(
  const unsigned int to_level,
  VectorType        &dst,
  const VectorType  &src) const
{
  prolongations[to_level]->vmult(dst, src);
}



template <typename VectorType, typename MatrixType>
void
MGTransferMatrices<VectorType, MatrixType>::prolongate_and_add(
  const unsigned int to_level,
  VectorType        &dst,
  const VectorType  &src) const
{
  prolongations[to_level]->vmult_add(dst, src);
}



template <typename VectorType, typename MatrixType>
void
MGTransferMatrices<VectorType, MatrixType>::restrict_and_add(
  const unsigned int from_level,
  VectorType        &dst,
  const VectorType  &src) const
{
  prolongations[from_level]->Tvmult_add(dst, src);
}



template <typename VectorType, typename MatrixType>
template <int dim, typename OtherVectorType, int spacedim>
void
MGTransferMatrices<VectorType, MatrixType>::copy_to_mg(
  const DoFHandler<dim, spacedim> &,
  MGLevelObject<VectorType> &dst,
  const OtherVectorType     &src) const
{
  const unsigned int min_level = prolongations.min_level();
  const unsigned int max_level = prolongations.max_level();
  dst.resize(min_level, max_level);
  for (unsigned int level = min_level; level < max_level; ++level)
    internal::MGTransferMatrices::reinit_vector(*prolongations[level + 1],
                                                true,
                                                dst[level]);
  if (max_level > min_level)
    internal::MGTransferMatrices::reinit_vector(*prolongations[max_level],
                                                false,
                                                dst[max_level]);
  dst[max_level] = src;
}



template <typename VectorType, typename MatrixType>
template <int dim, typename OtherVectorType, int spacedim>
void
MGTransferMatrices<VectorType, MatrixType>::copy_from_mg(
  const DoFHandler<dim, spacedim> &,
  OtherVectorType                 &dst,
  const MGLevelObject<VectorType> &src) const
{
  dst = src[src.max_level()];
}



template <typename VectorType, typename MatrixType>
template <int dim, typename OtherVectorType, int spacedim>
void
MGTransferMatrices<VectorType, MatrixType>::copy_from_mg_add(
  const DoFHandler<dim, spacedim> &,
  OtherVectorType                 &dst,
  const MGLevelObject<VectorType> &src) const
{
  dst += src[src.max_level()];
}

#endif // DOXYGEN

DEAL_II_NAMESPACE_CLOSE

#endif
