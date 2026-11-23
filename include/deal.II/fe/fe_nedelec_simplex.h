// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2018 - 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

#ifndef dealii_fe_nedelec_simplex_h
#define dealii_fe_nedelec_simplex_h

#include <deal.II/base/config.h>

#include <deal.II/base/derivative_form.h>
#//include <deal.II/base/polynomials_integrated_legendre_sz.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>

DEAL_II_NAMESPACE_OPEN

/**
 * @addtogroup fe
 * @{
 */

/**
 * This class represents an implementation of linear nedelec elements for triangles and tetrahedra.
 *
 */
template <int dim, int spacedim = dim>
class FE_NedelecSimplex : public FiniteElement<dim, dim>
{
public:
  static_assert(dim == spacedim,
                "FE_NedelecSimplex is only implemented for dim==spacedim!");

  /**
   * Constructor for the NedelecSZ element of given @p order. 
   * Currently only order = 0 thus linear polynomials are implemented.
   */
  FE_NedelecSimplex(const unsigned int order);

  virtual UpdateFlags
  requires_update_flags(const UpdateFlags update_flags) const override;

  virtual std::string
  get_name() const override;

  virtual std::unique_ptr<FiniteElement<dim, dim>>
  clone() const override;


protected:
  /**
   * The mapping kind to be used to map shape functions from the reference
   * cell to the mesh cell.
   */
  MappingKind mapping_kind;


  virtual std::unique_ptr<
    typename dealii::FiniteElement<dim, spacedim>::InternalDataBase>
  get_data(
    const UpdateFlags             update_flags,
    const Mapping<dim, spacedim> &mapping,
    const Quadrature<dim>        &quadrature,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;

  /**
   * Compute information about the shape functions on the cell denoted by the
   * first argument. Note that this function must recompute the cell-dependent
   * degrees of freedom, and so is not thread-safe at this time.
   */
  virtual void
  fill_fe_values(
    const typename Triangulation<dim, dim>::cell_iterator &cell,
    const CellSimilarity::Similarity                       cell_similarity,
    const Quadrature<dim>                                 &quadrature,
    const Mapping<dim, dim>                               &mapping,
    const typename Mapping<dim, dim>::InternalDataBase    &mapping_internal,
    const internal::FEValuesImplementation::MappingRelatedData<dim, dim>
                                                             &mapping_data,
    const typename FiniteElement<dim, dim>::InternalDataBase &fedata,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim, dim>
      &data) const override;

  using FiniteElement<dim, spacedim>::fill_fe_face_values;

  /**
   * Compute information about the shape functions on the cell and face denoted
   * by the first two arguments. Note that this function must recompute the
   * cell-dependent degrees of freedom, and so is not thread-safe at this time.
   */
  virtual void
  fill_fe_face_values(
    const typename Triangulation<dim, dim>::cell_iterator &cell,
    const unsigned int                                     face_no,
    const hp::QCollection<dim - 1>                        &quadrature,
    const Mapping<dim, dim>                               &mapping,
    const typename Mapping<dim, dim>::InternalDataBase    &mapping_internal,
    const internal::FEValuesImplementation::MappingRelatedData<dim, dim>
                                                             &mapping_data,
    const typename FiniteElement<dim, dim>::InternalDataBase &fedata,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim, dim>
      &data) const override;

        /**
   * Not implemented.
   */
  virtual void
  fill_fe_subface_values(
    const typename Triangulation<dim, dim>::cell_iterator &cell,
    const unsigned int                                     face_no,
    const unsigned int                                     sub_no,
    const Quadrature<dim - 1>                             &quadrature,
    const Mapping<dim, dim>                               &mapping,
    const typename Mapping<dim, dim>::InternalDataBase    &mapping_internal,
    const internal::FEValuesImplementation::MappingRelatedData<dim, dim>
                                                             &mapping_data,
    const typename FiniteElement<dim, dim>::InternalDataBase &fedata,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim, dim>
      &data) const override;

  class InternalData : public FiniteElement<dim, dim>::InternalDataBase
  {
  public:
    InternalData(std::vector<std::vector<Tensor<1,dim>>> &&values, std::vector<std::vector<Tensor<2,dim>>> &&grads):shape_values(std::move(values)), shape_grads(std::move(grads)){};
    /**
     * Storage for shape functions on the reference element. We only pre-compute
     * the values on the reference cell for the edge dofs and transform mapping and edge_sign for each cell individually
     *
     */
    std::vector<std::vector<Tensor<1, dim>>> shape_values;

    /**
     * Storage for shape function gradients on the reference element. We only
     * the gradients on the reference cell for the edge dofs and transform mapping and edge_sign for each cell individually
     *
     */
    std::vector<std::vector<Tensor<2, dim>>> shape_grads;
  };

private:
  /**
   * Internal function to return a vector of "dofs per object"
   * where the components of the returned vector refer to:
   * 0 = vertex
   * 1 = edge
   * 2 = face (which is a cell in 2d)
   * 3 = cell
   */
  static std::vector<unsigned int>
  get_dpo_vector(const unsigned int degree);

  /**
   * Returns the number of DoFs in the basis set.
   */
  unsigned int
  compute_num_dofs(const unsigned int degree) const;

  /**
   * fills the edge signs
   */
  std::vector<int>
  compute_edge_sign(const typename Triangulation<dim, dim>::cell_iterator &cell) const;

  std::vector<std::vector<Tensor<1, dim>>>
  get_shape_values(const std::vector<Point<dim>> &p_list) const;

  std::vector<std::vector<Tensor<2, dim>>>
  get_shape_grads(const std::vector<Point<dim>> &p_list) const;

};



/** @} */

DEAL_II_NAMESPACE_CLOSE

#endif
