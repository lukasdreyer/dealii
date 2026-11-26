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

#ifndef dealii_fe_nedelec_lowest_order_h
#define dealii_fe_nedelec_lowest_order_h

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
 * Base class of the lowest-order H(curl)-conforming Nédélec elements with one
 * DoF per line (FE_NedelecSimplex, FE_NedelecWedge). The DoF of a line is the
 * line integral of the tangential component in the direction of the line
 * object (from line->vertex(0) to line->vertex(1)): the shape function of
 * line l of the reference cell is directed from line_to_cell_vertices(l, 0)
 * to line_to_cell_vertices(l, 1) and gets the sign of
 * cell->line_orientation(l). This class implements everything that does not
 * depend on the shape of the cell (covariant mapping, signs, cell and face
 * values); derived classes provide the shape functions on their reference
 * cell.
 */
template <int dim, int spacedim = dim>
class FE_NedelecLowestOrderBase : public FiniteElement<dim, dim>
{
public:
  static_assert(dim == spacedim,
                "FE_NedelecLowestOrderBase is only implemented for "
                "dim==spacedim!");

  virtual UpdateFlags
  requires_update_flags(const UpdateFlags update_flags) const override;

  /**
   * Component @p component of the reference shape function @p i at the
   * point @p p of the reference cell (directed as described at
   * get_shape_values(), i.e. without the sign of a mesh cell).
   */
  virtual double
  shape_value_component(const unsigned int i,
                        const Point<dim>  &p,
                        const unsigned int component) const override;

  /**
   * The values of all reference shape functions at all points @p points of
   * the reference cell, indexed [shape function][point] (directed as
   * described at get_shape_values()). Much cheaper than
   * shape_value_component() for many functions and points, since all
   * functions are computed together.
   */
  std::vector<std::vector<Tensor<1, dim>>>
  reference_shape_values(const std::vector<Point<dim>> &points) const
  {
    return get_shape_values(points);
  }

  /**
   * With another lowest-order edge element derived from this class (on
   * another reference cell), the single DoF of a line is identical, since
   * both refer to the direction of the line object. This makes the DoFs of
   * hybrid meshes (e.g. FE_NedelecSimplex and FE_NedelecPyramid) conforming.
   */
  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_line_dof_identities(
    const FiniteElement<dim, spacedim> &fe_other) const override;

  /**
   * There are no DoFs on vertices.
   */
  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_vertex_dof_identities(
    const FiniteElement<dim, spacedim> &fe_other) const override;

  /**
   * There are no DoFs on quads.
   */
  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_quad_dof_identities(const FiniteElement<dim, spacedim> &fe_other,
                         const unsigned int face_no = 0) const override;

  /**
   * Lowest-order edge elements on different reference cells have the same
   * traces, so either can dominate.
   */
  virtual FiniteElementDomination::Domination
  compare_for_domination(const FiniteElement<dim, spacedim> &fe_other,
                         const unsigned int codim = 0) const override;

protected:
  /**
   * Constructor: one DoF per line of @p reference_cell, whose shape
   * functions are given by get_shape_values() and get_shape_grads() of the
   * derived class. Only @p order = 0 is implemented.
   */
  FE_NedelecLowestOrderBase(const ReferenceCell &reference_cell,
                            const unsigned int   order);

  /**
   * The mapping kind to be used to map shape functions from the reference
   * cell to the mesh cell.
   */
  MappingKind mapping_kind;

  /**
   * Values of the shape functions at the points @p p_list of the reference
   * cell, indexed [shape function][point]. Shape function l belongs to line
   * l of the reference cell and is directed from
   * line_to_cell_vertices(l, 0) to line_to_cell_vertices(l, 1).
   */
  virtual std::vector<std::vector<Tensor<1, dim>>>
  get_shape_values(const std::vector<Point<dim>> &p_list) const = 0;

  /**
   * Gradients of the shape functions, indexed [shape function][point] and
   * [component][direction].
   */
  virtual std::vector<std::vector<Tensor<2, dim>>>
  get_shape_grads(const std::vector<Point<dim>> &p_list) const = 0;

  /**
   * Hessians of the shape functions, indexed [shape function][point] and
   * [component][direction][direction].
   */
  virtual std::vector<std::vector<Tensor<3, dim>>>
  get_shape_hessians(const std::vector<Point<dim>> &p_list) const = 0;


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

  using FiniteElement<dim, dim>::fill_fe_subface_values;

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
    InternalData(std::vector<std::vector<Tensor<1, dim>>> &&values,
                 std::vector<std::vector<Tensor<2, dim>>> &&grads,
                 std::vector<std::vector<Tensor<3, dim>>> &&hessians)
      : shape_values(std::move(values))
      , shape_grads(std::move(grads))
      , shape_hessians(std::move(hessians))
    {}
    /**
     * Storage for shape functions on the reference element. We only pre-compute
     * the values on the reference cell for the edge dofs and transform mapping
     * and edge_sign for each cell individually
     *
     */
    std::vector<std::vector<Tensor<1, dim>>> shape_values;

    /**
     * Storage for shape function gradients on the reference element. We only
     * the gradients on the reference cell for the edge dofs and transform
     * mapping and edge_sign for each cell individually
     *
     */
    std::vector<std::vector<Tensor<2, dim>>> shape_grads;

    /**
     * Storage for the Hessians of the shape functions on the reference
     * element.
     */
    std::vector<std::vector<Tensor<3, dim>>> shape_hessians;
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
   * fills the edge signs
   */
  std::vector<int>
  compute_edge_sign(
    const typename Triangulation<dim, dim>::cell_iterator &cell) const;

  /**
   * Transform the values, gradients and Hessians on the reference cell at
   * the points offset, ..., offset + n_q_points - 1 of the quadrature of
   * get_data() to the mesh cell (covariant transformation and its
   * derivatives, multiplied by the signs of the lines). Used for cells and
   * faces.
   */
  void
  transform_shape_data(
    const std::vector<int>                             &edge_sign,
    const unsigned int                                  offset,
    const unsigned int                                  n_q_points,
    const Mapping<dim, dim>                            &mapping,
    const typename Mapping<dim, dim>::InternalDataBase &mapping_internal,
    const internal::FEValuesImplementation::MappingRelatedData<dim, dim>
                       &mapping_data,
    const InternalData &fe_data,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim, dim>
      &data) const;
};



/** @} */

DEAL_II_NAMESPACE_CLOSE

#endif
