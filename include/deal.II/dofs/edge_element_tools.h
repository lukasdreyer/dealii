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

#ifndef dealii_dofs_edge_element_tools_h
#define dealii_dofs_edge_element_tools_h

#include <deal.II/base/config.h>

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

#include <map>
#include <memory>
#include <vector>

DEAL_II_NAMESPACE_OPEN

/**
 * Tools for lowest-order edge elements whose line DoFs refer to the
 * direction of the line objects (from line->vertex(0) to line->vertex(1)),
 * such as FE_NedelecSimplex(0): the DoF of a line is the line integral of
 * the tangential component along the line object.
 */
namespace internal
{
  namespace EdgeElementTools
  {
    template <int dim, int spacedim>
    void
    assert_line_oriented_edge_element(const FiniteElement<dim, spacedim> &fe)
    {
      (void)fe;
      Assert(fe.reference_cell().is_simplex() && fe.n_dofs_per_vertex() == 0 &&
               fe.n_dofs_per_line() == 1 &&
               fe.n_dofs_per_cell() == fe.reference_cell().n_lines(),
             ExcMessage("This function is only implemented for lowest-order "
                        "edge elements on simplices whose DoFs refer to the "
                        "direction of the line objects (FE_NedelecSimplex)."));
    }



    /**
     * The entries of the row of the discrete gradient for every locally
     * owned edge DoF that is not constrained: -1 at the DoF of
     * line->vertex(0) and +1 at the DoF of line->vertex(1), with constrained
     * (hanging) vertex DoFs expressed by their constraints.
     */
    template <int dim, int spacedim, typename number>
    std::map<types::global_dof_index,
             std::map<types::global_dof_index, number>>
    discrete_gradient_rows(
      const DoFHandler<dim, spacedim> &edge_dof_handler,
      const dealii::AffineConstraints<number> &edge_constraints,
      const DoFHandler<dim, spacedim> &nodal_dof_handler,
      const dealii::AffineConstraints<number> &nodal_constraints)
    {
      assert_line_oriented_edge_element(edge_dof_handler.get_fe());
      Assert(nodal_dof_handler.get_fe().n_dofs_per_vertex() == 1 &&
               nodal_dof_handler.get_fe().n_dofs_per_cell() ==
                 nodal_dof_handler.get_fe().reference_cell().n_vertices(),
             ExcMessage("The nodal element must have exactly one DoF per "
                        "vertex and no others (FE_SimplexP(1))."));
      Assert(&edge_dof_handler.get_triangulation() ==
               &nodal_dof_handler.get_triangulation(),
             ExcMessage("Both DoFHandlers must use the same triangulation."));

      const auto &tria  = edge_dof_handler.get_triangulation();
      const auto &owned = edge_dof_handler.locally_owned_dofs();

      std::map<types::global_dof_index,
               std::map<types::global_dof_index, number>>
        rows;
      for (const auto &cell : edge_dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          for (unsigned int l = 0; l < cell->n_lines(); ++l)
            {
              const auto                    line = cell->line(l);
              const types::global_dof_index edge_dof =
                line->dof_index(0, cell->active_fe_index());
              if (owned.is_element(edge_dof) == false ||
                  edge_constraints.is_constrained(edge_dof) ||
                  rows.find(edge_dof) != rows.end())
                continue;

              auto &row = rows[edge_dof];
              const typename DoFHandler<dim, spacedim>::line_iterator
                nodal_line(&tria, 0, line->index(), &nodal_dof_handler);
              for (unsigned int v = 0; v < 2; ++v)
                {
                  const number sign = (v == 0) ? -1 : 1;
                  const types::global_dof_index vertex_dof =
                    nodal_line->vertex_dof_index(v, 0);
                  if (nodal_constraints.is_constrained(vertex_dof))
                    {
                      for (const auto &[dof, weight] :
                           *nodal_constraints.get_constraint_entries(
                             vertex_dof))
                        row[dof] += sign * weight;
                    }
                  else
                    row[vertex_dof] += sign;
                }
            }
      return rows;
    }
  } // namespace EdgeElementTools
} // namespace internal



namespace DoFTools
{
  /**
   * Sparsity pattern of the discrete gradient computed by
   * make_discrete_gradient(): rows are the DoFs of @p edge_dof_handler,
   * columns those of @p nodal_dof_handler. Only the rows of locally owned
   * edge DoFs are added.
   */
  template <int dim, int spacedim, typename number, typename SparsityPatternType>
  void
  make_discrete_gradient_sparsity_pattern(
    const DoFHandler<dim, spacedim> &edge_dof_handler,
    const AffineConstraints<number> &edge_constraints,
    const DoFHandler<dim, spacedim> &nodal_dof_handler,
    const AffineConstraints<number> &nodal_constraints,
    SparsityPatternType             &sparsity)
  {
    for (const auto &[row, entries] :
         dealii::internal::EdgeElementTools::discrete_gradient_rows(edge_dof_handler,
                                                            edge_constraints,
                                                            nodal_dof_handler,
                                                            nodal_constraints))
      for (const auto &entry : entries)
        sparsity.add(row, entry.first);
  }



  /**
   * The discrete gradient G from the nodal space of @p nodal_dof_handler
   * (FE_SimplexP(1)) to the edge element space of @p edge_dof_handler
   * (FE_NedelecSimplex(0)): for a nodal vector p, G p is the vector of edge
   * DoFs of grad p_h. Since the edge DoFs are line integrals along the line
   * objects, the row of an edge DoF has -1 at the DoF of line->vertex(0) and
   * +1 at the DoF of line->vertex(1). Hanging nodes are taken into account
   * on both sides: constrained vertex DoFs are expressed by their
   * constraints (@p nodal_constraints must contain the locally relevant
   * ones), and the rows of constrained edge DoFs (hanging or Dirichlet) are
   * zero, since these DoFs are not unknowns. This is the matrix required by
   * auxiliary space preconditioners like Hypre's AMS.
   *
   * @p matrix must have been initialized with a sparsity pattern containing
   * the one of make_discrete_gradient_sparsity_pattern(). Only the rows of
   * locally owned edge DoFs are set.
   */
  template <int dim, int spacedim, typename number, typename MatrixType>
  void
  make_discrete_gradient(const DoFHandler<dim, spacedim> &edge_dof_handler,
                         const AffineConstraints<number> &edge_constraints,
                         const DoFHandler<dim, spacedim> &nodal_dof_handler,
                         const AffineConstraints<number> &nodal_constraints,
                         MatrixType                      &matrix)
  {
    std::vector<types::global_dof_index> columns;
    std::vector<number>                  values;
    for (const auto &[row, entries] :
         dealii::internal::EdgeElementTools::discrete_gradient_rows(edge_dof_handler,
                                                            edge_constraints,
                                                            nodal_dof_handler,
                                                            nodal_constraints))
      {
        columns.clear();
        values.clear();
        for (const auto &[column, value] : entries)
          {
            columns.push_back(column);
            values.push_back(value);
          }
        matrix.set(row, columns, values);
      }
    matrix.compress(VectorOperation::insert);
  }
} // namespace DoFTools



namespace VectorTools
{
  /**
   * Tangential boundary values for lowest-order edge elements
   * (FE_NedelecSimplex(0)): for every line of a boundary face whose boundary
   * id is in @p function_map, the DoF is set to the line integral of the
   * tangential component of the function along the line object (components
   * @p first_vector_component to @p first_vector_component + dim - 1). This
   * is the canonical interpolation, so that n x u = n x g holds exactly for
   * functions g in the edge element space. The integral is computed with
   * Gauss points on the reference line of the cell, mapped by @p mapping,
   * so that curved boundaries are taken into account. Lines that are
   * already constrained (e.g. hanging lines) are not changed, so the hanging
   * node constraints should be computed first.
   */
  template <int dim, int spacedim, typename number>
  void
  interpolate_boundary_values_curl_conforming(
    const Mapping<dim, spacedim>    &mapping,
    const DoFHandler<dim, spacedim> &dof_handler,
    const std::map<types::boundary_id, const Function<spacedim, number> *>
                              &function_map,
    AffineConstraints<number> &constraints,
    const unsigned int         first_vector_component = 0)
  {
    static_assert(dim > 1, "Edge elements need at least two dimensions.");
    const FiniteElement<dim, spacedim> &fe = dof_handler.get_fe();
    dealii::internal::EdgeElementTools::assert_line_oriented_edge_element(fe);

    const ReferenceCell ref_cell = fe.reference_cell();
    const QGauss<1>     q_line(4);

    // one FEValues object per line of the reference cell, with the Gauss
    // points on that line
    std::vector<std::unique_ptr<FEValues<dim, spacedim>>> line_values;
    std::vector<Tensor<1, dim>>                           reference_tangents;
    for (unsigned int l = 0; l < ref_cell.n_lines(); ++l)
      {
        const Point<dim> v0 =
          ref_cell.template vertex<dim>(ref_cell.line_to_cell_vertices(l, 0));
        const Point<dim> v1 =
          ref_cell.template vertex<dim>(ref_cell.line_to_cell_vertices(l, 1));
        std::vector<Point<dim>> points;
        for (unsigned int q = 0; q < q_line.size(); ++q)
          points.push_back(v0 + q_line.point(q)[0] * (v1 - v0));
        line_values.push_back(std::make_unique<FEValues<dim, spacedim>>(
          mapping,
          fe,
          Quadrature<dim>(points),
          update_quadrature_points | update_jacobians));
        reference_tangents.push_back(v1 - v0);
      }

    Vector<number> g;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_artificial())
          continue;
        for (const unsigned int f : cell->face_indices())
          {
            if (cell->at_boundary(f) == false)
              continue;
            const auto function = function_map.find(cell->face(f)->boundary_id());
            if (function == function_map.end())
              continue;
            Assert(function->second->n_components >=
                     first_vector_component + spacedim,
                   ExcMessage("The boundary function has too few components."));
            g.reinit(function->second->n_components);

            for (unsigned int l = 0; l < cell->n_lines(); ++l)
              {
                // is line l of the cell a line of face f?
                bool on_face = false;
                if constexpr (dim == 2)
                  on_face = (cell->line_index(l) == cell->face(f)->index());
                else
                  for (const unsigned int k : cell->face(f)->line_indices())
                    on_face |=
                      (cell->line_index(l) == cell->face(f)->line_index(k));
                if (on_face == false)
                  continue;

                const types::global_dof_index dof =
                  cell->line(l)->dof_index(0, cell->active_fe_index());
                if (constraints.is_constrained(dof) ||
                    constraints.can_store_line(dof) == false)
                  continue;

                line_values[l]->reinit(cell);
                number integral = 0;
                for (unsigned int q = 0; q < q_line.size(); ++q)
                  {
                    // derivative of the mapped line with respect to the
                    // line parameter
                    const Tensor<1, spacedim> tangent =
                      apply_transformation(line_values[l]->jacobian(q),
                                           reference_tangents[l]);
                    function->second->vector_value(
                      line_values[l]->quadrature_point(q), g);
                    for (unsigned int d = 0; d < spacedim; ++d)
                      integral += q_line.weight(q) *
                                  g(first_vector_component + d) * tangent[d];
                  }
                // the DoF refers to the direction of the line object
                if (cell->line_orientation(l) !=
                    numbers::default_geometric_orientation)
                  integral = -integral;
                constraints.add_constraint(dof, {}, integral);
              }
          }
      }
  }
} // namespace VectorTools

DEAL_II_NAMESPACE_CLOSE

#endif
