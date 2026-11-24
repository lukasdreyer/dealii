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
#include <deal.II/fe/fe_nedelec_lowest_order.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

#include <map>
#include <memory>
#include <vector>

DEAL_II_NAMESPACE_OPEN

/**
 * Tools for lowest-order edge elements whose line DoFs refer to the
 * direction of the line objects (from line->vertex(0) to line->vertex(1)),
 * (the elements derived from FE_NedelecLowestOrderBase: FE_NedelecSimplex,
 * FE_NedelecWedge, FE_NedelecPyramid, FE_NedelecHypercube): the DoF of a
 * line is the line integral of
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
      Assert((dynamic_cast<const FE_NedelecLowestOrderBase<dim, spacedim> *>(
                &fe) != nullptr),
             ExcMessage("This function is only implemented for the "
                        "lowest-order edge elements derived from "
                        "FE_NedelecLowestOrderBase, whose DoFs refer to the "
                        "direction of the line objects."));
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
                    nodal_line->vertex_dof_index(
                      v, 0, nodal_line->nth_active_fe_index(0));
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



    /**
     * The rows of the prolongation from the edge element space of
     * @p coarse_dof_handler to the one of @p fine_dof_handler, for every
     * locally owned fine DoF: the DoF of a fine line is the line integral of
     * the coarse function along it (in the direction of the fine line
     * object). Every active coarse cell must either be active in the fine
     * triangulation or be refined once there, as for the meshes of
     * MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence().
     *
     * Since the covariant transformation keeps line integrals, the integral
     * is computed in the reference coordinates of the coarse cell. The
     * reference positions of the fine vertices follow from the topology:
     * the vertices of the coarse cell, the midpoints of its lines, the
     * centers of its refined quadrilateral faces and (hexahedra) its center.
     * As for the hanging node constraints, this assumes that the children
     * are the images of the reference children (straight lines, flat
     * faces). Fine lines inside an unrefined coarse cell get the identity
     * row.
     *
     * Constrained coarse DoFs (hanging or Dirichlet) are expressed by their
     * constraints, and the rows of constrained fine DoFs are zero, so that
     * the matrix maps the unknowns of the coarse space to those of the fine
     * one. @p coarse_constraints must contain the constraints of all coarse
     * DoFs of the cells that are used.
     */
    template <int dim, int spacedim, typename number>
    std::map<types::global_dof_index,
             std::map<types::global_dof_index, number>>
    prolongation_rows(
      const DoFHandler<dim, spacedim>         &coarse_dof_handler,
      const dealii::AffineConstraints<number> &coarse_constraints,
      const DoFHandler<dim, spacedim>         &fine_dof_handler,
      const dealii::AffineConstraints<number> &fine_constraints)
    {
      const auto &fine_tria = fine_dof_handler.get_triangulation();
      const auto &owned     = fine_dof_handler.locally_owned_dofs();

      // the coarse functions are polynomial along the fine lines on
      // simplices, wedges and hexahedra (degree at most 2), and rational on
      // pyramids, whose pole (the plane of the apex) is at least half a
      // coarse cell away from the lines of the children that do not end at
      // the apex
      const QGauss<1> polynomial_line_quadrature(2);
      const QGauss<1> rational_line_quadrature(10);

      std::map<types::global_dof_index,
               std::map<types::global_dof_index, number>>
        rows;

      std::map<unsigned int, Point<dim>> reference_points;
      std::vector<types::global_dof_index> coarse_dofs;
      std::vector<typename DoFHandler<dim, spacedim>::active_cell_iterator>
        fine_cells;

      for (const auto &coarse_cell : coarse_dof_handler.active_cell_iterators())
        {
          if (coarse_cell->is_artificial() ||
              fine_tria.contains_cell(coarse_cell->id()) == false)
            continue;

          const auto fine_tria_cell =
            fine_tria.create_cell_iterator(coarse_cell->id());
          const typename DoFHandler<dim, spacedim>::cell_iterator fine_cell(
            &fine_tria,
            fine_tria_cell->level(),
            fine_tria_cell->index(),
            &fine_dof_handler);

          fine_cells.clear();
          if (fine_cell->is_active())
            fine_cells.push_back(fine_cell);
          else
            for (unsigned int c = 0; c < fine_cell->n_children(); ++c)
              {
                Assert(fine_cell->child(c)->is_active(),
                       ExcMessage("The fine triangulation must refine every "
                                  "active coarse cell at most once."));
                fine_cells.push_back(fine_cell->child(c));
              }

          const auto &fe =
            dynamic_cast<const FE_NedelecLowestOrderBase<dim, spacedim> &>(
              coarse_cell->get_fe());
          const QGauss<1> &line_quadrature =
            (coarse_cell->reference_cell() == ReferenceCells::Pyramid) ?
              rational_line_quadrature :
              polynomial_line_quadrature;
          coarse_dofs.resize(fe.n_dofs_per_cell());
          coarse_cell->get_dof_indices(coarse_dofs);

          // reference positions of the vertices of the fine cells, keyed by
          // their vertex index in the fine triangulation (the fine cell has
          // the vertex numbering of the coarse cell)
          const ReferenceCell reference_cell = coarse_cell->reference_cell();
          reference_points.clear();
          for (const unsigned int v : reference_cell.vertex_indices())
            reference_points[fine_cell->vertex_index(v)] =
              reference_cell.template vertex<dim>(v);

          const auto reference_point = [&](const unsigned int vertex_index) {
            const auto it = reference_points.find(vertex_index);
            Assert(it != reference_points.end(), ExcInternalError());
            return it->second;
          };

          if (fine_cell->has_children())
            {
              for (unsigned int l = 0; l < fine_cell->n_lines(); ++l)
                {
                  const auto line = fine_cell->line(l);
                  Assert(line->has_children(), ExcInternalError());
                  reference_points[line->child(0)->vertex_index(1)] =
                    0.5 * (reference_point(line->vertex_index(0)) +
                           reference_point(line->vertex_index(1)));
                }

              // remaining new vertices: centers of refined quadrilateral
              // faces, then the center of the cell
              const auto add_centers = [&](const auto &object,
                                           const auto &children) {
                Point<dim> center;
                for (unsigned int v = 0; v < object->n_vertices(); ++v)
                  center += reference_point(object->vertex_index(v));
                center /= object->n_vertices();
                for (const auto &child : children)
                  for (unsigned int v = 0; v < child->n_vertices(); ++v)
                    reference_points.emplace(child->vertex_index(v), center);
              };

              if constexpr (dim == 3)
                for (const unsigned int f : fine_cell->face_indices())
                  {
                    const auto face = fine_cell->face(f);
                    if (face->has_children())
                      {
                        std::vector<
                          typename DoFHandler<dim, spacedim>::face_iterator>
                          children;
                        for (unsigned int c = 0; c < face->n_children(); ++c)
                          children.push_back(face->child(c));
                        add_centers(face, children);
                      }
                  }
              add_centers(fine_cell, fine_cells);
            }

          for (const auto &child : fine_cells)
            {
              if (child->is_artificial())
                continue;
              for (unsigned int l = 0; l < child->n_lines(); ++l)
                {
                  const auto                    line = child->line(l);
                  const types::global_dof_index fine_dof =
                    line->dof_index(0, child->active_fe_index());
                  if (owned.is_element(fine_dof) == false ||
                      rows.find(fine_dof) != rows.end())
                    continue;

                  auto &row = rows[fine_dof];
                  if (fine_constraints.is_constrained(fine_dof))
                    continue;

                  const Point<dim> start =
                    reference_point(line->vertex_index(0));
                  const Tensor<1, dim> tangent =
                    reference_point(line->vertex_index(1)) - start;

                  for (unsigned int L = 0; L < coarse_cell->n_lines(); ++L)
                    {
                      double value = 0;
                      for (unsigned int q = 0; q < line_quadrature.size(); ++q)
                        {
                          const Point<dim> x =
                            start + line_quadrature.point(q)[0] * tangent;
                          for (unsigned int d = 0; d < dim; ++d)
                            value += line_quadrature.weight(q) *
                                     fe.shape_value_component(L, x, d) *
                                     tangent[d];
                        }
                      if (coarse_cell->line_orientation(L) !=
                          numbers::default_geometric_orientation)
                        value = -value;
                      if (std::abs(value) < 1e-12)
                        continue;

                      const types::global_dof_index coarse_dof =
                        coarse_dofs[L];
                      if (coarse_constraints.is_constrained(coarse_dof))
                        {
                          for (const auto &[dof, weight] :
                               *coarse_constraints.get_constraint_entries(
                                 coarse_dof))
                            row[dof] += value * weight;
                        }
                      else
                        row[coarse_dof] += value;
                    }
                }
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



  /**
   * Sparsity pattern of the prolongation computed by
   * make_edge_element_prolongation(): rows are the DoFs of
   * @p fine_dof_handler, columns those of @p coarse_dof_handler. Only the
   * rows of locally owned fine DoFs are added.
   */
  template <int dim, int spacedim, typename number, typename SparsityPatternType>
  void
  make_edge_element_prolongation_sparsity_pattern(
    const DoFHandler<dim, spacedim> &coarse_dof_handler,
    const AffineConstraints<number> &coarse_constraints,
    const DoFHandler<dim, spacedim> &fine_dof_handler,
    const AffineConstraints<number> &fine_constraints,
    SparsityPatternType             &sparsity)
  {
    for (const auto &[row, entries] :
         dealii::internal::EdgeElementTools::prolongation_rows(
           coarse_dof_handler,
           coarse_constraints,
           fine_dof_handler,
           fine_constraints))
      for (const auto &entry : entries)
        sparsity.add(row, entry.first);
  }



  /**
   * The prolongation P from the lowest-order edge element space of
   * @p coarse_dof_handler to the one of @p fine_dof_handler, for geometric
   * multigrid with a sequence of meshes in which every active coarse cell
   * is active or refined once in the fine mesh. The fine DoFs of P u are the
   * line integrals of the coarse function u along the fine lines, so P is
   * the embedding of the coarse space where the spaces are nested
   * (tetrahedra, wedges, hexahedra) and the canonical interpolation
   * otherwise (pyramids, whose children include tetrahedra). Constrained
   * coarse DoFs are expressed by their constraints, and the rows of
   * constrained fine DoFs are zero, so P maps the unknowns of the coarse
   * space to those of the fine one; the restriction is P^T. See
   * internal::EdgeElementTools::prolongation_rows() for the details.
   *
   * @p matrix must have been initialized with a sparsity pattern containing
   * the one of make_edge_element_prolongation_sparsity_pattern(). Only the
   * rows of locally owned fine DoFs are set.
   */
  template <int dim, int spacedim, typename number, typename MatrixType>
  void
  make_edge_element_prolongation(
    const DoFHandler<dim, spacedim> &coarse_dof_handler,
    const AffineConstraints<number> &coarse_constraints,
    const DoFHandler<dim, spacedim> &fine_dof_handler,
    const AffineConstraints<number> &fine_constraints,
    MatrixType                      &matrix)
  {
    std::vector<types::global_dof_index> columns;
    std::vector<number>                  values;
    for (const auto &[row, entries] :
         dealii::internal::EdgeElementTools::prolongation_rows(
           coarse_dof_handler,
           coarse_constraints,
           fine_dof_handler,
           fine_constraints))
      {
        if (entries.empty())
          continue;
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
   * (FE_NedelecSimplex(0), FE_NedelecWedge(0), FE_NedelecPyramid(0), also
   * mixed in an hp::FECollection): for every line of a boundary face whose
   * boundary id is in @p function_map, the DoF is set to the line integral of
   * the tangential component of the function along the line object
   * (components @p first_vector_component to @p first_vector_component +
   * dim - 1). This is the canonical interpolation, so that n x u = n x g
   * holds exactly for functions g in the edge element space. The integral is
   * computed with Gauss points on the reference line of the cell, mapped by
   * the mapping with the active FE index of the cell (or the only one), so
   * that curved boundaries are taken into account. Lines that are already
   * constrained (e.g. hanging lines) are not changed, so the hanging node
   * constraints should be computed first.
   */
  template <int dim, int spacedim, typename number>
  void
  interpolate_boundary_values_curl_conforming(
    const hp::MappingCollection<dim, spacedim> &mapping_collection,
    const DoFHandler<dim, spacedim>            &dof_handler,
    const std::map<types::boundary_id, const Function<spacedim, number> *>
                              &function_map,
    AffineConstraints<number> &constraints,
    const unsigned int         first_vector_component = 0)
  {
    static_assert(dim > 1, "Edge elements need at least two dimensions.");
    const hp::FECollection<dim, spacedim> &fe_collection =
      dof_handler.get_fe_collection();
    for (unsigned int i = 0; i < fe_collection.size(); ++i)
      dealii::internal::EdgeElementTools::assert_line_oriented_edge_element(
        fe_collection[i]);
    Assert(mapping_collection.size() == 1 ||
             mapping_collection.size() == fe_collection.size(),
           ExcDimensionMismatch(mapping_collection.size(),
                                fe_collection.size()));

    const QGauss<1> q_line(4);

    // one FEValues object per element and line of its reference cell, with
    // the Gauss points on that line, created on first use
    std::map<std::pair<unsigned int, unsigned int>,
             std::unique_ptr<FEValues<dim, spacedim>>>
      line_values;
    const auto get_line_values = [&](const unsigned int fe_index,
                                     const unsigned int l)
      -> FEValues<dim, spacedim> & {
      auto &entry = line_values[{fe_index, l}];
      if (entry == nullptr)
        {
          const ReferenceCell ref_cell =
            fe_collection[fe_index].reference_cell();
          const Point<dim> v0 = ref_cell.template vertex<dim>(
            ref_cell.line_to_cell_vertices(l, 0));
          const Point<dim> v1 = ref_cell.template vertex<dim>(
            ref_cell.line_to_cell_vertices(l, 1));
          std::vector<Point<dim>> points;
          for (unsigned int q = 0; q < q_line.size(); ++q)
            points.push_back(v0 + q_line.point(q)[0] * (v1 - v0));
          entry = std::make_unique<FEValues<dim, spacedim>>(
            mapping_collection[mapping_collection.size() == 1 ? 0 : fe_index],
            fe_collection[fe_index],
            Quadrature<dim>(points),
            update_quadrature_points | update_jacobians);
        }
      return *entry;
    };

    Vector<number> g;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_artificial())
          continue;
        const ReferenceCell ref_cell = cell->reference_cell();
        for (const unsigned int f : cell->face_indices())
          {
            if (cell->at_boundary(f) == false)
              continue;
            const auto function =
              function_map.find(cell->face(f)->boundary_id());
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

                FEValues<dim, spacedim> &values =
                  get_line_values(cell->active_fe_index(), l);
                values.reinit(cell);
                const Tensor<1, dim> reference_tangent =
                  ref_cell.template vertex<dim>(
                    ref_cell.line_to_cell_vertices(l, 1)) -
                  ref_cell.template vertex<dim>(
                    ref_cell.line_to_cell_vertices(l, 0));
                number integral = 0;
                for (unsigned int q = 0; q < q_line.size(); ++q)
                  {
                    // derivative of the mapped line with respect to the
                    // line parameter
                    const Tensor<1, spacedim> tangent =
                      apply_transformation(values.jacobian(q),
                                           reference_tangent);
                    function->second->vector_value(values.quadrature_point(q),
                                                   g);
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



  /**
   * Same as above, with one mapping for all cells.
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
    interpolate_boundary_values_curl_conforming(
      hp::MappingCollection<dim, spacedim>(mapping),
      dof_handler,
      function_map,
      constraints,
      first_vector_component);
  }
} // namespace VectorTools

DEAL_II_NAMESPACE_CLOSE

#endif
