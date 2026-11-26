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
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_nedelec_lowest_order.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/cell_id_translator.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <array>
#include <map>
#include <memory>
#include <optional>
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
     * What the rows of the prolongation need from an active coarse cell:
     * its FE index, the DoF indices of its lines, and whether they are
     * constrained, with their constraint entries.
     */
    template <typename number>
    struct CoarseCellData
    {
      unsigned int                         fe_index = 0;
      std::vector<types::global_dof_index> dofs;
      std::vector<char>                    constrained;
      std::vector<std::vector<std::pair<types::global_dof_index, number>>>
        entries;
    };



    template <int dim, int spacedim, typename number>
    CoarseCellData<number>
    get_coarse_cell_data(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      const dealii::AffineConstraints<number> &constraints)
    {
      CoarseCellData<number> data;
      data.fe_index = cell->active_fe_index();
      data.dofs.resize(cell->get_fe().n_dofs_per_cell());
      cell->get_dof_indices(data.dofs);
      data.constrained.resize(data.dofs.size());
      data.entries.resize(data.dofs.size());
      for (unsigned int i = 0; i < data.dofs.size(); ++i)
        if (constraints.is_constrained(data.dofs[i]))
          {
            data.constrained[i] = 1;
            for (const auto &entry :
                 *constraints.get_constraint_entries(data.dofs[i]))
              data.entries[i].emplace_back(entry.first, entry.second);
          }
      return data;
    }



    // append the data of a coarse cell to an integer and a number buffer
    template <typename number>
    void
    pack_coarse_cell_data(const CoarseCellData<number>         &data,
                          std::vector<types::global_dof_index> &integers,
                          std::vector<number>                  &numbers)
    {
      integers.push_back(data.fe_index);
      integers.push_back(data.dofs.size());
      for (unsigned int i = 0; i < data.dofs.size(); ++i)
        {
          integers.push_back(data.dofs[i]);
          integers.push_back(data.constrained[i]);
          integers.push_back(data.entries[i].size());
          for (const auto &[dof, weight] : data.entries[i])
            {
              integers.push_back(dof);
              numbers.push_back(weight);
            }
        }
    }



    // read the data of a coarse cell from the buffers at the given positions
    template <typename number>
    CoarseCellData<number>
    unpack_coarse_cell_data(
      const std::vector<types::global_dof_index> &integers,
      const std::vector<number>                  &numbers,
      unsigned int                               &integer_position,
      unsigned int                               &number_position)
    {
      CoarseCellData<number> data;
      data.fe_index            = integers[integer_position++];
      const unsigned int n_dofs = integers[integer_position++];
      data.dofs.resize(n_dofs);
      data.constrained.resize(n_dofs);
      data.entries.resize(n_dofs);
      for (unsigned int i = 0; i < n_dofs; ++i)
        {
          data.dofs[i]                = integers[integer_position++];
          data.constrained[i]         = integers[integer_position++];
          const unsigned int n_entries = integers[integer_position++];
          for (unsigned int e = 0; e < n_entries; ++e)
            data.entries[i].emplace_back(integers[integer_position++],
                                         numbers[number_position++]);
        }
      return data;
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
     * one.
     *
     * In parallel, the two triangulations can be partitioned independently.
     * The coarse cell K of a locally owned fine cell (the cell itself or its
     * parent) is taken from the fine triangulation, which contains it with
     * the same vertices and lines (and so the same line directions); only
     * the DoF indices of its lines in the coarse space and their constraints
     * come from the owner of K in the coarse triangulation, if that is
     * another process. The owner is found with a CellIDTranslator, in which
     * every process registers its locally owned active coarse cells and
     * their (potential) children. @p coarse_constraints must contain the
     * constraints of the DoFs of the locally owned coarse cells.
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
      using active_cell_iterator =
        typename DoFHandler<dim, spacedim>::active_cell_iterator;

      const auto    &coarse_tria = coarse_dof_handler.get_triangulation();
      const auto    &fine_tria   = fine_dof_handler.get_triangulation();
      const auto    &owned       = fine_dof_handler.locally_owned_dofs();
      const MPI_Comm comm        = fine_dof_handler.get_mpi_communicator();

      // the locally owned active coarse cell with this id, if there is one
      const auto owned_coarse_cell =
        [&](const CellId &id) -> std::optional<active_cell_iterator> {
        if (coarse_tria.contains_cell(id) == false)
          return {};
        const auto tria_cell = coarse_tria.create_cell_iterator(id);
        if (tria_cell->is_active() == false ||
            tria_cell->is_locally_owned() == false)
          return {};
        return active_cell_iterator(&coarse_tria,
                                    tria_cell->level(),
                                    tria_cell->index(),
                                    &coarse_dof_handler);
      };

      // 1. the coarse cells K of the locally owned fine cells: the data of
      // the locally owned ones, and the fine cells whose K is elsewhere
      std::map<CellId, CoarseCellData<number>> coarse_data;
      std::vector<active_cell_iterator>        remote_fine_cells;
      for (const auto &fine_cell : fine_dof_handler.active_cell_iterators())
        if (fine_cell->is_locally_owned())
          {
            if (const auto coarse_cell = owned_coarse_cell(fine_cell->id()))
              coarse_data.emplace(fine_cell->id(),
                                  get_coarse_cell_data<dim, spacedim>(
                                    *coarse_cell, coarse_constraints));
            else if (fine_cell->level() > 0 &&
                     owned_coarse_cell(fine_cell->parent()->id()))
              coarse_data.emplace(fine_cell->parent()->id(),
                                  get_coarse_cell_data<dim, spacedim>(
                                    *owned_coarse_cell(
                                      fine_cell->parent()->id()),
                                    coarse_constraints));
            else
              remote_fine_cells.push_back(fine_cell);
          }

      // 2. get the data of the other coarse cells from their owners
      if (Utilities::MPI::n_mpi_processes(comm) > 1)
        {
          const internal::CellIDTranslator<dim> translator(
            coarse_tria.n_global_coarse_cells(),
            fine_tria.n_global_levels() + 1);

          IndexSet registered(translator.size());
          for (const auto &cell : coarse_dof_handler.active_cell_iterators())
            if (cell->is_locally_owned())
              {
                registered.add_index(translator.translate(cell));
                for (unsigned int c = 0;
                     c < cell->reference_cell().n_isotropic_children();
                     ++c)
                  registered.add_index(translator.translate(cell, c));
              }
          registered.compress();

          std::map<types::global_cell_index, active_cell_iterator> lookup;
          IndexSet lookup_set(translator.size());
          for (const auto &fine_cell : remote_fine_cells)
            {
              lookup.emplace(translator.translate(fine_cell), fine_cell);
              lookup_set.add_index(translator.translate(fine_cell));
            }
          lookup_set.compress();
          const std::vector<unsigned int> owners =
            Utilities::MPI::compute_index_owner(registered, lookup_set, comm);

          // requests: the index of the fine cell and of its parent
          std::map<unsigned int, std::vector<types::global_cell_index>>
                                                           requests;
          std::map<unsigned int, std::vector<active_cell_iterator>> requested;
          {
            unsigned int i = 0;
            for (const auto &[index, fine_cell] : lookup)
              {
                AssertIndexRange(owners[i], Utilities::MPI::n_mpi_processes(comm));
                requests[owners[i]].push_back(index);
                requests[owners[i]].push_back(
                  fine_cell->level() > 0 ?
                    translator.translate(fine_cell->parent()) :
                    index);
                requested[owners[i]].push_back(fine_cell);
                ++i;
              }
          }

          // answers: for each request, whether the coarse cell is the parent
          // and its data
          std::map<unsigned int,
                   std::pair<std::vector<types::global_dof_index>,
                             std::vector<number>>>
            answers;
          for (const auto &[rank, indices] :
               Utilities::MPI::some_to_some(comm, requests))
            {
              auto &[integers, numbers] = answers[rank];
              for (unsigned int i = 0; i < indices.size(); i += 2)
                {
                  auto coarse_cell =
                    owned_coarse_cell(translator.to_cell_id(indices[i]));
                  integers.push_back(coarse_cell ? 0 : 1);
                  if (!coarse_cell)
                    coarse_cell =
                      owned_coarse_cell(translator.to_cell_id(indices[i + 1]));
                  Assert(coarse_cell, ExcInternalError());
                  pack_coarse_cell_data(get_coarse_cell_data<dim, spacedim>(
                                          *coarse_cell, coarse_constraints),
                                        integers,
                                        numbers);
                }
            }

          for (const auto &[rank, buffers] :
               Utilities::MPI::some_to_some(comm, answers))
            {
              const auto  &[integers, numbers] = buffers;
              unsigned int integer_position = 0, number_position = 0;
              for (const auto &fine_cell : requested[rank])
                {
                  const bool is_parent = integers[integer_position++];
                  coarse_data.emplace(is_parent ? fine_cell->parent()->id() :
                                                  fine_cell->id(),
                                      unpack_coarse_cell_data(integers,
                                                              numbers,
                                                              integer_position,
                                                              number_position));
                }
            }
        }
      else
        AssertThrow(remote_fine_cells.empty(), ExcInternalError());

      // 3. the rows
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
      std::vector<active_cell_iterator>  fine_cells;
      // per coarse cell: the rows to fill (pointers into rows, whose nodes
      // do not move), the tangents of their lines, the quadrature points
      std::vector<std::map<types::global_dof_index, number> *> rows_of_cell;
      std::vector<Tensor<1, dim>>                              tangents;
      std::vector<Point<dim>>                                  points;

      for (const auto &[coarse_id, data] : coarse_data)
        {
          // the coarse cell K in the fine triangulation
          const auto fine_tria_cell = fine_tria.create_cell_iterator(coarse_id);
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
              coarse_dof_handler.get_fe(data.fe_index));
          const ReferenceCell reference_cell = fine_cell->reference_cell();
          Assert(fe.reference_cell() == reference_cell, ExcInternalError());
          const QGauss<1> &line_quadrature =
            (reference_cell == ReferenceCells::Pyramid) ?
              rational_line_quadrature :
              polynomial_line_quadrature;

          // reference positions of the vertices of the fine cells, keyed by
          // their vertex index in the fine triangulation
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

          // the fine lines that need a row here, with the quadrature points
          // along them in reference coordinates of K
          rows_of_cell.clear();
          tangents.clear();
          points.clear();
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
                  rows_of_cell.push_back(&row);
                  tangents.push_back(tangent);
                  for (unsigned int q = 0; q < line_quadrature.size(); ++q)
                    points.push_back(start +
                                     line_quadrature.point(q)[0] * tangent);
                }
            }
          if (rows_of_cell.empty())
            continue;

          // all coarse shape functions at all these points at once
          const std::vector<std::vector<Tensor<1, dim>>> values =
            fe.reference_shape_values(points);

          for (unsigned int k = 0; k < rows_of_cell.size(); ++k)
            for (unsigned int L = 0; L < fine_cell->n_lines(); ++L)
              {
                double value = 0;
                for (unsigned int q = 0; q < line_quadrature.size(); ++q)
                  value += line_quadrature.weight(q) *
                           (values[L][k * line_quadrature.size() + q] *
                            tangents[k]);
                // the coarse cell has the line directions of K in the fine
                // triangulation
                if (fine_cell->line_orientation(L) !=
                    numbers::default_geometric_orientation)
                  value = -value;
                if (std::abs(value) < 1e-12)
                  continue;

                auto &row = *rows_of_cell[k];
                if (data.constrained[L])
                  {
                    for (const auto &[dof, weight] : data.entries[L])
                      row[dof] += value * weight;
                  }
                else
                  row[data.dofs[L]] += value;
              }
        }

      // every locally owned fine DoF is a line of a locally owned fine cell
      AssertThrow(rows.size() == owned.n_elements(), ExcInternalError());
      return rows;
    }
  } // namespace EdgeElementTools
} // namespace internal



namespace EdgeElementTools
{
  /**
   * The reference points on the coarser neighbor of @p cell across face
   * @p face_no (a hanging face: the face of @p cell is a child of the face
   * of the neighbor) of the points of @p face_quadrature on this face, in
   * the order used by FEFaceValues on @p cell. With FEFaceValues on @p cell
   * and FEValues on the neighbor with these points, both sides of a hanging
   * face can be evaluated at the same points; this replaces FESubfaceValues
   * on the coarse side, which deal.II does not support for wedges and
   * pyramids (cells with faces of different shapes).
   *
   * The positions of the vertices of the face of @p cell in the reference
   * cell of the neighbor follow from the topology (vertices, midpoints of the
   * lines and, for quadrilaterals, the center of the face of the neighbor),
   * and the points are interpolated with the linear vertex functions of the
   * reference cell of @p cell. As for the hanging node constraints, this
   * assumes that the children of a face are the images of the reference
   * children (straight lines; flat faces or the d-linear mappings of
   * MappingFE with linear elements, for which the physical points of both
   * sides are the same).
   */
  template <int dim, int spacedim>
  std::vector<Point<dim>>
  face_points_on_coarser_neighbor(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int                                          face_no,
    const Quadrature<dim - 1>                                  &face_quadrature)
  {
    Assert(cell->neighbor_is_coarser(face_no),
           ExcMessage("The neighbor across this face must be coarser."));
    const auto neighbor = cell->neighbor(face_no);
    const unsigned int neighbor_face_no =
      cell->neighbor_of_coarser_neighbor(face_no).first;
    const auto          neighbor_face  = neighbor->face(neighbor_face_no);
    const ReferenceCell neighbor_shape = neighbor->reference_cell();

    // reference positions on the neighbor of the vertices of its face, of
    // the midpoints of the lines of the face and of the center
    std::map<unsigned int, Point<dim>> reference_points;
    for (const unsigned int v : neighbor->vertex_indices())
      reference_points[neighbor->vertex_index(v)] =
        neighbor_shape.template vertex<dim>(v);
    Point<dim> center;
    for (unsigned int v = 0; v < neighbor_face->n_vertices(); ++v)
      center += reference_points[neighbor_face->vertex_index(v)];
    center /= neighbor_face->n_vertices();
    if constexpr (dim == 3)
      for (unsigned int l = 0; l < neighbor_face->n_lines(); ++l)
        {
          const auto line = neighbor_face->line(l);
          Assert(line->has_children(), ExcInternalError());
          reference_points[line->child(0)->vertex_index(1)] =
            0.5 * (reference_points.at(line->vertex_index(0)) +
                   reference_points.at(line->vertex_index(1)));
        }
    else
      {
        // in 2d the face is a line, whose child 0 ends at the midpoint
        reference_points[neighbor_face->child(0)->vertex_index(1)] = center;
      }
    for (unsigned int c = 0; c < neighbor_face->n_children(); ++c)
      for (unsigned int v = 0; v < neighbor_face->child(c)->n_vertices(); ++v)
        reference_points.emplace(neighbor_face->child(c)->vertex_index(v),
                                 center);

    // the points on the reference cell of cell, and their images
    const Quadrature<dim> cell_points =
      QProjector<dim>::project_to_face(cell->reference_cell(),
                                       face_quadrature,
                                       face_no,
                                       cell->combined_face_orientation(
                                         face_no));
    std::vector<Point<dim>> points(cell_points.size());
    for (unsigned int q = 0; q < cell_points.size(); ++q)
      for (const unsigned int v : cell->vertex_indices())
        {
          const double phi = cell->reference_cell().d_linear_shape_function(
            cell_points.point(q), v);
          // (only the vertices of the face contribute on the face)
          if (std::abs(phi) > 1e-12)
            {
              const auto it = reference_points.find(cell->vertex_index(v));
              Assert(it != reference_points.end(), ExcInternalError());
              points[q] += phi * it->second;
            }
        }
    return points;
  }


  /**
   * Like face_points_on_coarser_neighbor(), but for any neighbor that is not
   * finer than @p cell (same level or coarser): the reference points on the
   * neighbor of the points of @p face_quadrature on face @p face_no of
   * @p cell, in the order used by FEFaceValues on @p cell.
   */
  template <int dim, int spacedim>
  std::vector<Point<dim>>
  face_points_on_neighbor(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int                                          face_no,
    const Quadrature<dim - 1>                                  &face_quadrature)
  {
    if (cell->neighbor_is_coarser(face_no))
      return face_points_on_coarser_neighbor<dim, spacedim>(cell,
                                                            face_no,
                                                            face_quadrature);
    Assert(cell->face(face_no)->has_children() == false,
           ExcMessage("The neighbor must not be finer."));
    const auto neighbor = cell->neighbor(face_no);
    // the vertices of the face are vertices of both cells
    std::map<unsigned int, Point<dim>> reference_points;
    for (const unsigned int v : neighbor->vertex_indices())
      reference_points[neighbor->vertex_index(v)] =
        neighbor->reference_cell().template vertex<dim>(v);
    const Quadrature<dim> cell_points =
      QProjector<dim>::project_to_face(cell->reference_cell(),
                                       face_quadrature,
                                       face_no,
                                       cell->combined_face_orientation(
                                         face_no));
    std::vector<Point<dim>> points(cell_points.size());
    for (unsigned int q = 0; q < cell_points.size(); ++q)
      for (const unsigned int v : cell->vertex_indices())
        {
          const double phi = cell->reference_cell().d_linear_shape_function(
            cell_points.point(q), v);
          if (std::abs(phi) > 1e-12)
            {
              const auto it = reference_points.find(cell->vertex_index(v));
              Assert(it != reference_points.end(), ExcInternalError());
              points[q] += phi * it->second;
            }
        }
    return points;
  }



  /**
   * Residual error indicators for curl(mu^{-1} curl u) + beta u = f with
   * the lowest-order edge elements (FE_NedelecLowestOrderBase), for
   * constants mu^{-1} and beta, with tangential Dirichlet conditions on the
   * whole boundary (no boundary terms): for every locally owned active cell
   * K,
   *
   *   eta_K^2 = h_K^2 || f - beta u_h - mu^{-1} curl curl u_h ||_K^2
   *           + h_K^2 || div (f - beta u_h) ||_K^2
   *           + 1/2 sum_{F interior faces of K}
   *               h_F ( || [n x mu^{-1} curl u_h] ||_F^2
   *                     + || [n . beta u_h] ||_F^2 )
   *
   * (Beck, Hiptmair, Hoppe, Wohlmuth 2000; Monk 2003). @p indicators gets
   * eta_K at the active cell index of the locally owned cells (it must have
   * n_active_cells() entries; the others are not changed). @p solution must
   * contain the values of the DoFs of the locally owned and ghost cells
   * (with the constraints applied). @p f must provide vector_value() and
   * vector_gradient() (for div f).
   *
   * Every cell computes the contributions of all its faces from both sides
   * (so every interior face is computed twice), which only needs the ghost
   * layer and gives the same values on all processes. Both sides are
   * evaluated at the same points with face_points_on_neighbor() (from the
   * fine side on hanging faces), which works on all cell shapes.
   * @p face_quadratures holds the face quadratures for triangles and
   * quadrilaterals (in this order); @p quadratures, @p mapping are indexed by
   * the active FE index.
   */
  template <typename VectorType>
  void
  estimate_residual(const hp::MappingCollection<3> &mapping,
                    const DoFHandler<3>            &dof_handler,
                    const VectorType               &solution,
                    const Function<3>              &f,
                    const double                    beta,
                    const double                    mu_inv,
                    const hp::QCollection<3>       &quadratures,
                    const std::array<Quadrature<2>, 2> &face_quadratures,
                    Vector<float>                  &indicators)
  {
    using cell_iterator = typename DoFHandler<3>::active_cell_iterator;
    const FEValuesExtractors::Vector E(0);
    AssertDimension(indicators.size(),
                    dof_handler.get_triangulation().n_active_cells());

    hp::FEValues<3> hp_fe_values(mapping,
                                 dof_handler.get_fe_collection(),
                                 quadratures,
                                 update_values | update_gradients |
                                   update_hessians | update_quadrature_points |
                                   update_JxW_values);

    std::vector<Tensor<1, 3>>             values, curls, f_values;
    std::vector<double>                   divergences;
    std::vector<Tensor<3, 3>>             hessians;
    std::vector<Vector<double>>           f_vector_values;
    std::vector<std::vector<Tensor<1, 3>>> f_gradients;

    // values and curls of u_h on cell at the reference points (or at the
    // points of fe_face_values if given)
    const auto evaluate_at = [&](const cell_iterator           &cell,
                                 const std::vector<Point<3>>   &points,
                                 std::vector<Tensor<1, 3>>     &u,
                                 std::vector<Tensor<1, 3>>     &curl_u) {
      const unsigned int i = cell->active_fe_index();
      FEValues<3>        fe_values(mapping[i],
                            dof_handler.get_fe(i),
                            Quadrature<3>(points),
                            update_values | update_gradients);
      fe_values.reinit(cell);
      u.resize(points.size());
      curl_u.resize(points.size());
      fe_values[E].get_function_values(solution, u);
      fe_values[E].get_function_curls(solution, curl_u);
    };

    // FEFaceValues for every FE index (with the face quadrature of each face
    // of its reference cell), created when first needed
    std::map<unsigned int, std::unique_ptr<FEFaceValues<3>>> face_values;
    const auto get_face_values = [&](const cell_iterator &cell,
                                     const unsigned int   face_no)
      -> FEFaceValues<3> & {
      const unsigned int i  = cell->active_fe_index();
      auto              &fv = face_values[i];
      if (!fv)
        {
          hp::QCollection<2> face_quadrature_collection;
          for (const unsigned int g : cell->face_indices())
            face_quadrature_collection.push_back(
              face_quadratures[cell->face(g)->reference_cell() ==
                                   ReferenceCells::Triangle ?
                                 0 :
                                 1]);
          fv = std::make_unique<FEFaceValues<3>>(
            mapping[i],
            dof_handler.get_fe(i),
            face_quadrature_collection,
            update_values | update_gradients | update_quadrature_points |
              update_normal_vectors | update_JxW_values);
        }
      fv->reinit(cell, face_no);
      return *fv;
    };

    // contribution h_F (||[n x mu^{-1} curl u]||^2 + ||[n . beta u]||^2) of
    // face face_no of cell, whose neighbor is not finer
    std::vector<Tensor<1, 3>> u_here, curl_here, u_there, curl_there;
    std::vector<Point<3>>     points_here;
    std::vector<unsigned int> matching;
    const auto face_contribution = [&](const cell_iterator &cell,
                                       const unsigned int   face_no) {
      const bool triangle =
        cell->face(face_no)->reference_cell() == ReferenceCells::Triangle;
      const Quadrature<2> &face_quadrature =
        face_quadratures[triangle ? 0 : 1];
      const unsigned int n_q = face_quadrature.size();

      FEFaceValues<3> &fe_face_values = get_face_values(cell, face_no);
      u_here.resize(n_q);
      curl_here.resize(n_q);
      fe_face_values[E].get_function_values(solution, u_here);
      fe_face_values[E].get_function_curls(solution, curl_here);
      points_here = fe_face_values.get_quadrature_points();
      std::vector<double> JxW(n_q);
      std::vector<Tensor<1, 3>> normals(n_q);
      for (unsigned int q = 0; q < n_q; ++q)
        {
          JxW[q]     = fe_face_values.JxW(q);
          normals[q] = fe_face_values.normal_vector(q);
        }

      const cell_iterator neighbor = cell->neighbor(face_no);
      if (cell->neighbor_is_coarser(face_no))
        // hanging face: the neighbor at the corresponding points
        evaluate_at(neighbor,
                    face_points_on_coarser_neighbor<3, 3>(cell,
                                                          face_no,
                                                          face_quadrature),
                    u_there,
                    curl_there);
      else
        {
          // conforming face: FEFaceValues on the neighbor, whose points are
          // the same up to their order
          FEFaceValues<3> &neighbor_values =
            get_face_values(neighbor, cell->neighbor_of_neighbor(face_no));
          std::vector<Tensor<1, 3>> u_n(n_q), curl_n(n_q);
          neighbor_values[E].get_function_values(solution, u_n);
          neighbor_values[E].get_function_curls(solution, curl_n);
          u_there.resize(n_q);
          curl_there.resize(n_q);
          for (unsigned int q = 0; q < n_q; ++q)
            {
              unsigned int best = 0;
              for (unsigned int p = 1; p < n_q; ++p)
                if (neighbor_values.quadrature_point(p).distance_square(
                      points_here[q]) <
                    neighbor_values.quadrature_point(best).distance_square(
                      points_here[q]))
                  best = p;
              Assert(neighbor_values.quadrature_point(best).distance(
                       points_here[q]) < 1e-10 * cell->diameter(),
                     ExcInternalError());
              u_there[q]    = u_n[best];
              curl_there[q] = curl_n[best];
            }
        }

      double jumps = 0;
      for (unsigned int q = 0; q < n_q; ++q)
        {
          const Tensor<1, 3> &n = normals[q];
          jumps += (mu_inv * mu_inv *
                      cross_product_3d(n, curl_here[q] - curl_there[q])
                        .norm_square() +
                    beta * beta *
                      Utilities::fixed_power<2>(n * (u_here[q] - u_there[q]))) *
                   JxW[q];
        }
      return cell->face(face_no)->diameter() * jumps;
    };

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          // cell terms
          hp_fe_values.reinit(cell);
          const FEValues<3> &fe_values = hp_fe_values.get_present_fe_values();
          const unsigned int n_q       = fe_values.n_quadrature_points;
          values.resize(n_q);
          divergences.resize(n_q);
          hessians.resize(n_q);
          f_vector_values.assign(n_q, Vector<double>(3));
          f_gradients.assign(n_q, std::vector<Tensor<1, 3>>(3));
          fe_values[E].get_function_values(solution, values);
          fe_values[E].get_function_divergences(solution, divergences);
          fe_values[E].get_function_hessians(solution, hessians);
          f.vector_value_list(fe_values.get_quadrature_points(),
                              f_vector_values);
          f.vector_gradient_list(fe_values.get_quadrature_points(),
                                 f_gradients);
          double element_residual = 0, divergence_residual = 0;
          for (unsigned int q = 0; q < n_q; ++q)
            {
              // curl curl u = grad div u - Laplace u
              Tensor<1, 3> curl_curl;
              for (unsigned int c = 0; c < 3; ++c)
                for (unsigned int j = 0; j < 3; ++j)
                  curl_curl[c] += hessians[q][j][c][j] - hessians[q][c][j][j];
              Tensor<1, 3> r;
              double       div_f = 0;
              for (unsigned int c = 0; c < 3; ++c)
                {
                  r[c] = f_vector_values[q](c) - beta * values[q][c] -
                         mu_inv * curl_curl[c];
                  div_f += f_gradients[q][c][c];
                }
              element_residual += r.norm_square() * fe_values.JxW(q);
              divergence_residual +=
                Utilities::fixed_power<2>(div_f - beta * divergences[q]) *
                fe_values.JxW(q);
            }
          const double h   = cell->diameter();
          double       eta = h * h * (element_residual + divergence_residual);

          // face terms (half of each face, from both of its sides)
          for (const unsigned int f_no : cell->face_indices())
            {
              if (cell->at_boundary(f_no))
                continue;
              if (cell->face(f_no)->has_children())
                {
                  // finer neighbors: the child faces, from the fine side
                  for (unsigned int sf = 0;
                       sf < cell->face(f_no)->n_children();
                       ++sf)
                    {
                      const auto fine =
                        cell->neighbor_child_on_subface(f_no, sf);
                      for (const unsigned int g : fine->face_indices())
                        if (!fine->at_boundary(g) &&
                            fine->neighbor_is_coarser(g) &&
                            fine->neighbor(g) == cell)
                          eta += 0.5 * face_contribution(fine, g);
                    }
                }
              else
                eta += 0.5 * face_contribution(cell, f_no);
            }
          indicators(cell->active_cell_index()) = std::sqrt(eta);
        }
  }
} // namespace EdgeElementTools



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
   *
   * With hanging nodes, a refined line can still carry a DoF, if a coarser
   * cell touches the boundary only along this line (e.g. a tetrahedron with
   * one edge on the boundary). Such a line is not a line of any active
   * boundary face, but its children on the boundary are constrained to it.
   * These lines are found by their boundary id (lines on the boundary carry
   * the boundary id of a face they lie on) and get the same values.
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

    // the line integral of the tangential component of the function along
    // line l of the cell, in the direction of the line object
    Vector<number> g;
    const auto     constrain_line = [&](const auto              &cell,
                                    const unsigned int        l,
                                    const Function<spacedim, number> &function) {
      const types::global_dof_index dof =
        cell->line(l)->dof_index(0, cell->active_fe_index());
      if (constraints.is_constrained(dof) ||
          constraints.can_store_line(dof) == false)
        return;
      Assert(function.n_components >= first_vector_component + spacedim,
             ExcMessage("The boundary function has too few components."));
      g.reinit(function.n_components);

      const ReferenceCell      ref_cell = cell->reference_cell();
      FEValues<dim, spacedim> &values =
        get_line_values(cell->active_fe_index(), l);
      values.reinit(cell);
      const Tensor<1, dim> reference_tangent =
        ref_cell.template vertex<dim>(ref_cell.line_to_cell_vertices(l, 1)) -
        ref_cell.template vertex<dim>(ref_cell.line_to_cell_vertices(l, 0));
      number integral = 0;
      for (unsigned int q = 0; q < q_line.size(); ++q)
        {
          // derivative of the mapped line with respect to the line parameter
          const Tensor<1, spacedim> tangent =
            apply_transformation(values.jacobian(q), reference_tangent);
          function.vector_value(values.quadrature_point(q), g);
          for (unsigned int d = 0; d < spacedim; ++d)
            integral +=
              q_line.weight(q) * g(first_vector_component + d) * tangent[d];
        }
      // the DoF refers to the direction of the line object
      if (cell->line_orientation(l) != numbers::default_geometric_orientation)
        integral = -integral;
      constraints.add_constraint(dof, {}, integral);
    };

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_artificial())
          continue;
        for (const unsigned int f : cell->face_indices())
          {
            if (cell->at_boundary(f) == false)
              continue;
            const auto function =
              function_map.find(cell->face(f)->boundary_id());
            if (function == function_map.end())
              continue;

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
                if (on_face)
                  constrain_line(cell, l, *function->second);
              }
          }

        // refined lines on the boundary that still carry a DoF of this cell
        if constexpr (dim == 3)
          for (unsigned int l = 0; l < cell->n_lines(); ++l)
            if (cell->line(l)->has_children())
              {
                const auto function =
                  function_map.find(cell->line(l)->boundary_id());
                if (function != function_map.end())
                  constrain_line(cell, l, *function->second);
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
