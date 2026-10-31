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

#ifndef dealii_fe_face_dof_tools_h
#define dealii_fe_face_dof_tools_h

#include <deal.II/base/config.h>

#include <deal.II/base/point.h>
#include <deal.II/base/table.h>

#include <deal.II/fe/fe.h>

#include <deal.II/grid/reference_cell.h>

#include <array>
#include <utility>
#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace internal
{
  /**
   * Tools for the DoFs of Lagrange elements on the lines and faces of cells,
   * which are described by their support points: the elements on pyramids,
   * wedges and simplices of arbitrary degree, which have to fit to each other
   * and to FE_Q (with equidistant support points) at the faces of hybrid
   * meshes.
   */
  namespace FEFaceDofTools
  {
    /**
     * The weights of the vertices of the triangle or quadrilateral
     * @p reference_cell (barycentric or bilinear) at the point @p p.
     */
    inline std::array<double, 4>
    vertex_weights(const ReferenceCell &reference_cell, const Point<2> &p)
    {
      if (reference_cell == ReferenceCells::Triangle)
        return {{1. - p[0] - p[1], p[0], p[1], 0.}};
      else
        {
          Assert(reference_cell == ReferenceCells::Quadrilateral,
                 ExcNotImplemented());
          return {{(1. - p[0]) * (1. - p[1]),
                   p[0] * (1. - p[1]),
                   (1. - p[0]) * p[1],
                   p[0] * p[1]}};
        }
    }



    /**
     * Map the point @p p on the triangle or quadrilateral @p reference_cell to
     * the cell whose vertices are @p vertices (in the order of the vertices of
     * @p reference_cell).
     */
    template <int dim>
    Point<dim>
    map_face_point(const ReferenceCell           &reference_cell,
                   const Point<2>                &p,
                   const std::vector<Point<dim>> &vertices)
    {
      const std::array<double, 4> w = vertex_weights(reference_cell, p);
      Point<dim>                  x;
      for (const unsigned int v : reference_cell.vertex_indices())
        x += w[v] * vertices[v];
      return x;
    }



    /**
     * Fill the table of
     * FiniteElement::adjust_line_dof_index_for_line_orientation() for
     * @p n_dofs_per_line DoFs ordered along the line.
     */
    inline void
    fill_line_dof_orientation_table(std::vector<int> &table,
                                    const unsigned int n_dofs_per_line)
    {
      table.resize(n_dofs_per_line);
      for (unsigned int i = 0; i < n_dofs_per_line; ++i)
        table[i] = static_cast<int>(n_dofs_per_line - 1 - i) -
                   static_cast<int>(i);
    }



    /**
     * Fill the table of
     * FiniteElement::adjust_quad_dof_index_for_face_orientation() of a face
     * of shape @p face_reference_cell whose interior DoFs have the support
     * points @p interior_points (on the reference triangle or quadrilateral):
     * in orientation o, the DoF at point p is the one at the point with the
     * same weights of the vertices permuted by o (as for FE_Q).
     */
    inline void
    fill_face_dof_orientation_table(Table<2, int>               &table,
                                    const ReferenceCell         &face_reference_cell,
                                    const std::vector<Point<2>> &interior_points)
    {
      const unsigned int n = interior_points.size();
      // the orientations of a triangle or quadrilateral as a face of a 3d cell
      const unsigned int n_orientations =
        face_reference_cell == ReferenceCells::Triangle ? 6 : 8;
      table.reinit(n, n_orientations);

      std::array<unsigned int, 4> vertex_ids = {{0, 1, 2, 3}};
      const unsigned int n_vertices = face_reference_cell.n_vertices();
      for (unsigned int o = 0; o < n_orientations; ++o)
        {
          const auto permutation =
            face_reference_cell.permute_by_combined_orientation(
              make_array_view(std::as_const(vertex_ids).data(),
                              std::as_const(vertex_ids).data() + n_vertices),
              static_cast<types::geometric_orientation>(o));
          for (unsigned int i = 0; i < n; ++i)
            {
              const std::array<double, 4> w =
                vertex_weights(face_reference_cell, interior_points[i]);
              Point<2> q;
              for (unsigned int k = 0; k < n_vertices; ++k)
                q += w[k] *
                     face_reference_cell.template vertex<2>(permutation[k]);
              unsigned int j = 0;
              for (; j < n; ++j)
                if (interior_points[j].distance(q) < 1e-10)
                  break;
              Assert(j < n,
                     ExcMessage("The support points inside the face are not "
                                "invariant under the face orientations."));
              table(i, o) = static_cast<int>(j) - static_cast<int>(i);
            }
        }
    }



    /**
     * The support points (in face coordinates) of the DoFs of @p fe inside
     * face @p face_no, i.e., without the ones on its vertices and lines.
     */
    template <int dim, int spacedim>
    std::vector<Point<dim - 1>>
    interior_face_support_points(const FiniteElement<dim, spacedim> &fe,
                                 const unsigned int                  face_no)
    {
      const auto  &points = fe.get_unit_face_support_points(face_no);
      const auto   face_reference_cell = fe.reference_cell().face_reference_cell(face_no);
      unsigned int first = face_reference_cell.n_vertices() * fe.n_dofs_per_vertex();
      if (dim == 3)
        first += face_reference_cell.n_lines() * fe.n_dofs_per_line();
      return std::vector<Point<dim - 1>>(points.begin() + first, points.end());
    }



    /**
     * The positions (from 0 to 1) of the DoFs of @p fe inside its lines, from
     * the support points of the first line of the first face.
     */
    template <int dim, int spacedim>
    std::vector<double>
    interior_line_positions(const FiniteElement<dim, spacedim> &fe)
    {
      const ReferenceCell reference_cell = fe.reference_cell();
      const Point<dim>    v0 =
        reference_cell.template vertex<dim>(reference_cell.line_to_cell_vertices(0, 0));
      const Point<dim> v1 =
        reference_cell.template vertex<dim>(reference_cell.line_to_cell_vertices(0, 1));
      const unsigned int first = reference_cell.n_vertices() * fe.n_dofs_per_vertex();
      std::vector<double> positions;
      for (unsigned int i = 0; i < fe.n_dofs_per_line(); ++i)
        positions.push_back(
          fe.get_unit_support_points()[first + i].distance(v0) / v1.distance(v0));
      return positions;
    }



    /**
     * The pairs (i, j) of DoFs at the same position, from the positions or
     * points of two elements on a line or face.
     */
    template <typename PointType>
    std::vector<std::pair<unsigned int, unsigned int>>
    identities_by_position(const std::vector<PointType> &positions_1,
                           const std::vector<PointType> &positions_2)
    {
      const auto distance = [](const PointType &a, const PointType &b) {
        if constexpr (std::is_same_v<PointType, double>)
          return std::abs(a - b);
        else
          return a.distance(b);
      };
      std::vector<std::pair<unsigned int, unsigned int>> identities;
      for (unsigned int i = 0; i < positions_1.size(); ++i)
        for (unsigned int j = 0; j < positions_2.size(); ++j)
          if (distance(positions_1[i], positions_2[j]) < 1e-10)
            identities.emplace_back(i, j);
      return identities;
    }



    /**
     * The face of @p fe with the shape @p face_reference_cell.
     */
    template <int dim, int spacedim>
    unsigned int
    face_with_shape(const FiniteElement<dim, spacedim> &fe,
                    const ReferenceCell                &face_reference_cell)
    {
      for (const unsigned int f : fe.reference_cell().face_indices())
        if (fe.reference_cell().face_reference_cell(f) == face_reference_cell)
          return f;
      return numbers::invalid_unsigned_int;
    }



    /**
     * The identities between the DoFs inside the lines of two Lagrange
     * elements, by the positions of their support points. Both elements need
     * support points.
     */
    template <int dim, int spacedim>
    std::vector<std::pair<unsigned int, unsigned int>>
    line_dof_identities(const FiniteElement<dim, spacedim> &fe_1,
                        const FiniteElement<dim, spacedim> &fe_2)
    {
      if (!fe_1.has_support_points() || !fe_2.has_support_points())
        return {};
      return identities_by_position(interior_line_positions(fe_1),
                                    interior_line_positions(fe_2));
    }



    /**
     * The identities between the DoFs inside face @p face_no of @p fe_1 and
     * the face of the same shape of @p fe_2, by the positions of their face
     * support points.
     */
    template <int dim, int spacedim>
    std::vector<std::pair<unsigned int, unsigned int>>
    face_dof_identities(const FiniteElement<dim, spacedim> &fe_1,
                        const FiniteElement<dim, spacedim> &fe_2,
                        const unsigned int                  face_no)
    {
      const unsigned int face_no_2 = face_with_shape(
        fe_2, fe_1.reference_cell().face_reference_cell(face_no));
      if (face_no_2 == numbers::invalid_unsigned_int ||
          !fe_1.has_face_support_points(face_no) ||
          !fe_2.has_face_support_points(face_no_2))
        return {};
      return identities_by_position(
        interior_face_support_points(fe_1, face_no),
        interior_face_support_points(fe_2, face_no_2));
    }
  } // namespace FEFaceDofTools
} // namespace internal

DEAL_II_NAMESPACE_CLOSE

#endif
