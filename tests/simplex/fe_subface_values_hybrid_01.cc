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


// FESubfaceValues with one face quadrature per face (an hp::QCollection, as
// for FEFaceValues) on the hanging faces of distorted hybrid meshes with
// local refinement: on the coarse side FESubfaceValues for each subface, on
// the fine side FEFaceValues of the child cell behind the subface. Checks
// - the same number of quadrature points, the same points and weights, and
//   opposite normals on both sides,
// - the same values of a continuous function (linear Lagrange elements, a
//   random vector with the hanging node constraints applied) on both sides,
// - the subfaces cover the face: the integrals of the normal vector (the
//   vector area, integrated exactly also on bilinear faces) over the subfaces
//   add up to the one over the face.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

#include <functional>
#include <map>

#include "../tests.h"

#include "./pyramid_meshes.h"
#include "./simplex_grids.h"
#include "./wedge_meshes.h"


struct Collections
{
  hp::FECollection<3>      fe{FE_SimplexP<3>(1),
                         FE_WedgeP<3>(1),
                         FE_PyramidP<3>(1),
                         FE_Q<3>(1)};
  hp::MappingCollection<3> mapping{MappingFE<3>(FE_SimplexP<3>(1)),
                                   MappingFE<3>(FE_WedgeP<3>(1)),
                                   MappingFE<3>(FE_PyramidP<3>(1)),
                                   MappingFE<3>(FE_Q<3>(1))};

  static unsigned int
  fe_index(const ReferenceCell &reference_cell)
  {
    if (reference_cell == ReferenceCells::Tetrahedron)
      return 0;
    if (reference_cell == ReferenceCells::Wedge)
      return 1;
    if (reference_cell == ReferenceCells::Pyramid)
      return 2;
    return 3;
  }
};



void
refine(Triangulation<3> &tria, const Point<3> &center, const double radius)
{
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->center().distance(center) < radius)
      cell->set_refine_flag();
  tria.prepare_coarsening_and_refinement();
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->refine_flag_set() &&
        cell->reference_cell() == ReferenceCells::Tetrahedron)
      cell->set_refine_choice(
        static_cast<std::uint8_t>(IsotropicRefinementChoice::cut_tet_68));
  tria.execute_coarsening_and_refinement();
}



// one face quadrature for each face of the given reference cell: Gauss rules
// with the same number of points per direction on triangles and
// quadrilaterals
hp::QCollection<2>
face_quadratures(const ReferenceCell &reference_cell)
{
  const QGaussSimplex<2> triangle(3);
  const QGauss<2>        quadrilateral(3);
  hp::QCollection<2>     quadratures;
  for (const unsigned int f : reference_cell.face_indices())
    quadratures.push_back(reference_cell.face_reference_cell(f) ==
                              ReferenceCells::Triangle ?
                            static_cast<const Quadrature<2> &>(triangle) :
                            quadrilateral);
  return quadratures;
}



double
function_value(const FEValuesBase<3>                      &fe_values,
               const std::vector<types::global_dof_index> &dofs,
               const Vector<double>                       &u,
               const unsigned int                          q)
{
  double value = 0;
  for (unsigned int k = 0; k < dofs.size(); ++k)
    value += u(dofs[k]) * fe_values.shape_value(k, q);
  return value;
}



void
check(Triangulation<3> &tria, const std::string &name)
{
  const Collections c;
  DoFHandler<3>     dof_handler(tria);
  for (const auto &cell : dof_handler.active_cell_iterators())
    cell->set_active_fe_index(Collections::fe_index(cell->reference_cell()));
  dof_handler.distribute_dofs(c.fe);
  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();
  Vector<double> u(dof_handler.n_dofs());
  for (unsigned int i = 0; i < u.size(); ++i)
    u(i) = random_value<double>(-1., 1.);
  constraints.distribute(u);

  const UpdateFlags flags = update_values | update_quadrature_points |
                            update_JxW_values | update_normal_vectors;
  std::vector<std::unique_ptr<FESubfaceValues<3>>> fe_subface_values;
  std::vector<std::unique_ptr<FEFaceValues<3>>>    fe_face_values;
  for (unsigned int i = 0; i < c.fe.size(); ++i)
    {
      const hp::QCollection<2> quadratures =
        face_quadratures(c.fe[i].reference_cell());
      fe_subface_values.push_back(std::make_unique<FESubfaceValues<3>>(
        c.mapping[i], c.fe[i], quadratures, flags));
      fe_face_values.push_back(std::make_unique<FEFaceValues<3>>(
        c.mapping[i], c.fe[i], quadratures, flags));
    }

  std::map<std::string, unsigned int> n_faces;
  bool                                same_number_of_points = true;
  double max_point_distance = 0, max_weight_difference = 0, max_normal_sum = 0,
         max_value_jump = 0, max_area_difference = 0;
  std::vector<types::global_dof_index> dofs, child_dofs;
  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const unsigned int f : cell->face_indices())
      if (!cell->at_boundary(f) && cell->face(f)->has_children())
        {
          const unsigned int i = cell->active_fe_index();
          dofs.resize(c.fe[i].n_dofs_per_cell());
          cell->get_dof_indices(dofs);
          fe_face_values[i]->reinit(cell, f);
          double       area = 0;
          Tensor<1, 3> vector_area;
          for (unsigned int q = 0; q < fe_face_values[i]->n_quadrature_points;
               ++q)
            {
              area += fe_face_values[i]->JxW(q);
              vector_area +=
                fe_face_values[i]->JxW(q) * fe_face_values[i]->normal_vector(q);
            }
          const double length = std::sqrt(area);

          Tensor<1, 3> subface_vector_area;
          for (unsigned int s = 0; s < cell->face(f)->n_children(); ++s)
            {
              const auto child = cell->neighbor_child_on_subface(f, s);
              ++n_faces[cell->face(f)->reference_cell().to_string() +
                        " subfaces between " +
                        cell->reference_cell().to_string() + " and " +
                        child->reference_cell().to_string()];

              // the face of the child behind the subface
              unsigned int g = numbers::invalid_unsigned_int;
              for (const unsigned int h : child->face_indices())
                if (!child->at_boundary(h) && child->neighbor(h) == cell)
                  g = h;
              AssertThrow(g != numbers::invalid_unsigned_int,
                          ExcInternalError());

              FESubfaceValues<3> &coarse = *fe_subface_values[i];
              coarse.reinit(cell, f, s);
              const unsigned int j    = child->active_fe_index();
              FEFaceValues<3>   &fine = *fe_face_values[j];
              fine.reinit(child, g);
              child_dofs.resize(c.fe[j].n_dofs_per_cell());
              child->get_dof_indices(child_dofs);

              if (coarse.n_quadrature_points != fine.n_quadrature_points)
                {
                  same_number_of_points = false;
                  continue;
                }
              for (unsigned int q = 0; q < coarse.n_quadrature_points; ++q)
                {
                  subface_vector_area +=
                    coarse.JxW(q) * coarse.normal_vector(q);
                  max_point_distance =
                    std::max(max_point_distance,
                             coarse.quadrature_point(q).distance(
                               fine.quadrature_point(q)) /
                               length);
                  max_weight_difference =
                    std::max(max_weight_difference,
                             std::abs(coarse.JxW(q) - fine.JxW(q)) / area);
                  max_normal_sum = std::max(
                    max_normal_sum,
                    (coarse.normal_vector(q) + fine.normal_vector(q)).norm());
                  max_value_jump =
                    std::max(max_value_jump,
                             std::abs(function_value(coarse, dofs, u, q) -
                                      function_value(fine, child_dofs, u, q)));
                }
            }
          max_area_difference =
            std::max(max_area_difference,
                     (subface_vector_area - vector_area).norm() / area);
        }

  const auto ok = [](const bool b) { return b ? "OK" : "WRONG"; };
  deallog << name << ":" << std::endl;
  for (const auto &[type, n] : n_faces)
    deallog << "  " << type << ": " << n << std::endl;
  deallog << "  same number of points: " << ok(same_number_of_points)
          << ", same points: " << ok(max_point_distance < 1e-12)
          << ", same weights: " << ok(max_weight_difference < 1e-12)
          << ", opposite normals: " << ok(max_normal_sum < 1e-12) << std::endl;
  deallog << "  continuous function: " << ok(max_value_jump < 1e-12)
          << ", subfaces cover the face: " << ok(max_area_difference < 1e-12)
          << std::endl;
  if (max_point_distance >= 1e-12 || max_weight_difference >= 1e-12 ||
      max_normal_sum >= 1e-12 || max_value_jump >= 1e-12 ||
      max_area_difference >= 1e-12)
    deallog << "  " << max_point_distance << ' ' << max_weight_difference << ' '
            << max_normal_sum << ' ' << max_value_jump << ' '
            << max_area_difference << std::endl;
}



int
main()
{
  initlog();

  const std::vector<
    std::pair<std::string, std::function<void(Triangulation<3> &)>>>
    meshes = {
      {"tetrahedra",
       [](Triangulation<3> &t) {
         GridGenerator::subdivided_hyper_cube_with_simplices(t, 2);
       }},
      {"wedges",
       [](Triangulation<3> &t) {
         GridGenerator::subdivided_hyper_cube_with_wedges(t, 2);
       }},
      {"hexahedra",
       [](Triangulation<3> &t) { GridGenerator::subdivided_hyper_cube(t, 2); }},
      {"pyramids", [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 0); }},
      {"pyramids and tetrahedra",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 2); }},
      {"hexahedra and pyramids",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 1); }},
      {"wedges and tetrahedra",
       [](Triangulation<3> &t) { make_wedge_tetrahedron_mesh(t, 2); }}};

  for (const auto &[name, make] : meshes)
    {
      Triangulation<3> tria;
      make(tria);
      GridTools::distort_random(0.15, tria, true, 42);
      const double size = GridTools::diameter(tria);
      refine(tria, Point<3>(0.2, 0.3, 0.25) * size, 0.45 * size);
      refine(tria, Point<3>(0.3, 0.2, 0.3) * size, 0.25 * size);
      check(tria, name);
    }
}
