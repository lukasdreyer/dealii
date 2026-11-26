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


// FEInterfaceValues with one face quadrature per element and face (the
// constructor taking a vector of hp::QCollection objects) on all interior
// faces of distorted hybrid meshes with local refinement: regular faces and
// hanging faces (subface on the coarse side) between all pairs of cell
// shapes. Checks
// - both sides have the same quadrature points,
// - the jump of a continuous function (linear Lagrange elements, a random
//   vector with the hanging node constraints applied) vanishes,
// - the interface weights add up to the area of all interior faces.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
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

  std::vector<hp::QCollection<2>> quadratures;
  for (unsigned int i = 0; i < c.fe.size(); ++i)
    quadratures.push_back(face_quadratures(c.fe[i].reference_cell()));
  const UpdateFlags flags = update_values | update_quadrature_points |
                            update_JxW_values | update_normal_vectors;
  FEInterfaceValues<3> fe_interface_values(c.mapping, c.fe, quadratures, flags);
  hp::FEFaceValues<3>  hp_fe_face_values(c.mapping, c.fe, quadratures, flags);

  std::map<std::string, unsigned int> n_faces;
  double max_point_distance = 0, max_jump = 0, interface_area = 0,
         face_area = 0;
  std::vector<double> jumps;
  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const unsigned int f : cell->face_indices())
      {
        if (cell->at_boundary(f))
          continue;
        const auto neighbor = cell->neighbor(f);
        const bool hanging  = cell->neighbor_is_coarser(f);
        // each regular face once, each hanging face from the fine side
        if (!hanging &&
            (cell->face(f)->has_children() ||
             neighbor->active_cell_index() < cell->active_cell_index()))
          continue;

        // the area of the face on this side
        hp_fe_face_values.reinit(cell, f);
        const FEFaceValues<3> &fe_face_values =
          hp_fe_face_values.get_present_fe_values();
        for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
          face_area += fe_face_values.JxW(q);

        if (hanging)
          {
            const auto [neighbor_face, neighbor_subface] =
              cell->neighbor_of_coarser_neighbor(f);
            fe_interface_values.reinit(cell,
                                       f,
                                       numbers::invalid_unsigned_int,
                                       neighbor,
                                       neighbor_face,
                                       neighbor_subface);
          }
        else
          fe_interface_values.reinit(cell,
                                     f,
                                     numbers::invalid_unsigned_int,
                                     neighbor,
                                     cell->neighbor_of_neighbor(f),
                                     numbers::invalid_unsigned_int);
        ++n_faces[std::string(hanging ? "hanging " : "regular ") +
                  cell->face(f)->reference_cell().to_string() +
                  " faces between " + cell->reference_cell().to_string() +
                  " and " + neighbor->reference_cell().to_string()];

        const FEFaceValuesBase<3> &side_0 =
          fe_interface_values.get_fe_face_values(0);
        const FEFaceValuesBase<3> &side_1 =
          fe_interface_values.get_fe_face_values(1);
        AssertThrow(side_0.n_quadrature_points ==
                        fe_interface_values.n_quadrature_points &&
                      side_1.n_quadrature_points ==
                        fe_interface_values.n_quadrature_points,
                    ExcInternalError());
        jumps.resize(fe_interface_values.n_quadrature_points);
        fe_interface_values.get_jump_in_function_values(u, jumps);
        for (const unsigned int q :
             fe_interface_values.quadrature_point_indices())
          {
            interface_area += fe_interface_values.JxW(q);
            max_point_distance = std::max(max_point_distance,
                                          side_0.quadrature_point(q).distance(
                                            side_1.quadrature_point(q)));
            max_jump           = std::max(max_jump, std::abs(jumps[q]));
          }
      }

  const auto ok = [](const bool b) { return b ? "OK" : "WRONG"; };
  deallog << name << ":" << std::endl;
  for (const auto &[type, n] : n_faces)
    deallog << "  " << type << ": " << n << std::endl;
  deallog << "  same points: " << ok(max_point_distance < 1e-12)
          << ", continuous function without jumps: " << ok(max_jump < 1e-12)
          << ", area of all faces: "
          << ok(std::abs(interface_area - face_area) < 1e-12 * face_area)
          << std::endl;
  if (max_point_distance >= 1e-12 || max_jump >= 1e-12 ||
      std::abs(interface_area - face_area) >= 1e-12 * face_area)
    deallog << "  " << max_point_distance << ' ' << max_jump << ' '
            << interface_area << ' ' << face_area << std::endl;
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
