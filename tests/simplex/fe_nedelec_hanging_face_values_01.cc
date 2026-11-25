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


// EdgeElementTools::face_points_on_coarser_neighbor(): both sides of the
// hanging faces of distorted hybrid meshes with local refinement (FEFaceValues
// on the fine cell, FEValues on the coarser neighbor at the returned points),
// for all four Nedelec elements:
// - the physical points of both sides are the same,
// - for a random vector with the hanging node constraints applied, the
//   tangential trace n x u and the normal component of the curl n . curl u
//   are the same on both sides (tangential continuity).

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/edge_element_tools.h>

#include <deal.II/fe/fe_nedelec_hypercube.h>
#include <deal.II/fe/fe_nedelec_pyramid.h>
#include <deal.II/fe/fe_nedelec_simplex.h>
#include <deal.II/fe/fe_nedelec_wedge.h>
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
  hp::FECollection<3>      fe{FE_NedelecSimplex<3>(0),
                         FE_NedelecWedge<3>(0),
                         FE_NedelecPyramid<3>(0),
                         FE_NedelecHypercube<3>(0)};
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



// the values and curls of the FE function u on a cell (with FEValues or
// FEFaceValues already reinitialized)
template <typename FEValuesType>
void
evaluate(const FEValuesType                         &fe_values,
         const std::vector<types::global_dof_index> &dofs,
         const Vector<double>                       &u,
         std::vector<Tensor<1, 3>>                  &values,
         std::vector<Tensor<1, 3>>                  &curls)
{
  const FEValuesExtractors::Vector E(0);
  values.assign(fe_values.n_quadrature_points, Tensor<1, 3>());
  curls.assign(fe_values.n_quadrature_points, Tensor<1, 3>());
  for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
    for (unsigned int k = 0; k < dofs.size(); ++k)
      {
        values[q] += u(dofs[k]) * fe_values[E].value(k, q);
        curls[q] += u(dofs[k]) * fe_values[E].curl(k, q);
      }
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

  const QGaussSimplex<2> triangle_quadrature(3);
  const QGauss<2>        quadrilateral_quadrature(3);

  std::map<std::string, unsigned int> n_faces;
  double max_point_distance = 0, max_trace_jump = 0, max_curl_jump = 0,
         max_size = 0;
  std::vector<types::global_dof_index> dofs, neighbor_dofs;
  std::vector<Tensor<1, 3>> values, curls, neighbor_values, neighbor_curls;
  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const unsigned int f : cell->face_indices())
      if (!cell->at_boundary(f) && cell->neighbor_is_coarser(f))
        {
          const auto neighbor = cell->neighbor(f);
          const bool triangle =
            cell->face(f)->reference_cell() == ReferenceCells::Triangle;
          const Quadrature<2> &face_quadrature =
            triangle ? static_cast<const Quadrature<2> &>(triangle_quadrature) :
                       quadrilateral_quadrature;
          ++n_faces[std::string(triangle ? "triangles" : "quadrilaterals") +
                    " between " + cell->reference_cell().to_string() +
                    " and " + neighbor->reference_cell().to_string()];

          // the fine side: one face quadrature for all faces of the cell
          // with the shape of this face is enough here
          hp::QCollection<2> quadratures;
          for (const unsigned int g : cell->face_indices())
            quadratures.push_back(
              cell->face(g)->reference_cell() == ReferenceCells::Triangle ?
                static_cast<const Quadrature<2> &>(triangle_quadrature) :
                quadrilateral_quadrature);
          const unsigned int i = cell->active_fe_index();
          FEFaceValues<3>    fe_face_values(c.mapping[i],
                                         c.fe[i],
                                         quadratures,
                                         update_values | update_gradients |
                                           update_quadrature_points |
                                           update_normal_vectors);
          fe_face_values.reinit(cell, f);
          dofs.resize(c.fe[i].n_dofs_per_cell());
          cell->get_dof_indices(dofs);
          evaluate(fe_face_values, dofs, u, values, curls);

          // the coarse side
          const std::vector<Point<3>> points =
            EdgeElementTools::face_points_on_coarser_neighbor<3, 3>(
              cell, f, face_quadrature);
          const unsigned int j = neighbor->active_fe_index();
          FEValues<3>        fe_values(c.mapping[j],
                                c.fe[j],
                                Quadrature<3>(points),
                                update_values | update_gradients |
                                  update_quadrature_points);
          fe_values.reinit(neighbor);
          neighbor_dofs.resize(c.fe[j].n_dofs_per_cell());
          neighbor->get_dof_indices(neighbor_dofs);
          evaluate(fe_values, neighbor_dofs, u, neighbor_values, neighbor_curls);

          for (unsigned int q = 0; q < points.size(); ++q)
            {
              const Tensor<1, 3> &n = fe_face_values.normal_vector(q);
              max_point_distance =
                std::max(max_point_distance,
                         fe_face_values.quadrature_point(q).distance(
                           fe_values.quadrature_point(q)));
              max_trace_jump = std::max(
                max_trace_jump,
                cross_product_3d(n, values[q] - neighbor_values[q]).norm());
              max_curl_jump =
                std::max(max_curl_jump,
                         std::abs(n * (curls[q] - neighbor_curls[q])));
              max_size =
                std::max({max_size, values[q].norm(), curls[q].norm()});
            }
        }

  deallog << name << ":" << std::endl;
  for (const auto &[type, n] : n_faces)
    deallog << "  hanging " << type << ": " << n << std::endl;
  deallog << "  same points: " << (max_point_distance < 1e-12 ? "OK" : "WRONG")
          << ", n x u continuous: "
          << (max_trace_jump < 1e-10 * max_size ? "OK" : "WRONG")
          << ", n . curl u continuous: "
          << (max_curl_jump < 1e-10 * max_size ? "OK" : "WRONG") << std::endl;
  if (max_point_distance >= 1e-12 || max_trace_jump >= 1e-10 * max_size ||
      max_curl_jump >= 1e-10 * max_size)
    deallog << "  " << max_point_distance << ' ' << max_trace_jump << ' '
            << max_curl_jump << ' ' << max_size << std::endl;
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
