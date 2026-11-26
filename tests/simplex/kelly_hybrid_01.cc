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


// KellyErrorEstimator with one face quadrature per finite element and face
// (the function taking a vector of hp::QCollection objects) on distorted
// hybrid meshes with local refinement (linear Lagrange elements, a random
// vector with the hanging node constraints applied). Compares the estimate
// of each cell with
//   eta_K^2 = diam(K)/24 sum_{faces F of K} int_F |[grad u] . n|^2
// computed with FEInterfaceValues (hanging faces as the sum over their
// subfaces). On meshes of a single cell shape, also compares with the
// function that takes one face quadrature per finite element.

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

#include <deal.II/numerics/error_estimator.h>

#include <functional>

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



const QGaussSimplex<2> triangle_quadrature(3);
const QGauss<2>        quadrilateral_quadrature(3);

// one face quadrature for each face of the given reference cell
hp::QCollection<2>
face_quadratures(const ReferenceCell &reference_cell)
{
  hp::QCollection<2> quadratures;
  for (const unsigned int f : reference_cell.face_indices())
    quadratures.push_back(
      reference_cell.face_reference_cell(f) == ReferenceCells::Triangle ?
        static_cast<const Quadrature<2> &>(triangle_quadrature) :
        quadrilateral_quadrature);
  return quadratures;
}



// int_F |[grad u] . n|^2 over the interface last initialized
double
jump_integral(const FEInterfaceValues<3> &fe_interface_values,
              const Vector<double>       &u)
{
  std::vector<Tensor<1, 3>> jumps(fe_interface_values.n_quadrature_points);
  fe_interface_values.get_jump_in_function_gradients(u, jumps);
  double integral = 0;
  for (const unsigned int q : fe_interface_values.quadrature_point_indices())
    integral += Utilities::fixed_power<2>(
                  jumps[q] * fe_interface_values.normal_vector(q)) *
                fe_interface_values.JxW(q);
  return integral;
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

  Vector<float> estimate(tria.n_active_cells());
  KellyErrorEstimator<3>::estimate(
    c.mapping, dof_handler, quadratures, {}, u, estimate);

  // the reference
  FEInterfaceValues<3> fe_interface_values(c.mapping,
                                           c.fe,
                                           quadratures,
                                           update_gradients |
                                             update_JxW_values |
                                             update_normal_vectors);
  Vector<double>       reference(tria.n_active_cells());
  unsigned int         n_hanging_faces = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      double sum = 0;
      for (const unsigned int f : cell->face_indices())
        {
          if (cell->at_boundary(f))
            continue;
          if (cell->face(f)->has_children())
            {
              // the hanging face from the coarse side: the sum over the
              // subfaces, each from the child behind it
              for (unsigned int s = 0; s < cell->face(f)->n_children(); ++s)
                {
                  const auto   child = cell->neighbor_child_on_subface(f, s);
                  unsigned int g     = numbers::invalid_unsigned_int;
                  for (const unsigned int h : child->face_indices())
                    if (child->face(h) == cell->face(f)->child(s))
                      g = h;
                  AssertThrow(g != numbers::invalid_unsigned_int,
                              ExcInternalError());
                  fe_interface_values.reinit(
                    child, g, numbers::invalid_unsigned_int, cell, f, s);
                  sum += jump_integral(fe_interface_values, u);
                }
              ++n_hanging_faces;
            }
          else if (cell->neighbor_is_coarser(f))
            {
              const auto [neighbor_face, neighbor_subface] =
                cell->neighbor_of_coarser_neighbor(f);
              fe_interface_values.reinit(cell,
                                         f,
                                         numbers::invalid_unsigned_int,
                                         cell->neighbor(f),
                                         neighbor_face,
                                         neighbor_subface);
              sum += jump_integral(fe_interface_values, u);
            }
          else
            {
              fe_interface_values.reinit(cell,
                                         f,
                                         numbers::invalid_unsigned_int,
                                         cell->neighbor(f),
                                         cell->neighbor_of_neighbor(f),
                                         numbers::invalid_unsigned_int);
              sum += jump_integral(fe_interface_values, u);
            }
        }
      reference(cell->active_cell_index()) =
        std::sqrt(cell->diameter() / 24 * sum);
    }

  double max_difference = 0;
  for (unsigned int i = 0; i < estimate.size(); ++i)
    max_difference =
      std::max(max_difference, std::abs(estimate(i) - reference(i)));
  const double tolerance = 1e-6 * reference.linfty_norm();

  deallog << name << ": " << tria.n_active_cells() << " cells, "
          << n_hanging_faces << " hanging faces, estimate as reference: "
          << (max_difference < tolerance ? "OK" : "WRONG") << std::endl;
  if (max_difference >= tolerance)
    deallog << "  " << max_difference << ' ' << reference.linfty_norm()
            << std::endl;

  // only tetrahedra or only hexahedra: the function with one face quadrature
  // per finite element gives the same
  bool single_shape = true;
  for (const auto &cell : tria.active_cell_iterators())
    single_shape &=
      cell->reference_cell() == tria.begin_active()->reference_cell();
  if (single_shape &&
      (tria.begin_active()->reference_cell() == ReferenceCells::Tetrahedron ||
       tria.begin_active()->reference_cell() == ReferenceCells::Hexahedron))
    {
      const unsigned int i =
        Collections::fe_index(tria.begin_active()->reference_cell());
      hp::QCollection<2> one_per_element;
      for (unsigned int j = 0; j < c.fe.size(); ++j)
        one_per_element.push_back(quadratures[i][0]);
      Vector<float> estimate_one_per_element(tria.n_active_cells());
      KellyErrorEstimator<3>::estimate(c.mapping,
                                       dof_handler,
                                       one_per_element,
                                       {},
                                       u,
                                       estimate_one_per_element);
      estimate_one_per_element -= estimate;
      deallog << "  one face quadrature per element gives the same: "
              << (estimate_one_per_element.linfty_norm() <=
                      1e-6 * estimate.linfty_norm() ?
                    "OK" :
                    "WRONG")
              << std::endl;
    }
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
