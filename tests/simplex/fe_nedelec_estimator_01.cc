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


// EdgeElementTools::estimate_residual() for curl curl u + u = f with
// u = (sin y, sin z, sin x) (curl curl u = u, div u = 0) and tangential
// Dirichlet values on distorted hybrid meshes with hanging nodes (one cell
// refined first, then all cells in every cycle): the estimate
// eta = (sum_K eta_K^2)^{1/2} and the error ||u - u_h||_{H(curl)} converge
// with first order, and their ratio (the effectivity index) stays bounded.
// (By default on the meshes of tetrahedra, pyramids and tetrahedra,
// hexahedra and pyramids, wedges and tetrahedra, which contain all four
// cell shapes; with the argument "all" also on the meshes of wedges,
// hexahedra and pyramids alone.)

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

#include <deal.II/grid/tria.h>

#include <functional>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_tools.h>

#include "../tests.h"

#include "./pyramid_meshes.h"
#include "./simplex_grids.h"
#include "./wedge_meshes.h"


// the elements, mappings and quadratures, with the FE index of a cell given
// by its reference cell
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
  hp::QCollection<3>       quadrature{QGaussSimplex<3>(3),
                                QGaussWedge<3>(3),
                                QGaussPyramid<3>(4),
                                QGauss<3>(3)};

  static unsigned int
  fe_index(const ReferenceCell &reference_cell)
  {
    if (reference_cell == ReferenceCells::Tetrahedron)
      return 0;
    if (reference_cell == ReferenceCells::Wedge)
      return 1;
    if (reference_cell == ReferenceCells::Pyramid)
      return 2;
    Assert(reference_cell == ReferenceCells::Hexahedron, ExcNotImplemented());
    return 3;
  }

  void
  distribute_dofs(DoFHandler<3> &dof_handler) const
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      cell->set_active_fe_index(fe_index(cell->reference_cell()));
    dof_handler.distribute_dofs(fe);
  }
};



Tensor<1, 3>
linear_field(const Point<3> &x)
{
  const Tensor<1, 3> a({0.3, -0.2, 0.5});
  const Tensor<1, 3> b({0.7, -0.4, 0.9});
  return a + cross_product_3d(b, x);
}

Tensor<1, 3>
exact_u(const Point<3> &p)
{
  return Tensor<1, 3>({std::sin(p[1]), std::sin(p[2]), std::sin(p[0])});
}

Tensor<1, 3>
exact_curl(const Point<3> &p)
{
  return Tensor<1, 3>({-std::cos(p[2]), -std::cos(p[0]), -std::cos(p[1])});
}

class ExactSolution : public Function<3>
{
public:
  ExactSolution()
    : Function<3>(3)
  {}

  void
  vector_value(const Point<3> &p, Vector<double> &values) const override
  {
    const Tensor<1, 3> u = exact_u(p);
    for (unsigned int d = 0; d < 3; ++d)
      values(d) = u[d];
  }
};



void
refine(Triangulation<3> &tria, const Point<3> &center, const double radius)
{
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->center().distance(center) < radius)
      cell->set_refine_flag();
  tria.prepare_coarsening_and_refinement();
  // tetrahedra can only be refined along the line (6,8), also the ones
  // flagged by the mesh smoothing
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->refine_flag_set() &&
        cell->reference_cell() == ReferenceCells::Tetrahedron)
      cell->set_refine_choice(
        static_cast<std::uint8_t>(IsotropicRefinementChoice::cut_tet_68));
  tria.execute_coarsening_and_refinement();
}



// f = 2 u and its gradient (for div f)
class RightHandSide : public Function<3>
{
public:
  RightHandSide()
    : Function<3>(3)
  {}

  void
  vector_value(const Point<3> &p, Vector<double> &values) const override
  {
    const Tensor<1, 3> u = exact_u(p);
    for (unsigned int d = 0; d < 3; ++d)
      values(d) = 2. * u[d];
  }

  void
  vector_gradient(const Point<3>            &p,
                  std::vector<Tensor<1, 3>> &gradients) const override
  {
    for (auto &g : gradients)
      g = Tensor<1, 3>();
    gradients[0][1] = 2. * std::cos(p[1]);
    gradients[1][2] = 2. * std::cos(p[2]);
    gradients[2][0] = 2. * std::cos(p[0]);
  }
};



void
check(Triangulation<3> &tria, const std::string &name, const unsigned int n_cycles)
{
  const Collections                c;
  const FEValuesExtractors::Vector E(0);
  const ExactSolution              exact_solution;
  const RightHandSide              rhs_function;
  const std::array<Quadrature<2>, 2> face_quadratures = {QGaussSimplex<2>(3),
                                                         QGauss<2>(3)};

  deallog << name << ":" << std::endl;
  double previous_error = 0, previous_eta = 0;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      if (cycle == 0)
        {
          // refine one cell, which creates hanging faces on every mesh
          tria.begin_active()->set_refine_flag();
          refine(tria, Point<3>(), -1.);
        }
      else
        refine(tria, Point<3>(), 1e10);

      DoFHandler<3> dof_handler(tria);
      c.distribute_dofs(dof_handler);
      AffineConstraints<double> constraints;
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      VectorTools::interpolate_boundary_values_curl_conforming(
        c.mapping, dof_handler, {{0, &exact_solution}}, constraints);
      constraints.close();

      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityPattern sparsity;
      sparsity.copy_from(dsp);
      SparseMatrix<double> matrix(sparsity);
      Vector<double>       rhs(dof_handler.n_dofs()), solution(rhs.size());

      hp::FEValues<3> hp_fe_values(c.mapping,
                                   c.fe,
                                   c.quadrature,
                                   update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values);
      FullMatrix<double>                   cell_matrix;
      Vector<double>                       cell_rhs;
      std::vector<types::global_dof_index> dof_indices;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          hp_fe_values.reinit(cell);
          const FEValues<3> &fe_values = hp_fe_values.get_present_fe_values();
          const unsigned int n_dofs    = cell->get_fe().n_dofs_per_cell();
          cell_matrix.reinit(n_dofs, n_dofs);
          cell_rhs.reinit(n_dofs);
          dof_indices.resize(n_dofs);
          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
              const Tensor<1, 3> f = 2. * exact_u(fe_values.quadrature_point(q));
              for (unsigned int i = 0; i < n_dofs; ++i)
                {
                  for (unsigned int j = 0; j < n_dofs; ++j)
                    cell_matrix(i, j) +=
                      (fe_values[E].curl(i, q) * fe_values[E].curl(j, q) +
                       fe_values[E].value(i, q) * fe_values[E].value(j, q)) *
                      fe_values.JxW(q);
                  cell_rhs(i) += f * fe_values[E].value(i, q) * fe_values.JxW(q);
                }
            }
          cell->get_dof_indices(dof_indices);
          constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, dof_indices, matrix, rhs);
        }

      SparseDirectUMFPACK direct;
      direct.initialize(matrix);
      direct.vmult(solution, rhs);
      constraints.distribute(solution);

      // the error in the H(curl) norm
      double error2 = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          hp_fe_values.reinit(cell);
          const FEValues<3> &fe_values = hp_fe_values.get_present_fe_values();
          dof_indices.resize(cell->get_fe().n_dofs_per_cell());
          cell->get_dof_indices(dof_indices);
          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
              Tensor<1, 3> u, curl;
              for (unsigned int i = 0; i < dof_indices.size(); ++i)
                {
                  u += solution(dof_indices[i]) * fe_values[E].value(i, q);
                  curl += solution(dof_indices[i]) * fe_values[E].curl(i, q);
                }
              const Point<3> &x = fe_values.quadrature_point(q);
              error2 += ((u - exact_u(x)).norm_square() +
                         (curl - exact_curl(x)).norm_square()) *
                        fe_values.JxW(q);
            }
        }
      const double error = std::sqrt(error2);

      Vector<float> indicators(tria.n_active_cells());
      EdgeElementTools::estimate_residual(c.mapping,
                                          dof_handler,
                                          solution,
                                          rhs_function,
                                          1.,
                                          1.,
                                          c.quadrature,
                                          face_quadratures,
                                          indicators);
      const double eta = indicators.l2_norm();

      deallog << "  cells=" << tria.n_active_cells()
              << " dofs=" << dof_handler.n_dofs() << std::fixed
              << std::setprecision(1) << " effectivity=" << eta / error;
      if (cycle > 0)
        deallog << std::setprecision(2)
                << " rate_error=" << std::log2(previous_error / error)
                << " rate_eta=" << std::log2(previous_eta / eta);
      deallog << std::defaultfloat << std::endl;
      previous_error = error;
      previous_eta   = eta;
    }
}



int
main(int argc, char **argv)
{
  initlog();
  // by default the meshes that contain all four cell shapes; with the
  // argument "all" also the meshes of one shape
  const bool all = (argc > 1 && std::string(argv[1]) == "all");

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
      {"pyramids", [](Triangulation<3> &t) { make_pyramid_mesh(t, 1, 0); }},
      {"pyramids and tetrahedra",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 1, 2); }},
      {"hexahedra and pyramids",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 1); }},
      {"wedges and tetrahedra",
       [](Triangulation<3> &t) { make_wedge_tetrahedron_mesh(t, 1); }}};

  for (const auto &[name, make] : meshes)
    {
      if (!all && (name == "wedges" || name == "hexahedra" || name == "pyramids"))
        continue;
      Triangulation<3> tria;
      make(tria);
      GridTools::distort_random(0.1, tria, true, 42);
      check(tria, name, 3);
    }
}
