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


// Basic properties of FE_NedelecSimplex(0) on distorted simplex meshes:
//
// 1. Duality: on every cell, the line integral of shape function i along
//    line l, taken in the direction of the line object (line->vertex(0) to
//    line->vertex(1)), is delta_il. Since every cell is checked against the
//    direction of the shared line object, this also checks the edge signs
//    and the tangential continuity across faces.
// 2. Exactness: for p in P1, u = sum_l (p(v1) - p(v0)) phi_l equals grad p,
//    and curl u = 0. This checks values and gradients of the shape
//    functions.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nedelec_simplex.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include "../tests.h"


template <int dim>
Tensor<1, dim>
gradient_of_p()
{
  Tensor<1, dim> g;
  for (unsigned int d = 0; d < dim; ++d)
    g[d] = 1.0 + 0.5 * d;
  return g;
}



template <int dim>
double
p_value(const Point<dim> &x)
{
  return 0.3 + gradient_of_p<dim>() * x;
}



template <int dim>
void
test(const unsigned int n_refinements)
{
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube_with_simplices(tria, 2);
  tria.refine_global(n_refinements);
  GridTools::distort_random(0.2, tria, false, 42);

  FE_NedelecSimplex<dim> fe(0);
  MappingFE<dim>         mapping(FE_SimplexP<dim>(1));
  const ReferenceCell    ref_cell = fe.reference_cell();

  // Quadrature on all lines of the reference cell, 2 Gauss points per line,
  // weights for the integral over the line parameter in [0, 1].
  const QGauss<1>         q_line(2);
  std::vector<Point<dim>> points;
  for (unsigned int l = 0; l < ref_cell.n_lines(); ++l)
    for (unsigned int q = 0; q < q_line.size(); ++q)
      {
        const Point<dim> v0 =
          ref_cell.template vertex<dim>(ref_cell.line_to_cell_vertices(l, 0));
        const Point<dim> v1 =
          ref_cell.template vertex<dim>(ref_cell.line_to_cell_vertices(l, 1));
        points.push_back(v0 + q_line.point(q)[0] * (v1 - v0));
      }
  FEValues<dim> fe_values_lines(mapping,
                                fe,
                                Quadrature<dim>(points),
                                update_values | update_quadrature_points);

  const QGaussSimplex<dim> q_cell(2);
  FEValues<dim>            fe_values_cell(mapping,
                               fe,
                               q_cell,
                               update_values | update_gradients |
                                 update_quadrature_points);

  unsigned int n_flipped = 0, n_lines = 0;
  double       max_duality_error = 0, max_value_error = 0, max_curl = 0;

  for (const auto &cell : tria.active_cell_iterators())
    {
      fe_values_lines.reinit(cell);
      fe_values_cell.reinit(cell);

      // 1. duality
      for (unsigned int l = 0; l < cell->n_lines(); ++l)
        {
          ++n_lines;
          if (cell->line_orientation(l) !=
              numbers::default_geometric_orientation)
            ++n_flipped;
          const Tensor<1, dim> tangent =
            cell->line(l)->vertex(1) - cell->line(l)->vertex(0);
          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            {
              double integral = 0;
              for (unsigned int q = 0; q < q_line.size(); ++q)
                {
                  const unsigned int qq = l * q_line.size() + q;
                  Tensor<1, dim>     value;
                  for (unsigned int d = 0; d < dim; ++d)
                    value[d] = fe_values_lines.shape_value_component(i, qq, d);
                  integral += q_line.weight(q) * (value * tangent);
                }
              max_duality_error =
                std::max(max_duality_error,
                         std::abs(integral - (i == l ? 1.0 : 0.0)));
            }
        }

      // 2. exactness for gradients of P1
      std::vector<double> dof_values(fe.n_dofs_per_cell());
      for (unsigned int l = 0; l < cell->n_lines(); ++l)
        {
          // the DoF of line l refers to the direction of the line object
          dof_values[l] =
            p_value(cell->line(l)->vertex(1)) - p_value(cell->line(l)->vertex(0));
        }
      for (unsigned int q = 0; q < q_cell.size(); ++q)
        {
          Tensor<1, dim> u;
          Tensor<2, dim> grad_u;
          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            for (unsigned int d = 0; d < dim; ++d)
              {
                u[d] += dof_values[i] * fe_values_cell.shape_value_component(i, q, d);
                grad_u[d] +=
                  dof_values[i] * fe_values_cell.shape_grad_component(i, q, d);
              }
          max_value_error =
            std::max(max_value_error, (u - gradient_of_p<dim>()).norm());
          // curl u = antisymmetric part of grad u
          max_curl = std::max(max_curl, (grad_u - transpose(grad_u)).norm());
        }
    }

  deallog << "dim=" << dim << " refinements=" << n_refinements
          << " cells=" << tria.n_active_cells() << " flipped lines seen by cells="
          << n_flipped << "/" << n_lines << std::endl;
  deallog << "  duality " << (max_duality_error < 1e-12 ? "OK" : "FAILED")
          << ", gradient reproduction "
          << (max_value_error < 1e-12 ? "OK" : "FAILED") << ", curl "
          << (max_curl < 1e-12 ? "OK" : "FAILED") << std::endl;
  if (max_duality_error >= 1e-12 || max_value_error >= 1e-12 ||
      max_curl >= 1e-12)
    deallog << "  errors: " << max_duality_error << ' ' << max_value_error
            << ' ' << max_curl << std::endl;
}



int
main()
{
  initlog();

  for (unsigned int r = 0; r < 2; ++r)
    test<2>(r);
  for (unsigned int r = 0; r < 2; ++r)
    test<3>(r);
}
