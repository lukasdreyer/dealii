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


// Basic properties of FE_NedelecWedge(0):
// 1. Duality on distorted wedge meshes: the line integral of shape function
//    i along line l, in the direction of the line object, is delta_il. This
//    also checks the signs and the tangential continuity.
// 2. Exactness on undistorted (affine) wedge meshes: the DoFs (line
//    integrals along the line objects) of
//    - u = a + b x x reproduce u, with curl u = 2 b,
//    - grad p with p in P1(x,y) x P1(z) (including the terms x z and y z)
//      reproduce grad p, with curl 0.
//    This checks the values and the gradients of the shape functions.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nedelec_wedge.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include "../tests.h"

#include "./simplex_grids.h"


const Tensor<1, 3> a({0.3, -0.2, 0.5});
const Tensor<1, 3> b({0.7, -0.4, 0.9});

Tensor<1, 3>
rotation_field(const Point<3> &x)
{
  return a + cross_product_3d(b, x);
}

double
p_value(const Point<3> &x)
{
  return 0.3 + x[0] + 1.5 * x[1] + 2. * x[2] + 0.7 * x[0] * x[2] -
         0.4 * x[1] * x[2];
}

Tensor<1, 3>
p_gradient(const Point<3> &x)
{
  return Tensor<1, 3>(
    {1. + 0.7 * x[2], 1.5 - 0.4 * x[2], 2. + 0.7 * x[0] - 0.4 * x[1]});
}



void
check_duality(const Triangulation<3> &tria, const std::string &name)
{
  const FE_NedelecWedge<3> fe(0);
  const MappingFE<3>       mapping(FE_WedgeP<3>(1));
  const ReferenceCell      ref_cell = ReferenceCells::Wedge;

  // Gauss points on all lines of the reference cell
  const QGauss<1>       q_line(2);
  std::vector<Point<3>> points;
  for (unsigned int l = 0; l < ref_cell.n_lines(); ++l)
    for (unsigned int q = 0; q < q_line.size(); ++q)
      {
        const Point<3> v0 =
          ref_cell.vertex<3>(ref_cell.line_to_cell_vertices(l, 0));
        const Point<3> v1 =
          ref_cell.vertex<3>(ref_cell.line_to_cell_vertices(l, 1));
        points.push_back(v0 + q_line.point(q)[0] * (v1 - v0));
      }
  FEValues<3> fe_values(mapping,
                        fe,
                        Quadrature<3>(points),
                        update_values | update_jacobians);

  unsigned int n_flipped = 0, n_lines = 0;
  double       max_error = 0;
  for (const auto &cell : tria.active_cell_iterators())
    {
      fe_values.reinit(cell);
      for (unsigned int l = 0; l < cell->n_lines(); ++l)
        {
          ++n_lines;
          const bool flipped = cell->line_orientation(l) !=
                               numbers::default_geometric_orientation;
          n_flipped += flipped;
          // tangent of the mapped reference line, in the direction of the
          // line object
          const Tensor<1, 3> reference_tangent =
            ref_cell.vertex<3>(ref_cell.line_to_cell_vertices(l, 1)) -
            ref_cell.vertex<3>(ref_cell.line_to_cell_vertices(l, 0));
          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            {
              double integral = 0;
              for (unsigned int q = 0; q < q_line.size(); ++q)
                {
                  const unsigned int qq      = l * q_line.size() + q;
                  const Tensor<1, 3> tangent = apply_transformation(
                    fe_values.jacobian(qq), reference_tangent);
                  Tensor<1, 3> value;
                  for (unsigned int d = 0; d < 3; ++d)
                    value[d] = fe_values.shape_value_component(i, qq, d);
                  integral += q_line.weight(q) * (value * tangent);
                }
              if (flipped)
                integral = -integral;
              max_error = std::max(max_error,
                                   std::abs(integral - (i == l ? 1. : 0.)));
            }
        }
    }
  deallog << name << ": cells=" << tria.n_active_cells()
          << " flipped lines seen by cells=" << n_flipped << "/" << n_lines
          << ", duality " << (max_error < 1e-12 ? "OK" : "FAILED")
          << std::endl;
}



void
check_exactness(const Triangulation<3> &tria, const std::string &name)
{
  const FE_NedelecWedge<3> fe(0);
  const MappingFE<3>       mapping(FE_WedgeP<3>(1));
  const QGaussWedge<3>     quadrature(3);
  FEValues<3>              fe_values(mapping,
                        fe,
                        quadrature,
                        update_values | update_gradients |
                          update_quadrature_points);

  double max_rotation_error = 0, max_curl_error = 0, max_gradient_error = 0,
         max_gradient_curl = 0;
  std::vector<double> rotation_dofs(fe.n_dofs_per_cell()),
    gradient_dofs(fe.n_dofs_per_cell());
  for (const auto &cell : tria.active_cell_iterators())
    {
      fe_values.reinit(cell);
      for (unsigned int l = 0; l < cell->n_lines(); ++l)
        {
          // DoFs: line integrals along the line object; u is linear along
          // the line, so the midpoint rule is exact
          const auto     line = cell->line(l);
          const Point<3> mid  = line->center();
          rotation_dofs[l] =
            rotation_field(mid) * (line->vertex(1) - line->vertex(0));
          gradient_dofs[l] = p_value(line->vertex(1)) - p_value(line->vertex(0));
        }
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          Tensor<1, 3> u, v;
          Tensor<2, 3> grad_u, grad_v;
          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            for (unsigned int d = 0; d < 3; ++d)
              {
                u[d] +=
                  rotation_dofs[i] * fe_values.shape_value_component(i, q, d);
                grad_u[d] +=
                  rotation_dofs[i] * fe_values.shape_grad_component(i, q, d);
                v[d] +=
                  gradient_dofs[i] * fe_values.shape_value_component(i, q, d);
                grad_v[d] +=
                  gradient_dofs[i] * fe_values.shape_grad_component(i, q, d);
              }
          const Point<3> &x = fe_values.quadrature_point(q);
          max_rotation_error =
            std::max(max_rotation_error, (u - rotation_field(x)).norm());
          const Tensor<1, 3> curl({grad_u[2][1] - grad_u[1][2],
                                   grad_u[0][2] - grad_u[2][0],
                                   grad_u[1][0] - grad_u[0][1]});
          max_curl_error = std::max(max_curl_error, (curl - 2. * b).norm());
          max_gradient_error =
            std::max(max_gradient_error, (v - p_gradient(x)).norm());
          max_gradient_curl =
            std::max(max_gradient_curl, (grad_v - transpose(grad_v)).norm());
        }
    }
  deallog << name << ": cells=" << tria.n_active_cells() << ", a + b x x "
          << (max_rotation_error < 1e-12 ? "OK" : "FAILED") << ", curl "
          << (max_curl_error < 1e-12 ? "OK" : "FAILED") << ", grad p "
          << (max_gradient_error < 1e-12 ? "OK" : "FAILED")
          << ", curl grad p "
          << (max_gradient_curl < 1e-12 ? "OK" : "FAILED") << std::endl;
  if (std::max({max_rotation_error,
                max_curl_error,
                max_gradient_error,
                max_gradient_curl}) >= 1e-12)
    deallog << "  errors: " << max_rotation_error << ' ' << max_curl_error
            << ' ' << max_gradient_error << ' ' << max_gradient_curl
            << std::endl;
}



int
main()
{
  initlog();

  for (unsigned int r = 0; r < 2; ++r)
    {
      Triangulation<3> tria;
      GridGenerator::subdivided_hyper_cube_with_wedges(tria, 2);
      GridTools::distort_random(0.2, tria, false, 42);
      tria.refine_global(r);
      check_duality(tria, "distorted, refined " + std::to_string(r));
    }
  for (unsigned int r = 0; r < 2; ++r)
    {
      Triangulation<3> tria;
      GridGenerator::subdivided_hyper_cube_with_wedges(tria, 2);
      tria.refine_global(r);
      check_exactness(tria, "affine, refined " + std::to_string(r));
    }
}
