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


// Basic properties of FE_NedelecPyramid(0) on the pyramid mesh of
// pyramid_meshes.h (cubes split into 6 pyramids, with permuted vertices):
// 1. Duality on a distorted mesh: the line integral of shape function i
//    along line l, in the direction of the line object, is delta_il.
// 2. Exactness on the undistorted (affine) mesh: the DoFs of u = a + b x x
//    reproduce u, with curl u = 2 b, and those of grad p for linear p
//    reproduce grad p.
// 3. Stokes' theorem on every cell: int_K curl phi_i = int_dK n x phi_i,
//    which checks the gradients of the rational shape functions and the
//    quadrature.
// 4. At the apex (where the rational expressions are 0/0, e.g. in the
//    patches of DataOut) the values and curls are finite and equal to their
//    limits along the axis of the pyramid. (The full gradients of the
//    functions of the apex lines grow like 1/(1 - z) towards the apex, as
//    N_apex times second derivatives of the base vertex functions; this
//    part is symmetric and does not enter the curl.)

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nedelec_pyramid.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/q_collection.h>

#include "../tests.h"

#include "./pyramid_meshes.h"


const Tensor<1, 3> a({0.3, -0.2, 0.5});
const Tensor<1, 3> b({0.7, -0.4, 0.9});
const Tensor<1, 3> grad_p({1., 1.5, -2.});

Tensor<1, 3>
rotation_field(const Point<3> &x)
{
  return a + cross_product_3d(b, x);
}

Tensor<1, 3>
curl_of(const Tensor<2, 3> &grad)
{
  return Tensor<1, 3>({grad[2][1] - grad[1][2],
                       grad[0][2] - grad[2][0],
                       grad[1][0] - grad[0][1]});
}



void
check_duality(const Triangulation<3> &tria)
{
  const FE_NedelecPyramid<3> fe(0);
  const MappingFE<3>         mapping(FE_PyramidP<3>(1));
  const ReferenceCell        ref_cell = ReferenceCells::Pyramid;

  // Gauss points on all lines of the reference cell
  const QGauss<1>       q_line(4);
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
  deallog << "distorted: cells=" << tria.n_active_cells()
          << " flipped lines seen by cells=" << n_flipped << "/" << n_lines
          << ", duality " << (max_error < 1e-12 ? "OK" : "FAILED")
          << std::endl;
}



void
check_exactness_and_stokes(const Triangulation<3> &tria)
{
  const FE_NedelecPyramid<3> fe(0);
  const MappingFE<3>         mapping(FE_PyramidP<3>(1));
  const QGaussPyramid<3>     quadrature(4);
  FEValues<3>                fe_values(mapping,
                        fe,
                        quadrature,
                        update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);
  // one quadrature per face: the base, then four triangles
  const hp::QCollection<2> face_quadrature{QGauss<2>(4),
                                           QGaussSimplex<2>(4),
                                           QGaussSimplex<2>(4),
                                           QGaussSimplex<2>(4),
                                           QGaussSimplex<2>(4)};
  FEFaceValues<3> fe_face_values(mapping,
                                 fe,
                                 face_quadrature,
                                 update_values | update_normal_vectors |
                                   update_JxW_values);

  double max_rotation = 0, max_curl = 0, max_gradient = 0, max_stokes = 0;
  std::vector<double> rotation_dofs(fe.n_dofs_per_cell()),
    gradient_dofs(fe.n_dofs_per_cell());
  for (const auto &cell : tria.active_cell_iterators())
    {
      fe_values.reinit(cell);
      for (unsigned int l = 0; l < cell->n_lines(); ++l)
        {
          const auto         line = cell->line(l);
          const Tensor<1, 3> t    = line->vertex(1) - line->vertex(0);
          rotation_dofs[l]        = rotation_field(line->center()) * t;
          gradient_dofs[l]        = grad_p * t;
        }

      std::vector<Tensor<1, 3>> volume_integrals(fe.n_dofs_per_cell());
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          Tensor<1, 3> u, v;
          Tensor<2, 3> grad_u;
          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            {
              Tensor<2, 3> grad_i;
              for (unsigned int d = 0; d < 3; ++d)
                {
                  u[d] +=
                    rotation_dofs[i] * fe_values.shape_value_component(i, q, d);
                  v[d] +=
                    gradient_dofs[i] * fe_values.shape_value_component(i, q, d);
                  grad_i[d] = fe_values.shape_grad_component(i, q, d);
                }
              grad_u += rotation_dofs[i] * grad_i;
              volume_integrals[i] += curl_of(grad_i) * fe_values.JxW(q);
            }
          const Point<3> &x = fe_values.quadrature_point(q);
          max_rotation = std::max(max_rotation, (u - rotation_field(x)).norm());
          max_curl     = std::max(max_curl, (curl_of(grad_u) - 2. * b).norm());
          max_gradient = std::max(max_gradient, (v - grad_p).norm());
        }

      // Stokes: int_K curl phi = int_dK n x phi
      std::vector<Tensor<1, 3>> boundary_integrals(fe.n_dofs_per_cell());
      for (const unsigned int f : cell->face_indices())
        {
          fe_face_values.reinit(cell, f);
          for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
              {
                Tensor<1, 3> value;
                for (unsigned int d = 0; d < 3; ++d)
                  value[d] = fe_face_values.shape_value_component(i, q, d);
                boundary_integrals[i] +=
                  cross_product_3d(fe_face_values.normal_vector(q), value) *
                  fe_face_values.JxW(q);
              }
        }
      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        max_stokes = std::max(max_stokes,
                              (volume_integrals[i] - boundary_integrals[i])
                                .norm());
    }
  deallog << "affine: cells=" << tria.n_active_cells() << ", a + b x x "
          << (max_rotation < 1e-12 ? "OK" : "FAILED") << ", curl "
          << (max_curl < 1e-12 ? "OK" : "FAILED") << ", grad p "
          << (max_gradient < 1e-12 ? "OK" : "FAILED") << ", Stokes "
          << (max_stokes < 1e-12 ? "OK" : "FAILED") << std::endl;
  if (std::max({max_rotation, max_curl, max_gradient, max_stokes}) >= 1e-12)
    deallog << "  errors: " << max_rotation << ' ' << max_curl << ' '
            << max_gradient << ' ' << max_stokes << std::endl;
}



void
check_apex()
{
  Triangulation<3> tria;
  GridGenerator::reference_cell(tria, ReferenceCells::Pyramid);
  const FE_NedelecPyramid<3> fe(0);
  const MappingFE<3>         mapping(FE_PyramidP<3>(1));
  // the apex and a point below it on the axis
  const std::vector<Point<3>> points = {Point<3>(0, 0, 1),
                                        Point<3>(0, 0, 1 - 1e-6)};
  FEValues<3> fe_values(mapping,
                        fe,
                        Quadrature<3>(points),
                        update_values | update_gradients);
  fe_values.reinit(tria.begin_active());
  const FEValuesExtractors::Vector E(0);
  bool   finite = true;
  double difference = 0, size = 0;
  for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
    {
      const Tensor<1, 3> value = fe_values[E].value(i, 0);
      const Tensor<1, 3> curl  = fe_values[E].curl(i, 0);
      for (unsigned int c = 0; c < 3; ++c)
        finite = finite && std::isfinite(value[c]) && std::isfinite(curl[c]);
      difference = std::max({difference,
                             (value - fe_values[E].value(i, 1)).norm(),
                             (curl - fe_values[E].curl(i, 1)).norm()});
      size       = std::max({size, value.norm(), curl.norm()});
    }
  deallog << "apex: values and curls finite: " << (finite ? "OK" : "FAILED")
          << ", equal to the limits along the axis: "
          << (difference < 1e-5 * size ? "OK" : "FAILED") << std::endl;
}



int
main()
{
  initlog();
  {
    Triangulation<3> tria;
    make_pyramid_mesh(tria, 2, 0);
    GridTools::distort_random(0.2, tria, false, 42);
    check_duality(tria);
  }
  {
    Triangulation<3> tria;
    make_pyramid_mesh(tria, 2, 0);
    check_exactness_and_stokes(tria);
  }
  check_apex();
}
