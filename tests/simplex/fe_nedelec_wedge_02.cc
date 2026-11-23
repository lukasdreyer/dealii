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


// Hanging node constraints and convergence of FE_NedelecWedge(0) on
// adaptively refined wedge meshes, with hanging triangular and
// quadrilateral faces:
// 1. exactness: the DoFs of u = a + b x x (affine mesh) satisfy the
//    constraints,
// 2. tangential continuity of a random constrained vector across all hanging
//    faces,
// 3. convergence for curl curl u + u = f with natural boundary conditions
//    (u = (sin y, sin z, sin x), curl curl u = u) on a distorted mesh where a
//    region is refined once and then all cells in every cycle: first order
//    in L2 for u and curl u.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nedelec_wedge.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include "../tests.h"

#include "./simplex_grids.h"


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

// one quadrature per face of the wedge: two triangles, three quadrilaterals
hp::QCollection<2>
face_quadratures(const unsigned int n)
{
  return hp::QCollection<2>{QGaussSimplex<2>(n),
                            QGaussSimplex<2>(n),
                            QGauss<2>(n),
                            QGauss<2>(n),
                            QGauss<2>(n)};
}

void
refine_region(Triangulation<3> &tria, const Point<3> &center, const double r)
{
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->center().distance(center) < r)
      cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();
}



void
check_constraints()
{
  Triangulation<3> tria;
  GridGenerator::subdivided_hyper_cube_with_wedges(tria, 3);
  for (unsigned int cycle = 0; cycle < 2; ++cycle)
    refine_region(tria, Point<3>(0.3 + 0.3 * cycle, 0.4, 0.5), 0.35);

  const FE_NedelecWedge<3> fe(0);
  const MappingFE<3>       mapping(FE_WedgeP<3>(1));
  DoFHandler<3>            dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  // 1. exactness
  Vector<double> exact(dof_handler.n_dofs());
  for (const auto &cell : dof_handler.active_cell_iterators())
    for (unsigned int l = 0; l < cell->n_lines(); ++l)
      {
        const auto line = cell->line(l);
        exact(line->dof_index(0)) =
          linear_field(line->center()) * (line->vertex(1) - line->vertex(0));
      }
  Vector<double> distributed = exact;
  constraints.distribute(distributed);
  distributed -= exact;

  // 2. tangential continuity
  Vector<double> random(dof_handler.n_dofs());
  for (auto &value : random)
    value = random_value<double>(-1., 1.);
  constraints.distribute(random);

  FEFaceValues<3> fine_values(mapping,
                              fe,
                              face_quadratures(2),
                              update_values | update_quadrature_points |
                                update_normal_vectors);
  std::vector<types::global_dof_index> fine_dofs(fe.n_dofs_per_cell()),
    coarse_dofs(fe.n_dofs_per_cell());
  unsigned int n_triangles = 0, n_quads = 0;
  double       max_jump = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const unsigned int f : cell->face_indices())
      if (!cell->at_boundary(f) && cell->neighbor_is_coarser(f))
        {
          if (cell->face(f)->reference_cell() == ReferenceCells::Triangle)
            ++n_triangles;
          else
            ++n_quads;
          const auto coarse = cell->neighbor(f);
          fine_values.reinit(cell, f);
          cell->get_dof_indices(fine_dofs);
          coarse->get_dof_indices(coarse_dofs);

          const unsigned int    n_q = fine_values.n_quadrature_points;
          std::vector<Point<3>> unit_points(n_q);
          for (unsigned int q = 0; q < n_q; ++q)
            unit_points[q] =
              mapping.transform_real_to_unit_cell(coarse,
                                                  fine_values.quadrature_point(
                                                    q));
          FEValues<3> coarse_values(mapping,
                                    fe,
                                    Quadrature<3>(unit_points),
                                    update_values);
          coarse_values.reinit(coarse);
          for (unsigned int q = 0; q < n_q; ++q)
            {
              Tensor<1, 3> jump;
              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                for (unsigned int d = 0; d < 3; ++d)
                  jump[d] +=
                    random(fine_dofs[i]) *
                      fine_values.shape_value_component(i, q, d) -
                    random(coarse_dofs[i]) *
                      coarse_values.shape_value_component(i, q, d);
              const Tensor<1, 3> &n = fine_values.normal_vector(q);
              max_jump = std::max(max_jump, (jump - (jump * n) * n).norm());
            }
        }

  deallog << "constraints: cells=" << tria.n_active_cells()
          << " hanging triangles=" << n_triangles
          << " hanging quadrilaterals=" << n_quads
          << " constraints=" << constraints.n_constraints() << std::endl;
  deallog << "  exactness "
          << (distributed.linfty_norm() < 1e-12 ? "OK" : "FAILED")
          << ", tangential continuity " << (max_jump < 1e-12 ? "OK" : "FAILED")
          << std::endl;
  if (distributed.linfty_norm() >= 1e-12 || max_jump >= 1e-12)
    deallog << "  errors: " << distributed.linfty_norm() << ' ' << max_jump
            << std::endl;
}



void
check_convergence(const unsigned int n_cycles)
{
  Triangulation<3> tria;
  GridGenerator::subdivided_hyper_cube_with_wedges(tria, 2, 0., 2.);
  GridTools::distort_random(0.2, tria, false, 42);

  const FE_NedelecWedge<3>         fe(0);
  const MappingFE<3>               mapping(FE_WedgeP<3>(1));
  const QGaussWedge<3>             q_cell(3);
  const hp::QCollection<2>         q_face = face_quadratures(3);
  const FEValuesExtractors::Vector E(0);

  double previous_error_u = 0, previous_error_curl = 0;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      if (cycle == 0)
        refine_region(tria, Point<3>(0.6, 0., 0.), 0.8);
      else
        tria.refine_global(1);

      DoFHandler<3> dof_handler(tria);
      dof_handler.distribute_dofs(fe);
      AffineConstraints<double> constraints;
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();

      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityPattern sparsity;
      sparsity.copy_from(dsp);
      SparseMatrix<double> matrix(sparsity);
      Vector<double>       rhs(dof_handler.n_dofs()), solution(rhs.size());

      FEValues<3>     fe_values(mapping,
                            fe,
                            q_cell,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
      FEFaceValues<3> fe_face_values(mapping,
                                     fe,
                                     q_face,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);
      const unsigned int                   n_dofs = fe.n_dofs_per_cell();
      FullMatrix<double>                   cell_matrix(n_dofs, n_dofs);
      Vector<double>                       cell_rhs(n_dofs);
      std::vector<types::global_dof_index> dof_indices(n_dofs);
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          fe_values.reinit(cell);
          cell_matrix = 0;
          cell_rhs    = 0;
          for (unsigned int q = 0; q < q_cell.size(); ++q)
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
          for (const unsigned int face : cell->face_indices())
            if (cell->at_boundary(face))
              {
                fe_face_values.reinit(cell, face);
                for (unsigned int q = 0; q < fe_face_values.n_quadrature_points;
                     ++q)
                  {
                    // int curl curl u . v = int curl u . curl v
                    //                       + int (n x curl u) . v
                    const Tensor<1, 3> g =
                      cross_product_3d(fe_face_values.normal_vector(q),
                                       exact_curl(
                                         fe_face_values.quadrature_point(q)));
                    for (unsigned int i = 0; i < n_dofs; ++i)
                      cell_rhs(i) -= g * fe_face_values[E].value(i, q) *
                                     fe_face_values.JxW(q);
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

      double error_u = 0, error_curl = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(dof_indices);
          for (unsigned int q = 0; q < q_cell.size(); ++q)
            {
              Tensor<1, 3> u, curl;
              for (unsigned int i = 0; i < n_dofs; ++i)
                {
                  u += solution(dof_indices[i]) * fe_values[E].value(i, q);
                  curl += solution(dof_indices[i]) * fe_values[E].curl(i, q);
                }
              const Point<3> &x = fe_values.quadrature_point(q);
              error_u += (u - exact_u(x)).norm_square() * fe_values.JxW(q);
              error_curl +=
                (curl - exact_curl(x)).norm_square() * fe_values.JxW(q);
            }
        }
      error_u    = std::sqrt(error_u);
      error_curl = std::sqrt(error_curl);

      deallog << "convergence: cells=" << tria.n_active_cells()
              << " dofs=" << dof_handler.n_dofs()
              << " constraints=" << constraints.n_constraints();
      if (cycle > 0)
        deallog << std::fixed << std::setprecision(2)
                << " rate_u=" << std::log2(previous_error_u / error_u)
                << " rate_curl=" << std::log2(previous_error_curl / error_curl)
                << std::defaultfloat;
      deallog << std::endl;
      previous_error_u    = error_u;
      previous_error_curl = error_curl;
    }
}



int
main()
{
  initlog();
  check_constraints();
  check_convergence(3);
}
