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


// Convergence of FE_NedelecSimplex(0) for curl curl u + u = f on distorted
// simplex meshes with global refinement. The boundary condition is the
// natural one, n x curl u = n x curl u_exact, so that no projection of
// tangential boundary values is needed. The exact solution satisfies
// curl curl u = u, so f = 2 u:
//   2d: u = (sin y, sin x),        curl u = cos x - cos y
//   3d: u = (sin y, sin z, sin x), curl u = -(cos z, cos x, cos y)
// Expected: first order in L2 for u and for curl u.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nedelec_simplex.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include "../tests.h"


Tensor<1, 2>
exact_u(const Point<2> &p)
{
  return Tensor<1, 2>({std::sin(p[1]), std::sin(p[0])});
}

Tensor<1, 3>
exact_u(const Point<3> &p)
{
  return Tensor<1, 3>({std::sin(p[1]), std::sin(p[2]), std::sin(p[0])});
}

// the curl as a vector with one component in 2d and three in 3d
Tensor<1, 1>
exact_curl(const Point<2> &p)
{
  return Tensor<1, 1>({std::cos(p[0]) - std::cos(p[1])});
}

Tensor<1, 3>
exact_curl(const Point<3> &p)
{
  return Tensor<1, 3>({-std::cos(p[2]), -std::cos(p[0]), -std::cos(p[1])});
}

// n x curl u in 3d; in 2d the boundary term of
// int curl curl u . v = int curl u curl v + int_boundary c (n_y v_x - n_x v_y)
// with c = curl u, written as boundary_vector . v
Tensor<1, 2>
boundary_vector(const Point<2> &p, const Tensor<1, 2> &n)
{
  const double c = exact_curl(p)[0];
  return Tensor<1, 2>({c * n[1], -c * n[0]});
}

Tensor<1, 3>
boundary_vector(const Point<3> &p, const Tensor<1, 3> &n)
{
  return cross_product_3d(n, exact_curl(p));
}



template <int dim>
void
test(const unsigned int n_cycles)
{
  constexpr unsigned int n_curl = (dim == 2) ? 1 : 3;

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube_with_simplices(tria, 2, 0., 2.);
  GridTools::distort_random(0.2, tria, false, 42);

  const FE_NedelecSimplex<dim>      fe(0);
  const MappingFE<dim>              mapping(FE_SimplexP<dim>(1));
  const QGaussSimplex<dim>          q_cell(3);
  const QGaussSimplex<dim - 1>      q_face(3);
  const FEValuesExtractors::Vector  E(0);

  double previous_error_u = 0, previous_error_curl = 0;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      if (cycle > 0)
        tria.refine_global(1);

      DoFHandler<dim> dof_handler(tria);
      dof_handler.distribute_dofs(fe);

      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp);
      SparsityPattern sparsity;
      sparsity.copy_from(dsp);
      SparseMatrix<double> matrix(sparsity);
      Vector<double>       rhs(dof_handler.n_dofs()), solution(rhs.size());

      FEValues<dim>     fe_values(mapping,
                              fe,
                              q_cell,
                              update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);
      FEFaceValues<dim> fe_face_values(mapping,
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
              const Tensor<1, dim> f =
                2. * exact_u(fe_values.quadrature_point(q));
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
                for (unsigned int q = 0; q < q_face.size(); ++q)
                  {
                    const Tensor<1, dim> g =
                      boundary_vector(fe_face_values.quadrature_point(q),
                                      fe_face_values.normal_vector(q));
                    for (unsigned int i = 0; i < n_dofs; ++i)
                      cell_rhs(i) -= g * fe_face_values[E].value(i, q) *
                                     fe_face_values.JxW(q);
                  }
              }
          cell->get_dof_indices(dof_indices);
          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              for (unsigned int j = 0; j < n_dofs; ++j)
                matrix.add(dof_indices[i], dof_indices[j], cell_matrix(i, j));
              rhs(dof_indices[i]) += cell_rhs(i);
            }
        }

      SparseDirectUMFPACK direct;
      direct.initialize(matrix);
      direct.vmult(solution, rhs);

      double error_u = 0, error_curl = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(dof_indices);
          for (unsigned int q = 0; q < q_cell.size(); ++q)
            {
              Tensor<1, dim>    u;
              Tensor<1, n_curl> curl;
              for (unsigned int i = 0; i < n_dofs; ++i)
                {
                  u += solution(dof_indices[i]) * fe_values[E].value(i, q);
                  const auto c = fe_values[E].curl(i, q);
                  for (unsigned int d = 0; d < n_curl; ++d)
                    curl[d] += solution(dof_indices[i]) * c[d];
                }
              const Point<dim> &x = fe_values.quadrature_point(q);
              error_u += (u - exact_u(x)).norm_square() * fe_values.JxW(q);
              error_curl +=
                (curl - exact_curl(x)).norm_square() * fe_values.JxW(q);
            }
        }
      error_u    = std::sqrt(error_u);
      error_curl = std::sqrt(error_curl);

      deallog << "dim=" << dim << " cells=" << tria.n_active_cells()
              << " dofs=" << dof_handler.n_dofs();
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
  test<2>(5);
  test<3>(3);
}
