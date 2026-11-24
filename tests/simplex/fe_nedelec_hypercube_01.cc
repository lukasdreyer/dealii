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


// FE_NedelecHypercube(0) on quadrilaterals and hexahedra:
// 1. duality on distorted meshes: the line integral of shape function i
//    along line l, in the direction of the line object, is delta_il,
// 2. exactness on affine meshes: the DoFs of u = a + b x x (2d:
//    a + b (-y, x)) and of grad p for multilinear p reproduce them,
// 3. hanging node constraints (hanging lines in 2d, hanging quadrilaterals
//    in 3d): exactness for a + b x x and tangential continuity of a random
//    constrained vector,
// 4. convergence in 3d for curl curl u + u = f with tangential Dirichlet
//    values (u = (sin y, sin z, sin x)), where a region is refined once and
//    then all cells in every cycle.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/edge_element_tools.h>

#include <deal.II/fe/fe_nedelec_hypercube.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include "../tests.h"


Tensor<1, 2>
linear_field(const Point<2> &x)
{
  return Tensor<1, 2>({0.3 - 0.7 * x[1], -0.2 + 0.7 * x[0]});
}

Tensor<1, 3>
linear_field(const Point<3> &x)
{
  const Tensor<1, 3> a({0.3, -0.2, 0.5});
  const Tensor<1, 3> b({0.7, -0.4, 0.9});
  return a + cross_product_3d(b, x);
}

// a multilinear function and its gradient
template <int dim>
double
p_value(const Point<dim> &x)
{
  double value = 0.3, product = 1.;
  for (unsigned int d = 0; d < dim; ++d)
    {
      value += (1. + 0.5 * d) * x[d];
      product *= x[d];
    }
  if (dim == 3)
    value += 0.7 * x[0] * x[1] - 0.4 * x[1] * x[2];
  return value + 0.9 * product;
}

template <int dim>
Tensor<1, dim>
p_gradient(const Point<dim> &x)
{
  Tensor<1, dim> g;
  for (unsigned int d = 0; d < dim; ++d)
    {
      double product = 1.;
      for (unsigned int j = 0; j < dim; ++j)
        if (j != d)
          product *= x[j];
      g[d] = 1. + 0.5 * d + 0.9 * product;
    }
  if (dim == 3)
    {
      g[0] += 0.7 * x[1];
      g[1] += 0.7 * x[0] - 0.4 * x[2];
      g[2] += -0.4 * x[1];
    }
  return g;
}



template <int dim>
void
check_duality_and_exactness()
{
  const FE_NedelecHypercube<dim> fe(0);
  const MappingQ<dim>            mapping(1);
  const ReferenceCell ref_cell = ReferenceCells::get_hypercube<dim>();

  // duality on a distorted mesh
  {
    Triangulation<dim> tria;
    GridGenerator::subdivided_hyper_cube(tria, 3);
    GridTools::distort_random(0.2, tria, false, 42);
    tria.refine_global(1);

    const QGauss<1>         q_line(3);
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
    FEValues<dim> fe_values(mapping,
                            fe,
                            Quadrature<dim>(points),
                            update_values | update_jacobians);
    double max_error = 0;
    for (const auto &cell : tria.active_cell_iterators())
      {
        fe_values.reinit(cell);
        for (unsigned int l = 0; l < cell->n_lines(); ++l)
          {
            const Tensor<1, dim> reference_tangent =
              ref_cell.template vertex<dim>(
                ref_cell.line_to_cell_vertices(l, 1)) -
              ref_cell.template vertex<dim>(
                ref_cell.line_to_cell_vertices(l, 0));
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
              {
                double integral = 0;
                for (unsigned int q = 0; q < q_line.size(); ++q)
                  {
                    const unsigned int   qq      = l * q_line.size() + q;
                    const Tensor<1, dim> tangent = apply_transformation(
                      fe_values.jacobian(qq), reference_tangent);
                    Tensor<1, dim> value;
                    for (unsigned int d = 0; d < dim; ++d)
                      value[d] = fe_values.shape_value_component(i, qq, d);
                    integral += q_line.weight(q) * (value * tangent);
                  }
                if (cell->line_orientation(l) !=
                    numbers::default_geometric_orientation)
                  integral = -integral;
                max_error = std::max(max_error,
                                     std::abs(integral - (i == l ? 1. : 0.)));
              }
          }
      }
    deallog << "dim=" << dim << " distorted: duality "
            << (max_error < 1e-12 ? "OK" : "FAILED") << std::endl;
  }

  // exactness on an affine mesh
  {
    Triangulation<dim> tria;
    GridGenerator::subdivided_hyper_cube(tria, 2);
    tria.refine_global(1);
    const QGauss<dim> quadrature(3);
    FEValues<dim>     fe_values(mapping,
                            fe,
                            quadrature,
                            update_values | update_quadrature_points);
    double              max_linear = 0, max_gradient = 0;
    std::vector<double> linear_dofs(fe.n_dofs_per_cell()),
      gradient_dofs(fe.n_dofs_per_cell());
    for (const auto &cell : tria.active_cell_iterators())
      {
        fe_values.reinit(cell);
        for (unsigned int l = 0; l < cell->n_lines(); ++l)
          {
            const auto line = cell->line(l);
            linear_dofs[l] =
              linear_field(line->center()) * (line->vertex(1) - line->vertex(0));
            gradient_dofs[l] =
              p_value(line->vertex(1)) - p_value(line->vertex(0));
          }
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            Tensor<1, dim> u, v;
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
              for (unsigned int d = 0; d < dim; ++d)
                {
                  u[d] +=
                    linear_dofs[i] * fe_values.shape_value_component(i, q, d);
                  v[d] +=
                    gradient_dofs[i] * fe_values.shape_value_component(i, q, d);
                }
            const Point<dim> &x = fe_values.quadrature_point(q);
            max_linear   = std::max(max_linear, (u - linear_field(x)).norm());
            max_gradient = std::max(max_gradient, (v - p_gradient(x)).norm());
          }
      }
    deallog << "dim=" << dim << " affine: linear field "
            << (max_linear < 1e-12 ? "OK" : "FAILED") << ", grad p "
            << (max_gradient < 1e-12 ? "OK" : "FAILED") << std::endl;
  }
}



template <int dim>
void
check_constraints()
{
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria, 3);
  for (unsigned int cycle = 0; cycle < 2; ++cycle)
    {
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->center().distance(Point<dim>() +
                                    (0.3 + 0.3 * cycle) *
                                      Point<dim>::unit_vector(0)) < 0.35)
          cell->set_refine_flag();
      tria.execute_coarsening_and_refinement();
    }

  const FE_NedelecHypercube<dim> fe(0);
  const MappingQ<dim>            mapping(1);
  DoFHandler<dim>                dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

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

  Vector<double> random(dof_handler.n_dofs());
  for (auto &value : random)
    value = random_value<double>(-1., 1.);
  constraints.distribute(random);

  const QGauss<dim - 1>                q_face(2);
  FEFaceValues<dim>                    fine_values(mapping,
                                fe,
                                q_face,
                                update_values | update_quadrature_points |
                                  update_normal_vectors);
  std::vector<types::global_dof_index> fine_dofs(fe.n_dofs_per_cell()),
    coarse_dofs(fe.n_dofs_per_cell());
  unsigned int n_hanging = 0;
  double       max_jump  = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const unsigned int f : cell->face_indices())
      if (!cell->at_boundary(f) && cell->neighbor_is_coarser(f))
        {
          ++n_hanging;
          const auto coarse = cell->neighbor(f);
          fine_values.reinit(cell, f);
          cell->get_dof_indices(fine_dofs);
          coarse->get_dof_indices(coarse_dofs);
          std::vector<Point<dim>> unit_points(q_face.size());
          for (unsigned int q = 0; q < q_face.size(); ++q)
            unit_points[q] = mapping.transform_real_to_unit_cell(
              coarse, fine_values.quadrature_point(q));
          FEValues<dim> coarse_values(mapping,
                                      fe,
                                      Quadrature<dim>(unit_points),
                                      update_values);
          coarse_values.reinit(coarse);
          for (unsigned int q = 0; q < q_face.size(); ++q)
            {
              Tensor<1, dim> jump;
              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                for (unsigned int d = 0; d < dim; ++d)
                  jump[d] += random(fine_dofs[i]) *
                               fine_values.shape_value_component(i, q, d) -
                             random(coarse_dofs[i]) *
                               coarse_values.shape_value_component(i, q, d);
              const Tensor<1, dim> &n = fine_values.normal_vector(q);
              max_jump = std::max(max_jump, (jump - (jump * n) * n).norm());
            }
        }
  deallog << "dim=" << dim << " constraints: cells=" << tria.n_active_cells()
          << " hanging faces=" << n_hanging
          << " constraints=" << constraints.n_constraints() << ", exactness "
          << (distributed.linfty_norm() < 1e-12 ? "OK" : "FAILED")
          << ", tangential continuity " << (max_jump < 1e-12 ? "OK" : "FAILED")
          << std::endl;
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
    values(0) = std::sin(p[1]);
    values(1) = std::sin(p[2]);
    values(2) = std::sin(p[0]);
  }
};



void
check_convergence(const unsigned int n_cycles)
{
  Triangulation<3> tria;
  GridGenerator::subdivided_hyper_cube(tria, 2, 0., 2.);
  GridTools::distort_random(0.2, tria, true, 42);

  const FE_NedelecHypercube<3>     fe(0);
  const MappingQ<3>                mapping(1);
  const QGauss<3>                  q_cell(3);
  const FEValuesExtractors::Vector E(0);
  const ExactSolution              exact_solution;
  Vector<double>                   exact_value(3);

  double previous_error = 0;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      if (cycle == 0)
        {
          for (const auto &cell : tria.active_cell_iterators())
            if (cell->center().distance(Point<3>(0.6, 0., 0.)) < 0.8)
              cell->set_refine_flag();
          tria.execute_coarsening_and_refinement();
        }
      else
        tria.refine_global(1);

      DoFHandler<3> dof_handler(tria);
      dof_handler.distribute_dofs(fe);
      AffineConstraints<double> constraints;
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      VectorTools::interpolate_boundary_values_curl_conforming(
        mapping, dof_handler, {{0, &exact_solution}}, constraints);
      constraints.close();

      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityPattern sparsity;
      sparsity.copy_from(dsp);
      SparseMatrix<double> matrix(sparsity);
      Vector<double>       rhs(dof_handler.n_dofs()), solution(rhs.size());

      FEValues<3>        fe_values(mapping,
                            fe,
                            q_cell,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
      const unsigned int n_dofs = fe.n_dofs_per_cell();
      FullMatrix<double> cell_matrix(n_dofs, n_dofs);
      Vector<double>     cell_rhs(n_dofs);
      std::vector<types::global_dof_index> dof_indices(n_dofs);
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          fe_values.reinit(cell);
          cell_matrix = 0;
          cell_rhs    = 0;
          for (unsigned int q = 0; q < q_cell.size(); ++q)
            {
              exact_solution.vector_value(fe_values.quadrature_point(q),
                                          exact_value);
              const Tensor<1, 3> f(
                {2. * exact_value(0), 2. * exact_value(1), 2. * exact_value(2)});
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

      double error = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(dof_indices);
          for (unsigned int q = 0; q < q_cell.size(); ++q)
            {
              Tensor<1, 3> u;
              for (unsigned int i = 0; i < n_dofs; ++i)
                u += solution(dof_indices[i]) * fe_values[E].value(i, q);
              exact_solution.vector_value(fe_values.quadrature_point(q),
                                          exact_value);
              for (unsigned int d = 0; d < 3; ++d)
                error += std::pow(u[d] - exact_value(d), 2) * fe_values.JxW(q);
            }
        }
      error = std::sqrt(error);
      deallog << "dim=3 convergence: cells=" << tria.n_active_cells()
              << " dofs=" << dof_handler.n_dofs()
              << " constraints=" << constraints.n_constraints();
      if (cycle > 0)
        deallog << std::fixed << std::setprecision(2)
                << " rate_u=" << std::log2(previous_error / error)
                << std::defaultfloat;
      deallog << std::endl;
      previous_error = error;
    }
}



int
main()
{
  initlog();
  check_duality_and_exactness<2>();
  check_duality_and_exactness<3>();
  check_constraints<2>();
  check_constraints<3>();
  check_convergence(3);
}
