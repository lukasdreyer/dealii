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


// Convergence of FE_NedelecSimplex(0) with hanging nodes and tangential
// Dirichlet boundary values in a parallel::distributed::Triangulation:
// curl curl u + u = f on distorted simplex meshes with n x u = n x u_exact,
// set by VectorTools::interpolate_boundary_values_curl_conforming(). In the first cycle a region is refined once, so
// that the mesh has hanging faces; every later cycle refines all cells, so
// that the hanging faces stay and the mesh size halves. With correct
// hanging node constraints, u and curl u converge with first order in L2.
// The meshes are only refined, so the output does not depend on the number
// of processes.

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/edge_element_tools.h>

#include <deal.II/fe/fe_nedelec_simplex.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

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

template <int dim>
class ExactSolution : public Function<dim>
{
public:
  ExactSolution()
    : Function<dim>(dim)
  {}

  void
  vector_value(const Point<dim> &p, Vector<double> &values) const override
  {
    const Tensor<1, dim> u = exact_u(p);
    for (unsigned int d = 0; d < dim; ++d)
      values(d) = u[d];
  }
};



template <int dim>
void
test(const unsigned int n_cycles)
{
  constexpr unsigned int n_curl = (dim == 2) ? 1 : 3;
  const MPI_Comm         comm   = MPI_COMM_WORLD;

  Triangulation<dim> coarse;
  GridGenerator::subdivided_hyper_cube_with_simplices(coarse, 2, 0., 2.);
  GridTools::distort_random(0.2, coarse, false, 42);
  parallel::distributed::Triangulation<dim> tria(comm);
  tria.copy_triangulation(coarse);

  const FE_NedelecSimplex<dim>     fe(0);
  const MappingFE<dim>             mapping(FE_SimplexP<dim>(1));
  const QGaussSimplex<dim>         q_cell(3);
  const FEValuesExtractors::Vector E(0);
  DoFHandler<dim>                  dof_handler(tria);

  double previous_error_u = 0, previous_error_curl = 0;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      if (cycle == 0)
        {
          // refine a region once, which creates the hanging faces
          for (const auto &cell : tria.active_cell_iterators())
            if (cell->is_locally_owned() &&
                cell->center().distance(Point<dim>() +
                                        0.6 * Point<dim>::unit_vector(0)) <
                  0.8)
              cell->set_refine_flag();
          tria.execute_coarsening_and_refinement();
        }
      else
        tria.refine_global(1);

      dof_handler.distribute_dofs(fe);
      const IndexSet &owned = dof_handler.locally_owned_dofs();
      const IndexSet  relevant =
        DoFTools::extract_locally_relevant_dofs(dof_handler);

      AffineConstraints<double> constraints(owned, relevant);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      const ExactSolution<dim> exact_solution;
      VectorTools::interpolate_boundary_values_curl_conforming(
        mapping, dof_handler, {{0, &exact_solution}}, constraints);
      constraints.close();

      DynamicSparsityPattern dsp(relevant);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(dsp, owned, comm, relevant);
      TrilinosWrappers::SparseMatrix matrix;
      matrix.reinit(owned, owned, dsp, comm);
      TrilinosWrappers::MPI::Vector rhs(owned, comm), solution(owned, comm);

      FEValues<dim>     fe_values(mapping,
                              fe,
                              q_cell,
                              update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);
      const unsigned int                   n_dofs = fe.n_dofs_per_cell();
      FullMatrix<double>                   cell_matrix(n_dofs, n_dofs);
      Vector<double>                       cell_rhs(n_dofs);
      std::vector<types::global_dof_index> dof_indices(n_dofs);

      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
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
                    cell_rhs(i) +=
                      f * fe_values[E].value(i, q) * fe_values.JxW(q);
                  }
              }
            cell->get_dof_indices(dof_indices);
            constraints.distribute_local_to_global(
              cell_matrix, cell_rhs, dof_indices, matrix, rhs);
          }
      matrix.compress(VectorOperation::add);
      rhs.compress(VectorOperation::add);

      SolverControl                  control;
      TrilinosWrappers::SolverDirect direct(control);
      direct.solve(matrix, solution, rhs);
      constraints.distribute(solution);

      TrilinosWrappers::MPI::Vector ghosted(owned, relevant, comm);
      ghosted = solution;

      double error_u = 0, error_curl = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(dof_indices);
            for (unsigned int q = 0; q < q_cell.size(); ++q)
              {
                Tensor<1, dim>    u;
                Tensor<1, n_curl> curl;
                for (unsigned int i = 0; i < n_dofs; ++i)
                  {
                    const double value = ghosted(dof_indices[i]);
                    u += value * fe_values[E].value(i, q);
                    const auto c = fe_values[E].curl(i, q);
                    for (unsigned int d = 0; d < n_curl; ++d)
                      curl[d] += value * c[d];
                  }
                const Point<dim> &x = fe_values.quadrature_point(q);
                error_u += (u - exact_u(x)).norm_square() * fe_values.JxW(q);
                error_curl +=
                  (curl - exact_curl(x)).norm_square() * fe_values.JxW(q);
              }
          }
      error_u    = std::sqrt(Utilities::MPI::sum(error_u, comm));
      error_curl = std::sqrt(Utilities::MPI::sum(error_curl, comm));

      unsigned int n_owned_constraints = 0;
      for (const auto &line : constraints.get_lines())
        if (owned.is_element(line.index))
          ++n_owned_constraints;

      deallog << "dim=" << dim << " cells=" << tria.n_global_active_cells()
              << " dofs=" << dof_handler.n_dofs() << " constraints="
              << Utilities::MPI::sum(n_owned_constraints, comm);
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
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  MPILogInitAll                    log;

  test<2>(5);
  test<3>(3);
}
