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


// Hypre's auxiliary space Maxwell solver (AMS, through PETSc) for
// FE_NedelecSimplex(0) with hanging nodes in a
// parallel::distributed::Triangulation: CG preconditioned with AMS for
// curl curl u + beta u = f with tangential Dirichlet values, on distorted
// simplex meshes where a region is refined once and then all cells in
// every cycle. AMS gets the discrete gradient from
// DoFTools::make_discrete_gradient() and the edge vectors of the constant
// fields, G x, G y (and G z), from the vertex coordinates. The number of
// iterations should be bounded independently of the mesh size and beta.
//
// Optional arguments (for runs outside the test suite):
//   dim n_cycles beta
// Without arguments, a small 2d and 3d case is run; the output contains the
// L2 errors and whether the iteration counts stay below a bound, since the
// exact counts depend on the partition.

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

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
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/sparsity_tools.h>

#include <petscpc.h>

#include "../tests.h"


/**
 * AMS through PETSc's interface to Hypre.
 */
class PreconditionAMS : public PETScWrappers::PreconditionBase
{
public:
  void
  initialize(const PETScWrappers::MPI::SparseMatrix &matrix,
             const PETScWrappers::MPI::SparseMatrix &gradient,
             const std::vector<PETScWrappers::MPI::Vector> &constant_fields)
  {
    create_pc_with_mat(matrix);
    PetscErrorCode ierr = PCSetType(pc, PCHYPRE);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    ierr = PCHYPRESetType(pc, "ams");
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    ierr = PCHYPRESetDiscreteGradient(pc, gradient);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    ierr = PCHYPRESetEdgeConstantVectors(
      pc,
      constant_fields[0],
      constant_fields[1],
      constant_fields.size() > 2 ? static_cast<Vec>(constant_fields[2]) :
                                   nullptr);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    ierr = PCSetUp(pc);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
  }
};



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

// curl curl u = u for the solutions above



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



// returns the maximal number of iterations over the cycles
template <int dim>
unsigned int
test(const unsigned int n_cycles, const double beta, const bool verbose)
{
  const MPI_Comm comm = MPI_COMM_WORLD;

  Triangulation<dim> coarse;
  GridGenerator::subdivided_hyper_cube_with_simplices(coarse, 2, 0., 2.);
  GridTools::distort_random(0.2, coarse, false, 42);
  parallel::distributed::Triangulation<dim> tria(comm);
  tria.copy_triangulation(coarse);

  const FE_NedelecSimplex<dim>     fe(0);
  const FE_SimplexP<dim>           fe_nodal(1);
  const MappingFE<dim>             mapping(FE_SimplexP<dim>(1));
  const QGaussSimplex<dim>         q_cell(3);
  const FEValuesExtractors::Vector E(0);
  DoFHandler<dim>                  dof_handler(tria), nodal_dh(tria);

  unsigned int max_iterations = 0;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      if (cycle == 0)
        {
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

      Timer timer;
      dof_handler.distribute_dofs(fe);
      nodal_dh.distribute_dofs(fe_nodal);
      const IndexSet &owned = dof_handler.locally_owned_dofs();
      const IndexSet  relevant =
        DoFTools::extract_locally_relevant_dofs(dof_handler);
      const IndexSet &nodal_owned = nodal_dh.locally_owned_dofs();
      const IndexSet  nodal_relevant =
        DoFTools::extract_locally_relevant_dofs(nodal_dh);

      AffineConstraints<double> constraints(owned, relevant);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      const ExactSolution<dim> exact_solution;
      VectorTools::interpolate_boundary_values_curl_conforming(
        mapping, dof_handler, {{0, &exact_solution}}, constraints);
      constraints.close();
      AffineConstraints<double> nodal_constraints(nodal_owned, nodal_relevant);
      DoFTools::make_hanging_node_constraints(nodal_dh, nodal_constraints);
      nodal_constraints.close();

      // system matrix
      DynamicSparsityPattern dsp(relevant);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(dsp, owned, comm, relevant);
      PETScWrappers::MPI::SparseMatrix matrix;
      matrix.reinit(owned, owned, dsp, comm);
      PETScWrappers::MPI::Vector rhs(owned, comm), solution(owned, comm);

      FEValues<dim> fe_values(mapping,
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
                  (1. + beta) * exact_u(fe_values.quadrature_point(q));
                for (unsigned int i = 0; i < n_dofs; ++i)
                  {
                    for (unsigned int j = 0; j < n_dofs; ++j)
                      cell_matrix(i, j) +=
                        (fe_values[E].curl(i, q) * fe_values[E].curl(j, q) +
                         beta * fe_values[E].value(i, q) *
                           fe_values[E].value(j, q)) *
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

      // discrete gradient and the edge vectors of the constant fields
      DynamicSparsityPattern gradient_dsp(dof_handler.n_dofs(),
                                          nodal_dh.n_dofs(),
                                          owned);
      DoFTools::make_discrete_gradient_sparsity_pattern(
        dof_handler, constraints, nodal_dh, nodal_constraints, gradient_dsp);
      PETScWrappers::MPI::SparseMatrix gradient;
      gradient.reinit(owned, nodal_owned, gradient_dsp, comm);
      DoFTools::make_discrete_gradient(
        dof_handler, constraints, nodal_dh, nodal_constraints, gradient);

      const std::map<types::global_dof_index, Point<dim>> support_points =
        DoFTools::map_dofs_to_support_points(mapping, nodal_dh);
      std::vector<PETScWrappers::MPI::Vector> constant_fields(dim);
      for (unsigned int d = 0; d < dim; ++d)
        {
          PETScWrappers::MPI::Vector coordinate(nodal_owned, comm);
          for (const types::global_dof_index dof : nodal_owned)
            coordinate(dof) = support_points.at(dof)[d];
          coordinate.compress(VectorOperation::insert);
          constant_fields[d].reinit(owned, comm);
          gradient.vmult(constant_fields[d], coordinate);
        }
      const double setup_time = timer.wall_time();

      timer.restart();
      PreconditionAMS preconditioner;
      preconditioner.initialize(matrix, gradient, constant_fields);
      const double ams_setup_time = timer.wall_time();

      timer.restart();
      SolverControl solver_control(1000, 1e-10 * rhs.l2_norm());
      PETScWrappers::SolverCG solver(solver_control);
      solver.solve(matrix, solution, rhs, preconditioner);
      constraints.distribute(solution);
      const double solve_time = timer.wall_time();
      max_iterations = std::max(max_iterations, solver_control.last_step());

      // L2 error of u, as a check of the solution
      PETScWrappers::MPI::Vector ghosted(owned, relevant, comm);
      ghosted       = solution;
      double error2 = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(dof_indices);
            for (unsigned int q = 0; q < q_cell.size(); ++q)
              {
                Tensor<1, dim> u;
                for (unsigned int i = 0; i < n_dofs; ++i)
                  u += static_cast<double>(ghosted(dof_indices[i])) *
                       fe_values[E].value(i, q);
                error2 += (u - exact_u(fe_values.quadrature_point(q)))
                            .norm_square() *
                          fe_values.JxW(q);
              }
          }
      const double error = std::sqrt(Utilities::MPI::sum(error2, comm));

      if (verbose && Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << "dim=" << dim << " beta=" << beta << " cycle=" << cycle
                  << " cells=" << tria.n_global_active_cells()
                  << " dofs=" << dof_handler.n_dofs()
                  << " iterations=" << solver_control.last_step()
                  << " L2 error=" << error << " setup=" << setup_time
                  << "s ams setup=" << ams_setup_time
                  << "s solve=" << solve_time << "s" << std::endl;
      if (!verbose)
        deallog << "dim=" << dim << " beta=" << beta
                << " dofs=" << dof_handler.n_dofs() << " L2 error "
                << std::setprecision(3) << error << std::setprecision(6)
                << std::endl;
    }
  return max_iterations;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  if (argc == 4)
    {
      const unsigned int dim      = std::stoi(argv[1]);
      const unsigned int n_cycles = std::stoi(argv[2]);
      const double       beta     = std::stod(argv[3]);
      if (dim == 2)
        test<2>(n_cycles, beta, true);
      else
        test<3>(n_cycles, beta, true);
      return 0;
    }

  MPILogInitAll log;
  for (const double beta : {1e-2, 1., 1e2})
    {
      const unsigned int it2 = test<2>(4, beta, false);
      deallog << "  at most 30 iterations: " << (it2 <= 30 ? "OK" : "FAILED")
              << std::endl;
      const unsigned int it3 = test<3>(3, beta, false);
      deallog << "  at most 30 iterations: " << (it3 <= 30 ? "OK" : "FAILED")
              << std::endl;
    }
}
