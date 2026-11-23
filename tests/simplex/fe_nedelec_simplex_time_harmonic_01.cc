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


// Time-harmonic Maxwell with absorption for FE_NedelecSimplex(0) with
// hanging nodes in a parallel::distributed::Triangulation:
//   curl curl E - omega^2 E + i omega sigma E = f   in (0,1)^dim,
//   n x E = 0                                        on the boundary,
// with the exact solution E = (1 + i/2) E0,
//   2d: E0 = (sin pi y, sin pi x),                         curl curl E0 = pi^2 E0,
//   3d: E0 = (sin pi y sin pi z, sin pi x sin pi z, sin pi x sin pi y),
//                                                          curl curl E0 = 2 pi^2 E0.
// For omega^2 above the smallest Maxwell eigenvalue the problem is
// indefinite. With E = E_r + i E_i, the real block system is
//   [A  -B] [E_r]   [F_r]
//   [B   A] [E_i] = [F_i],   A = K - omega^2 M,  B = omega sigma M,
// solved by GMRES with the block diagonal preconditioner diag(P, P), where
// P = K + (omega^2 + omega sigma) M is positive definite and approximated by
// Hypre's AMS (through PETSc).
//
// Optional arguments (for runs outside the test suite):
//   dim n_cycles omega sigma [exact]
// where "exact" replaces one application of AMS by an inner CG solve with
// AMS to a tight tolerance (and GMRES by FGMRES).
// Without arguments, small 2d and 3d cases are run; the output contains the
// L2 errors and whether the number of iterations stays below a bound, since
// the exact counts depend on the partition.

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
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <petscpc.h>

#include "../tests.h"


/**
 * AMS through PETSc's interface to Hypre.
 */
class PreconditionAMS : public PETScWrappers::PreconditionBase
{
public:
  void
  initialize(const PETScWrappers::MPI::SparseMatrix        &matrix,
             const PETScWrappers::MPI::SparseMatrix        &gradient,
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



/**
 * The real block form [A -B; B A] of A + i B.
 */
struct ComplexBlockMatrix
{
  const PETScWrappers::MPI::SparseMatrix &A;
  const PETScWrappers::MPI::SparseMatrix &B;

  void
  vmult(PETScWrappers::MPI::BlockVector       &dst,
        const PETScWrappers::MPI::BlockVector &src) const
  {
    PETScWrappers::MPI::Vector tmp(dst.block(0));
    A.vmult(dst.block(0), src.block(0));
    B.vmult(tmp, src.block(1));
    dst.block(0) -= tmp;
    B.vmult(dst.block(1), src.block(0));
    A.vmult_add(dst.block(1), src.block(1));
  }
};



/**
 * diag(P^-1, P^-1), where P^-1 is either one application of AMS or (to
 * separate the quality of the block preconditioner from that of AMS) an
 * inner CG solve with AMS to a tight tolerance.
 */
struct BlockDiagonalPreconditioner
{
  const PETScWrappers::MPI::SparseMatrix &P;
  const PreconditionAMS                  &ams;
  const bool                              inner_solve;

  void
  apply(PETScWrappers::MPI::Vector       &dst,
        const PETScWrappers::MPI::Vector &src) const
  {
    if (inner_solve)
      {
        SolverControl control(1000, 1e-10 * src.l2_norm());
        SolverCG<PETScWrappers::MPI::Vector> cg(control);
        dst = 0;
        cg.solve(P, dst, src, ams);
      }
    else
      ams.vmult(dst, src);
  }

  void
  vmult(PETScWrappers::MPI::BlockVector       &dst,
        const PETScWrappers::MPI::BlockVector &src) const
  {
    apply(dst.block(0), src.block(0));
    apply(dst.block(1), src.block(1));
  }
};



Tensor<1, 2>
exact_e0(const Point<2> &p)
{
  return Tensor<1, 2>(
    {std::sin(numbers::PI * p[1]), std::sin(numbers::PI * p[0])});
}

Tensor<1, 3>
exact_e0(const Point<3> &p)
{
  const double sx = std::sin(numbers::PI * p[0]),
               sy = std::sin(numbers::PI * p[1]),
               sz = std::sin(numbers::PI * p[2]);
  return Tensor<1, 3>({sy * sz, sx * sz, sx * sy});
}

// curl curl E0 = lambda E0
template <int dim>
constexpr double lambda = (dim - 1) * numbers::PI * numbers::PI;



// returns the maximal number of iterations over the cycles
template <int dim>
unsigned int
test(const unsigned int n_cycles,
     const double       omega,
     const double       sigma,
     const bool         verbose,
     const bool         exact_p = false)
{
  const MPI_Comm comm = MPI_COMM_WORLD;

  Triangulation<dim> coarse;
  GridGenerator::subdivided_hyper_cube_with_simplices(coarse, 2);
  // keep the boundary: the exact solution has n x E = 0 on the unit cube
  GridTools::distort_random(0.2, coarse, true, 42);
  parallel::distributed::Triangulation<dim> tria(comm);
  tria.copy_triangulation(coarse);

  const FE_NedelecSimplex<dim>     fe(0);
  const FE_SimplexP<dim>           fe_nodal(1);
  const MappingFE<dim>             mapping(FE_SimplexP<dim>(1));
  const QGaussSimplex<dim>         q_cell(3);
  const FEValuesExtractors::Vector E(0);
  DoFHandler<dim>                  dof_handler(tria), nodal_dh(tria);

  // E = (1 + i/2) E0, f = (lambda - omega^2 + i omega sigma) E
  const double shift = lambda<dim> - omega * omega;
  const double f_r   = shift - 0.5 * omega * sigma;
  const double f_i   = 0.5 * shift + omega * sigma;

  unsigned int max_iterations = 0;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      if (cycle == 0)
        {
          for (const auto &cell : tria.active_cell_iterators())
            if (cell->is_locally_owned() &&
                cell->center().distance(Point<dim>() +
                                        0.3 * Point<dim>::unit_vector(0)) <
                  0.4)
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

      // hanging nodes and perfect conductor (n x E = 0), the same for the
      // real and the imaginary part
      AffineConstraints<double> constraints(owned, relevant);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      const Functions::ZeroFunction<dim> zero(dim);
      VectorTools::interpolate_boundary_values_curl_conforming(
        mapping, dof_handler, {{0, &zero}}, constraints);
      constraints.close();
      AffineConstraints<double> nodal_constraints(nodal_owned, nodal_relevant);
      DoFTools::make_hanging_node_constraints(nodal_dh, nodal_constraints);
      nodal_constraints.close();

      DynamicSparsityPattern dsp(relevant);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(dsp, owned, comm, relevant);
      PETScWrappers::MPI::SparseMatrix A, B, P;
      A.reinit(owned, owned, dsp, comm);
      B.reinit(owned, owned, dsp, comm);
      P.reinit(owned, owned, dsp, comm);
      PETScWrappers::MPI::BlockVector rhs(std::vector<IndexSet>{owned, owned},
                                          comm),
        solution(std::vector<IndexSet>{owned, owned}, comm);

      FEValues<dim> fe_values(mapping,
                              fe,
                              q_cell,
                              update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);
      const unsigned int n_dofs = fe.n_dofs_per_cell();
      FullMatrix<double> stiffness(n_dofs, n_dofs), mass(n_dofs, n_dofs),
        cell_matrix(n_dofs, n_dofs);
      Vector<double>                       rhs_r(n_dofs), rhs_i(n_dofs);
      std::vector<types::global_dof_index> dof_indices(n_dofs);
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            stiffness = 0;
            mass      = 0;
            rhs_r     = 0;
            rhs_i     = 0;
            for (unsigned int q = 0; q < q_cell.size(); ++q)
              {
                const Tensor<1, dim> e0 =
                  exact_e0(fe_values.quadrature_point(q));
                for (unsigned int i = 0; i < n_dofs; ++i)
                  {
                    for (unsigned int j = 0; j < n_dofs; ++j)
                      {
                        stiffness(i, j) += fe_values[E].curl(i, q) *
                                           fe_values[E].curl(j, q) *
                                           fe_values.JxW(q);
                        mass(i, j) += fe_values[E].value(i, q) *
                                      fe_values[E].value(j, q) *
                                      fe_values.JxW(q);
                      }
                    const double v = e0 * fe_values[E].value(i, q) *
                                     fe_values.JxW(q);
                    rhs_r(i) += f_r * v;
                    rhs_i(i) += f_i * v;
                  }
              }
            cell->get_dof_indices(dof_indices);

            // A = K - omega^2 M and the right hand sides
            cell_matrix = stiffness;
            cell_matrix.add(-omega * omega, mass);
            constraints.distribute_local_to_global(cell_matrix,
                                                   dof_indices,
                                                   A);
            constraints.distribute_local_to_global(rhs_r,
                                                   dof_indices,
                                                   rhs.block(0));
            constraints.distribute_local_to_global(rhs_i,
                                                   dof_indices,
                                                   rhs.block(1));

            // B = omega sigma M, without diagonal entries for constrained
            // DoFs, since it couples the real and the imaginary part
            cell_matrix = 0;
            cell_matrix.add(omega * sigma, mass);
            constraints.distribute_local_to_global(cell_matrix,
                                                   dof_indices,
                                                   dof_indices,
                                                   B);

            // the positive definite P = K + (omega^2 + omega sigma) M
            cell_matrix = stiffness;
            cell_matrix.add(omega * omega + omega * sigma, mass);
            constraints.distribute_local_to_global(cell_matrix,
                                                   dof_indices,
                                                   P);
          }
      A.compress(VectorOperation::add);
      B.compress(VectorOperation::add);
      P.compress(VectorOperation::add);
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
      PreconditionAMS ams;
      ams.initialize(P, gradient, constant_fields);
      const double ams_setup_time = timer.wall_time();

      timer.restart();
      SolverControl solver_control(2000, 1e-8 * rhs.l2_norm());
      const BlockDiagonalPreconditioner preconditioner{P, ams, exact_p};
      if (exact_p)
        {
          // the inner solves are not a fixed linear operator
          SolverFGMRES<PETScWrappers::MPI::BlockVector>::AdditionalData data;
          data.max_basis_size = 200;
          SolverFGMRES<PETScWrappers::MPI::BlockVector> solver(solver_control,
                                                               data);
          solver.solve(ComplexBlockMatrix{A, B},
                       solution,
                       rhs,
                       preconditioner);
        }
      else
        {
          SolverGMRES<PETScWrappers::MPI::BlockVector>::AdditionalData data;
          data.max_basis_size        = 200;
          data.right_preconditioning = true;
          SolverGMRES<PETScWrappers::MPI::BlockVector> solver(solver_control,
                                                              data);
          solver.solve(ComplexBlockMatrix{A, B},
                       solution,
                       rhs,
                       preconditioner);
        }
      constraints.distribute(solution.block(0));
      constraints.distribute(solution.block(1));
      const double solve_time = timer.wall_time();
      max_iterations = std::max(max_iterations, solver_control.last_step());

      // L2 errors of the real and the imaginary part
      PETScWrappers::MPI::Vector ghosted_r(owned, relevant, comm),
        ghosted_i(owned, relevant, comm);
      ghosted_r      = solution.block(0);
      ghosted_i      = solution.block(1);
      double error_r = 0, error_i = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(dof_indices);
            for (unsigned int q = 0; q < q_cell.size(); ++q)
              {
                Tensor<1, dim> e_r, e_i;
                for (unsigned int i = 0; i < n_dofs; ++i)
                  {
                    e_r += static_cast<double>(ghosted_r(dof_indices[i])) *
                           fe_values[E].value(i, q);
                    e_i += static_cast<double>(ghosted_i(dof_indices[i])) *
                           fe_values[E].value(i, q);
                  }
                const Tensor<1, dim> e0 =
                  exact_e0(fe_values.quadrature_point(q));
                error_r += (e_r - e0).norm_square() * fe_values.JxW(q);
                error_i += (e_i - 0.5 * e0).norm_square() * fe_values.JxW(q);
              }
          }
      error_r = std::sqrt(Utilities::MPI::sum(error_r, comm));
      error_i = std::sqrt(Utilities::MPI::sum(error_i, comm));

      if (verbose && Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << (exact_p ? "inner CG+AMS " : "AMS ") << "dim=" << dim
                  << " omega=" << omega << " sigma=" << sigma
                  << " cycle=" << cycle
                  << " cells=" << tria.n_global_active_cells()
                  << " dofs=" << 2 * dof_handler.n_dofs()
                  << " iterations=" << solver_control.last_step()
                  << " L2 errors=" << error_r << ' ' << error_i
                  << " setup=" << setup_time << "s ams setup=" << ams_setup_time
                  << "s solve=" << solve_time << "s" << std::endl;
      if (!verbose)
        deallog << "dim=" << dim << " omega=" << omega << " sigma=" << sigma
                << " dofs=" << 2 * dof_handler.n_dofs() << " L2 errors "
                << std::setprecision(3) << error_r << ' ' << error_i
                << std::setprecision(6) << std::endl;
    }
  return max_iterations;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  if (argc == 5 || argc == 6)
    {
      const bool exact_p = (argc == 6 && std::string(argv[5]) == "exact");
      const unsigned int dim      = std::stoi(argv[1]);
      const unsigned int n_cycles = std::stoi(argv[2]);
      const double       omega    = std::stod(argv[3]);
      const double       sigma    = std::stod(argv[4]);
      if (dim == 2)
        test<2>(n_cycles, omega, sigma, true, exact_p);
      else
        test<3>(n_cycles, omega, sigma, true, exact_p);
      return 0;
    }

  MPILogInitAll log;
  // omega^2 = 64 is above the smallest eigenvalues pi^2 (2d), 2 pi^2 (3d).
  // The bound on the iterations only guards against a breakdown: one AMS
  // application per block is not robust for the indefinite problem (the
  // iterations grow with the mesh size, but not with an exact inverse of P).
  const unsigned int it2 = test<2>(5, 8., 1., false);
  deallog << "  at most 500 iterations: " << (it2 <= 500 ? "OK" : "FAILED")
          << std::endl;
  const unsigned int it3 = test<3>(3, 8., 1., false);
  deallog << "  at most 500 iterations: " << (it3 <= 500 ? "OK" : "FAILED")
          << std::endl;
}
