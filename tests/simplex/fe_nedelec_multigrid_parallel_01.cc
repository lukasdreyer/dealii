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


// Geometric multigrid in parallel for curl curl u + beta u = f with
// FE_NedelecSimplex(0) on an adaptively refined
// parallel::distributed::Triangulation (t8code, hanging nodes), with
// Trilinos matrices:
// - the meshes of the levels from
//   MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence()
//   (copies of the fine mesh, coarsened once per level and repartitioned,
//   so a coarse cell can be owned by another process than its children; the
//   prolongation gets the coarse DoF indices of such cells from their
//   owners),
// - the prolongation of DoFTools::make_edge_element_prolongation() in
//   MGTransferMatrices,
// - MGSmootherHiptmair with TrilinosWrappers::PreconditionSSOR; the nodal
//   operator G^T A G is assembled directly as beta (grad p, grad q), which
//   is the same operator,
// - CG with PreconditionMG (one V(1,1) cycle), compared with the edge
//   smoother alone.
// The Frobenius norms of the prolongations and discrete gradients depend
// only on the meshes and must be the same for all numbers of processes.

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
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_smoother_hiptmair.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrices.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"


using VectorType = TrilinosWrappers::MPI::Vector;
using MatrixType = TrilinosWrappers::SparseMatrix;


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



// the DoFHandlers and constraints of one level
template <int dim>
struct Level
{
  Level(const Triangulation<dim> &tria)
    : edge_dh(tria)
    , nodal_dh(tria)
  {}

  DoFHandler<dim>           edge_dh;
  DoFHandler<dim>           nodal_dh;
  AffineConstraints<double> edge_constraints;
  AffineConstraints<double> nodal_constraints;
  VectorType                rhs;
};



class CoarseSolver : public MGCoarseGridBase<VectorType>
{
public:
  CoarseSolver(const MatrixType &matrix)
    : control(1, 0)
    , direct(control)
  {
    direct.initialize(matrix);
  }

  void
  operator()(const unsigned int,
             VectorType       &dst,
             const VectorType &src) const override
  {
    direct.solve(dst, src);
  }

private:
  mutable SolverControl                         control;
  mutable TrilinosWrappers::SolverDirect        direct;
};



// edge operator A, right-hand side (on all levels; only the finest one is
// used), nodal operator beta (grad p, grad q) = G^T A G, and the discrete
// gradient G
template <int dim>
void
setup_level(Level<dim>  &l,
            const double beta,
            MatrixType  &a,
            MatrixType  &s,
            MatrixType  &g)
{
  const MPI_Comm               comm = l.edge_dh.get_mpi_communicator();
  const FE_NedelecSimplex<dim> fe(0);
  const FE_SimplexP<dim>       nodal_fe(1);
  const MappingFE<dim>         mapping(nodal_fe);
  const QGaussSimplex<dim>     quadrature(3);

  l.edge_dh.distribute_dofs(fe);
  l.nodal_dh.distribute_dofs(nodal_fe);
  const IndexSet &owned = l.edge_dh.locally_owned_dofs();
  const IndexSet  relevant =
    DoFTools::extract_locally_relevant_dofs(l.edge_dh);
  const IndexSet &nodal_owned = l.nodal_dh.locally_owned_dofs();
  const IndexSet  nodal_relevant =
    DoFTools::extract_locally_relevant_dofs(l.nodal_dh);

  const ExactSolution<dim> exact_solution;
  l.edge_constraints.reinit(owned, relevant);
  DoFTools::make_hanging_node_constraints(l.edge_dh, l.edge_constraints);
  VectorTools::interpolate_boundary_values_curl_conforming(
    mapping, l.edge_dh, {{0, &exact_solution}}, l.edge_constraints);
  l.edge_constraints.close();
  l.nodal_constraints.reinit(nodal_owned, nodal_relevant);
  DoFTools::make_hanging_node_constraints(l.nodal_dh, l.nodal_constraints);
  VectorTools::interpolate_boundary_values(mapping,
                                           l.nodal_dh,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           l.nodal_constraints);
  l.nodal_constraints.close();

  const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

  // edge operator and right-hand side
  {
    TrilinosWrappers::SparsityPattern sparsity(owned, owned, relevant, comm);
    DoFTools::make_sparsity_pattern(
      l.edge_dh, sparsity, l.edge_constraints, false, my_rank);
    sparsity.compress();
    a.reinit(sparsity);
    l.rhs.reinit(owned, comm);

    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    const FEValuesExtractors::Vector     E(0);
    const unsigned int                   n_dofs = fe.n_dofs_per_cell();
    FullMatrix<double>                   cell_matrix(n_dofs, n_dofs);
    Vector<double>                       cell_rhs(n_dofs);
    std::vector<types::global_dof_index> dof_indices(n_dofs);
    for (const auto &cell : l.edge_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell_matrix = 0;
          cell_rhs    = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
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
          l.edge_constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, dof_indices, a, l.rhs);
        }
    a.compress(VectorOperation::add);
    l.rhs.compress(VectorOperation::add);
  }

  // nodal operator
  {
    TrilinosWrappers::SparsityPattern sparsity(nodal_owned,
                                               nodal_owned,
                                               nodal_relevant,
                                               comm);
    DoFTools::make_sparsity_pattern(
      l.nodal_dh, sparsity, l.nodal_constraints, false, my_rank);
    sparsity.compress();
    s.reinit(sparsity);

    FEValues<dim>      fe_values(mapping,
                            nodal_fe,
                            quadrature,
                            update_gradients | update_JxW_values);
    const unsigned int n_dofs = nodal_fe.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(n_dofs, n_dofs);
    std::vector<types::global_dof_index> dof_indices(n_dofs);
    for (const auto &cell : l.nodal_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell_matrix = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            for (unsigned int i = 0; i < n_dofs; ++i)
              for (unsigned int j = 0; j < n_dofs; ++j)
                cell_matrix(i, j) += beta * fe_values.shape_grad(i, q) *
                                     fe_values.shape_grad(j, q) *
                                     fe_values.JxW(q);
          cell->get_dof_indices(dof_indices);
          l.nodal_constraints.distribute_local_to_global(cell_matrix,
                                                         dof_indices,
                                                         s);
        }
    s.compress(VectorOperation::add);
  }

  // discrete gradient
  {
    TrilinosWrappers::SparsityPattern sparsity(owned, nodal_owned, comm);
    DoFTools::make_discrete_gradient_sparsity_pattern(l.edge_dh,
                                                      l.edge_constraints,
                                                      l.nodal_dh,
                                                      l.nodal_constraints,
                                                      sparsity);
    sparsity.compress();
    g.reinit(sparsity);
    DoFTools::make_discrete_gradient(l.edge_dh,
                                     l.edge_constraints,
                                     l.nodal_dh,
                                     l.nodal_constraints,
                                     g);
  }
}



template <int dim>
void
setup_prolongation(const Level<dim> &coarse,
                   const Level<dim> &fine,
                   MatrixType       &p)
{
  TrilinosWrappers::SparsityPattern sparsity(
    fine.edge_dh.locally_owned_dofs(),
    coarse.edge_dh.locally_owned_dofs(),
    fine.edge_dh.get_mpi_communicator());
  DoFTools::make_edge_element_prolongation_sparsity_pattern(
    coarse.edge_dh,
    coarse.edge_constraints,
    fine.edge_dh,
    fine.edge_constraints,
    sparsity);
  sparsity.compress();
  p.reinit(sparsity);
  DoFTools::make_edge_element_prolongation(coarse.edge_dh,
                                           coarse.edge_constraints,
                                           fine.edge_dh,
                                           fine.edge_constraints,
                                           p);
}



// refine the locally owned cells within radius of center
template <int dim>
void
refine(parallel::distributed::Triangulation<dim> &tria,
       const Point<dim>                          &center,
       const double                               radius)
{
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center().distance(center) < radius)
      cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();
}



// the fine mesh: a distorted coarse mesh, n_global uniform refinements, then
// two local ones
template <int dim>
void
make_mesh(parallel::distributed::Triangulation<dim> &tria,
          const unsigned int                         n_global)
{
  Triangulation<dim> coarse;
  GridGenerator::subdivided_hyper_cube_with_simplices(coarse, 2);
  GridTools::distort_random(0.2, coarse, true, 42);
  tria.copy_triangulation(coarse);
  for (unsigned int i = 0; i < n_global; ++i)
    refine(tria, Point<dim>(), 10.);
  refine(tria, Point<dim>() + 0.3 * Point<dim>::unit_vector(0), 0.5);
  refine(tria, Point<dim>() + 0.3 * Point<dim>::unit_vector(0), 0.3);
}



// Without repartitioning (no_automatic_repartitioning), coarsening can meet
// families that are split between processes, which are not coarsened (and
// must not be passed to t8code for coarsening). Coarsen the mesh and a copy
// of it (copy_triangulation(), the forest copy of t8code) twice.
template <int dim>
void
check_coarsening_without_repartitioning()
{
  parallel::distributed::Triangulation<dim> tria(
    MPI_COMM_WORLD,
    Triangulation<dim>::none,
    parallel::distributed::Triangulation<
      dim>::Settings::no_automatic_repartitioning);
  make_mesh(tria, 1);
  tria.repartition();
  parallel::distributed::Triangulation<dim> copy(MPI_COMM_WORLD);
  copy.copy_triangulation(tria);
  deallog << "dim=" << dim << " without repartitioning: "
          << tria.n_global_active_cells() << " cells";
  for (unsigned int i = 0; i < 2; ++i)
    {
      tria.coarsen_global();
      copy.coarsen_global();
      deallog << ", coarsened " << tria.n_global_active_cells() << " (copy "
              << copy.n_global_active_cells() << ")";
    }
  deallog << std::endl;
}



template <int dim>
void
test(const unsigned int n_global,
     const double       beta,
     const bool         verbose)
{
  const MPI_Comm comm = MPI_COMM_WORLD;

  parallel::distributed::Triangulation<dim> tria(comm);
  make_mesh(tria, n_global);

  const auto trias =
    MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(tria);
  const unsigned int n_levels = trias.size();

  std::vector<std::unique_ptr<Level<dim>>> levels;
  MGLevelObject<MatrixType> a(0, n_levels - 1), s(0, n_levels - 1),
    g(0, n_levels - 1), p(0, n_levels - 1);
  for (unsigned int l = 0; l < n_levels; ++l)
    {
      levels.push_back(std::make_unique<Level<dim>>(*trias[l]));
      setup_level(*levels[l], beta, a[l], s[l], g[l]);
      if (l > 0)
        setup_prolongation(*levels[l - 1], *levels[l], p[l]);
    }
  const Level<dim> &fine = *levels.back();

  if (!verbose)
    {
      deallog << "dim=" << dim << " beta=" << beta << " levels=" << n_levels
              << std::endl;
      for (unsigned int l = 0; l < n_levels; ++l)
        {
          deallog << "  level " << l << ": cells "
                  << trias[l]->n_global_active_cells() << ", edge DoFs "
                  << levels[l]->edge_dh.n_dofs() << ", |G|_F "
                  << std::setprecision(10) << g[l].frobenius_norm();
          if (l > 0)
            deallog << ", |P|_F " << p[l].frobenius_norm();
          deallog << std::setprecision(6) << std::endl;
        }
    }

  mg::Matrix<VectorType> matrix(a);
  MGTransferMatrices<VectorType, MatrixType> transfer;
  transfer.initialize(p);
  CoarseSolver coarse_solver(a[0]);

  using Relaxation = TrilinosWrappers::PreconditionSSOR;
  MGSmootherHiptmair<VectorType, MatrixType, Relaxation> hiptmair;
  hiptmair.initialize(a, g, s);
  MGSmootherPrecondition<MatrixType, Relaxation, VectorType> edge_only;
  edge_only.initialize(a);

  for (const bool nodal_step : {true, false})
    {
      const MGSmootherBase<VectorType> &smoother =
        nodal_step ? static_cast<const MGSmootherBase<VectorType> &>(hiptmair) :
                     static_cast<const MGSmootherBase<VectorType> &>(edge_only);
      Multigrid<VectorType> mg(
        matrix, coarse_solver, transfer, smoother, smoother, 0, n_levels - 1);
      PreconditionMG<dim, VectorType, MGTransferMatrices<VectorType, MatrixType>>
        preconditioner(fine.edge_dh, mg, transfer);

      VectorType    solution(fine.edge_dh.locally_owned_dofs(), comm);
      SolverControl control(1000, 1e-10 * fine.rhs.l2_norm(), false, false);
      SolverCG<VectorType> cg(control);
      try
        {
          cg.solve(a[n_levels - 1], solution, fine.rhs, preconditioner);
        }
      catch (const SolverControl::NoConvergence &)
        {}

      // L2 error of the solution
      fine.edge_constraints.distribute(solution);
      VectorType ghosted(fine.edge_dh.locally_owned_dofs(),
                         DoFTools::extract_locally_relevant_dofs(
                           fine.edge_dh),
                         comm);
      ghosted = solution;
      const MappingFE<dim>      mapping(FE_SimplexP<dim>(1));
      const QGaussSimplex<dim>  quadrature(3);
      FEValues<dim>             fe_values(mapping,
                              fine.edge_dh.get_fe(),
                              quadrature,
                              update_values | update_quadrature_points |
                                update_JxW_values);
      std::vector<Tensor<1, dim>> values(quadrature.size());
      double                      error2 = 0;
      for (const auto &cell : fine.edge_dh.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values[FEValuesExtractors::Vector(0)].get_function_values(
              ghosted, values);
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              error2 += (values[q] - exact_u(fe_values.quadrature_point(q)))
                          .norm_square() *
                        fe_values.JxW(q);
          }
      const double error = std::sqrt(Utilities::MPI::sum(error2, comm));

      if (verbose)
        {
          if (Utilities::MPI::this_mpi_process(comm) == 0)
            std::cout << "dim=" << dim << " beta=" << beta
                      << " levels=" << n_levels
                      << " dofs=" << fine.edge_dh.n_dofs()
                      << (nodal_step ? " hiptmair" : " edge only")
                      << " iterations=" << control.last_step()
                      << " L2 error=" << error << std::endl;
        }
      else
        deallog << "  " << (nodal_step ? "hiptmair" : "edge only")
                << ": iterations " << control.last_step() << ", L2 error "
                << std::setprecision(3) << error << std::setprecision(6)
                << std::endl;
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  if (argc == 4)
    {
      const unsigned int dim      = std::stoi(argv[1]);
      const unsigned int n_global = std::stoi(argv[2]);
      const double       beta     = std::stod(argv[3]);
      for (unsigned int n = 0; n <= n_global; ++n)
        if (dim == 2)
          test<2>(n, beta, true);
        else
          test<3>(n, beta, true);
      return 0;
    }

  MPILogInitAll log;
  check_coarsening_without_repartitioning<2>();
  check_coarsening_without_repartitioning<3>();
  for (const double beta : {1e-2, 1., 1e2})
    {
      test<2>(2, beta, false);
      test<3>(1, beta, false);
    }
}
