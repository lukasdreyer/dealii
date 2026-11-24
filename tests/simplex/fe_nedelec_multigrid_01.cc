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


// Geometric multigrid for curl curl u + beta u = f with FE_NedelecSimplex(0)
// on a sequence of meshes (each one the previous one refined once), with
// the prolongation of DoFTools::make_edge_element_prolongation() and
// Hiptmair's hybrid smoother (MGSmootherHiptmair with PreconditionSSOR): a
// symmetric Gauss-Seidel step on the edge operator A, one on the nodal
// operator G^T A G (G the discrete gradient of
// DoFTools::make_discrete_gradient(), G^T A G from
// MGTools::make_hiptmair_nodal_matrix()), corrected by G, and again one on
// A.
// CG with one V(1,1) cycle as preconditioner; the iteration counts should
// be bounded independently of the number of levels and of beta. For
// comparison, the same multigrid with one symmetric Gauss-Seidel step on A
// (MGSmootherPrecondition), which does not damp the gradients (the kernel
// of curl).
//
// Optional arguments (for runs outside the test suite):
//   dim n_levels beta

#include <deal.II/base/quadrature_lib.h>

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
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_smoother_hiptmair.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/vector_tools.h>

#include <memory>

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



// everything that lives on one mesh of the sequence (the matrices are in
// MGLevelObjects)
template <int dim>
struct Level
{
  Triangulation<dim>        tria;
  DoFHandler<dim>           edge_dh;
  DoFHandler<dim>           nodal_dh;
  AffineConstraints<double> edge_constraints;
  AffineConstraints<double> nodal_constraints;

  SparsityPattern sparsity_a, sparsity_g, sparsity_s, sparsity_p;
  Vector<double>  rhs;
};



template <int dim>
using Levels = std::vector<std::unique_ptr<Level<dim>>>;



// the transfer with the prolongation matrices (prolongations[l] from
// level l - 1 to level l)
class EdgeTransfer : public MGTransferBase<Vector<double>>
{
public:
  EdgeTransfer(const MGLevelObject<SparseMatrix<double>> &prolongations)
    : prolongations(prolongations)
  {}

  void
  prolongate(const unsigned int    to_level,
             Vector<double>       &dst,
             const Vector<double> &src) const override
  {
    prolongations[to_level].vmult(dst, src);
  }

  void
  prolongate_and_add(const unsigned int    to_level,
                     Vector<double>       &dst,
                     const Vector<double> &src) const override
  {
    prolongations[to_level].vmult_add(dst, src);
  }

  void
  restrict_and_add(const unsigned int    from_level,
                   Vector<double>       &dst,
                   const Vector<double> &src) const override
  {
    prolongations[from_level].Tvmult_add(dst, src);
  }

private:
  const MGLevelObject<SparseMatrix<double>> &prolongations;
};



class CoarseSolver : public MGCoarseGridBase<Vector<double>>
{
public:
  CoarseSolver(const SparseMatrix<double> &matrix)
  {
    direct.initialize(matrix);
  }

  void
  operator()(const unsigned int,
             Vector<double>       &dst,
             const Vector<double> &src) const override
  {
    direct.vmult(dst, src);
  }

private:
  SparseDirectUMFPACK direct;
};



// one V-cycle as a preconditioner
template <int dim>
class PreconditionVCycle
{
public:
  PreconditionVCycle(const Levels<dim> &levels, Multigrid<Vector<double>> &mg)
    : levels(levels)
    , mg(mg)
  {}

  void
  vmult(Vector<double> &dst, const Vector<double> &src) const
  {
    const unsigned int max_level = levels.size() - 1;
    mg.defect.resize(0, max_level);
    for (unsigned int l = 0; l < max_level; ++l)
      mg.defect[l].reinit(levels[l]->edge_dh.n_dofs());
    mg.defect[max_level] = src;
    mg.cycle();
    dst = mg.solution[max_level];
  }

private:
  const Levels<dim>         &levels;
  Multigrid<Vector<double>> &mg;
};



template <int dim>
void
setup_level(Level<dim>           &l,
            const double          beta,
            SparseMatrix<double> &a,
            SparseMatrix<double> &g,
            SparseMatrix<double> &s)
{
  const FE_NedelecSimplex<dim> fe(0);
  const FE_SimplexP<dim>       nodal_fe(1);
  const MappingFE<dim>         mapping(nodal_fe);
  const QGaussSimplex<dim>     quadrature(3);

  l.edge_dh.reinit(l.tria);
  l.edge_dh.distribute_dofs(fe);
  l.nodal_dh.reinit(l.tria);
  l.nodal_dh.distribute_dofs(nodal_fe);

  const ExactSolution<dim> exact_solution;
  DoFTools::make_hanging_node_constraints(l.edge_dh, l.edge_constraints);
  VectorTools::interpolate_boundary_values_curl_conforming(
    mapping, l.edge_dh, {{0, &exact_solution}}, l.edge_constraints);
  l.edge_constraints.close();
  DoFTools::make_hanging_node_constraints(l.nodal_dh, l.nodal_constraints);
  VectorTools::interpolate_boundary_values(
    mapping, l.nodal_dh, 0, Functions::ZeroFunction<dim>(), l.nodal_constraints);
  l.nodal_constraints.close();

  // edge operator and right-hand side
  {
    DynamicSparsityPattern dsp(l.edge_dh.n_dofs());
    DoFTools::make_sparsity_pattern(l.edge_dh, dsp, l.edge_constraints, false);
    l.sparsity_a.copy_from(dsp);
    a.reinit(l.sparsity_a);
    l.rhs.reinit(l.edge_dh.n_dofs());

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
  }

  // discrete gradient and nodal operator G^T A G
  {
    DynamicSparsityPattern dsp(l.edge_dh.n_dofs(), l.nodal_dh.n_dofs());
    DoFTools::make_discrete_gradient_sparsity_pattern(
      l.edge_dh, l.edge_constraints, l.nodal_dh, l.nodal_constraints, dsp);
    l.sparsity_g.copy_from(dsp);
    g.reinit(l.sparsity_g);
    DoFTools::make_discrete_gradient(
      l.edge_dh, l.edge_constraints, l.nodal_dh, l.nodal_constraints, g);
    MGTools::make_hiptmair_nodal_matrix(a, g, l.sparsity_s, s);
  }
}



template <int dim>
void
setup_prolongation(const Level<dim>     &coarse,
                   Level<dim>           &fine,
                   SparseMatrix<double> &p)
{
  DynamicSparsityPattern dsp(fine.edge_dh.n_dofs(), coarse.edge_dh.n_dofs());
  DoFTools::make_edge_element_prolongation_sparsity_pattern(
    coarse.edge_dh,
    coarse.edge_constraints,
    fine.edge_dh,
    fine.edge_constraints,
    dsp);
  fine.sparsity_p.copy_from(dsp);
  p.reinit(fine.sparsity_p);
  DoFTools::make_edge_element_prolongation(coarse.edge_dh,
                                           coarse.edge_constraints,
                                           fine.edge_dh,
                                           fine.edge_constraints,
                                           p);
}



// returns the number of CG iterations with the Hiptmair smoother and with
// the edge smoother only
template <int dim>
std::pair<unsigned int, unsigned int>
test(const unsigned int n_levels, const double beta, const bool verbose)
{
  Levels<dim>                         levels;
  MGLevelObject<SparseMatrix<double>> a(0, n_levels - 1), g(0, n_levels - 1),
    s(0, n_levels - 1), p(0, n_levels - 1);
  for (unsigned int l = 0; l < n_levels; ++l)
    {
      levels.push_back(std::make_unique<Level<dim>>());
      if (l == 0)
        {
          GridGenerator::subdivided_hyper_cube_with_simplices(
            levels[0]->tria, 2);
          GridTools::distort_random(0.2, levels[0]->tria, true, 42);
        }
      else
        {
          levels[l]->tria.copy_triangulation(levels[l - 1]->tria);
          levels[l]->tria.refine_global(1);
        }
      setup_level(*levels[l], beta, a[l], g[l], s[l]);
      if (l > 0)
        setup_prolongation(*levels[l - 1], *levels[l], p[l]);
    }

  const Level<dim>            &fine = *levels.back();
  const SparseMatrix<double>  &fine_matrix = a[n_levels - 1];
  mg::Matrix<Vector<double>>   matrix(a);
  EdgeTransfer                 transfer(p);
  CoarseSolver                 coarse(a[0]);

  using Relaxation = PreconditionSSOR<SparseMatrix<double>>;
  MGSmootherHiptmair<Vector<double>, SparseMatrix<double>, Relaxation>
    hiptmair;
  hiptmair.initialize(a, g, s);
  MGSmootherPrecondition<SparseMatrix<double>, Relaxation, Vector<double>>
    edge_only;
  edge_only.initialize(a);

  std::pair<unsigned int, unsigned int> iterations;
  for (const bool nodal_step : {true, false})
    {
      const MGSmootherBase<Vector<double>> &smoother =
        nodal_step ?
          static_cast<const MGSmootherBase<Vector<double>> &>(hiptmair) :
          static_cast<const MGSmootherBase<Vector<double>> &>(edge_only);
      Multigrid<Vector<double>> mg(
        matrix, coarse, transfer, smoother, smoother, 0, n_levels - 1);
      PreconditionVCycle<dim> preconditioner(levels, mg);

      Vector<double> solution(fine.edge_dh.n_dofs());
      SolverControl  control(1000, 1e-10 * fine.rhs.l2_norm(), false, false);
      SolverCG<Vector<double>> cg(control);
      try
        {
          cg.solve(fine_matrix, solution, fine.rhs, preconditioner);
        }
      catch (const SolverControl::NoConvergence &)
        {}
      fine.edge_constraints.distribute(solution);

      (nodal_step ? iterations.first : iterations.second) =
        control.last_step();

      if (verbose)
        std::cout << "dim=" << dim << " beta=" << beta
                  << " levels=" << n_levels
                  << " dofs=" << fine.edge_dh.n_dofs()
                  << (nodal_step ? " hiptmair" : " edge only")
                  << " iterations=" << control.last_step() << std::endl;
    }
  return iterations;
}



int
main(int argc, char **argv)
{
  initlog();

  if (argc == 4)
    {
      const unsigned int dim      = std::stoi(argv[1]);
      const unsigned int n_levels = std::stoi(argv[2]);
      const double       beta     = std::stod(argv[3]);
      for (unsigned int n = 1; n <= n_levels; ++n)
        if (dim == 2)
          test<2>(n, beta, true);
        else
          test<3>(n, beta, true);
      return 0;
    }

  for (const double beta : {1e-2, 1., 1e2})
    {
      for (unsigned int n = 2; n <= 5; ++n)
        {
          const auto [hiptmair, edge] = test<2>(n, beta, false);
          deallog << "dim=2 beta=" << beta << " levels=" << n
                  << " iterations hiptmair=" << hiptmair
                  << " edge only=" << edge << std::endl;
        }
      for (unsigned int n = 2; n <= 3; ++n)
        {
          const auto [hiptmair, edge] = test<3>(n, beta, false);
          deallog << "dim=3 beta=" << beta << " levels=" << n
                  << " iterations hiptmair=" << hiptmair
                  << " edge only=" << edge << std::endl;
        }
    }
}
