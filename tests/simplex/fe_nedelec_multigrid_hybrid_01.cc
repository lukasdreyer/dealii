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


// Geometric multigrid with the Hiptmair smoother for curl curl u + beta u = f
// on hybrid meshes (FE_NedelecSimplex, FE_NedelecWedge, FE_NedelecPyramid,
// FE_NedelecHypercube in one hp::FECollection), serial. Level 0 has hanging
// nodes (one cell refined), every further level is the previous one refined
// globally. The nodal space is the hp collection FE_SimplexP(1),
// FE_WedgeP(1), FE_PyramidP(1), FE_Q(1), and the nodal operator of the
// smoother is assembled as beta (grad p, grad q).
// 1. Discrete exact sequence on every level: the assembled nodal operator
//    equals G^T A G (G from DoFTools::make_discrete_gradient()) on the
//    unconstrained nodal DoFs.
// 2. CG with one V(1,1) cycle (symmetric Gauss-Seidel relaxations, direct
//    coarse solve): iteration counts over the number of levels, with the
//    Hiptmair smoother and with the edge smoother alone. The spaces are
//    nested except where pyramids are refined (their children include
//    tetrahedra), where the prolongation is the canonical interpolation.
//
// Optional arguments (for runs outside the test suite):
//   mesh n_levels beta     (mesh: index into the list of meshes in main())

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/edge_element_tools.h>

#include <deal.II/fe/fe_nedelec_hypercube.h>
#include <deal.II/fe/fe_nedelec_pyramid.h>
#include <deal.II/fe/fe_nedelec_simplex.h>
#include <deal.II/fe/fe_nedelec_wedge.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

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

#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_smoother_hiptmair.h>
#include <deal.II/multigrid/mg_transfer_matrices.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/vector_tools.h>

#include <functional>
#include <memory>

#include "../tests.h"

#include "./pyramid_meshes.h"
#include "./simplex_grids.h"
#include "./wedge_meshes.h"


// the elements, nodal elements, mappings and quadratures, with the FE index
// of a cell given by its reference cell
struct Collections
{
  hp::FECollection<3>      fe{FE_NedelecSimplex<3>(0),
                         FE_NedelecWedge<3>(0),
                         FE_NedelecPyramid<3>(0),
                         FE_NedelecHypercube<3>(0)};
  hp::FECollection<3>      nodal_fe{FE_SimplexP<3>(1),
                               FE_WedgeP<3>(1),
                               FE_PyramidP<3>(1),
                               FE_Q<3>(1)};
  hp::MappingCollection<3> mapping{MappingFE<3>(FE_SimplexP<3>(1)),
                                   MappingFE<3>(FE_WedgeP<3>(1)),
                                   MappingFE<3>(FE_PyramidP<3>(1)),
                                   MappingFE<3>(FE_Q<3>(1))};
  hp::QCollection<3>       quadrature{QGaussSimplex<3>(3),
                                QGaussWedge<3>(3),
                                QGaussPyramid<3>(4),
                                QGauss<3>(3)};

  static unsigned int
  fe_index(const ReferenceCell &reference_cell)
  {
    if (reference_cell == ReferenceCells::Tetrahedron)
      return 0;
    if (reference_cell == ReferenceCells::Wedge)
      return 1;
    if (reference_cell == ReferenceCells::Pyramid)
      return 2;
    Assert(reference_cell == ReferenceCells::Hexahedron, ExcNotImplemented());
    return 3;
  }

  void
  distribute_dofs(DoFHandler<3> &dof_handler, const bool nodal) const
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      cell->set_active_fe_index(fe_index(cell->reference_cell()));
    dof_handler.distribute_dofs(nodal ? nodal_fe : fe);
  }
};



Tensor<1, 3>
exact_u(const Point<3> &p)
{
  return Tensor<1, 3>({std::sin(p[1]), std::sin(p[2]), std::sin(p[0])});
}

// curl curl u = u
class ExactSolution : public Function<3>
{
public:
  ExactSolution()
    : Function<3>(3)
  {}

  void
  vector_value(const Point<3> &p, Vector<double> &values) const override
  {
    const Tensor<1, 3> u = exact_u(p);
    for (unsigned int d = 0; d < 3; ++d)
      values(d) = u[d];
  }
};



void
refine(Triangulation<3> &tria)
{
  tria.prepare_coarsening_and_refinement();
  // tetrahedra can only be refined along the line (6,8), also the ones
  // flagged by the mesh smoothing
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->refine_flag_set() &&
        cell->reference_cell() == ReferenceCells::Tetrahedron)
      cell->set_refine_choice(
        static_cast<std::uint8_t>(IsotropicRefinementChoice::cut_tet_68));
  tria.execute_coarsening_and_refinement();
}



struct Level
{
  Triangulation<3>          tria;
  DoFHandler<3>             edge_dh;
  DoFHandler<3>             nodal_dh;
  AffineConstraints<double> edge_constraints;
  AffineConstraints<double> nodal_constraints;

  SparsityPattern sparsity_a, sparsity_g, sparsity_s, sparsity_p,
    sparsity_gtag;
  Vector<double> rhs;
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



// edge operator A and right-hand side, nodal operator beta (grad p, grad q),
// discrete gradient G; returns the largest difference between the nodal
// operator and G^T A G on the unconstrained nodal DoFs (applied to a random
// vector), relative to the size of the result
double
setup_level(Level                &l,
            const double          beta,
            SparseMatrix<double> &a,
            SparseMatrix<double> &s,
            SparseMatrix<double> &g)
{
  const Collections                c;
  const FEValuesExtractors::Vector E(0);
  const ExactSolution              exact_solution;

  l.edge_dh.reinit(l.tria);
  c.distribute_dofs(l.edge_dh, false);
  l.nodal_dh.reinit(l.tria);
  c.distribute_dofs(l.nodal_dh, true);

  DoFTools::make_hanging_node_constraints(l.edge_dh, l.edge_constraints);
  VectorTools::interpolate_boundary_values_curl_conforming(
    c.mapping, l.edge_dh, {{0, &exact_solution}}, l.edge_constraints);
  l.edge_constraints.close();
  DoFTools::make_hanging_node_constraints(l.nodal_dh, l.nodal_constraints);
  VectorTools::interpolate_boundary_values(c.mapping,
                                           l.nodal_dh,
                                           0,
                                           Functions::ZeroFunction<3>(),
                                           l.nodal_constraints);
  l.nodal_constraints.close();

  // edge operator and right-hand side
  {
    DynamicSparsityPattern dsp(l.edge_dh.n_dofs());
    DoFTools::make_sparsity_pattern(l.edge_dh, dsp, l.edge_constraints, false);
    l.sparsity_a.copy_from(dsp);
    a.reinit(l.sparsity_a);
    l.rhs.reinit(l.edge_dh.n_dofs());

    hp::FEValues<3> hp_fe_values(c.mapping,
                                 c.fe,
                                 c.quadrature,
                                 update_values | update_gradients |
                                   update_quadrature_points |
                                   update_JxW_values);
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> dof_indices;
    for (const auto &cell : l.edge_dh.active_cell_iterators())
      {
        hp_fe_values.reinit(cell);
        const FEValues<3> &fe_values = hp_fe_values.get_present_fe_values();
        const unsigned int n_dofs    = cell->get_fe().n_dofs_per_cell();
        cell_matrix.reinit(n_dofs, n_dofs);
        cell_rhs.reinit(n_dofs);
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
          {
            const Tensor<1, 3> f =
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
        dof_indices.resize(n_dofs);
        cell->get_dof_indices(dof_indices);
        l.edge_constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, dof_indices, a, l.rhs);
      }
  }

  // nodal operator
  {
    DynamicSparsityPattern dsp(l.nodal_dh.n_dofs());
    DoFTools::make_sparsity_pattern(l.nodal_dh,
                                    dsp,
                                    l.nodal_constraints,
                                    false);
    l.sparsity_s.copy_from(dsp);
    s.reinit(l.sparsity_s);

    hp::FEValues<3> hp_fe_values(c.mapping,
                                 c.nodal_fe,
                                 c.quadrature,
                                 update_gradients | update_JxW_values);
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> dof_indices;
    for (const auto &cell : l.nodal_dh.active_cell_iterators())
      {
        hp_fe_values.reinit(cell);
        const FEValues<3> &fe_values = hp_fe_values.get_present_fe_values();
        const unsigned int n_dofs    = cell->get_fe().n_dofs_per_cell();
        cell_matrix.reinit(n_dofs, n_dofs);
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              cell_matrix(i, j) += beta * fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) *
                                   fe_values.JxW(q);
        dof_indices.resize(n_dofs);
        cell->get_dof_indices(dof_indices);
        l.nodal_constraints.distribute_local_to_global(cell_matrix,
                                                       dof_indices,
                                                       s);
      }
  }

  // discrete gradient
  {
    DynamicSparsityPattern dsp(l.edge_dh.n_dofs(), l.nodal_dh.n_dofs());
    DoFTools::make_discrete_gradient_sparsity_pattern(
      l.edge_dh, l.edge_constraints, l.nodal_dh, l.nodal_constraints, dsp);
    l.sparsity_g.copy_from(dsp);
    g.reinit(l.sparsity_g);
    DoFTools::make_discrete_gradient(
      l.edge_dh, l.edge_constraints, l.nodal_dh, l.nodal_constraints, g);
  }

  // compare with G^T A G
  SparseMatrix<double> gtag;
  MGTools::make_hiptmair_nodal_matrix(a, g, l.sparsity_gtag, gtag);
  Vector<double> p(l.nodal_dh.n_dofs()), y1(p.size()), y2(p.size());
  for (unsigned int i = 0; i < p.size(); ++i)
    p(i) = l.nodal_constraints.is_constrained(i) ?
             0. :
             random_value<double>(-1., 1.);
  s.vmult(y1, p);
  gtag.vmult(y2, p);
  double difference = 0, size = 0;
  for (unsigned int i = 0; i < p.size(); ++i)
    if (!l.nodal_constraints.is_constrained(i))
      {
        difference = std::max(difference, std::abs(y1(i) - y2(i)));
        size       = std::max(size, std::abs(y1(i)));
      }
  // (on coarse meshes all vertices can be constrained)
  return (size > 0) ? difference / size : difference;
}



void
setup_prolongation(const Level &coarse, Level &fine, SparseMatrix<double> &p)
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
std::pair<unsigned int, unsigned int>
test(const std::function<void(Triangulation<3> &)> &make,
     const unsigned int                             n_levels,
     const double                                   beta,
     const bool                                     verbose)
{
  std::vector<std::unique_ptr<Level>> levels;
  MGLevelObject<SparseMatrix<double>> a(0, n_levels - 1), s(0, n_levels - 1),
    g(0, n_levels - 1), p(0, n_levels - 1);
  double max_difference = 0;
  for (unsigned int l = 0; l < n_levels; ++l)
    {
      levels.push_back(std::make_unique<Level>());
      Triangulation<3> &tria = levels[l]->tria;
      if (l == 0)
        {
          make(tria);
          // one refined cell: hanging faces on all levels
          tria.begin_active()->set_refine_flag();
        }
      else
        {
          tria.copy_triangulation(levels[l - 1]->tria);
          for (const auto &cell : tria.active_cell_iterators())
            cell->set_refine_flag();
        }
      refine(tria);
      max_difference =
        std::max(max_difference, setup_level(*levels[l], beta, a[l], s[l], g[l]));
      if (l > 0)
        setup_prolongation(*levels[l - 1], *levels[l], p[l]);
    }
  if (!verbose)
    deallog << "  levels=" << n_levels << ": G^T A G = nodal operator "
            << (max_difference < 1e-10 ? "OK" : "FAILED");
  else
    std::cout << "  G^T A G - nodal operator (relative): " << max_difference
              << std::endl;

  const Level                     &fine = *levels.back();
  mg::Matrix<Vector<double>>       matrix(a);
  MGTransferMatrices<Vector<double>, SparseMatrix<double>> transfer;
  transfer.initialize(p);
  CoarseSolver coarse(a[0]);

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
      PreconditionMG<3,
                     Vector<double>,
                     MGTransferMatrices<Vector<double>, SparseMatrix<double>>>
        preconditioner(fine.edge_dh, mg, transfer);

      Vector<double> solution(fine.edge_dh.n_dofs());
      SolverControl  control(1000, 1e-10 * fine.rhs.l2_norm(), false, false);
      SolverCG<Vector<double>> cg(control);
      try
        {
          cg.solve(a[n_levels - 1], solution, fine.rhs, preconditioner);
        }
      catch (const SolverControl::NoConvergence &)
        {}

      (nodal_step ? iterations.first : iterations.second) =
        control.last_step();
      if (verbose)
        std::cout << "  levels=" << n_levels
                  << " dofs=" << fine.edge_dh.n_dofs()
                  << (nodal_step ? " hiptmair" : " edge only")
                  << " iterations=" << control.last_step() << std::endl;
    }
  if (!verbose)
    deallog << ", iterations hiptmair=" << iterations.first
            << " edge only=" << iterations.second << std::endl;
  return iterations;
}



int
main(int argc, char **argv)
{
  initlog();

  const std::vector<
    std::pair<std::string, std::function<void(Triangulation<3> &)>>>
    meshes = {
      {"tetrahedra",
       [](Triangulation<3> &t) {
         GridGenerator::subdivided_hyper_cube_with_simplices(t, 2);
       }},
      {"wedges",
       [](Triangulation<3> &t) {
         GridGenerator::subdivided_hyper_cube_with_wedges(t, 2);
       }},
      {"hexahedra",
       [](Triangulation<3> &t) { GridGenerator::subdivided_hyper_cube(t, 2); }},
      {"pyramids", [](Triangulation<3> &t) { make_pyramid_mesh(t, 1, 0); }},
      {"pyramids and tetrahedra",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 1, 2); }},
      {"hexahedra and pyramids",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 1); }},
      {"wedges and tetrahedra",
       [](Triangulation<3> &t) { make_wedge_tetrahedron_mesh(t, 1); }}};

  if (argc == 4)
    {
      const unsigned int mesh     = std::stoi(argv[1]);
      const unsigned int n_levels = std::stoi(argv[2]);
      const double       beta     = std::stod(argv[3]);
      std::cout << meshes[mesh].first << std::endl;
      for (unsigned int n = 2; n <= n_levels; ++n)
        test(meshes[mesh].second, n, beta, true);
      return 0;
    }

  for (const auto &[name, make] : meshes)
    {
      deallog << name << std::endl;
      for (unsigned int n = 2; n <= 3; ++n)
        test(make, n, 1., false);
    }
}
