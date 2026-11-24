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
// on hybrid meshes in parallel (parallel::distributed::Triangulation with
// t8code, Trilinos): FE_NedelecSimplex, FE_NedelecWedge, FE_NedelecPyramid,
// FE_NedelecHypercube in one hp::FECollection, nodal hp collection
// FE_SimplexP(1), FE_WedgeP(1), FE_PyramidP(1), FE_Q(1). The fine mesh is
// the coarse mesh with one cell refined (hanging nodes) and then refined
// globally; the levels come from
// MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence()
// (repartitioned, so the prolongation communicates; on the coarse levels
// some processes own no cells). The nodal operator is assembled as
// beta (grad p, grad q) (equal to G^T A G, checked in
// fe_nedelec_multigrid_hybrid_01). Output: the level meshes and the
// Frobenius norms of the discrete gradients and prolongations, which must be
// the same for all numbers of processes, and the CG iterations with one
// V(1,1) cycle (Trilinos SSOR relaxations) with the Hiptmair smoother and
// the edge smoother alone.
//
// Optional arguments (for runs outside the test suite):
//   mesh n_global beta     (mesh: index into the list of meshes in main())

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

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

#include <functional>
#include <memory>

#include "../tests.h"

#include "./pyramid_meshes.h"
#include "./simplex_grids.h"
#include "./wedge_meshes.h"


using VectorType = TrilinosWrappers::MPI::Vector;
using MatrixType = TrilinosWrappers::SparseMatrix;


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
      if (cell->is_locally_owned())
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
refine(parallel::distributed::Triangulation<3> &tria)
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
  Level(const Triangulation<3> &tria)
    : edge_dh(tria)
    , nodal_dh(tria)
  {}

  DoFHandler<3>             edge_dh;
  DoFHandler<3>             nodal_dh;
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
  mutable SolverControl                  control;
  mutable TrilinosWrappers::SolverDirect direct;
};



// edge operator A and right-hand side, nodal operator beta (grad p, grad q),
// discrete gradient G
void
setup_level(Level       &l,
            const double beta,
            MatrixType  &a,
            MatrixType  &s,
            MatrixType  &g)
{
  const Collections                c;
  const FEValuesExtractors::Vector E(0);
  const ExactSolution              exact_solution;
  const MPI_Comm                   comm = l.edge_dh.get_mpi_communicator();
  const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

  c.distribute_dofs(l.edge_dh, false);
  c.distribute_dofs(l.nodal_dh, true);
  const IndexSet &owned = l.edge_dh.locally_owned_dofs();
  const IndexSet  relevant =
    DoFTools::extract_locally_relevant_dofs(l.edge_dh);
  const IndexSet &nodal_owned = l.nodal_dh.locally_owned_dofs();
  const IndexSet  nodal_relevant =
    DoFTools::extract_locally_relevant_dofs(l.nodal_dh);

  l.edge_constraints.reinit(owned, relevant);
  DoFTools::make_hanging_node_constraints(l.edge_dh, l.edge_constraints);
  VectorTools::interpolate_boundary_values_curl_conforming(
    c.mapping, l.edge_dh, {{0, &exact_solution}}, l.edge_constraints);
  l.edge_constraints.close();
  l.nodal_constraints.reinit(nodal_owned, nodal_relevant);
  DoFTools::make_hanging_node_constraints(l.nodal_dh, l.nodal_constraints);
  VectorTools::interpolate_boundary_values(c.mapping,
                                           l.nodal_dh,
                                           0,
                                           Functions::ZeroFunction<3>(),
                                           l.nodal_constraints);
  l.nodal_constraints.close();

  // edge operator and right-hand side
  {
    TrilinosWrappers::SparsityPattern sparsity(owned, owned, relevant, comm);
    DoFTools::make_sparsity_pattern(
      l.edge_dh, sparsity, l.edge_constraints, false, my_rank);
    sparsity.compress();
    a.reinit(sparsity);
    l.rhs.reinit(owned, comm);

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
      if (cell->is_locally_owned())
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

    hp::FEValues<3> hp_fe_values(c.mapping,
                                 c.nodal_fe,
                                 c.quadrature,
                                 update_gradients | update_JxW_values);
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> dof_indices;
    for (const auto &cell : l.nodal_dh.active_cell_iterators())
      if (cell->is_locally_owned())
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



void
setup_prolongation(const Level &coarse, const Level &fine, MatrixType &p)
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



void
test(const std::string                             &name,
     const std::function<void(Triangulation<3> &)> &make,
     const unsigned int                             n_global,
     const double                                   beta,
     const bool                                     verbose)
{
  const MPI_Comm comm = MPI_COMM_WORLD;

  Triangulation<3> coarse;
  make(coarse);
  parallel::distributed::Triangulation<3> tria(comm);
  tria.copy_triangulation(coarse);
  // one refined cell (the first coarse cell): hanging faces on all levels
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned() && cell->index() == 0)
      cell->set_refine_flag();
  refine(tria);
  for (unsigned int i = 0; i < n_global; ++i)
    {
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          cell->set_refine_flag();
      refine(tria);
    }

  const auto trias =
    MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(tria);
  const unsigned int n_levels = trias.size();

  std::vector<std::unique_ptr<Level>> levels;
  MGLevelObject<MatrixType> a(0, n_levels - 1), s(0, n_levels - 1),
    g(0, n_levels - 1), p(0, n_levels - 1);
  for (unsigned int l = 0; l < n_levels; ++l)
    {
      levels.push_back(std::make_unique<Level>(*trias[l]));
      setup_level(*levels[l], beta, a[l], s[l], g[l]);
      if (l > 0)
        setup_prolongation(*levels[l - 1], *levels[l], p[l]);
    }
  const Level &fine = *levels.back();

  if (!verbose)
    {
      deallog << name << ": levels=" << n_levels << std::endl;
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

  mg::Matrix<VectorType>                     matrix(a);
  MGTransferMatrices<VectorType, MatrixType> transfer;
  transfer.initialize(p);
  CoarseSolver coarse_solver(a[0]);

  using Relaxation = TrilinosWrappers::PreconditionSSOR;
  MGSmootherHiptmair<VectorType, MatrixType, Relaxation> hiptmair;
  hiptmair.initialize(a, g, s);
  MGSmootherPrecondition<MatrixType, Relaxation, VectorType> edge_only;
  edge_only.initialize(a);

  // the iteration counts of the edge smoother alone depend on rounding, so
  // only the comparison with the Hiptmair smoother is printed
  unsigned int hiptmair_iterations = 0;
  for (const bool nodal_step : {true, false})
    {
      const MGSmootherBase<VectorType> &smoother =
        nodal_step ? static_cast<const MGSmootherBase<VectorType> &>(hiptmair) :
                     static_cast<const MGSmootherBase<VectorType> &>(edge_only);
      Multigrid<VectorType> mg(
        matrix, coarse_solver, transfer, smoother, smoother, 0, n_levels - 1);
      PreconditionMG<3, VectorType, MGTransferMatrices<VectorType, MatrixType>>
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

      if (verbose)
        {
          if (Utilities::MPI::this_mpi_process(comm) == 0)
            std::cout << name << " levels=" << n_levels
                      << " dofs=" << fine.edge_dh.n_dofs()
                      << (nodal_step ? " hiptmair" : " edge only")
                      << " iterations=" << control.last_step() << std::endl;
        }
      else
        {
          if (nodal_step)
            {
              hiptmair_iterations = control.last_step();
              deallog << "  hiptmair: iterations " << control.last_step()
                      << std::endl;
            }
          else
            deallog << "  edge only: more iterations "
                    << (control.last_step() > hiptmair_iterations ? "yes" :
                                                                    "no")
                    << std::endl;
        }
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

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
      const unsigned int n_global = std::stoi(argv[2]);
      const double       beta     = std::stod(argv[3]);
      for (unsigned int n = 0; n <= n_global; ++n)
        test(meshes[mesh].first, meshes[mesh].second, n, beta, true);
      return 0;
    }

  MPILogInitAll log;
  for (const auto &[name, make] : meshes)
    test(name, make, 1, 1., false);
}
