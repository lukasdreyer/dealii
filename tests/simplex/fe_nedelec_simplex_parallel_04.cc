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


// The discrete gradient G from FE_SimplexP(1) to FE_NedelecSimplex(0)
// (DoFTools::make_discrete_gradient) on adaptively refined, distorted
// simplex meshes in a parallel::distributed::Triangulation: for a random
// nodal vector p satisfying the hanging node constraints, u = G p followed
// by distributing the edge constraints equals grad p_h at every quadrature
// point of every locally owned cell. Also checks that the rows of
// constrained edge DoFs are zero.

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
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include "../tests.h"


template <int dim>
void
test()
{
  const MPI_Comm comm = MPI_COMM_WORLD;

  Triangulation<dim> coarse;
  GridGenerator::subdivided_hyper_cube_with_simplices(coarse, 3);
  GridTools::distort_random(0.2, coarse, false, 42);
  parallel::distributed::Triangulation<dim> tria(comm);
  tria.copy_triangulation(coarse);
  for (unsigned int cycle = 0; cycle < 3; ++cycle)
    {
      const Point<dim> center =
        Point<dim>() + (0.2 + 0.25 * cycle) * Point<dim>::unit_vector(0) +
        0.4 * Point<dim>::unit_vector(1);
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned() && cell->center().distance(center) < 0.3)
          cell->set_refine_flag();
      tria.execute_coarsening_and_refinement();
    }

  const FE_NedelecSimplex<dim> fe_edge(0);
  const FE_SimplexP<dim>       fe_nodal(1);
  const MappingFE<dim>         mapping(FE_SimplexP<dim>(1));

  DoFHandler<dim> edge_dh(tria), nodal_dh(tria);
  edge_dh.distribute_dofs(fe_edge);
  nodal_dh.distribute_dofs(fe_nodal);
  const IndexSet &edge_owned = edge_dh.locally_owned_dofs();
  const IndexSet  edge_relevant =
    DoFTools::extract_locally_relevant_dofs(edge_dh);
  const IndexSet &nodal_owned = nodal_dh.locally_owned_dofs();
  const IndexSet  nodal_relevant =
    DoFTools::extract_locally_relevant_dofs(nodal_dh);

  AffineConstraints<double> edge_constraints(edge_owned, edge_relevant);
  DoFTools::make_hanging_node_constraints(edge_dh, edge_constraints);
  edge_constraints.close();
  AffineConstraints<double> nodal_constraints(nodal_owned, nodal_relevant);
  DoFTools::make_hanging_node_constraints(nodal_dh, nodal_constraints);
  nodal_constraints.close();

  DynamicSparsityPattern dsp(edge_dh.n_dofs(), nodal_dh.n_dofs(), edge_owned);
  DoFTools::make_discrete_gradient_sparsity_pattern(
    edge_dh, edge_constraints, nodal_dh, nodal_constraints, dsp);
  TrilinosWrappers::SparseMatrix gradient;
  gradient.reinit(edge_owned, nodal_owned, dsp, comm);
  DoFTools::make_discrete_gradient(
    edge_dh, edge_constraints, nodal_dh, nodal_constraints, gradient);

  // a random nodal vector that satisfies the hanging node constraints
  TrilinosWrappers::MPI::Vector p(nodal_owned, comm);
  for (const types::global_dof_index dof : nodal_owned)
    p(dof) = random_value<double>(-1., 1.);
  nodal_constraints.distribute(p);

  TrilinosWrappers::MPI::Vector u(edge_owned, comm);
  gradient.vmult(u, p);

  // the rows of constrained edge DoFs are zero
  double max_constrained = 0;
  for (const types::global_dof_index dof : edge_owned)
    if (edge_constraints.is_constrained(dof))
      max_constrained = std::max(max_constrained, std::abs(u(dof)));
  edge_constraints.distribute(u);

  TrilinosWrappers::MPI::Vector u_ghosted(edge_owned, edge_relevant, comm),
    p_ghosted(nodal_owned, nodal_relevant, comm);
  u_ghosted = u;
  p_ghosted = p;

  const QGaussSimplex<dim> quadrature(2);
  FEValues<dim> edge_values(mapping, fe_edge, quadrature, update_values);
  FEValues<dim> nodal_values(mapping, fe_nodal, quadrature, update_gradients);
  std::vector<types::global_dof_index> edge_dofs(fe_edge.n_dofs_per_cell()),
    nodal_dofs(fe_nodal.n_dofs_per_cell());
  double max_error = 0;
  for (const auto &cell : edge_dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const auto nodal_cell = cell->as_dof_handler_iterator(nodal_dh);
        edge_values.reinit(cell);
        nodal_values.reinit(nodal_cell);
        cell->get_dof_indices(edge_dofs);
        nodal_cell->get_dof_indices(nodal_dofs);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            Tensor<1, dim> value, grad;
            for (unsigned int i = 0; i < edge_dofs.size(); ++i)
              for (unsigned int d = 0; d < dim; ++d)
                value[d] += static_cast<double>(u_ghosted(edge_dofs[i])) *
                            edge_values.shape_value_component(i, q, d);
            for (unsigned int i = 0; i < nodal_dofs.size(); ++i)
              grad += static_cast<double>(p_ghosted(nodal_dofs[i])) *
                      nodal_values.shape_grad(i, q);
            max_error = std::max(max_error, (value - grad).norm());
          }
      }

  unsigned int n_edge_constraints = 0, n_nodal_constraints = 0;
  for (const auto &line : edge_constraints.get_lines())
    if (edge_owned.is_element(line.index))
      ++n_edge_constraints;
  for (const auto &line : nodal_constraints.get_lines())
    if (nodal_owned.is_element(line.index))
      ++n_nodal_constraints;

  deallog << "dim=" << dim << " cells=" << tria.n_global_active_cells()
          << " edges=" << edge_dh.n_dofs() << " vertices=" << nodal_dh.n_dofs()
          << " constrained edges="
          << Utilities::MPI::sum(n_edge_constraints, comm)
          << " constrained vertices="
          << Utilities::MPI::sum(n_nodal_constraints, comm) << std::endl;
  deallog << "  constrained rows "
          << (Utilities::MPI::max(max_constrained, comm) < 1e-14 ? "zero" :
                                                                   "NONZERO")
          << ", G p = grad p_h "
          << (Utilities::MPI::max(max_error, comm) < 1e-12 ? "OK" : "FAILED")
          << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  MPILogInitAll                    log;

  test<2>();
  test<3>();
}
