// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


// Transfer a finite element function through adaptive refinement, coarsening
// and the repartitioning after each step of a
// parallel::distributed::Triangulation with simplices and hypercubes (also
// mixed) on coarse meshes with several trees. The function is the interpolant
// of a quadratic polynomial, which is exact on every cell, so after each
// transfer it must equal the interpolant on the new mesh. A difference
// indicates wrong parent-child relations or transferred data assigned to wrong
// cells.

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <functional>

#include "adaptive_meshes.h"


template <int dim>
class Quadratic : public Function<dim>
{
public:
  double
  value(const Point<dim> &p, const unsigned int) const override
  {
    double value = 1;
    for (unsigned int d = 0; d < dim; ++d)
      value += (d + 1) * p[d] + p[d] * p[(d + 1) % dim];
    return value;
  }
};



template <int dim>
void
test()
{
  using VectorType = LinearAlgebra::distributed::Vector<double>;

  const unsigned int n_cycles  = 4;
  const unsigned int max_level = dim == 2 ? 5 : 3;

  for (unsigned int i = 0; i < coarse_mesh_names<dim>().size(); ++i)
    {
      deallog.push(coarse_mesh_names<dim>()[i]);

      parallel::distributed::Triangulation<dim> tria(
        MPI_COMM_WORLD, Triangulation<dim>::limit_level_difference_at_vertices);
      make_coarse_mesh(tria, i);
      const MPI_Comm         comm = tria.get_mpi_communicator();
      const BoundingBox<dim> box  = GridTools::compute_bounding_box(tria);
      const double           radius =
        0.25 * box.get_boundary_points().first.distance(
                 box.get_boundary_points().second);

      const hp::MappingCollection<dim> mapping =
        make_mapping_collection<dim>();
      const hp::FECollection<dim> fe = make_fe_collection<dim>(2);

      DoFHandler<dim> dof_handler(tria);
      set_fe_indices(dof_handler);
      dof_handler.distribute_dofs(fe);
      VectorType solution(dof_handler.locally_owned_dofs(),
                          DoFTools::extract_locally_relevant_dofs(dof_handler),
                          comm);
      VectorTools::interpolate(mapping, dof_handler, Quadratic<dim>(), solution);

      // transfer the solution through the mesh change, return the error
      const auto transfer_through = [&](const std::function<void()> &change) {
        solution.update_ghost_values();
        SolutionTransfer<dim, VectorType> transfer(dof_handler);
        transfer.prepare_for_coarsening_and_refinement(solution);
        change();

        dof_handler.distribute_dofs(fe);
        const IndexSet relevant_dofs =
          DoFTools::extract_locally_relevant_dofs(dof_handler);
        AffineConstraints<double> constraints(
          dof_handler.locally_owned_dofs(), relevant_dofs);
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        constraints.close();

        // both interpolate() functions need ghost entries for the locally
        // relevant dofs
        VectorType transferred(dof_handler.locally_owned_dofs(),
                               relevant_dofs,
                               comm);
        transfer.interpolate(transferred);
        constraints.distribute(transferred);

        VectorType interpolated(dof_handler.locally_owned_dofs(),
                                relevant_dofs,
                                comm);
        VectorTools::interpolate(mapping,
                                 dof_handler,
                                 Quadratic<dim>(),
                                 interpolated);
        interpolated -= transferred;

        solution.reinit(dof_handler.locally_owned_dofs(),
                        relevant_dofs,
                        comm);
        solution.copy_locally_owned_data_from(transferred);
        return interpolated.linfty_norm();
      };
      const auto print_error = [](const double error) {
        if (error < 1e-10)
          deallog << "exact" << std::endl;
        else
          deallog << "error " << error << std::endl;
      };

      for (unsigned int k = 0; k < n_cycles; ++k)
        {
          const double error = transfer_through([&]() {
            refine_around(tria,
                          refinement_center(box, k, n_cycles),
                          radius / (1 + k),
                          max_level);
          });
          deallog << "cells " << tria.n_global_active_cells() << ", dofs "
                  << dof_handler.n_dofs() << ", ";
          print_error(error);
        }

      // the meshes while coarsening depend on the number of processes, so
      // only print the largest error
      double max_error = 0;
      while (tria.n_global_levels() > 1)
        max_error = std::max(max_error, transfer_through([&]() {
                               coarsen_all(tria);
                             }));
      deallog << "coarsening: ";
      print_error(max_error);

      deallog.pop();
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  mpi_initlog();

  deallog.push("2d");
  test<2>();
  deallog.pop();

  deallog.push("3d");
  test<3>();
  deallog.pop();
}
