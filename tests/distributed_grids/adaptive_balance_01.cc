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


// Check the 2:1 balance across faces, edges and vertices
// (limit_level_difference_at_vertices) of an adaptively refined
// parallel::distributed::Triangulation with simplices and hypercubes on coarse
// meshes with several trees (also mixed), while refining and coarsening, by
// brute force: the levels of all active cells touching a locally owned cell
// differ by at most one from its level. The refinement criterion asks for deep
// refinement at vertices of the coarse mesh, which are often shared by several
// trees, so balance has to refine many cells, also across tree boundaries.

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <array>

#include "adaptive_meshes.h"


// Return the global numbers of unbalanced pairs and of artificial cells
// touching locally owned cells.
template <int dim>
std::array<unsigned int, 2>
check_balance(const parallel::distributed::Triangulation<dim> &tria)
{
  unsigned int n_unbalanced = 0, n_artificial_neighbors = 0;

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto &other : tria.active_cell_iterators())
        if (other != cell && cells_touch<dim>(cell, other))
          {
            if (other->is_artificial())
              ++n_artificial_neighbors;
            else if (std::abs(cell->level() - other->level()) > 1)
              ++n_unbalanced;
          }

  const MPI_Comm comm = tria.get_mpi_communicator();
  return {{Utilities::MPI::sum(n_unbalanced, comm),
           Utilities::MPI::sum(n_artificial_neighbors, comm)}};
}



void
print_errors(const std::array<unsigned int, 2> &errors)
{
  deallog << "unbalanced pairs " << errors[0] << ", artificial neighbors "
          << errors[1] << std::endl;
}



template <int dim>
void
test()
{
  const unsigned int n_cycles  = 3;
  const unsigned int max_level = dim == 2 ? 6 : 4;

  for (unsigned int i = 0; i < coarse_mesh_names<dim>().size(); ++i)
    {
      deallog.push(coarse_mesh_names<dim>()[i]);

      parallel::distributed::Triangulation<dim> tria(
        MPI_COMM_WORLD, Triangulation<dim>::limit_level_difference_at_vertices);
      make_coarse_mesh(tria, i);
      const std::vector<Point<dim>> coarse_vertices = tria.get_vertices();

      // refine only the cells containing a vertex of the coarse mesh,
      // repeatedly, so that only balance refines the surrounding cells
      for (unsigned int k = 0; k < n_cycles; ++k)
        {
          const Point<dim> center =
            coarse_vertices[k * (coarse_vertices.size() - 1) / (n_cycles - 1)];
          for (unsigned int l = 0; l < max_level; ++l)
            {
              for (const auto &cell : tria.active_cell_iterators())
                if (cell->is_locally_owned() &&
                    point_in_cell<dim>(cell, center))
                  cell->set_refine_flag();
              tria.execute_coarsening_and_refinement();
            }
          deallog << "cells " << tria.n_global_active_cells() << ", levels "
                  << tria.n_global_levels() << ", ";
          print_errors(check_balance(tria));
        }

      // the meshes while coarsening depend on the number of processes, so
      // only print the sum of the errors
      std::array<unsigned int, 2> errors = {{0, 0}};
      while (coarsen_all(tria))
        {
          const std::array<unsigned int, 2> round_errors = check_balance(tria);
          for (unsigned int e = 0; e < 2; ++e)
            errors[e] += round_errors[e];
        }
      deallog << "coarsening: ";
      print_errors(errors);

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
