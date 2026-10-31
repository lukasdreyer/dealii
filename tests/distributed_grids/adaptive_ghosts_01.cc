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


// Check the ghost layer of an adaptively refined
// parallel::distributed::Triangulation with simplices and hypercubes on coarse
// meshes with several trees (also mixed), while refining and coarsening, by
// brute force: every active cell that touches a locally owned cell (across a
// face, an edge or a vertex, also across tree boundaries) must be a ghost
// cell, every ghost cell must touch a locally owned cell, and the owner of
// each ghost cell must have the same cell.

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_tools.h>

#include <array>

#include "adaptive_meshes.h"


// Return the global numbers of missing, superfluous and wrong ghosts.
template <int dim>
std::array<unsigned int, 3>
check_ghosts(const parallel::distributed::Triangulation<dim> &tria)
{
  using cell_iterator =
    typename parallel::distributed::Triangulation<dim>::active_cell_iterator;

  unsigned int n_missing_ghosts = 0, n_superfluous_ghosts = 0,
               n_wrong_ghosts = 0;

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_ghost())
      {
        bool touches_owned_cell = false;
        for (const auto &other : tria.active_cell_iterators())
          if (other->is_locally_owned() && cells_touch<dim>(cell, other))
            {
              touches_owned_cell = true;
              break;
            }
        if (!touches_owned_cell)
          ++n_superfluous_ghosts;
      }
    else if (cell->is_artificial())
      {
        for (const auto &other : tria.active_cell_iterators())
          if (other->is_locally_owned() && cells_touch<dim>(cell, other))
            {
              ++n_missing_ghosts;
              break;
            }
      }

  // the owner sends the center of each cell that is a ghost elsewhere
  GridTools::exchange_cell_data_to_ghosts<
    Point<dim>,
    parallel::distributed::Triangulation<dim>>(
    tria,
    [](const cell_iterator &cell) { return cell->center(); },
    [&](const cell_iterator &cell, const Point<dim> &center) {
      if (center.distance(cell->center()) > 1e-12 * cell->diameter())
        ++n_wrong_ghosts;
    });

  const MPI_Comm comm = tria.get_mpi_communicator();
  return {{Utilities::MPI::sum(n_missing_ghosts, comm),
           Utilities::MPI::sum(n_superfluous_ghosts, comm),
           Utilities::MPI::sum(n_wrong_ghosts, comm)}};
}



void
print_errors(const std::array<unsigned int, 3> &errors)
{
  deallog << "missing ghosts " << errors[0] << ", superfluous ghosts "
          << errors[1] << ", wrong ghosts " << errors[2] << std::endl;
}



template <int dim>
void
test()
{
  const unsigned int n_cycles  = 4;
  const unsigned int max_level = dim == 2 ? 5 : 3;

  for (unsigned int i = 0; i < coarse_mesh_names<dim>().size(); ++i)
    {
      deallog.push(coarse_mesh_names<dim>()[i]);

      parallel::distributed::Triangulation<dim> tria(
        MPI_COMM_WORLD, Triangulation<dim>::limit_level_difference_at_vertices);
      make_coarse_mesh(tria, i);
      const BoundingBox<dim> box = GridTools::compute_bounding_box(tria);
      const double           radius =
        0.25 * box.get_boundary_points().first.distance(
                 box.get_boundary_points().second);
      deallog << "cells " << tria.n_global_active_cells() << ", ";
      print_errors(check_ghosts(tria));

      for (unsigned int k = 0; k < n_cycles; ++k)
        {
          refine_around(tria,
                        refinement_center(box, k, n_cycles),
                        radius / (1 + k),
                        max_level);
          deallog << "cells " << tria.n_global_active_cells() << ", ";
          print_errors(check_ghosts(tria));
        }

      // the meshes while coarsening depend on the number of processes, so
      // only print the sum of the errors
      std::array<unsigned int, 3> errors = {{0, 0, 0}};
      while (coarsen_all(tria))
        {
          const std::array<unsigned int, 3> round_errors = check_ghosts(tria);
          for (unsigned int e = 0; e < 3; ++e)
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
