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


// Adaptive refinement and coarsening of a parallel::distributed::Triangulation
// with simplices and hypercubes (also mixed) on coarse meshes with several
// trees. The refinement criterion only depends on the geometry, so the output
// must not depend on the number of processes. Prints global quantities of the
// mesh after each refinement cycle: number of active cells and levels, volume,
// and moments of the cell centers and levels. Finally, all cells are
// coarsened, which must give back the coarse mesh (the intermediate meshes
// depend on the number of processes, see adaptive_meshes.h).

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include "adaptive_meshes.h"


template <int dim>
void
print_mesh(const parallel::distributed::Triangulation<dim> &tria)
{
  double         volume = 0, level_moment = 0;
  Tensor<1, dim> center_moment;
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const double measure = cell->measure();
        volume += measure;
        level_moment += cell->level() * measure;
        center_moment += cell->center() * measure;
      }

  // round the sums, whose last digits depend on the order of summation and
  // thus on the number of processes
  const auto round = [](const double x) {
    return std::round(x * 1e10) / 1e10;
  };
  const MPI_Comm       comm = tria.get_mpi_communicator();
  const Tensor<1, dim> center_moment_sum =
    Utilities::MPI::sum(center_moment, comm);
  Tensor<1, dim> rounded_center_moment;
  for (unsigned int d = 0; d < dim; ++d)
    rounded_center_moment[d] = round(center_moment_sum[d]);
  deallog << "cells " << tria.n_global_active_cells() << ", levels "
          << tria.n_global_levels() << ", volume "
          << round(Utilities::MPI::sum(volume, comm)) << ", level moment "
          << round(Utilities::MPI::sum(level_moment, comm))
          << ", center moment " << rounded_center_moment << std::endl;
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
      const unsigned int     n_coarse_cells = tria.n_global_active_cells();
      const BoundingBox<dim> box = GridTools::compute_bounding_box(tria);
      const double           radius =
        0.25 * box.get_boundary_points().first.distance(
                 box.get_boundary_points().second);
      print_mesh(tria);

      for (unsigned int k = 0; k < n_cycles; ++k)
        {
          refine_around(tria,
                        refinement_center(box, k, n_cycles),
                        radius / (1 + k),
                        max_level);
          print_mesh(tria);
        }

      // coarsen everything; the intermediate meshes depend on the number of
      // processes, so only print the final one
      for (unsigned int round = 0; coarsen_all(tria); ++round)
        AssertThrow(round < 4 * max_level, ExcInternalError());
      AssertThrow(tria.n_global_active_cells() == n_coarse_cells,
                  ExcInternalError());
      print_mesh(tria);

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
