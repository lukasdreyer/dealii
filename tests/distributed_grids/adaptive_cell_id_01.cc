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


// Check the CellIds of an adaptively refined
// parallel::distributed::Triangulation with simplices and hypercubes on coarse
// meshes with several trees (also mixed). Locally: create_cell_iterator() returns the cell
// of each CellId, the binary and string representations round trip, and the
// CellId of the parent is a prefix. Globally: the CellIds of all active cells
// are unique, and the list of CellIds with the cell centers (printed as a
// hash) does not depend on the number of processes after refinement. While
// coarsening, only the errors are counted.

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/cell_id.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <utility>

#include "adaptive_meshes.h"


// Return the number of errors (wrong ids, and on process 0 also duplicate
// or missing ones) and, on process 0, the hash of the sorted list of the
// CellIds with the cell centers.
template <int dim>
std::pair<unsigned int, std::uint64_t>
check_cell_ids(const parallel::distributed::Triangulation<dim> &tria)
{
  unsigned int             n_wrong_ids = 0;
  std::vector<std::string> owned_cells;
  for (const auto &cell : tria.active_cell_iterators())
    if (!cell->is_artificial())
      {
        const CellId id = cell->id();
        if (tria.create_cell_iterator(id) != cell ||
            CellId(id.template to_binary<dim>()) != id ||
            CellId(id.to_string()) != id ||
            (cell->level() > 0 && !cell->parent()->id().is_parent_of(id)))
          ++n_wrong_ids;

        if (cell->is_locally_owned())
          {
            std::ostringstream entry;
            entry << id << ' ' << std::fixed << std::setprecision(6)
                  << cell->center();
            owned_cells.push_back(entry.str());
          }
      }

  const MPI_Comm comm = tria.get_mpi_communicator();
  const std::vector<std::vector<std::string>> all_cells =
    Utilities::MPI::gather(comm, owned_cells);
  unsigned int  n_errors = Utilities::MPI::sum(n_wrong_ids, comm);
  std::uint64_t hash     = 14695981039346656037ull;

  if (Utilities::MPI::this_mpi_process(comm) == 0)
    {
      std::vector<std::string> cells;
      for (const auto &cells_of_process : all_cells)
        cells.insert(cells.end(),
                     cells_of_process.begin(),
                     cells_of_process.end());
      std::sort(cells.begin(), cells.end());
      if (std::adjacent_find(cells.begin(), cells.end()) != cells.end() ||
          cells.size() != tria.n_global_active_cells())
        ++n_errors;

      // FNV-1a hash of the sorted list
      for (const auto &entry : cells)
        for (const char c : entry + '\n')
          hash = (hash ^ static_cast<unsigned char>(c)) * 1099511628211ull;
    }
  return {n_errors, hash};
}



template <int dim>
void
print_cell_ids(const parallel::distributed::Triangulation<dim> &tria)
{
  const auto [n_errors, hash] = check_cell_ids(tria);
  deallog << "cells " << tria.n_global_active_cells() << ", errors "
          << n_errors << ", hash " << std::hex << hash << std::dec
          << std::endl;
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
      print_cell_ids(tria);

      for (unsigned int k = 0; k < n_cycles; ++k)
        {
          refine_around(tria,
                        refinement_center(box, k, n_cycles),
                        radius / (1 + k),
                        max_level);
          print_cell_ids(tria);
        }

      // the meshes while coarsening depend on the number of processes, so
      // only print the number of errors
      unsigned int n_errors = 0;
      while (coarsen_all(tria))
        n_errors += check_cell_ids(tria).first;
      deallog << "coarsening: errors " << n_errors << std::endl;

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
