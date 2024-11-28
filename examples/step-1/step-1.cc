/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 1999 - 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 */

// @sect3{Include files}

// The most fundamental class in the library is the Triangulation class, which
// is declared here:
#include <deal.II/distributed/tria.h>
// Here are some functions to generate standard grids:
#include <deal.II/grid/grid_generator.h>
// Output of grids in various graphics formats:
#include <deal.II/grid/grid_out.h>
#include <deal.II/base/utilities.h>

// This is needed for C++ output:
#include <iostream>
#include <fstream>
// And this for the declarations of the `std::sqrt` and `std::fabs` functions:
#include <cmath>
int main(int argc, char **argv)
{
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPI_Comm                         mpi_communicator(MPI_COMM_WORLD);

  parallel::distributed::Triangulation<2> triangulation(mpi_communicator);
  GridGenerator::hyper_cube(triangulation);

  std::cout << triangulation.n_active_cells() << std::endl;

  for (const auto &cell: triangulation.active_cell_iterators()){
    cell->set_material_id(cell->subdomain_id());
  }

  std::ofstream out("grid." + std::to_string(Utilities::MPI::this_mpi_process(triangulation.get_communicator())) + ".vtk");
  GridOut       grid_out;
  grid_out.write_vtk(triangulation, out);  
}
