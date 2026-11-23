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


// Edge elements need a direction of every line that all processes agree on.
// This test checks on adaptively refined and coarsened
// parallel::distributed::Triangulations (with t8code: tetrahedra, wedges,
// pyramids and hexahedra; with p4est: hexahedra) that
// 1. the direction of the line objects (line->vertex(0) to line->vertex(1))
//    is the same on all processes that have the line (owned or ghost cells),
// 2. cell->line_orientation(l) is the default orientation exactly when the
//    line object starts at the cell's local vertex
//    line_to_cell_vertices(l, 0).
// For comparison, it also counts the shared lines on which the order of the
// two vertex indices (used by FE_NedelecSZ) differs between processes. This
// number is printed to stdout only, since it depends on the partition.
// Lines are matched across processes by their endpoint coordinates.

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <iostream>
#include <map>

#include "../tests.h"

#ifdef DEAL_II_WITH_T8CODE
#  include "pyramid_meshes.h"
#endif


// [0,n]^2 x [0,n]: every unit square is split into two triangles, which are
// extruded to wedges. The diagonal alternates, so that the triangles have
// different vertex orders.
void
make_wedge_mesh(Triangulation<3> &tria, const unsigned int n)
{
  std::vector<Point<3>> vertices;
  for (unsigned int z = 0; z <= n; ++z)
    for (unsigned int y = 0; y <= n; ++y)
      for (unsigned int x = 0; x <= n; ++x)
        vertices.emplace_back(x, y, z);
  const auto v = [n](unsigned int x, unsigned int y, unsigned int z) {
    return x + (n + 1) * (y + (n + 1) * z);
  };
  std::vector<CellData<3>> cells;
  for (unsigned int z = 0; z < n; ++z)
    for (unsigned int y = 0; y < n; ++y)
      for (unsigned int x = 0; x < n; ++x)
        {
          // triangles in counterclockwise order seen from above
          std::array<std::array<unsigned int, 3>, 2> triangles;
          if ((x + y) % 2 == 0)
            triangles = {{{{v(x, y, 0), v(x + 1, y, 0), v(x + 1, y + 1, 0)}},
                          {{v(x, y, 0), v(x + 1, y + 1, 0), v(x, y + 1, 0)}}}};
          else
            triangles = {{{{v(x + 1, y, 0), v(x + 1, y + 1, 0), v(x, y, 0)}},
                          {{v(x, y + 1, 0), v(x, y, 0), v(x + 1, y + 1, 0)}}}};
          for (const auto &t : triangles)
            {
              CellData<3> cell(6);
              const unsigned int layer = v(0, 0, 1);
              for (unsigned int i = 0; i < 3; ++i)
                {
                  cell.vertices[i]     = t[i] + z * layer;
                  cell.vertices[i + 3] = t[i] + (z + 1) * layer;
                }
              cells.push_back(cell);
            }
        }
  tria.create_triangulation(vertices, cells, SubCellData());
}



// A line as seen by one process: its endpoints sorted lexicographically,
// whether the line object starts at the smaller endpoint, and whether the
// vertex index of the smaller endpoint is the smaller one.
struct LineKey
{
  std::array<long, 6> coords;
  bool
  operator<(const LineKey &other) const
  {
    return coords < other.coords;
  }
};



void
check(parallel::distributed::Triangulation<3> &tria,
      const std::string                        &name,
      const unsigned int                        n_cycles)
{
  const MPI_Comm     comm = tria.get_mpi_communicator();
  const unsigned int rank = Utilities::MPI::this_mpi_process(comm);

  double max_coordinate = 0;
  for (const auto &v : tria.get_vertices())
    max_coordinate = std::max(max_coordinate, v.norm());

  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      // refine in a shell that moves with the cycle, coarsen elsewhere
      const Point<3> center(0.3 * max_coordinate, 0.4 * max_coordinate, 0.2 * max_coordinate);
      const double   radius = (0.2 + 0.15 * cycle) * max_coordinate;
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const double d = cell->center().distance(center);
            if (std::abs(d - radius) < 0.6 * cell->diameter() &&
                cell->level() < 3)
              cell->set_refine_flag();
            else if (cell->level() > 0 && std::abs(d - radius) > 0.25 * max_coordinate)
              cell->set_coarsen_flag();
          }
      tria.execute_coarsening_and_refinement();

      // collect the lines of the owned and ghost cells
      const double scale = 1e8 / max_coordinate;
      std::map<LineKey, std::pair<bool, bool>> lines;
      unsigned int n_flag_errors = 0;
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned() || cell->is_ghost())
          for (unsigned int l = 0; l < cell->n_lines(); ++l)
            {
              const auto         line = cell->line(l);
              const unsigned int v0 =
                cell->reference_cell().line_to_cell_vertices(l, 0);
              const bool starts_at_local_v0 =
                line->vertex_index(0) == cell->vertex_index(v0);
              const bool default_orientation =
                cell->line_orientation(l) ==
                numbers::default_geometric_orientation;
              if (starts_at_local_v0 != default_orientation)
                ++n_flag_errors;

              std::array<std::array<long, 3>, 2> p;
              for (unsigned int i = 0; i < 2; ++i)
                for (unsigned int d = 0; d < 3; ++d)
                  p[i][d] = std::lround(line->vertex(i)[d] * scale);
              const bool object_starts_at_smaller = p[0] < p[1];
              const bool index_of_smaller_is_smaller =
                (line->vertex_index(0) < line->vertex_index(1)) ==
                object_starts_at_smaller;
              if (!object_starts_at_smaller)
                std::swap(p[0], p[1]);
              LineKey key;
              for (unsigned int d = 0; d < 3; ++d)
                {
                  key.coords[d]     = p[0][d];
                  key.coords[d + 3] = p[1][d];
                }
              lines[key] = {object_starts_at_smaller,
                            index_of_smaller_is_smaller};
            }

      // send everything to rank 0 as a flat vector
      std::vector<long> flat;
      for (const auto &[key, dirs] : lines)
        {
          flat.insert(flat.end(), key.coords.begin(), key.coords.end());
          flat.push_back(dirs.first);
          flat.push_back(dirs.second);
        }
      const auto all_flat = Utilities::MPI::gather(comm, flat, 0);
      const unsigned int n_flag_errors_total =
        Utilities::MPI::sum(n_flag_errors, comm);

      if (rank == 0)
        {
          // for every line: the directions seen by the processes
          std::map<LineKey, std::vector<std::pair<bool, bool>>> seen;
          for (const auto &f : all_flat)
            for (unsigned int i = 0; i < f.size(); i += 8)
              {
                LineKey key;
                std::copy(f.begin() + i, f.begin() + i + 6, key.coords.begin());
                seen[key].emplace_back(f[i + 6] != 0, f[i + 7] != 0);
              }
          unsigned int n_shared = 0, n_direction_errors = 0,
                       n_index_disagreements = 0;
          for (const auto &[key, dirs] : seen)
            if (dirs.size() > 1)
              {
                ++n_shared;
                bool same_direction = true, same_index_order = true;
                for (const auto &d : dirs)
                  {
                    same_direction &= (d.first == dirs[0].first);
                    same_index_order &= (d.second == dirs[0].second);
                  }
                n_direction_errors += !same_direction;
                n_index_disagreements += !same_index_order;
              }

          deallog << name << " cycle " << cycle << ": line directions "
                  << (n_direction_errors == 0 ? "agree" : "DISAGREE")
                  << ", line_orientation flags "
                  << (n_flag_errors_total == 0 ? "OK" : "WRONG") << std::endl;
          std::cout << name << " cycle " << cycle
                    << ": cells=" << tria.n_global_active_cells()
                    << " lines=" << seen.size() << " shared=" << n_shared
                    << " direction errors=" << n_direction_errors
                    << " flag errors=" << n_flag_errors_total
                    << " vertex index order differs=" << n_index_disagreements
                    << std::endl;
        }
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  MPILogInitAll                    log;

  const auto make = [](const std::string &name, const auto &generator) {
    parallel::distributed::Triangulation<3> tria(MPI_COMM_WORLD);
    generator(tria);
    check(tria, name, 4);
  };

  make("hex", [](Triangulation<3> &tria) {
    GridGenerator::subdivided_hyper_cube(tria, 4);
  });
#ifdef DEAL_II_WITH_T8CODE
  make("tet", [](Triangulation<3> &tria) {
    GridGenerator::subdivided_hyper_cube_with_simplices(tria, 3);
  });
  make("wedge", [](Triangulation<3> &tria) { make_wedge_mesh(tria, 3); });
  for (unsigned int mesh = 0; mesh < 3; ++mesh)
    make("pyramid mesh " + std::to_string(mesh),
         [mesh](Triangulation<3> &tria) { make_pyramid_mesh(tria, 3, mesh); });
#endif
}
