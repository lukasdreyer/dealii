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


// The neighbor settings of parallel::distributed::Triangulation with
// t8code: set_balance_type(), set_ghost_layer() and set_ghost_overlap().
//
// A cube of 6^dim quadrilaterals or hexahedra is refined once and then three
// times at the cell above the center vertex p = (0.5, ..., 0.5): the cell that
// has p as its lowest vertex. The cell below p touches it only at p, so
// without a balance across vertices the levels at p jump. On axis-parallel boxes, two cells touch exactly
// where their bounding boxes intersect, and the dimension of the contact is
// the number of directions with an overlap of positive length. This gives
// independent checks:
// - balance: every two cells that touch with a contact of at least the
//   dimension of the balance type differ by at most one level (owned cells
//   against owned and ghost cells, with the vertex ghost layer);
// - ghost layer: every ghost cell touches an owned cell with a contact of at
//   least the dimension of the ghost type, and the vertex ghost layer
//   contains the others;
// - overlap: the ghost cells contain the vertex ghost layer and are within
//   one level-0 cell (sharing a vertex) of an owned level-0 cell.
// Settings that deal.II cannot use are rejected.

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <set>

#include "../tests.h"


using ConnectionType = parallel::distributed::ConnectionType;

std::string
name(const ConnectionType type)
{
  return type == ConnectionType::faces ? "faces" :
         type == ConnectionType::edges ? "edges" :
                                       "vertices";
}

// the minimal dimension of a contact of this type (in 2d, t8code's edges are
// vertices)
template <int dim>
int
min_contact_dim(const ConnectionType type)
{
  return type == ConnectionType::faces ? dim - 1 :
         type == ConnectionType::edges ? (dim == 3 ? 1 : 0) :
                                       0;
}

// the dimension of the contact of two boxes, -1 if they do not touch
template <int dim>
int
contact_dim(const BoundingBox<dim> &a, const BoundingBox<dim> &b)
{
  const double tol     = 1e-12;
  int          contact = 0;
  for (unsigned int d = 0; d < dim; ++d)
    {
      const double overlap =
        std::min(a.get_boundary_points().second[d],
                 b.get_boundary_points().second[d]) -
        std::max(a.get_boundary_points().first[d],
                 b.get_boundary_points().first[d]);
      if (overlap < -tol)
        return -1;
      if (overlap > tol)
        ++contact;
    }
  return contact;
}

// a cube of 6^dim cells, refined once and then three times at the cell above
// the center vertex
template <int dim>
void
create(parallel::distributed::Triangulation<dim> &tria)
{
  Triangulation<dim> coarse;
  GridGenerator::subdivided_hyper_cube(coarse, 6);
  tria.copy_triangulation(coarse);
  tria.refine_global(1);
  for (unsigned int cycle = 0; cycle < 3; ++cycle)
    {
      Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = 0.5;
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            bool above = true;
            for (unsigned int d = 0; d < dim; ++d)
              above = above && cell->center()[d] > p[d];
            for (const unsigned int v : cell->vertex_indices())
              if (above && cell->vertex(v).distance(p) < 1e-12)
                cell->set_refine_flag();
          }
      tria.execute_coarsening_and_refinement();
    }
}

template <int dim>
std::set<CellId>
ghost_ids(const parallel::distributed::Triangulation<dim> &tria)
{
  std::set<CellId> ids;
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_ghost())
      ids.insert(cell->id());
  return ids;
}

template <int dim>
void
test_balance(const ConnectionType type)
{
  const MPI_Comm                            comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<dim> tria(comm);
  tria.set_balance_type(type);
  create(tria);

  // violations of the balance type and of the vertex balance
  unsigned int violations = 0, vertex_violations = 0;
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto &other : tria.active_cell_iterators())
        if (!other->is_artificial() && other != cell &&
            std::abs(cell->level() - other->level()) > 1)
          {
            const int contact =
              contact_dim(cell->bounding_box(), other->bounding_box());
            violations += (contact >= min_contact_dim<dim>(type));
            vertex_violations += (contact >= 0);
          }
  deallog << dim << "d balance across " << name(type) << ": "
          << tria.n_global_active_cells() << " cells, violations "
          << Utilities::MPI::sum(violations, comm)
          << ", level jumps across vertices "
          << (Utilities::MPI::sum(vertex_violations, comm) > 0 ? "yes" : "no")
          << std::endl;
}

template <int dim>
void
test_ghost_layer(const ConnectionType type)
{
  const MPI_Comm                            comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<dim> tria(comm), reference(comm);
  tria.set_ghost_layer(type);
  create(tria);
  create(reference);

  unsigned int not_touching = 0;
  for (const auto &ghost : tria.active_cell_iterators())
    if (ghost->is_ghost())
      {
        bool touches = false;
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned() &&
              contact_dim(cell->bounding_box(), ghost->bounding_box()) >=
                min_contact_dim<dim>(type))
            touches = true;
        not_touching += !touches;
      }
  const std::set<CellId> ghosts = ghost_ids(tria), vertex = ghost_ids(reference);
  const bool subset = std::includes(vertex.begin(), vertex.end(),
                                    ghosts.begin(), ghosts.end());
  deallog << dim << "d ghost layer across " << name(type) << ": "
          << Utilities::MPI::sum<unsigned int>(ghosts.size(), comm)
          << " ghost cells (vertex layer "
          << Utilities::MPI::sum<unsigned int>(vertex.size(), comm)
          << "), not touching an owned cell "
          << Utilities::MPI::sum(not_touching, comm)
          << ", in the vertex layer on all processes "
          << (Utilities::MPI::min<unsigned int>(subset, comm) ? "yes" : "no")
          << ", same mesh "
          << (tria.n_global_active_cells() == reference.n_global_active_cells() ?
                "yes" :
                "no")
          << std::endl;
}

template <int dim>
void
test_overlap()
{
  const MPI_Comm                            comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<dim> tria(comm), reference(comm);
  tria.set_ghost_overlap(0, 1);
  create(tria);
  create(reference);

  // the level-0 cells with an owned descendant, and their vertices
  std::set<unsigned int> owned_vertices;
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        auto coarse = typename Triangulation<dim>::cell_iterator(cell);
        while (coarse->level() > 0)
          coarse = coarse->parent();
        for (const unsigned int v : coarse->vertex_indices())
          owned_vertices.insert(coarse->vertex_index(v));
      }
  // every ghost lies in a level-0 cell that shares a vertex with one of them
  unsigned int too_far = 0;
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_ghost())
      {
        auto coarse = typename Triangulation<dim>::cell_iterator(cell);
        while (coarse->level() > 0)
          coarse = coarse->parent();
        bool near = false;
        for (const unsigned int v : coarse->vertex_indices())
          near = near || owned_vertices.count(coarse->vertex_index(v)) > 0;
        too_far += !near;
      }
  const std::set<CellId> ghosts = ghost_ids(tria), vertex = ghost_ids(reference);
  const bool contains = std::includes(ghosts.begin(), ghosts.end(),
                                      vertex.begin(), vertex.end());
  deallog << dim << "d overlap of one level-0 layer: "
          << Utilities::MPI::sum<unsigned int>(ghosts.size(), comm)
          << " ghost cells, outside the overlap "
          << Utilities::MPI::sum(too_far, comm)
          << ", contains the vertex layer on all processes "
          << (Utilities::MPI::min<unsigned int>(contains, comm) ? "yes" : "no")
          << std::endl;
}

template <int dim>
void
test_rejected()
{
  const MPI_Comm comm = MPI_COMM_WORLD;
  if (dim == 3)
    {
      parallel::distributed::Triangulation<dim> tria(comm);
      try
        {
          tria.set_balance_type(ConnectionType::faces);
          deallog << "3d balance across faces: accepted" << std::endl;
        }
      catch (const std::exception &)
        {
          deallog << "3d balance across faces: rejected" << std::endl;
        }
    }
  // multigrid needs the vertex balance
  parallel::distributed::Triangulation<dim> tria(
    comm,
    Triangulation<dim>::none,
    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
  const ConnectionType weaker =
    dim == 3 ? ConnectionType::edges : ConnectionType::faces;
  try
    {
      tria.set_balance_type(weaker);
      deallog << dim << "d multigrid with balance across " << name(weaker)
              << ": accepted" << std::endl;
    }
  catch (const std::exception &)
    {
      deallog << dim << "d multigrid with balance across " << name(weaker)
              << ": rejected" << std::endl;
    }
}

template <int dim>
void
test()
{
  if (dim == 2)
    {
      test_balance<dim>(ConnectionType::faces);
      test_balance<dim>(ConnectionType::vertices);
    }
  else
    {
      test_balance<dim>(ConnectionType::edges);
      test_balance<dim>(ConnectionType::vertices);
    }
  for (const ConnectionType type :
       {ConnectionType::faces, ConnectionType::edges, ConnectionType::vertices})
    test_ghost_layer<dim>(type);
  test_overlap<dim>();
  test_rejected<dim>();
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  MPILogInitAll                    log;

  test<2>();
  test<3>();
}
