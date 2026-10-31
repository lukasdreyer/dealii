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


// Helpers for the tests of adaptive parallel::distributed::Triangulation
// meshes with simplices and hypercubes (adaptive_*.cc): coarse meshes with
// several trees, among them trees with non-standard orientation and mixed
// meshes, a refinement criterion that only depends on the geometry (so that
// the mesh does not depend on the number of processes), coarsening back to the
// coarse mesh, and a geometric test whether two cells touch.
//
// Coarsening does depend on the number of processes: deal.II only keeps the
// coarsen flags of a ghost family if all its children are known, and the mesh
// smoothing then keeps the locally owned cells next to it from coarsening. So
// the tests print only results that do not depend on the number of processes
// while coarsening.

#include <deal.II/base/bounding_box.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>

#include <string>
#include <vector>

#include "../tests.h"

#include "../simplex/wedge_meshes.h"


// The names of the coarse meshes of make_coarse_mesh() in dimension dim.
template <int dim>
std::vector<std::string>
coarse_mesh_names()
{
  if constexpr (dim == 2)
    return {"triangles",
            "triangles from rotated quads",
            "rotated quads",
            "quads and triangles"};
  else
    return {"tetrahedra",
            "tetrahedra from rotated hexahedra",
            "rotated hexahedra",
            "rotated wedges",
            "rotated hexahedra and wedges",
            "rotated wedges below tetrahedra"};
}



// Create coarse mesh number i of coarse_mesh_names() in tria.
template <int dim>
void
make_coarse_mesh(Triangulation<dim> &tria, const unsigned int i)
{
  Triangulation<dim> hypercubes;
  if constexpr (dim == 2)
    GridGenerator::non_standard_orientation_mesh(hypercubes, 1);
  else
    GridGenerator::non_standard_orientation_mesh(
      hypercubes, false, true, true, false);

  if (i == 0)
    GridGenerator::subdivided_hyper_cube_with_simplices(tria, 2);
  else if (i == 1)
    GridGenerator::convert_hypercube_to_simplex_mesh(hypercubes, tria);
  else if (i == 2)
    tria.copy_triangulation(hypercubes);
  else if constexpr (dim == 2)
    {
      // [0,3]x[0,2]: quads for x < 2, triangles with alternating diagonals
      // for x > 2
      std::vector<Point<2>> vertices;
      for (unsigned int y = 0; y < 3; ++y)
        for (unsigned int x = 0; x < 4; ++x)
          vertices.emplace_back(x, y);
      const auto vertex = [](const unsigned int x, const unsigned int y) {
        return 4 * y + x;
      };

      std::vector<CellData<2>> cells;
      for (unsigned int y = 0; y < 2; ++y)
        {
          for (unsigned int x = 0; x < 2; ++x)
            {
              CellData<2> quad(4);
              quad.vertices = {
                vertex(x, y), vertex(x + 1, y), vertex(x, y + 1), vertex(x + 1, y + 1)};
              cells.push_back(quad);
            }
          const unsigned int v00 = vertex(2, y), v10 = vertex(3, y),
                             v01 = vertex(2, y + 1), v11 = vertex(3, y + 1);
          CellData<2> triangle_1(3), triangle_2(3);
          if (y == 0)
            {
              triangle_1.vertices = {v00, v10, v11};
              triangle_2.vertices = {v00, v11, v01};
            }
          else
            {
              triangle_1.vertices = {v00, v10, v01};
              triangle_2.vertices = {v10, v11, v01};
            }
          cells.push_back(triangle_1);
          cells.push_back(triangle_2);
        }
      tria.create_triangulation(vertices, cells, SubCellData());
    }
  else if (i < 5)
    make_wedge_mesh(tria, 2, i == 4);
  else
    make_wedge_tetrahedron_mesh(tria, 2);
}



// The continuous elements of the given degree for the reference cells of
// dim: the hypercube element first, the simplex element second, in 3d the
// wedge element third.
template <int dim>
hp::FECollection<dim>
make_fe_collection(const unsigned int degree)
{
  hp::FECollection<dim> fe{FE_Q<dim>(degree), FE_SimplexP<dim>(degree)};
  if constexpr (dim == 3)
    fe.push_back(FE_WedgeP<dim>(degree));
  return fe;
}



// The linear mappings in the order of make_fe_collection().
template <int dim>
hp::MappingCollection<dim>
make_mapping_collection()
{
  hp::MappingCollection<dim> mapping(
    ReferenceCells::get_hypercube<dim>()
      .template get_default_linear_mapping<dim>(),
    ReferenceCells::get_simplex<dim>()
      .template get_default_linear_mapping<dim>());
  if constexpr (dim == 3)
    mapping.push_back(
      ReferenceCells::Wedge.template get_default_linear_mapping<dim>());
  return mapping;
}



// Choose the element of make_fe_collection() of each locally owned cell by
// its reference cell.
template <int dim>
void
set_fe_indices(DoFHandler<dim> &dof_handler)
{
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(
        cell->reference_cell().is_simplex() ? 1 :
        cell->reference_cell().is_hyper_cube() ? 0 :
                                                 2);
}



// Whether the closed point p lies in the closed cell (up to a relative
// tolerance).
template <int dim, typename CellIterator>
bool
point_in_cell(const CellIterator &cell, const Point<dim> &p)
{
  const double tolerance = 1e-10;
  BoundingBox<dim> box(cell->bounding_box());
  box.extend(tolerance * cell->diameter());
  if (!box.point_inside(p))
    return false;

  const Mapping<dim> &mapping =
    cell->reference_cell().template get_default_linear_mapping<dim>();
  try
    {
      const Point<dim> unit = mapping.transform_real_to_unit_cell(cell, p);
      return cell->reference_cell().contains_point(unit, tolerance);
    }
  catch (const typename Mapping<dim>::ExcTransformationFailed &)
    {
      return false;
    }
}



// Refine the locally owned cells containing center or closer than radius to
// it up to max_level.
template <int dim>
void
refine_around(parallel::distributed::Triangulation<dim> &tria,
              const Point<dim>                          &center,
              const double                               radius,
              const unsigned int                         max_level)
{
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned() &&
        cell->level() < static_cast<int>(max_level) &&
        (cell->center().distance(center) < radius ||
         point_in_cell<dim>(cell, center)))
      cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();
}



// Coarsen all locally owned cells once. Returns false if the mesh is the
// coarse mesh already.
template <int dim>
bool
coarsen_all(parallel::distributed::Triangulation<dim> &tria)
{
  if (tria.n_global_levels() == 1)
    return false;
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_coarsen_flag();
  tria.execute_coarsening_and_refinement();
  return true;
}



// The center of the refinement in cycle k of n_cycles: moves along the
// diagonal of the bounding box of the coarse mesh.
template <int dim>
Point<dim>
refinement_center(const BoundingBox<dim> &box,
                  const unsigned int      k,
                  const unsigned int      n_cycles)
{
  const double s = 0.2 + 0.6 * k / std::max(n_cycles - 1, 1u);
  return box.get_boundary_points().first +
         s * (box.get_boundary_points().second -
              box.get_boundary_points().first);
}



// Whether the closures of two cells of the same nested mesh intersect: then
// a vertex of one of them lies in the other one.
template <int dim, typename CellIterator>
bool
cells_touch(const CellIterator &cell_1, const CellIterator &cell_2)
{
  BoundingBox<dim> box_1(cell_1->bounding_box());
  box_1.extend(1e-10 * cell_1->diameter());
  if (box_1.get_neighbor_type(cell_2->bounding_box()) ==
      NeighborType::not_neighbors)
    return false;

  for (const unsigned int v : cell_1->vertex_indices())
    if (point_in_cell<dim>(cell_2, cell_1->vertex(v)))
      return true;
  for (const unsigned int v : cell_2->vertex_indices())
    if (point_in_cell<dim>(cell_1, cell_2->vertex(v)))
      return true;
  return false;
}
