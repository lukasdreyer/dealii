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


// Refinement and coarsening of meshes of pyramids, of pyramids and
// hexahedra, and of pyramids and tetrahedra with faces in all orientations:
// pyramids are split into 6 pyramids and 4 tetrahedra, the tetrahedra inside
// pyramids into 4 tetrahedra and 2 pyramids. Check
// ReferenceCell::child_cell_on_face(), that every face is seen in standard
// orientation by at least one of its two cells, the volumes of the children,
// CellAccessor::neighbor_child_on_subface() on an adaptively refined mesh,
// and coarsening back to the coarse mesh.

#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <map>
#include <sstream>
#include <string>

#include "../tests.h"

#include "pyramid_meshes.h"


void
refine(Triangulation<3> &tria)
{
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->reference_cell() == ReferenceCells::Tetrahedron)
      cell->set_refine_choice(
        static_cast<std::uint8_t>(IsotropicRefinementChoice::cut_tet_68));
  tria.execute_coarsening_and_refinement();
}

std::string
kind(const Triangulation<3>::cell_iterator &cell)
{
  std::string k = cell->reference_cell().to_string();
  if (cell->reference_cell() == ReferenceCells::Tetrahedron && cell->level() > 0 &&
      (cell->parent()->reference_cell() == ReferenceCells::Pyramid ||
       cell->parent()->refinement_case() == RefinementCase<3>::cut_z))
    k += "(in pyramid)";
  return k;
}

// child_cell_on_face: the child at subface s of face f contains that subface
void
check_child_cell_on_face(Triangulation<3> &tria)
{
  std::map<std::string, std::pair<unsigned, unsigned>> stats;
  for (const auto &cell : tria.cell_iterators())
    if (cell->has_children())
      for (const unsigned f : cell->face_indices())
        for (unsigned s = 0; s < cell->face(f)->n_children(); ++s)
          {
            const unsigned c = cell->reference_cell().child_cell_on_face(
              f, s, cell->combined_face_orientation(f), cell->refinement_case());
            bool owns = false;
            for (const unsigned j : cell->child(c)->face_indices())
              owns |= (cell->child(c)->face(j) == cell->face(f)->child(s));
            auto &st = stats[kind(cell) + " " + cell->face(f)->reference_cell().to_string()];
            ++st.first;
            st.second += !owns;
          }
  for (const auto &[k, st] : stats)
    deallog << "  child_cell_on_face " << k << ": " << st.second << " errors of "
              << st.first << std::endl;
}

// every face between two cells of the same level is seen in standard
// orientation by at least one of them
void
check_orientation(const Triangulation<3> &tria)
{
  std::map<std::string, unsigned> violations;
  unsigned                        n = 0, total = 0;
  for (const auto &cell : tria.cell_iterators())
    for (const unsigned f : cell->face_indices())
      {
        if (cell->at_boundary(f) || cell->neighbor(f)->level() != cell->level())
          continue;
        const auto neighbor = cell->neighbor(f);
        if (neighbor->index() < cell->index())
          continue;
        ++total;
        const unsigned nf = cell->neighbor_of_neighbor(f);
        if (cell->combined_face_orientation(f) != 0 &&
            neighbor->combined_face_orientation(nf) != 0)
          {
            ++n;
            std::ostringstream key;
            if (cell->level() == 0)
              key << "coarse";
            else if (cell->parent() == neighbor->parent())
              key << "inner " << kind(cell->parent());
            else
              key << "outer " << kind(cell->parent()) << " / "
                  << kind(neighbor->parent());
            key << " " << cell->face(f)->reference_cell().to_string();
            ++violations[key.str()];
          }
      }
  deallog << "  orientation: " << n << " of " << total
            << " faces with both sides non-standard" << std::endl;
  for (const auto &[k, c] : violations)
    deallog << "    " << c << "x " << k << std::endl;
}

void
check_volumes(const Triangulation<3> &tria)
{
  unsigned bad = 0, negative = 0;
  for (const auto &cell : tria.cell_iterators())
    {
      if (cell->measure() <= 0)
        ++negative;
      if (cell->has_children())
        {
          double sum = 0;
          for (unsigned c = 0; c < cell->n_children(); ++c)
            sum += cell->child(c)->measure();
          bad += std::abs(sum - cell->measure()) > 1e-12 * cell->measure();
        }
    }
  deallog << "  volumes: " << negative << " non-positive cells, " << bad
            << " parents with wrong child volume sum" << std::endl;
}

void
check_subfaces(const Triangulation<3> &tria)
{
  unsigned n = 0, bad = 0;
  for (const auto &cell : tria.active_cell_iterators())
    for (const unsigned f : cell->face_indices())
      if (!cell->at_boundary(f) && cell->neighbor(f)->has_children())
        for (unsigned s = 0; s < cell->face(f)->n_children(); ++s)
          {
            ++n;
            const auto child = cell->neighbor_child_on_subface(f, s);
            bool found = false;
            for (const unsigned j : child->face_indices())
              found |= (child->face(j) == cell->face(f)->child(s));
            bad += !found;
          }
  deallog << "  neighbor_child_on_subface: " << bad << " errors of " << n << std::endl;
}

int
main()
{
  initlog();

  const char *names[] = {"pyramids", "pyramids and hexahedra", "pyramids below tetrahedra"};
  for (unsigned int m = 0; m < 3; ++m)
    {
      Triangulation<3> tria;
      make_pyramid_mesh(tria, 2, m);
      const unsigned n_coarse = tria.n_active_cells();
      deallog << names[m] << ": " << n_coarse << " cells" << std::endl;
      check_orientation(tria);
      for (unsigned r = 0; r < 2; ++r)
        {
          for (const auto &cell : tria.active_cell_iterators())
            cell->set_refine_flag();
          refine(tria);
        }
      deallog << "  after 2 global refinements: " << tria.n_active_cells() << " cells"
                << std::endl;
      check_child_cell_on_face(tria);
      check_orientation(tria);
      check_volumes(tria);

      // coarsen back to the coarse mesh
      for (unsigned r = 0; r < 2; ++r)
        {
          for (const auto &cell : tria.active_cell_iterators())
            cell->set_coarsen_flag();
          tria.execute_coarsening_and_refinement();
        }
      deallog << "  coarsened: " << tria.n_active_cells() << " cells (coarse "
                << n_coarse << ")" << std::endl;

      // adaptive: refine around a point three times, then check the subfaces
      for (unsigned r = 0; r < 3; ++r)
        {
          for (const auto &cell : tria.active_cell_iterators())
            if (cell->center().distance(Point<3>(0.7, 0.6, 0.4)) < 0.8)
              cell->set_refine_flag();
          refine(tria);
        }
      deallog << "  adaptive: " << tria.n_active_cells() << " cells" << std::endl;
      check_subfaces(tria);
      check_orientation(tria);
    }
}
