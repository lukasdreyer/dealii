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


// Check ReferenceCell::child_cell_on_face() geometrically: after one
// refinement, the child returned for a subface (numbered as a child of the
// face) must have this subface as one of its faces. The meshes consist of
// wedges, hexahedra and tetrahedra whose vertices are permuted by symmetries
// of their reference cells, so that the faces occur in all orientations, and
// of tetrahedra from a mesh of rotated hexahedra.

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <map>
#include <string>
#include <vector>

#include "../tests.h"

#include "wedge_meshes.h"


void
check(const std::string &name, Triangulation<3> &tria)
{
  // tetrahedra can only be refined along the line (6,8)
  for (const auto &cell : tria.active_cell_iterators())
    {
      cell->set_refine_flag();
      if (cell->reference_cell() == ReferenceCells::Tetrahedron)
        cell->set_refine_choice(
          static_cast<std::uint8_t>(IsotropicRefinementChoice::cut_tet_68));
    }
  tria.execute_coarsening_and_refinement();

  // per reference cell and face shape: number of faces in each orientation
  // and the number of wrong children
  std::map<std::string, std::map<unsigned int, unsigned int>> orientations;
  std::map<std::string, unsigned int>                         errors;
  for (const auto &cell : tria.cell_iterators_on_level(0))
    for (const unsigned int f : cell->face_indices())
      {
        const std::string kind =
          cell->reference_cell().to_string() + " " +
          cell->reference_cell().face_reference_cell(f).to_string();
        const auto orientation = cell->combined_face_orientation(f);
        ++orientations[kind][orientation];
        errors[kind] += 0;

        for (unsigned int s = 0; s < cell->face(f)->n_children(); ++s)
          {
            const auto child = cell->child(
              cell->reference_cell().child_cell_on_face(f, s, orientation));
            bool found = false;
            for (const unsigned int j : child->face_indices())
              if (child->face(j) == cell->face(f)->child(s))
                found = true;
            if (!found)
              ++errors[kind];
          }
      }

  deallog.push(name);
  for (const auto &[kind, counts] : orientations)
    {
      deallog << kind << ": orientations";
      for (const auto &[orientation, count] : counts)
        deallog << ' ' << orientation << ':' << count;
      deallog << ", errors " << errors[kind] << std::endl;
    }
  deallog.pop();
}



int
main()
{
  initlog();

  {
    Triangulation<3> hypercubes, tria;
    GridGenerator::non_standard_orientation_mesh(
      hypercubes, false, true, true, false);
    GridGenerator::convert_hypercube_to_simplex_mesh(hypercubes, tria);
    check("tetrahedra from rotated hexahedra", tria);
  }
  {
    Triangulation<3> tria;
    make_wedge_mesh(tria, 3, false);
    check("rotated wedges", tria);
  }
  {
    Triangulation<3> tria;
    make_wedge_mesh(tria, 3, true);
    check("rotated hexahedra and wedges", tria);
  }
  {
    Triangulation<3> tria;
    make_wedge_tetrahedron_mesh(tria, 3);
    check("rotated wedges below tetrahedra", tria);
  }
}
