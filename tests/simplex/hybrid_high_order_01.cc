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


// FE_Q (with equidistant points), FE_SimplexP, FE_WedgeP and FE_PyramidP of
// degree 3 and 4 on adaptively refined meshes of wedges, hexahedra,
// tetrahedra and pyramids with faces in all orientations: the shape
// functions are nodal, the DoFs of the faces seen from the cells are the
// ones of the faces, a function with arbitrary DoF values is continuous
// after the hanging node constraints, and a polynomial of the degree is
// interpolated exactly.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <algorithm>

#include "../tests.h"

#include "pyramid_meshes.h"
#include "wedge_meshes.h"


// a polynomial of total degree p
class Poly : public Function<3>
{
public:
  Poly(unsigned p) : p(p) {}
  double value(const Point<3> &x, const unsigned int) const override
  {
    return std::pow(1 + 0.3 * x[0] - 0.7 * x[1] + 0.5 * x[2], p) + x[0] * x[1] * (p >= 2) - x[2];
  }
  unsigned p;
};

unsigned fe_index(const ReferenceCell &r)
{
  return r == ReferenceCells::Hexahedron ? 0 : r == ReferenceCells::Tetrahedron ? 1 : r == ReferenceCells::Wedge ? 2 : 3;
}

double value_at(const DoFHandler<3>::active_cell_iterator &cell, const Point<3> &p, const Vector<double> &v)
{
  const auto &mapping = cell->reference_cell().get_default_linear_mapping<3>();
  const Point<3> u = mapping.transform_real_to_unit_cell(cell, p);
  std::vector<types::global_dof_index> dofs(cell->get_fe().n_dofs_per_cell());
  cell->get_dof_indices(dofs);
  double s = 0;
  for (unsigned i = 0; i < dofs.size(); ++i)
    s += v[dofs[i]] * cell->get_fe().shape_value(i, u);
  return s;
}

void check_nodal(const FiniteElement<3> &fe)
{
  double err = 0;
  for (unsigned i = 0; i < fe.n_dofs_per_cell(); ++i)
    for (unsigned j = 0; j < fe.n_dofs_per_cell(); ++j)
      err = std::max(err, std::abs(fe.shape_value(i, fe.get_unit_support_points()[j]) - (i == j)));
  deallog << "nodal " << fe.get_name() << ": " << (err < 1e-12 ? "yes" : "no") << std::endl;
}

void check_mesh(const std::string &name, Triangulation<3> &tria, const unsigned p)
{
  const hp::FECollection<3> fe{FE_Q<3>(QIterated<1>(QTrapezoid<1>(), p)), FE_SimplexP<3>(p), FE_WedgeP<3>(p), FE_PyramidP<3>(p)};
  const hp::MappingCollection<3> mapping(ReferenceCells::Hexahedron.get_default_linear_mapping<3>(),
                                         ReferenceCells::Tetrahedron.get_default_linear_mapping<3>(),
                                         ReferenceCells::Wedge.get_default_linear_mapping<3>(),
                                         ReferenceCells::Pyramid.get_default_linear_mapping<3>());
  // refine adaptively twice to get hanging nodes
  for (unsigned r = 0; r < 2; ++r)
    {
      unsigned c = 0;
      for (const auto &cell : tria.active_cell_iterators())
        if (c++ % 3 == 0)
          cell->set_refine_flag();
      tria.prepare_coarsening_and_refinement();
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->reference_cell() == ReferenceCells::Tetrahedron)
          cell->set_refine_choice(static_cast<std::uint8_t>(IsotropicRefinementChoice::cut_tet_68));
      tria.execute_coarsening_and_refinement();
    }

  DoFHandler<3> dh(tria);
  for (const auto &cell : dh.active_cell_iterators())
    cell->set_active_fe_index(fe_index(cell->reference_cell()));
  dh.distribute_dofs(fe);
  AffineConstraints<double> c;
  DoFTools::make_hanging_node_constraints(dh, c);
  c.close();

  // face dofs through the cell and through the face
  unsigned face_dof_errors = 0;
  for (const auto &cell : dh.active_cell_iterators())
    for (const unsigned f : cell->face_indices())
      {
        const auto &cfe = cell->get_fe();
        std::vector<types::global_dof_index> face_dofs(cfe.n_dofs_per_face(f)), cell_dofs(cfe.n_dofs_per_cell());
        cell->face(f)->get_dof_indices(face_dofs, cell->active_fe_index());
        cell->get_dof_indices(cell_dofs);
        // the same DoFs (as sets: face_to_cell_index() does not know the
        // orientations of the lines inside the face)
        std::vector<types::global_dof_index> via_cell;
        for (unsigned i = 0; i < face_dofs.size(); ++i)
          via_cell.push_back(cell_dofs[cfe.face_to_cell_index(i, f, cell->combined_face_orientation(f))]);
        std::sort(face_dofs.begin(), face_dofs.end());
        std::sort(via_cell.begin(), via_cell.end());
        face_dof_errors += (face_dofs != via_cell);
      }

  // continuity of a function with arbitrary dof values
  Vector<double> v(dh.n_dofs());
  for (unsigned i = 0; i < v.size(); ++i)
    v[i] = std::sin(0.7 * i) + 0.1 * i;
  c.distribute(v);
  double jump = 0;
  for (const auto &cell : dh.active_cell_iterators())
    for (const unsigned f : cell->face_indices())
      {
        if (cell->at_boundary(f) || cell->neighbor(f)->has_children())
          continue;
        const auto face = cell->face(f);
        std::vector<Point<3>> pts;
        // points on the face: combinations of its vertices
        const unsigned n = face->n_vertices();
        for (unsigned a = 0; a < 7; ++a)
          for (unsigned b = 0; b + a < 7; ++b)
            {
              Point<3> x;
              if (n == 3)
                x = (a * face->vertex(1) + b * face->vertex(2) + (6 - a - b) * face->vertex(0)) / 6.;
              else
                {
                  const double s = a / 6., t = b / 6.;
                  x = (1 - s) * (1 - t) * face->vertex(0) + s * (1 - t) * face->vertex(1) + (1 - s) * t * face->vertex(2) + s * t * face->vertex(3);
                }
              pts.push_back(x);
            }
        for (const auto &x : pts)
          jump = std::max(jump, std::abs(value_at(cell, x, v) - value_at(cell->neighbor(f), x, v)));
      }

  // interpolation of a polynomial of degree p
  Vector<double> u(dh.n_dofs());
  VectorTools::interpolate(mapping, dh, Poly(p), u);
  Vector<double> u2 = u;
  c.distribute(u2);
  u2 -= u;
  double interp = 0;
  for (const auto &cell : dh.active_cell_iterators())
    for (const auto &x : {cell->center(), (cell->center() + cell->vertex(0)) / 2.})
      interp = std::max(interp, std::abs(value_at(cell, x, u) - Poly(p).value(x, 0)));

  deallog << name << ": cells " << tria.n_active_cells() << ", dofs "
          << dh.n_dofs() << ", constraints " << c.n_constraints()
          << ", face dof errors " << face_dof_errors << ", "
          << (jump < 1e-8 ? "continuous" : "jump") << ", "
          << (interp < 1e-10 && u2.linfty_norm() < 1e-10 ? "exact" : "not exact")
          << std::endl;
}



int
main()
{
  initlog();

  for (unsigned int p = 3; p <= 4; ++p)
    {
      deallog.push("degree " + std::to_string(p));
      check_nodal(FE_SimplexP<3>(p));
      check_nodal(FE_WedgeP<3>(p));
      check_nodal(FE_PyramidP<3>(p));
      const char *wedge_names[] = {"rotated wedges",
                                   "rotated hexahedra and wedges",
                                   "wedges below tetrahedra"};
      for (unsigned int m = 0; m < 3; ++m)
        {
          Triangulation<3> tria;
          if (m < 2)
            make_wedge_mesh(tria, 2, m == 1);
          else
            make_wedge_tetrahedron_mesh(tria, 2);
          check_mesh(wedge_names[m], tria, p);
        }
      const char *pyramid_names[] = {"rotated pyramids",
                                     "rotated pyramids and hexahedra",
                                     "rotated pyramids below tetrahedra"};
      for (unsigned int m = 0; m < 3; ++m)
        {
          Triangulation<3> tria;
          make_pyramid_mesh(tria, 2, m);
          check_mesh(pyramid_names[m], tria, p);
        }
      deallog.pop();
    }
}
