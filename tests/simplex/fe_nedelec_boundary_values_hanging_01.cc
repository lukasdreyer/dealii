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


// VectorTools::interpolate_boundary_values_curl_conforming with hanging
// nodes on hybrid meshes: after the hanging node constraints and the
// tangential boundary values of g = a + b x x, every line with DoFs on a
// boundary face of any level must be constrained to the line integral of g.
// This includes the refined lines on refined boundary faces that still carry
// a DoF because a coarser cell touches the boundary only along this line
// (e.g. a tetrahedron with one edge on the boundary): such a line is not a
// line of any active boundary face. If it stays free, its children on the
// boundary (constrained to half of it) violate n x u = n x g, and the edge
// space no longer matches the nodal space with zero boundary values, which
// slows down multigrid with a Hiptmair smoother.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/edge_element_tools.h>

#include <deal.II/fe/fe_nedelec_hypercube.h>
#include <deal.II/fe/fe_nedelec_pyramid.h>
#include <deal.II/fe/fe_nedelec_simplex.h>
#include <deal.II/fe/fe_nedelec_wedge.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>

#include <deal.II/lac/affine_constraints.h>

#include <functional>
#include <set>

#include "../tests.h"

#include "./pyramid_meshes.h"
#include "./wedge_meshes.h"


class LinearField : public Function<3>
{
public:
  LinearField()
    : Function<3>(3)
  {}

  void
  vector_value(const Point<3> &x, Vector<double> &values) const override
  {
    const Tensor<1, 3> v = value(x);
    for (unsigned int d = 0; d < 3; ++d)
      values(d) = v[d];
  }

  static Tensor<1, 3>
  value(const Point<3> &x)
  {
    const Tensor<1, 3> a({1., -0.5, 0.25}), b({0.2, 0.3, 0.7});
    return a + cross_product_3d(b, x);
  }
};



// refine the cells that have a boundary face (tetrahedra along the line
// (6,8), as in the adaptive tests); the cells that touch the boundary only
// along an edge stay coarse
void
refine_boundary_cells(Triangulation<3> &tria)
{
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->at_boundary())
      cell->set_refine_flag();
  tria.prepare_coarsening_and_refinement();
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->refine_flag_set() &&
        cell->reference_cell() == ReferenceCells::Tetrahedron)
      cell->set_refine_choice(
        static_cast<std::uint8_t>(IsotropicRefinementChoice::cut_tet_68));
  tria.execute_coarsening_and_refinement();
}



void
test(const std::string                              &name,
     const std::function<void(Triangulation<3> &)> &make)
{
  Triangulation<3> tria;
  make(tria);
  refine_boundary_cells(tria);

  const hp::FECollection<3> fe{FE_NedelecSimplex<3>(0),
                               FE_NedelecWedge<3>(0),
                               FE_NedelecPyramid<3>(0),
                               FE_NedelecHypercube<3>(0)};
  const hp::MappingCollection<3> mapping{MappingFE<3>(FE_SimplexP<3>(1)),
                                         MappingFE<3>(FE_WedgeP<3>(1)),
                                         MappingFE<3>(FE_PyramidP<3>(1)),
                                         MappingFE<3>(FE_Q<3>(1))};
  DoFHandler<3>             dof_handler(tria);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      const ReferenceCell r = cell->reference_cell();
      cell->set_active_fe_index(r == ReferenceCells::Tetrahedron ? 0 :
                                r == ReferenceCells::Wedge       ? 1 :
                                r == ReferenceCells::Pyramid     ? 2 :
                                                                   3);
    }
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  const LinearField g;
  VectorTools::interpolate_boundary_values_curl_conforming(mapping,
                                                           dof_handler,
                                                           {{0, &g}},
                                                           constraints);
  constraints.close();

  // all lines with DoFs on boundary faces of all levels
  const QGauss<1>                   quadrature(2);
  std::set<types::global_dof_index> checked;
  unsigned int                      n_refined = 0, n_wrong = 0;
  for (const auto &cell : dof_handler.cell_iterators())
    for (const unsigned int f : cell->face_indices())
      if (cell->at_boundary(f))
        for (unsigned int l = 0; l < cell->face(f)->n_lines(); ++l)
          {
            const auto line = cell->face(f)->line(l);
            if (line->n_active_fe_indices() == 0)
              continue;
            const types::global_dof_index dof =
              line->dof_index(0, line->nth_active_fe_index(0));
            if (checked.insert(dof).second == false)
              continue;
            if (line->has_children())
              ++n_refined;
            const Point<3> x0 = line->vertex(0), x1 = line->vertex(1);
            double         integral = 0;
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              integral += LinearField::value(
                            x0 + quadrature.point(q)[0] * (x1 - x0)) *
                          (x1 - x0) * quadrature.weight(q);
            if (constraints.is_constrained(dof) == false ||
                constraints.get_constraint_entries(dof)->empty() == false ||
                std::abs(constraints.get_inhomogeneity(dof) - integral) >
                  1e-12)
              ++n_wrong;
          }
  deallog << name << ": " << checked.size() << " boundary lines, "
          << n_refined << " of them refined, "
          << (n_wrong == 0 ? "all constrained to the line integral" :
                             std::to_string(n_wrong) + " wrong")
          << std::endl;
}



int
main()
{
  initlog();
  test("tetrahedra", [](Triangulation<3> &t) {
    GridGenerator::subdivided_hyper_cube_with_simplices(t, 2, 0., 2.);
  });
  test("pyramids", [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 0); });
  test("hexahedra and pyramids",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 1); });
  test("pyramids and tetrahedra",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 2); });
  test("wedges and tetrahedra",
       [](Triangulation<3> &t) { make_wedge_tetrahedron_mesh(t, 2); });
  test("hexahedra", [](Triangulation<3> &t) {
    GridGenerator::subdivided_hyper_cube(t, 2, 0., 2.);
  });
}
