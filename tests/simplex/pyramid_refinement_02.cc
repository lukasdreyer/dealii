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


// Adaptive refinement and coarsening of meshes of pyramids, of pyramids and
// hexahedra, and of pyramids below tetrahedra with faces in all orientations,
// with FE_Q/FE_PyramidP/FE_SimplexP of degree 1 and 2. Pyramids have
// tetrahedra among their children (and the tetrahedra inside pyramids have
// pyramids): the DoFHandler has to give them the corresponding element, and
// SolutionTransfer has to interpolate between the elements. A function in
// the space of all elements must be reproduced exactly by the hanging node
// constraints and by SolutionTransfer through refinement and coarsening.

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <functional>
#include <string>

#include "../tests.h"

#include "pyramid_meshes.h"


class TestFunction : public Function<3>
{
public:
  TestFunction(const bool quadratic)
    : quadratic(quadratic)
  {}

  double
  value(const Point<3> &p, const unsigned int) const override
  {
    return 1 + p[0] + 2 * p[1] + 3 * p[2] +
           (quadratic ? p[0] * p[1] + p[1] * p[2] + p[2] * p[0] : 0.);
  }

private:
  const bool quadratic;
};



// mesh 0: pyramids, 1: pyramids and hexahedra, 2: pyramids below tetrahedra
void
test(const std::string &name,
     const unsigned int mesh,
     const unsigned int degree,
     const bool         quadratic)
{
  deallog.push(name + ", degree " + std::to_string(degree) +
               (quadratic ? ", quadratic" : ", linear"));

  Triangulation<3> tria(Triangulation<3>::limit_level_difference_at_vertices);
  make_pyramid_mesh(tria, 2, mesh);

  const hp::FECollection<3>      fe{FE_Q<3>(degree),
                                FE_PyramidP<3>(degree),
                                FE_SimplexP<3>(degree)};
  const hp::MappingCollection<3> mapping(MappingQ<3>(1),
                                         MappingFE<3>(FE_PyramidP<3>(1)),
                                         MappingFE<3>(FE_SimplexP<3>(1)));
  const TestFunction             function(quadratic);

  // the active FE indices are only set on the coarse mesh: the DoFHandler
  // gives the children of other types the corresponding elements
  DoFHandler<3> dof_handler(tria);
  for (const auto &cell : dof_handler.active_cell_iterators())
    cell->set_active_fe_index(
      cell->reference_cell() == ReferenceCells::Pyramid     ? 1 :
      cell->reference_cell() == ReferenceCells::Tetrahedron ? 2 :
                                                              0);
  dof_handler.distribute_dofs(fe);

  Vector<double> solution(dof_handler.n_dofs());
  VectorTools::interpolate(mapping, dof_handler, function, solution);

  // set the refinement flags with set_flags, refine and coarsen the mesh,
  // transfer the solution, and print the difference to the interpolated
  // function
  const auto transfer_through = [&](const std::function<void()> &set_flags) {
    set_flags();
    SolutionTransfer<3, Vector<double>> transfer(dof_handler);
    tria.prepare_coarsening_and_refinement();
    // tetrahedra can only be refined along the line (6,8), also the ones
    // flagged by the mesh smoothing
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->refine_flag_set() &&
          cell->reference_cell() == ReferenceCells::Tetrahedron)
        cell->set_refine_choice(
          static_cast<std::uint8_t>(IsotropicRefinementChoice::cut_tet_68));
    transfer.prepare_for_coarsening_and_refinement(solution);
    tria.execute_coarsening_and_refinement();
    dof_handler.distribute_dofs(fe);

    unsigned int wrong_elements = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      wrong_elements += (cell->get_fe().reference_cell() !=
                         cell->reference_cell());

    Vector<double> transferred(dof_handler.n_dofs());
    transfer.interpolate(transferred);

    AffineConstraints<double> constraints;
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();
    constraints.distribute(transferred);

    Vector<double> interpolated(dof_handler.n_dofs());
    VectorTools::interpolate(mapping, dof_handler, function, interpolated);
    constraints.distribute(interpolated);
    interpolated -= transferred;

    double volume = 0;
    for (const auto &cell : tria.active_cell_iterators())
      volume += cell->measure();

    deallog << "cells " << tria.n_active_cells() << ", levels "
            << tria.n_levels() << ", volume " << volume << ", dofs "
            << dof_handler.n_dofs() << ", constraints "
            << constraints.n_constraints() << ", wrong elements "
            << wrong_elements << ", "
            << (interpolated.linfty_norm() < 1e-10 ? "exact" : "not exact")
            << std::endl;

    solution = transferred;
  };

  for (unsigned int cycle = 0; cycle < 3; ++cycle)
    transfer_through([&]() {
      unsigned int c = 0;
      for (const auto &cell : tria.active_cell_iterators())
        if (c++ % 3 == 0)
          cell->set_refine_flag();
    });

  while (tria.n_levels() > 1)
    transfer_through([&]() {
      for (const auto &cell : tria.active_cell_iterators())
        cell->set_coarsen_flag();
    });

  deallog.pop();
}



int
main()
{
  initlog();

  const char *names[] = {"pyramids",
                         "pyramids and hexahedra",
                         "pyramids below tetrahedra"};
  for (unsigned int mesh = 0; mesh < 3; ++mesh)
    {
      test(names[mesh], mesh, 1, false);
      test(names[mesh], mesh, 2, false);
      test(names[mesh], mesh, 2, true);
    }
}
