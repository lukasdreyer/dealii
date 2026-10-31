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


// Adaptive refinement and coarsening of meshes of wedges, of hexahedra and
// wedges, and of wedges and tetrahedra with faces in all orientations: a
// quadratic function is in the space of FE_Q(2)/FE_SimplexP(2)/FE_WedgeP(2),
// so it must be reproduced exactly by the hanging node constraints and by
// SolutionTransfer through refinement and coarsening.

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_wedge_p.h>
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

#include "wedge_meshes.h"


class Quadratic : public Function<3>
{
public:
  double
  value(const Point<3> &p, const unsigned int) const override
  {
    return 1 + p[0] + 2 * p[1] + 3 * p[2] + p[0] * p[1] + p[1] * p[2] +
           p[2] * p[0];
  }
};



// mesh 0: wedges, 1: hexahedra and wedges, 2: wedges below tetrahedra
void
test(const std::string &name, const unsigned int mesh)
{
  deallog.push(name);

  Triangulation<3> tria(Triangulation<3>::limit_level_difference_at_vertices);
  if (mesh < 2)
    make_wedge_mesh(tria, 2, mesh == 1);
  else
    make_wedge_tetrahedron_mesh(tria, 2);

  const hp::FECollection<3>      fe(FE_Q<3>(2),
                                FE_WedgeP<3>(2),
                                FE_SimplexP<3>(2));
  const hp::MappingCollection<3> mapping(MappingQ<3>(1),
                                         MappingFE<3>(FE_WedgeP<3>(1)),
                                         MappingFE<3>(FE_SimplexP<3>(1)));
  DoFHandler<3>                  dof_handler(tria);
  const auto                     distribute_dofs = [&]() {
    for (const auto &cell : dof_handler.active_cell_iterators())
      cell->set_active_fe_index(
        cell->reference_cell() == ReferenceCells::Wedge       ? 1 :
        cell->reference_cell() == ReferenceCells::Tetrahedron ? 2 :
                                                                0);
    dof_handler.distribute_dofs(fe);
  };
  distribute_dofs();

  Vector<double> solution(dof_handler.n_dofs());
  VectorTools::interpolate(mapping, dof_handler, Quadratic(), solution);

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
    distribute_dofs();

    Vector<double> transferred(dof_handler.n_dofs());
    transfer.interpolate(transferred);

    AffineConstraints<double> constraints;
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();
    constraints.distribute(transferred);

    Vector<double> interpolated(dof_handler.n_dofs());
    VectorTools::interpolate(mapping, dof_handler, Quadratic(), interpolated);
    constraints.distribute(interpolated);
    interpolated -= transferred;

    double volume = 0;
    for (const auto &cell : tria.active_cell_iterators())
      volume += cell->measure();

    deallog << "cells " << tria.n_active_cells() << ", levels "
            << tria.n_levels() << ", volume " << volume << ", dofs "
            << dof_handler.n_dofs() << ", constraints "
            << constraints.n_constraints() << ", "
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

  test("wedges", 0);
  test("hexahedra and wedges", 1);
  test("wedges below tetrahedra", 2);
}
