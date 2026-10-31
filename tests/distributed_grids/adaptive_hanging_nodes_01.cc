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


// Check the hanging node constraints of continuous elements on an adaptively
// refined parallel::distributed::Triangulation with simplices and hypercubes
// (also mixed) on coarse meshes with several trees, while refining and
// coarsening: after distributing the constraints of a vector with arbitrary
// values, the finite element function must be continuous across all faces,
// in particular across hanging faces whose children are numbered and oriented
// as the refinement of the coarser cell's face. Continuity is checked at
// points of each face, evaluated on both cells.

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "adaptive_meshes.h"


// Values of the finite element function at the physical points on a cell.
template <int dim>
std::vector<double>
values_at(const typename DoFHandler<dim>::active_cell_iterator &cell,
          const std::vector<Point<dim>>                        &points,
          const LinearAlgebra::distributed::Vector<double>     &vector)
{
  const Mapping<dim> &mapping =
    cell->reference_cell().template get_default_linear_mapping<dim>();
  std::vector<Point<dim>> unit_points;
  for (const auto &p : points)
    unit_points.push_back(mapping.transform_real_to_unit_cell(cell, p));

  FEValues<dim>       fe_values(mapping,
                          cell->get_fe(),
                          Quadrature<dim>(unit_points),
                          update_values);
  std::vector<double> values(points.size());
  fe_values.reinit(cell);
  fe_values.get_function_values(vector, values);
  return values;
}



// Return the largest jump of the finite element function across a face, and
// print the numbers of dofs and constrained dofs if print_dofs is set.
template <int dim>
double
check_continuity(const parallel::distributed::Triangulation<dim> &tria,
                 const unsigned int                               degree,
                 const bool                                       print_dofs)
{
  const MPI_Comm comm = tria.get_mpi_communicator();

  DoFHandler<dim> dof_handler(tria);
  set_fe_indices(dof_handler);
  dof_handler.distribute_dofs(make_fe_collection<dim>(degree));
  const IndexSet &owned_dofs = dof_handler.locally_owned_dofs();
  const IndexSet  relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  AffineConstraints<double> constraints(owned_dofs, relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  // arbitrary values, which only depend on the global dof index
  LinearAlgebra::distributed::Vector<double> vector(owned_dofs,
                                                    relevant_dofs,
                                                    comm);
  for (const auto i : owned_dofs)
    vector[i] = std::sin(0.7 * i) + 0.1 * i;
  constraints.distribute(vector);
  vector.update_ghost_values();

  unsigned int n_constrained = 0;
  for (const auto i : owned_dofs)
    if (constraints.is_constrained(i))
      ++n_constrained;

  // compare the values on both sides of each face, from the finer side
  double max_jump = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (!cell->is_artificial())
      for (const unsigned int f : cell->face_indices())
        {
          if (cell->at_boundary(f))
            continue;
          const auto neighbor = cell->neighbor(f);
          if (neighbor->has_children() || neighbor->is_artificial() ||
              (!cell->is_locally_owned() && !neighbor->is_locally_owned()))
            continue;

          // the face center, the face vertices, and points between them
          const auto              face = cell->face(f);
          std::vector<Point<dim>> points{face->center()};
          for (const unsigned int v : face->vertex_indices())
            {
              points.push_back(face->vertex(v));
              points.push_back((2 * face->center() + face->vertex(v)) / 3);
              points.push_back(
                (face->vertex(v) +
                 face->vertex((v + 1) % face->n_vertices())) /
                2);
            }

          const std::vector<double> values_1 =
            values_at<dim>(cell, points, vector);
          const std::vector<double> values_2 =
            values_at<dim>(neighbor, points, vector);
          for (unsigned int q = 0; q < points.size(); ++q)
            max_jump =
              std::max(max_jump, std::abs(values_1[q] - values_2[q]));
        }
  max_jump = Utilities::MPI::max(max_jump, comm);
  n_constrained = Utilities::MPI::sum(n_constrained, comm);

  if (print_dofs)
    deallog << "cells " << tria.n_global_active_cells() << ", degree "
            << degree << ", dofs " << dof_handler.n_dofs()
            << ", constrained dofs " << n_constrained << ", ";
  return max_jump;
}



void
print_jump(const double max_jump)
{
  if (max_jump < 1e-10)
    deallog << "continuous" << std::endl;
  else
    deallog << "jump " << max_jump << std::endl;
}



template <int dim>
void
test()
{
  const unsigned int n_cycles  = 3;
  const unsigned int max_level = dim == 2 ? 4 : 3;

  for (unsigned int i = 0; i < coarse_mesh_names<dim>().size(); ++i)
    {
      deallog.push(coarse_mesh_names<dim>()[i]);

      parallel::distributed::Triangulation<dim> tria(
        MPI_COMM_WORLD, Triangulation<dim>::limit_level_difference_at_vertices);
      make_coarse_mesh(tria, i);
      const BoundingBox<dim> box = GridTools::compute_bounding_box(tria);
      const double           radius =
        0.25 * box.get_boundary_points().first.distance(
                 box.get_boundary_points().second);

      for (unsigned int k = 0; k < n_cycles; ++k)
        {
          refine_around(tria,
                        refinement_center(box, k, n_cycles),
                        radius / (1 + k),
                        max_level);
          for (const unsigned int degree : {1, 2})
            print_jump(check_continuity(tria, degree, true));
        }

      // the meshes while coarsening depend on the number of processes, so
      // only print the largest jump
      double max_jump = 0;
      while (coarsen_all(tria))
        for (const unsigned int degree : {1, 2})
          max_jump = std::max(max_jump, check_continuity(tria, degree, false));
      deallog << "coarsening: ";
      print_jump(max_jump);

      deallog.pop();
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  mpi_initlog();

  deallog.push("2d");
  test<2>();
  deallog.pop();

  deallog.push("3d");
  test<3>();
  deallog.pop();
}
