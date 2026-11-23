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


// Hanging node constraints of FE_NedelecSimplex(0) on adaptively refined,
// distorted simplex meshes in a parallel::distributed::Triangulation:
// 1. the constraints of every locally active line agree on all processes
//    (AffineConstraints::is_consistent_in_parallel()),
// 2. exactness: the DoFs of u = a + b x x satisfy the constraints,
// 3. tangential continuity of a random constrained vector across all
//    hanging faces at locally owned fine cells, also where the coarse
//    neighbor is a ghost cell,
// 4. the mesh and the number of constraints equal those of a serial
//    Triangulation refined in the same way.
// The meshes are only refined, so they do not depend on the number of
// processes.

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nedelec_simplex.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "../tests.h"


Tensor<1, 2>
linear_field(const Point<2> &x)
{
  return Tensor<1, 2>({0.3 - 0.7 * x[1], -0.2 + 0.7 * x[0]});
}

Tensor<1, 3>
linear_field(const Point<3> &x)
{
  const Tensor<1, 3> a({0.3, -0.2, 0.5});
  const Tensor<1, 3> b({0.7, -0.4, 0.9});
  return a + cross_product_3d(b, x);
}



template <int dim>
void
make_coarse_mesh(Triangulation<dim> &tria)
{
  GridGenerator::subdivided_hyper_cube_with_simplices(tria, 3);
  GridTools::distort_random(0.2, tria, false, 42);
}



// refine the cells near a point that moves with the cycle; the same flags on
// the serial and the parallel triangulation (only the locally owned cells
// of the latter)
template <int dim>
void
refine(Triangulation<dim> &tria, const unsigned int cycle)
{
  const Point<dim> center =
    Point<dim>() + (0.2 + 0.25 * cycle) * Point<dim>::unit_vector(0) +
    0.4 * Point<dim>::unit_vector(1);
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center().distance(center) < 0.3)
      cell->set_refine_flag();
  tria.prepare_coarsening_and_refinement();
  // deal.II only implements the (6,8) cut of tetrahedra
  if (dim == 3)
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->refine_flag_set())
        cell->set_refine_choice(
          static_cast<std::uint8_t>(IsotropicRefinementChoice::cut_tet_68));
  tria.execute_coarsening_and_refinement();
}



template <int dim>
void
test()
{
  const MPI_Comm comm = MPI_COMM_WORLD;

  Triangulation<dim> coarse;
  make_coarse_mesh(coarse);
  parallel::distributed::Triangulation<dim> tria(comm);
  tria.copy_triangulation(coarse);

  // The parallel triangulation (t8code) limits the level difference at
  // vertices in 2d and at edges in 3d. deal.II has no smoothing for edges
  // only; on this 3d mesh, no smoothing gives the same mesh, which the check
  // below verifies by the number of cells.
  Triangulation<dim> serial(
    dim == 2 ? Triangulation<dim>::limit_level_difference_at_vertices :
               Triangulation<dim>::none);
  make_coarse_mesh(serial);

  for (unsigned int cycle = 0; cycle < 3; ++cycle)
    {
      refine(tria, cycle);
      refine(serial, cycle);
    }

  const FE_NedelecSimplex<dim> fe(0);
  const MappingFE<dim>         mapping(FE_SimplexP<dim>(1));

  // serial reference for the number of constraints
  DoFHandler<dim> serial_dof_handler(serial);
  serial_dof_handler.distribute_dofs(fe);
  AffineConstraints<double> serial_constraints;
  DoFTools::make_hanging_node_constraints(serial_dof_handler,
                                          serial_constraints);
  serial_constraints.close();

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  const IndexSet &owned    = dof_handler.locally_owned_dofs();
  const IndexSet  relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
  const IndexSet  active   = DoFTools::extract_locally_active_dofs(dof_handler);

  AffineConstraints<double> constraints(owned, relevant);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  // 1. consistency across processes
  const bool consistent = constraints.is_consistent_in_parallel(
    Utilities::MPI::all_gather(comm, owned), active, comm);

  unsigned int n_owned_constraints = 0;
  for (const auto &line : constraints.get_lines())
    if (owned.is_element(line.index))
      ++n_owned_constraints;

  // 2. exactness
  using VectorType = LinearAlgebra::distributed::Vector<double>;
  VectorType exact(owned, relevant, comm);
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (!cell->is_artificial())
      for (unsigned int l = 0; l < cell->n_lines(); ++l)
        {
          const auto                    line = cell->line(l);
          const types::global_dof_index dof  = line->dof_index(0);
          if (owned.is_element(dof))
            exact(dof) = linear_field(line->center()) *
                         (line->vertex(1) - line->vertex(0));
        }
  VectorType distributed(owned, relevant, comm);
  distributed = exact;
  constraints.distribute(distributed);
  distributed -= exact;
  const double exactness_error = distributed.linfty_norm();

  // 3. tangential continuity
  VectorType random(owned, relevant, comm);
  for (const types::global_dof_index dof : owned)
    random(dof) = random_value<double>(-1., 1.);
  constraints.distribute(random);
  random.update_ghost_values();

  const QGaussSimplex<dim - 1> q_face(2);
  FEFaceValues<dim>            fine_values(mapping,
                                fe,
                                q_face,
                                update_values | update_quadrature_points |
                                  update_normal_vectors);
  std::vector<types::global_dof_index> fine_dofs(fe.n_dofs_per_cell()),
    coarse_dofs(fe.n_dofs_per_cell());
  unsigned int n_hanging_faces = 0, n_ghost_coarse = 0;
  double       max_jump        = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const unsigned int f : cell->face_indices())
        if (!cell->at_boundary(f) && cell->neighbor_is_coarser(f))
          {
            ++n_hanging_faces;
            const auto coarse_cell = cell->neighbor(f);
            if (coarse_cell->is_ghost())
              ++n_ghost_coarse;
            fine_values.reinit(cell, f);
            cell->get_dof_indices(fine_dofs);
            coarse_cell->get_dof_indices(coarse_dofs);

            std::vector<Point<dim>> unit_points(q_face.size());
            for (unsigned int q = 0; q < q_face.size(); ++q)
              unit_points[q] = mapping.transform_real_to_unit_cell(
                coarse_cell, fine_values.quadrature_point(q));
            FEValues<dim> coarse_values(mapping,
                                        fe,
                                        Quadrature<dim>(unit_points),
                                        update_values);
            coarse_values.reinit(coarse_cell);

            for (unsigned int q = 0; q < q_face.size(); ++q)
              {
                Tensor<1, dim> jump;
                for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                  for (unsigned int d = 0; d < dim; ++d)
                    jump[d] +=
                      random(fine_dofs[i]) *
                        fine_values.shape_value_component(i, q, d) -
                      random(coarse_dofs[i]) *
                        coarse_values.shape_value_component(i, q, d);
                const Tensor<1, dim> &n = fine_values.normal_vector(q);
                max_jump = std::max(max_jump, (jump - (jump * n) * n).norm());
              }
          }

  const bool exact_ok =
    Utilities::MPI::max(exactness_error, comm) < 1e-12;
  const bool continuous = Utilities::MPI::max(max_jump, comm) < 1e-12;
  const unsigned int n_constraints = Utilities::MPI::sum(n_owned_constraints, comm);

  deallog << "dim=" << dim << " cells=" << tria.n_global_active_cells()
          << " dofs=" << dof_handler.n_dofs()
          << " hanging faces=" << Utilities::MPI::sum(n_hanging_faces, comm)
          << " constraints=" << n_constraints << std::endl;
  deallog << "  consistent " << (consistent ? "OK" : "FAILED")
          << ", exactness " << (exact_ok ? "OK" : "FAILED")
          << ", tangential continuity " << (continuous ? "OK" : "FAILED")
          << ", same number of constraints as serial "
          << (n_constraints == serial_constraints.n_constraints() &&
                  tria.n_global_active_cells() == serial.n_active_cells() ?
                "OK" :
                "FAILED")
          << std::endl;
  // depends on the partition, so only to stdout
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "dim=" << dim << " hanging faces with a ghost coarse cell: "
              << Utilities::MPI::sum(n_ghost_coarse, comm) << std::endl;
  else
    Utilities::MPI::sum(n_ghost_coarse, comm);
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  MPILogInitAll                    log;

  test<2>();
  test<3>();
}
