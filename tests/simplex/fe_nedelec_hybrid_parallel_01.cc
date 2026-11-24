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


// The lowest-order Nedelec elements on hybrid meshes (FE_NedelecSimplex,
// FE_NedelecWedge, FE_NedelecPyramid in one hp::FECollection) with hanging
// nodes in a parallel::distributed::Triangulation, on a mesh of pyramids, of
// pyramids and tetrahedra, and of wedges and tetrahedra:
// 1. the constraints of every locally active line agree on all processes,
// 2. exactness: the DoFs of u = a + b x x satisfy the constraints,
// 3. tangential continuity of a random constrained vector across all faces
//    of locally owned cells (hanging faces, faces between cells of different
//    types, faces to ghost cells),
// 4. the mesh and the number of constraints equal those of a serial
//    Triangulation refined in the same way.

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nedelec_pyramid.h>
#include <deal.II/fe/fe_nedelec_simplex.h>
#include <deal.II/fe/fe_nedelec_wedge.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <functional>

#include "../tests.h"

#include "./pyramid_meshes.h"
#include "./wedge_meshes.h"


const hp::FECollection<3>      fe_collection{FE_NedelecSimplex<3>(0),
                                        FE_NedelecWedge<3>(0),
                                        FE_NedelecPyramid<3>(0)};
const hp::MappingCollection<3> mapping_collection{
  MappingFE<3>(FE_SimplexP<3>(1)),
  MappingFE<3>(FE_WedgeP<3>(1)),
  MappingFE<3>(FE_PyramidP<3>(1))};

unsigned int
fe_index(const ReferenceCell &reference_cell)
{
  if (reference_cell == ReferenceCells::Tetrahedron)
    return 0;
  if (reference_cell == ReferenceCells::Wedge)
    return 1;
  return 2;
}

Tensor<1, 3>
linear_field(const Point<3> &x)
{
  const Tensor<1, 3> a({0.3, -0.2, 0.5});
  const Tensor<1, 3> b({0.7, -0.4, 0.9});
  return a + cross_product_3d(b, x);
}



void
refine(Triangulation<3> &tria, const Point<3> &center, const double radius)
{
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center().distance(center) < radius)
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
distribute_dofs(DoFHandler<3> &dof_handler)
{
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(fe_index(cell->reference_cell()));
  dof_handler.distribute_dofs(fe_collection);
}



template <typename VectorType>
std::vector<Tensor<1, 3>>
evaluate(const DoFHandler<3>::active_cell_iterator &cell,
         const VectorType                          &u,
         const std::vector<Point<3>>               &x)
{
  const unsigned int    i = cell->active_fe_index();
  std::vector<Point<3>> unit_points;
  for (const Point<3> &p : x)
    unit_points.push_back(
      mapping_collection[i].transform_real_to_unit_cell(cell, p));
  FEValues<3> fe_values(mapping_collection[i],
                        fe_collection[i],
                        Quadrature<3>(unit_points),
                        update_values);
  fe_values.reinit(cell);
  std::vector<types::global_dof_index> dofs(
    fe_collection[i].n_dofs_per_cell());
  cell->get_dof_indices(dofs);
  std::vector<Tensor<1, 3>> values(x.size());
  for (unsigned int q = 0; q < x.size(); ++q)
    for (unsigned int k = 0; k < dofs.size(); ++k)
      for (unsigned int d = 0; d < 3; ++d)
        values[q][d] += u(dofs[k]) * fe_values.shape_value_component(k, q, d);
  return values;
}



void
test(const std::string                              &name,
     const std::function<void(Triangulation<3> &)> &make_mesh)
{
  const MPI_Comm comm = MPI_COMM_WORLD;

  Triangulation<3> coarse;
  make_mesh(coarse);
  parallel::distributed::Triangulation<3> tria(comm);
  tria.copy_triangulation(coarse);
  // the parallel triangulation (t8code) limits the level difference at
  // edges; deal.II has no such smoothing, so the meshes are compared below
  Triangulation<3> serial;
  make_mesh(serial);
  for (const auto &[center, radius] :
       std::vector<std::pair<Point<3>, double>>{{Point<3>(0.5, 0.5, 0.8), 0.7},
                                                {Point<3>(0.8, 0.6, 1.), 0.5}})
    {
      refine(tria, center, radius);
      refine(serial, center, radius);
    }

  DoFHandler<3> dof_handler(tria);
  distribute_dofs(dof_handler);
  const IndexSet &owned    = dof_handler.locally_owned_dofs();
  const IndexSet  relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
  const IndexSet  active   = DoFTools::extract_locally_active_dofs(dof_handler);
  AffineConstraints<double> constraints(owned, relevant);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  DoFHandler<3> serial_dof_handler(serial);
  distribute_dofs(serial_dof_handler);
  AffineConstraints<double> serial_constraints;
  DoFTools::make_hanging_node_constraints(serial_dof_handler,
                                          serial_constraints);
  serial_constraints.close();

  // 1. consistency
  const bool consistent = constraints.is_consistent_in_parallel(
    Utilities::MPI::all_gather(comm, owned), active, comm);

  // 2. exactness
  using VectorType = LinearAlgebra::distributed::Vector<double>;
  VectorType exact(owned, relevant, comm);
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned() || cell->is_ghost())
      for (unsigned int l = 0; l < cell->n_lines(); ++l)
        {
          const auto                    line = cell->line(l);
          const types::global_dof_index dof =
            line->dof_index(0, cell->active_fe_index());
          if (owned.is_element(dof))
            exact(dof) =
              linear_field(line->center()) * (line->vertex(1) - line->vertex(0));
        }
  VectorType distributed(owned, relevant, comm);
  distributed = exact;
  constraints.distribute(distributed);
  distributed -= exact;
  const double exactness_error =
    Utilities::MPI::max(distributed.linfty_norm(), comm);

  // 3. tangential continuity
  VectorType random(owned, relevant, comm);
  for (const types::global_dof_index dof : owned)
    random(dof) = random_value<double>(-1., 1.);
  constraints.distribute(random);
  random.update_ghost_values();

  double max_jump = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const unsigned int f : cell->face_indices())
        {
          if (cell->at_boundary(f) || cell->neighbor(f)->has_children())
            continue; // finer neighbors: checked from their side
          const auto neighbor = cell->neighbor(f);
          const auto face     = cell->face(f);
          std::vector<Point<3>> x;
          if (face->reference_cell() == ReferenceCells::Triangle)
            {
              for (const auto &w : std::vector<std::array<double, 3>>{
                     {{1. / 3, 1. / 3, 1. / 3}},
                     {{0.6, 0.2, 0.2}},
                     {{0.2, 0.6, 0.2}},
                     {{0.2, 0.2, 0.6}}})
                x.push_back(Point<3>(w[0] * face->vertex(0) +
                                     w[1] * face->vertex(1) +
                                     w[2] * face->vertex(2)));
            }
          else
            for (const double s : {0.2, 0.7})
              for (const double t : {0.3, 0.8})
                x.push_back(Point<3>((1 - s) * (1 - t) * face->vertex(0) +
                                     s * (1 - t) * face->vertex(1) +
                                     (1 - s) * t * face->vertex(2) +
                                     s * t * face->vertex(3)));
          Tensor<1, 3> n = cross_product_3d(face->vertex(1) - face->vertex(0),
                                            face->vertex(2) - face->vertex(0));
          n /= n.norm();
          const auto u1 = evaluate(cell, random, x);
          const auto u2 = evaluate(neighbor, random, x);
          for (unsigned int q = 0; q < x.size(); ++q)
            {
              const Tensor<1, 3> jump = u1[q] - u2[q];
              max_jump = std::max(max_jump, (jump - (jump * n) * n).norm());
            }
        }
  max_jump = Utilities::MPI::max(max_jump, comm);

  unsigned int n_owned_constraints = 0;
  for (const auto &line : constraints.get_lines())
    if (owned.is_element(line.index))
      ++n_owned_constraints;
  const unsigned int n_constraints =
    Utilities::MPI::sum(n_owned_constraints, comm);

  deallog << name << ": cells=" << tria.n_global_active_cells()
          << " dofs=" << dof_handler.n_dofs()
          << " constraints=" << n_constraints << std::endl;
  deallog << "  consistent " << (consistent ? "OK" : "FAILED")
          << ", exactness " << (exactness_error < 1e-12 ? "OK" : "FAILED")
          << ", tangential continuity " << (max_jump < 1e-12 ? "OK" : "FAILED")
          << ", same mesh and constraints as serial "
          << (tria.n_global_active_cells() == serial.n_active_cells() &&
                  n_constraints == serial_constraints.n_constraints() ?
                "OK" :
                "FAILED")
          << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  MPILogInitAll                    log;

  test("pyramids", [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 0); });
  test("pyramids and tetrahedra",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 2); });
  test("wedges and tetrahedra",
       [](Triangulation<3> &t) { make_wedge_tetrahedron_mesh(t, 2); });
}
