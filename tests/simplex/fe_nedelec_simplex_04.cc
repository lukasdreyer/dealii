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
// distorted simplex meshes:
// 1. Exactness: u = a + b x x (in 2d a + b (-y, x)) lies in the lowest-order
//    Nedelec space. Its DoFs (line integrals along the line objects) satisfy
//    the constraints, i.e. distribute() reproduces the constrained DoFs.
// 2. Tangential continuity: for a random DoF vector, after distribute() the
//    tangential component of the coarse and the fine side agree on every
//    hanging face (compared at the face quadrature points of the fine
//    cells).

#include <deal.II/base/quadrature_lib.h>

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
#include <deal.II/lac/vector.h>

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
Tensor<1, dim>
fe_value(const FEValues<dim>                         &fe_values,
         const std::vector<types::global_dof_index> &dof_indices,
         const Vector<double>                       &vector,
         const unsigned int                          q)
{
  Tensor<1, dim> u;
  for (unsigned int i = 0; i < dof_indices.size(); ++i)
    for (unsigned int d = 0; d < dim; ++d)
      u[d] += vector(dof_indices[i]) * fe_values.shape_value_component(i, q, d);
  return u;
}



template <int dim>
void
test()
{
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube_with_simplices(tria, 2);
  GridTools::distort_random(0.2, tria, false, 42);
  for (unsigned int cycle = 0; cycle < 3; ++cycle)
    {
      const Point<dim> center = Point<dim>() + 0.3 * (cycle + 1) * Point<dim>::unit_vector(0);
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->center().distance(center) < 0.35)
          cell->set_refine_flag();
      tria.prepare_coarsening_and_refinement();
      // deal.II only implements the (6,8) cut of tetrahedra, which it would
      // not choose on distorted cells
      if (dim == 3)
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->refine_flag_set())
            cell->set_refine_choice(
              static_cast<std::uint8_t>(IsotropicRefinementChoice::cut_tet_68));
      tria.execute_coarsening_and_refinement();
    }

  const FE_NedelecSimplex<dim> fe(0);
  const MappingFE<dim>         mapping(FE_SimplexP<dim>(1));
  DoFHandler<dim>              dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  // 1. exactness
  Vector<double> exact(dof_handler.n_dofs());
  for (const auto &cell : dof_handler.active_cell_iterators())
    for (unsigned int l = 0; l < cell->n_lines(); ++l)
      {
        const auto           line = cell->line(l);
        const Tensor<1, dim> pq   = line->vertex(1) - line->vertex(0);
        exact(line->dof_index(0)) = linear_field(line->center()) * pq;
      }
  Vector<double> distributed = exact;
  constraints.distribute(distributed);
  distributed -= exact;

  // 2. tangential continuity
  Vector<double> random(dof_handler.n_dofs());
  for (auto &value : random)
    value = random_value<double>(-1., 1.);
  constraints.distribute(random);

  const QGaussSimplex<dim - 1> q_face(2);
  FEFaceValues<dim>            fine_values(mapping,
                                fe,
                                q_face,
                                update_values | update_quadrature_points |
                                  update_normal_vectors);
  std::vector<types::global_dof_index> fine_dofs(fe.n_dofs_per_cell()),
    coarse_dofs(fe.n_dofs_per_cell());
  unsigned int n_hanging_faces = 0;
  double       max_jump        = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const unsigned int f : cell->face_indices())
      if (!cell->at_boundary(f) && cell->neighbor_is_coarser(f))
        {
          ++n_hanging_faces;
          const auto coarse = cell->neighbor(f);
          fine_values.reinit(cell, f);
          cell->get_dof_indices(fine_dofs);
          coarse->get_dof_indices(coarse_dofs);

          std::vector<Point<dim>> unit_points(q_face.size());
          for (unsigned int q = 0; q < q_face.size(); ++q)
            unit_points[q] = mapping.transform_real_to_unit_cell(
              coarse, fine_values.quadrature_point(q));
          FEValues<dim> coarse_values(mapping,
                                      fe,
                                      Quadrature<dim>(unit_points),
                                      update_values);
          coarse_values.reinit(coarse);

          for (unsigned int q = 0; q < q_face.size(); ++q)
            {
              Tensor<1, dim> u_fine;
              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                for (unsigned int d = 0; d < dim; ++d)
                  u_fine[d] += random(fine_dofs[i]) *
                               fine_values.shape_value_component(i, q, d);
              const Tensor<1, dim> jump =
                u_fine - fe_value(coarse_values, coarse_dofs, random, q);
              const Tensor<1, dim> &n = fine_values.normal_vector(q);
              max_jump = std::max(max_jump, (jump - (jump * n) * n).norm());
            }
        }

  deallog << "dim=" << dim << " cells=" << tria.n_active_cells()
          << " hanging faces=" << n_hanging_faces
          << " constraints=" << constraints.n_constraints() << std::endl;
  deallog << "  exactness " << (distributed.linfty_norm() < 1e-12 ? "OK" : "FAILED")
          << ", tangential continuity " << (max_jump < 1e-12 ? "OK" : "FAILED")
          << std::endl;
  if (distributed.linfty_norm() >= 1e-12 || max_jump >= 1e-12)
    deallog << "  errors: " << distributed.linfty_norm() << ' ' << max_jump
            << std::endl;
}



int
main()
{
  initlog();
  test<2>();
  test<3>();
}
