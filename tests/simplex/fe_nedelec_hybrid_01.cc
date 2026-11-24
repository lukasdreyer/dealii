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
// FE_NedelecWedge, FE_NedelecPyramid, FE_NedelecHypercube in one
// hp::FECollection) with hanging
// nodes, on a mesh of pyramids (whose refinement creates tetrahedra), of
// pyramids and tetrahedra, and of wedges and tetrahedra:
// 1. exactness: the DoFs of u = a + b x x (affine meshes) satisfy the
//    constraints,
// 2. tangential continuity of a random constrained vector across all
//    interior faces: hanging faces and faces between cells of different
//    types,
// 3. convergence for curl curl u + u = f with tangential Dirichlet values
//    (u = (sin y, sin z, sin x), curl curl u = u), where one cell is refined
//    first and then all cells in every cycle: first order in L2 for u and
//    curl u.

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
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/tria.h>

#include <functional>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include "../tests.h"

#include "./pyramid_meshes.h"
#include "./wedge_meshes.h"


// the elements, mappings and quadratures, with the FE index of a cell given
// by its reference cell
struct Collections
{
  hp::FECollection<3>      fe{FE_NedelecSimplex<3>(0),
                         FE_NedelecWedge<3>(0),
                         FE_NedelecPyramid<3>(0),
                         FE_NedelecHypercube<3>(0)};
  hp::MappingCollection<3> mapping{MappingFE<3>(FE_SimplexP<3>(1)),
                                   MappingFE<3>(FE_WedgeP<3>(1)),
                                   MappingFE<3>(FE_PyramidP<3>(1)),
                                   MappingFE<3>(FE_Q<3>(1))};
  hp::QCollection<3>       quadrature{QGaussSimplex<3>(3),
                                QGaussWedge<3>(3),
                                QGaussPyramid<3>(4),
                                QGauss<3>(3)};

  static unsigned int
  fe_index(const ReferenceCell &reference_cell)
  {
    if (reference_cell == ReferenceCells::Tetrahedron)
      return 0;
    if (reference_cell == ReferenceCells::Wedge)
      return 1;
    if (reference_cell == ReferenceCells::Pyramid)
      return 2;
    Assert(reference_cell == ReferenceCells::Hexahedron, ExcNotImplemented());
    return 3;
  }

  void
  distribute_dofs(DoFHandler<3> &dof_handler) const
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      cell->set_active_fe_index(fe_index(cell->reference_cell()));
    dof_handler.distribute_dofs(fe);
  }
};



Tensor<1, 3>
linear_field(const Point<3> &x)
{
  const Tensor<1, 3> a({0.3, -0.2, 0.5});
  const Tensor<1, 3> b({0.7, -0.4, 0.9});
  return a + cross_product_3d(b, x);
}

Tensor<1, 3>
exact_u(const Point<3> &p)
{
  return Tensor<1, 3>({std::sin(p[1]), std::sin(p[2]), std::sin(p[0])});
}

Tensor<1, 3>
exact_curl(const Point<3> &p)
{
  return Tensor<1, 3>({-std::cos(p[2]), -std::cos(p[0]), -std::cos(p[1])});
}

class ExactSolution : public Function<3>
{
public:
  ExactSolution()
    : Function<3>(3)
  {}

  void
  vector_value(const Point<3> &p, Vector<double> &values) const override
  {
    const Tensor<1, 3> u = exact_u(p);
    for (unsigned int d = 0; d < 3; ++d)
      values(d) = u[d];
  }
};



void
refine(Triangulation<3> &tria, const Point<3> &center, const double radius)
{
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->center().distance(center) < radius)
      cell->set_refine_flag();
  tria.prepare_coarsening_and_refinement();
  // tetrahedra can only be refined along the line (6,8), also the ones
  // flagged by the mesh smoothing
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->refine_flag_set() &&
        cell->reference_cell() == ReferenceCells::Tetrahedron)
      cell->set_refine_choice(
        static_cast<std::uint8_t>(IsotropicRefinementChoice::cut_tet_68));
  tria.execute_coarsening_and_refinement();
}



// the values of the FE function with the DoF values u on cell at the
// physical points x
std::vector<Tensor<1, 3>>
evaluate(const Collections                        &c,
         const DoFHandler<3>::active_cell_iterator &cell,
         const Vector<double>                     &u,
         const std::vector<Point<3>>              &x)
{
  const unsigned int    i = cell->active_fe_index();
  std::vector<Point<3>> unit_points;
  for (const Point<3> &p : x)
    unit_points.push_back(c.mapping[i].transform_real_to_unit_cell(cell, p));
  FEValues<3> fe_values(c.mapping[i],
                        c.fe[i],
                        Quadrature<3>(unit_points),
                        update_values);
  fe_values.reinit(cell);
  std::vector<types::global_dof_index> dofs(c.fe[i].n_dofs_per_cell());
  cell->get_dof_indices(dofs);
  std::vector<Tensor<1, 3>> values(x.size());
  for (unsigned int q = 0; q < x.size(); ++q)
    for (unsigned int k = 0; k < dofs.size(); ++k)
      for (unsigned int d = 0; d < 3; ++d)
        values[q][d] += u(dofs[k]) * fe_values.shape_value_component(k, q, d);
  return values;
}



void
check_constraints(Triangulation<3> &tria, const std::string &name)
{
  const Collections c;
  DoFHandler<3>     dof_handler(tria);
  c.distribute_dofs(dof_handler);
  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  // 1. exactness
  Vector<double> exact(dof_handler.n_dofs());
  for (const auto &cell : dof_handler.active_cell_iterators())
    for (unsigned int l = 0; l < cell->n_lines(); ++l)
      {
        const auto line = cell->line(l);
        exact(line->dof_index(0, cell->active_fe_index())) =
          linear_field(line->center()) * (line->vertex(1) - line->vertex(0));
      }
  Vector<double> distributed = exact;
  constraints.distribute(distributed);
  distributed -= exact;

  // 2. tangential continuity across all interior faces, compared at points
  // of the face of the finer (or same level) cell
  Vector<double> random(dof_handler.n_dofs());
  for (auto &value : random)
    value = random_value<double>(-1., 1.);
  constraints.distribute(random);

  unsigned int n_hanging_triangles = 0, n_hanging_quads = 0,
               n_mixed_interfaces = 0;
  double max_jump = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const unsigned int f : cell->face_indices())
      {
        if (cell->at_boundary(f))
          continue;
        const bool hanging = cell->neighbor_is_coarser(f);
        if (!hanging && (cell->neighbor(f)->has_children() ||
                         cell->neighbor(f)->index() < cell->index()))
          continue; // visit every conforming face once, from one side
        const auto neighbor = cell->neighbor(f);
        const auto face     = cell->face(f);
        if (hanging)
          (face->reference_cell() == ReferenceCells::Triangle ?
             n_hanging_triangles :
             n_hanging_quads)++;
        if (neighbor->reference_cell() != cell->reference_cell())
          ++n_mixed_interfaces;

        // points in the (planar) face and its normal
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
              x.push_back(Point<3>(
                (1 - s) * (1 - t) * face->vertex(0) +
                s * (1 - t) * face->vertex(1) + (1 - s) * t * face->vertex(2) +
                s * t * face->vertex(3)));
        Tensor<1, 3> n = cross_product_3d(face->vertex(1) - face->vertex(0),
                                          face->vertex(2) - face->vertex(0));
        n /= n.norm();

        const auto u1 = evaluate(c, cell, random, x);
        const auto u2 = evaluate(c, neighbor, random, x);
        for (unsigned int q = 0; q < x.size(); ++q)
          {
            const Tensor<1, 3> jump = u1[q] - u2[q];
            max_jump = std::max(max_jump, (jump - (jump * n) * n).norm());
          }
      }

  unsigned int n_types[4] = {0, 0, 0, 0};
  for (const auto &cell : tria.active_cell_iterators())
    ++n_types[Collections::fe_index(cell->reference_cell())];
  deallog << name << ": tetrahedra=" << n_types[0] << " wedges=" << n_types[1]
          << " pyramids=" << n_types[2] << " hexahedra=" << n_types[3]
          << " hanging triangles=" << n_hanging_triangles
          << " hanging quadrilaterals=" << n_hanging_quads
          << " faces between different cells=" << n_mixed_interfaces
          << " constraints=" << constraints.n_constraints() << std::endl;
  deallog << "  exactness "
          << (distributed.linfty_norm() < 1e-12 ? "OK" : "FAILED")
          << ", tangential continuity " << (max_jump < 1e-12 ? "OK" : "FAILED")
          << std::endl;
  if (distributed.linfty_norm() >= 1e-12 || max_jump >= 1e-12)
    deallog << "  errors: " << distributed.linfty_norm() << ' ' << max_jump
            << std::endl;
}



void
check_convergence(Triangulation<3>  &tria,
                  const Point<3>    &center,
                  const std::string &name,
                  const unsigned int n_cycles)
{
  const Collections                c;
  const FEValuesExtractors::Vector E(0);
  const ExactSolution              exact_solution;

  double previous_error_u = 0, previous_error_curl = 0;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      if (cycle == 0)
        {
          // refine one cell, which creates hanging faces on every mesh
          tria.begin_active()->set_refine_flag();
          refine(tria, center, 0.);
        }
      else
        refine(tria, center, 1e10);

      DoFHandler<3> dof_handler(tria);
      c.distribute_dofs(dof_handler);
      AffineConstraints<double> constraints;
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      VectorTools::interpolate_boundary_values_curl_conforming(
        c.mapping, dof_handler, {{0, &exact_solution}}, constraints);
      constraints.close();

      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityPattern sparsity;
      sparsity.copy_from(dsp);
      SparseMatrix<double> matrix(sparsity);
      Vector<double>       rhs(dof_handler.n_dofs()), solution(rhs.size());

      hp::FEValues<3> hp_fe_values(c.mapping,
                                   c.fe,
                                   c.quadrature,
                                   update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values);
      FullMatrix<double>                   cell_matrix;
      Vector<double>                       cell_rhs;
      std::vector<types::global_dof_index> dof_indices;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          hp_fe_values.reinit(cell);
          const FEValues<3> &fe_values = hp_fe_values.get_present_fe_values();
          const unsigned int n_dofs    = cell->get_fe().n_dofs_per_cell();
          cell_matrix.reinit(n_dofs, n_dofs);
          cell_rhs.reinit(n_dofs);
          dof_indices.resize(n_dofs);
          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
              const Tensor<1, 3> f = 2. * exact_u(fe_values.quadrature_point(q));
              for (unsigned int i = 0; i < n_dofs; ++i)
                {
                  for (unsigned int j = 0; j < n_dofs; ++j)
                    cell_matrix(i, j) +=
                      (fe_values[E].curl(i, q) * fe_values[E].curl(j, q) +
                       fe_values[E].value(i, q) * fe_values[E].value(j, q)) *
                      fe_values.JxW(q);
                  cell_rhs(i) += f * fe_values[E].value(i, q) * fe_values.JxW(q);
                }
            }
          cell->get_dof_indices(dof_indices);
          constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, dof_indices, matrix, rhs);
        }

      SparseDirectUMFPACK direct;
      direct.initialize(matrix);
      direct.vmult(solution, rhs);
      constraints.distribute(solution);

      double error_u = 0, error_curl = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          hp_fe_values.reinit(cell);
          const FEValues<3> &fe_values = hp_fe_values.get_present_fe_values();
          dof_indices.resize(cell->get_fe().n_dofs_per_cell());
          cell->get_dof_indices(dof_indices);
          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
              Tensor<1, 3> u, curl;
              for (unsigned int i = 0; i < dof_indices.size(); ++i)
                {
                  u += solution(dof_indices[i]) * fe_values[E].value(i, q);
                  curl += solution(dof_indices[i]) * fe_values[E].curl(i, q);
                }
              const Point<3> &x = fe_values.quadrature_point(q);
              error_u += (u - exact_u(x)).norm_square() * fe_values.JxW(q);
              error_curl +=
                (curl - exact_curl(x)).norm_square() * fe_values.JxW(q);
            }
        }
      error_u    = std::sqrt(error_u);
      error_curl = std::sqrt(error_curl);

      deallog << name << " convergence: cells=" << tria.n_active_cells()
              << " dofs=" << dof_handler.n_dofs()
              << " constraints=" << constraints.n_constraints();
      if (cycle > 0)
        deallog << std::fixed << std::setprecision(2)
                << " rate_u=" << std::log2(previous_error_u / error_u)
                << " rate_curl=" << std::log2(previous_error_curl / error_curl)
                << std::defaultfloat;
      deallog << std::endl;
      previous_error_u    = error_u;
      previous_error_curl = error_curl;
    }
}



int
main()
{
  initlog();

  // the meshes with n x n (x n or x 2) unit cubes
  const std::vector<
    std::pair<std::string,
              std::function<void(Triangulation<3> &, const unsigned int)>>>
    meshes = {{"pyramids",
               [](Triangulation<3> &t, const unsigned int n) {
                 make_pyramid_mesh(t, n, 0);
               }},
              {"pyramids and tetrahedra",
               [](Triangulation<3> &t, const unsigned int n) {
                 make_pyramid_mesh(t, n, 2);
               }},
              {"hexahedra and pyramids",
               [](Triangulation<3> &t, const unsigned int n) {
                 make_pyramid_mesh(t, n, 1);
               }},
              {"wedges and tetrahedra",
               [](Triangulation<3> &t, const unsigned int n) {
                 make_wedge_tetrahedron_mesh(t, n);
               }}};

  for (const auto &[name, make] : meshes)
    {
      Triangulation<3> tria;
      make(tria, 2);
      refine(tria, Point<3>(0.5, 0.5, 0.8), 0.7);
      refine(tria, Point<3>(0.8, 0.6, 1.), 0.5);
      check_constraints(tria, name);
    }
  for (const auto &[name, make] : meshes)
    {
      // with one unit cube, the mesh of hexahedra and pyramids would only
      // consist of a hexahedron
      Triangulation<3> tria;
      make(tria, name == "hexahedra and pyramids" ? 2 : 1);
      check_convergence(tria, Point<3>(0.2, 0.2, 0.3), name, 3);
    }
}
