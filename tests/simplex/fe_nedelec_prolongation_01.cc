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


// DoFTools::make_edge_element_prolongation() between two meshes for
// geometric multigrid: the coarse mesh has hanging nodes, and the fine mesh
// is a copy of it in which some cells are refined once (more hanging nodes).
// 1. P reproduces the DoFs of u = a + b x x on the unconstrained fine DoFs,
//    also when the constrained coarse DoFs are zero (they are expressed by
//    their constraints), and the rows of constrained fine DoFs are zero,
// 2. for a random coarse vector (with its constraints applied), the
//    unconstrained fine DoFs of P u_H are the line integrals of the coarse
//    function along the fine lines, and the fine function (with the fine
//    constraints applied) equals the coarse function at the quadrature
//    points of all fine cells except in pyramids: the spaces are nested on
//    tetrahedra, wedges and hexahedra, but the rational pyramid functions
//    are not in the space of the children (which include tetrahedra).
// Meshes: tetrahedra, pyramids, pyramids and tetrahedra, hexahedra and
// pyramids, wedges and tetrahedra, hexahedra.

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

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <functional>

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



// the DoFs of the linear field: line integrals along the line objects
// (exact with the midpoint rule, since the field is affine)
Vector<double>
interpolate_linear_field(const DoFHandler<3> &dof_handler)
{
  Vector<double> u(dof_handler.n_dofs());
  for (const auto &cell : dof_handler.active_cell_iterators())
    for (unsigned int l = 0; l < cell->n_lines(); ++l)
      {
        const auto line = cell->line(l);
        u(line->dof_index(0, cell->active_fe_index())) =
          linear_field(line->center()) * (line->vertex(1) - line->vertex(0));
      }
  return u;
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
check(const Triangulation<3> &coarse_tria,
      const Triangulation<3> &fine_tria,
      const std::string      &name)
{
  const Collections c;

  DoFHandler<3> coarse_dh(coarse_tria), fine_dh(fine_tria);
  c.distribute_dofs(coarse_dh);
  c.distribute_dofs(fine_dh);

  AffineConstraints<double> coarse_constraints, fine_constraints;
  DoFTools::make_hanging_node_constraints(coarse_dh, coarse_constraints);
  DoFTools::make_hanging_node_constraints(fine_dh, fine_constraints);
  coarse_constraints.close();
  fine_constraints.close();

  DynamicSparsityPattern dsp(fine_dh.n_dofs(), coarse_dh.n_dofs());
  DoFTools::make_edge_element_prolongation_sparsity_pattern(
    coarse_dh, coarse_constraints, fine_dh, fine_constraints, dsp);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);
  SparseMatrix<double> prolongation(sparsity);
  DoFTools::make_edge_element_prolongation(
    coarse_dh, coarse_constraints, fine_dh, fine_constraints, prolongation);

  deallog << name << ": coarse " << coarse_tria.n_active_cells()
          << " cells, " << coarse_dh.n_dofs() << " DoFs ("
          << coarse_constraints.n_constraints() << " constrained), fine "
          << fine_tria.n_active_cells() << " cells, " << fine_dh.n_dofs()
          << " DoFs (" << fine_constraints.n_constraints()
          << " constrained), " << prolongation.n_nonzero_elements()
          << " nonzeros" << std::endl;

  // 1. the linear field
  {
    const Vector<double> u_fine = interpolate_linear_field(fine_dh);
    for (const bool zero_constrained : {false, true})
      {
        Vector<double> u_coarse = interpolate_linear_field(coarse_dh);
        if (zero_constrained)
          for (const auto &line : coarse_constraints.get_lines())
            u_coarse(line.index) = 0;
        Vector<double> p_u(fine_dh.n_dofs());
        prolongation.vmult(p_u, u_coarse);

        double error = 0, constrained_rows = 0;
        for (unsigned int i = 0; i < fine_dh.n_dofs(); ++i)
          if (fine_constraints.is_constrained(i))
            constrained_rows = std::max(constrained_rows, std::abs(p_u(i)));
          else
            error = std::max(error, std::abs(p_u(i) - u_fine(i)));
        deallog << "  a + b x x"
                << (zero_constrained ? ", constrained coarse DoFs zero" : "")
                << ": max error "
                << (error < 1e-12 ? std::string("< 1e-12") :
                                    std::to_string(error))
                << ", constrained fine rows "
                << (constrained_rows == 0 ? "zero" : "NONZERO") << std::endl;
      }
  }

  // 2. a random coarse function
  {
    Vector<double> u_coarse(coarse_dh.n_dofs());
    for (unsigned int i = 0; i < u_coarse.size(); ++i)
      u_coarse(i) = random_value<double>(-1., 1.);
    coarse_constraints.distribute(u_coarse);
    Vector<double> u_fine(fine_dh.n_dofs());
    prolongation.vmult(u_fine, u_coarse);
    fine_constraints.distribute(u_fine);

    // maximum difference on the fine cells in pyramids and in all other
    // coarse cells
    double error = 0, error_pyramids = 0, error_line_integrals = 0;
    for (const auto &fine_cell : fine_dh.active_cell_iterators())
      {
        // the coarse cell: the same cell or its parent
        const CellId coarse_id = coarse_tria.contains_cell(fine_cell->id()) ?
                                   fine_cell->id() :
                                   fine_cell->parent()->id();
        const auto   coarse_tria_cell = coarse_tria.create_cell_iterator(
          coarse_id);
        const DoFHandler<3>::active_cell_iterator coarse_cell(
          &coarse_tria,
          coarse_tria_cell->level(),
          coarse_tria_cell->index(),
          &coarse_dh);

        const unsigned int i = fine_cell->active_fe_index();
        FEValues<3>        fe_values(c.mapping[i],
                              c.fe[i],
                              c.quadrature[i],
                              update_quadrature_points);
        fe_values.reinit(fine_cell);
        const std::vector<Point<3>> &x = fe_values.get_quadrature_points();
        const auto fine_values   = evaluate(c, fine_cell, u_fine, x);
        const auto coarse_values = evaluate(c, coarse_cell, u_coarse, x);
        double &e = (coarse_cell->reference_cell() == ReferenceCells::Pyramid) ?
                      error_pyramids :
                      error;
        for (unsigned int q = 0; q < x.size(); ++q)
          e = std::max(e, (fine_values[q] - coarse_values[q]).norm());

        // the unconstrained fine DoFs are the line integrals of the coarse
        // function along the fine lines (high-order rule in physical space)
        const QGauss<1> line_quadrature(16);
        for (unsigned int l = 0; l < fine_cell->n_lines(); ++l)
          {
            const auto                    line = fine_cell->line(l);
            const types::global_dof_index dof  = line->dof_index(0, i);
            if (fine_constraints.is_constrained(dof))
              continue;
            const Tensor<1, 3> tangent = line->vertex(1) - line->vertex(0);
            std::vector<Point<3>> points;
            for (const Point<1> &t : line_quadrature.get_points())
              points.push_back(line->vertex(0) + t[0] * tangent);
            const auto values   = evaluate(c, coarse_cell, u_coarse, points);
            double     integral = 0;
            for (unsigned int q = 0; q < points.size(); ++q)
              integral += line_quadrature.weight(q) * values[q] * tangent;
            error_line_integrals =
              std::max(error_line_integrals, std::abs(integral - u_fine(dof)));
          }
      }
    deallog << "  random coarse function: max difference "
            << (error < 1e-10 ? std::string("< 1e-10") :
                                std::to_string(error))
            << ", in pyramids (not nested) " << error_pyramids << std::endl;
    deallog << "  fine DoFs = line integrals of the coarse function: max error "
            << (error_line_integrals < 1e-10 ?
                  std::string("< 1e-10") :
                  std::to_string(error_line_integrals))
            << std::endl;
  }
}



int
main()
{
  initlog();

  const std::vector<
    std::pair<std::string, std::function<void(Triangulation<3> &)>>>
    meshes = {{"tetrahedra", [](Triangulation<3> &t) {
                 GridGenerator::subdivided_hyper_cube_with_simplices(t, 2);
                 GridTools::distort_random(0.15, t, false, 42);
               }},
              {"pyramids",
               [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 0); }},
              {"pyramids and tetrahedra",
               [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 2); }},
              {"hexahedra and pyramids",
               [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 1); }},
              {"wedges and tetrahedra",
               [](Triangulation<3> &t) { make_wedge_tetrahedron_mesh(t, 2); }},
              {"hexahedra", [](Triangulation<3> &t) {
                 GridGenerator::subdivided_hyper_cube(t, 2);
               }}};

  for (const auto &[name, make] : meshes)
    {
      Triangulation<3> coarse_tria;
      make(coarse_tria);
      refine(coarse_tria, Point<3>(0.3, 0.3, 0.3), 0.6);

      Triangulation<3> fine_tria;
      fine_tria.copy_triangulation(coarse_tria);
      refine(fine_tria, Point<3>(0.7, 0.6, 0.5), 0.5);

      check(coarse_tria, fine_tria, name);
    }
}
