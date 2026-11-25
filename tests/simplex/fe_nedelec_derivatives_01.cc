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


// Gradients and Hessians of the lowest-order Nedelec elements
// (FE_NedelecSimplex, FE_NedelecWedge, FE_NedelecPyramid,
// FE_NedelecHypercube) on distorted (non-affine) hybrid meshes, where cells
// see lines in non-default orientation (sign -1): compared with central
// differences in physical space of the values (gradients) and of the
// gradients (Hessians) computed by FEValues, at two points of every cell.
// On every face of every cell, the values, gradients and Hessians of
// FEFaceValues are the same as those of FEValues at the same points.

#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

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

#include <functional>

#include "../tests.h"

#include "./pyramid_meshes.h"
#include "./simplex_grids.h"
#include "./wedge_meshes.h"


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

  static unsigned int
  fe_index(const ReferenceCell &reference_cell)
  {
    if (reference_cell == ReferenceCells::Tetrahedron)
      return 0;
    if (reference_cell == ReferenceCells::Wedge)
      return 1;
    if (reference_cell == ReferenceCells::Pyramid)
      return 2;
    return 3;
  }
};



// two points inside the reference cell (away from the apex of the pyramid)
std::vector<Point<3>>
reference_points(const ReferenceCell &reference_cell)
{
  if (reference_cell == ReferenceCells::Tetrahedron)
    return {Point<3>(0.2, 0.3, 0.25), Point<3>(0.1, 0.15, 0.6)};
  if (reference_cell == ReferenceCells::Wedge)
    return {Point<3>(0.2, 0.3, 0.4), Point<3>(0.55, 0.15, 0.8)};
  if (reference_cell == ReferenceCells::Pyramid)
    return {Point<3>(0.2, -0.3, 0.3), Point<3>(-0.1, 0.25, 0.55)};
  return {Point<3>(0.3, 0.6, 0.45), Point<3>(0.8, 0.2, 0.7)};
}



void
check(Triangulation<3> &tria, const std::string &name, const bool hessians)
{
  const Collections c;
  const double      h = 1e-5;

  double       max_gradient_error = 0, max_gradient = 0;
  double       max_hessian_error = 0, max_hessian = 0;
  unsigned int n_cells_with_flipped_lines = 0;
  for (const auto &cell : tria.active_cell_iterators())
    {
      const unsigned int i     = Collections::fe_index(cell->reference_cell());
      const auto        &fe    = c.fe[i];
      const auto        &mapping = c.mapping[i];
      for (unsigned int l = 0; l < cell->n_lines(); ++l)
        if (cell->line_orientation(l) != numbers::default_geometric_orientation)
          {
            ++n_cells_with_flipped_lines;
            break;
          }

      for (const Point<3> &p : reference_points(cell->reference_cell()))
        {
          // the point, and the points at distance h in all directions (in
          // physical space)
          const Point<3>        x = mapping.transform_unit_to_real_cell(cell, p);
          std::vector<Point<3>> points = {p};
          for (unsigned int k = 0; k < 3; ++k)
            for (const double sign : {1., -1.})
              points.push_back(mapping.transform_real_to_unit_cell(
                cell, x + sign * h * Point<3>::unit_vector(k)));

          FEValues<3> fe_values(mapping,
                                fe,
                                Quadrature<3>(points),
                                update_values | update_gradients |
                                  (hessians ? update_hessians :
                                              update_default));
          fe_values.reinit(cell);
          for (unsigned int s = 0; s < fe.n_dofs_per_cell(); ++s)
            for (unsigned int comp = 0; comp < 3; ++comp)
              {
                const Tensor<1, 3> gradient =
                  fe_values.shape_grad_component(s, 0, comp);
                max_gradient = std::max(max_gradient, gradient.norm());
                for (unsigned int k = 0; k < 3; ++k)
                  {
                    const double difference =
                      (fe_values.shape_value_component(s, 1 + 2 * k, comp) -
                       fe_values.shape_value_component(s, 2 + 2 * k, comp)) /
                      (2 * h);
                    max_gradient_error =
                      std::max(max_gradient_error,
                               std::abs(difference - gradient[k]));
                  }
                if (hessians)
                  {
                    const Tensor<2, 3> hessian =
                      fe_values.shape_hessian_component(s, 0, comp);
                    max_hessian = std::max(max_hessian, hessian.norm());
                    for (unsigned int k = 0; k < 3; ++k)
                      {
                        const Tensor<1, 3> difference =
                          (fe_values.shape_grad_component(s, 1 + 2 * k, comp) -
                           fe_values.shape_grad_component(s, 2 + 2 * k, comp)) /
                          (2 * h);
                        for (unsigned int j = 0; j < 3; ++j)
                          max_hessian_error =
                            std::max(max_hessian_error,
                                     std::abs(difference[j] - hessian[k][j]));
                      }
                  }
              }
        }
    }

  // faces: FEFaceValues against FEValues at the same points
  double max_face_difference = 0, max_face_size = 0;
  for (const auto &cell : tria.active_cell_iterators())
    {
      const unsigned int i = Collections::fe_index(cell->reference_cell());
      const auto        &fe      = c.fe[i];
      const auto        &mapping = c.mapping[i];
      // one face quadrature per face of the reference cell
      hp::QCollection<2> face_quadrature;
      for (const unsigned int f : cell->face_indices())
        {
          if (cell->face(f)->reference_cell() == ReferenceCells::Triangle)
            face_quadrature.push_back(QGaussSimplex<2>(2));
          else
            face_quadrature.push_back(QGauss<2>(2));
        }
      const UpdateFlags flags = update_values | update_gradients |
                                update_quadrature_points |
                                (hessians ? update_hessians : update_default);
      FEFaceValues<3> fe_face_values(mapping, fe, face_quadrature, flags);
      for (const unsigned int f : cell->face_indices())
        {
          fe_face_values.reinit(cell, f);
          // the points of the face quadrature on the reference cell, as used
          // by FEFaceValues
          const Quadrature<3> points = QProjector<3>::project_to_face(
            cell->reference_cell(),
            face_quadrature[f],
            f,
            cell->combined_face_orientation(f));
          FEValues<3> fe_values(mapping, fe, points, flags);
          fe_values.reinit(cell);
          for (unsigned int q = 0; q < points.size(); ++q)
            max_face_difference =
              std::max(max_face_difference,
                       fe_values.quadrature_point(q).distance(
                         fe_face_values.quadrature_point(q)));
          for (unsigned int q = 0; q < points.size(); ++q)
            for (unsigned int s = 0; s < fe.n_dofs_per_cell(); ++s)
              for (unsigned int comp = 0; comp < 3; ++comp)
                {
                  max_face_size = std::max(
                    {max_face_size,
                     std::abs(fe_values.shape_value_component(s, q, comp)),
                     fe_values.shape_grad_component(s, q, comp).norm()});
                  max_face_difference = std::max(
                    {max_face_difference,
                     std::abs(fe_face_values.shape_value_component(s, q, comp) -
                              fe_values.shape_value_component(s, q, comp)),
                     (fe_face_values.shape_grad_component(s, q, comp) -
                      fe_values.shape_grad_component(s, q, comp))
                       .norm()});
                  if (hessians)
                    max_face_difference = std::max(
                      max_face_difference,
                      (fe_face_values.shape_hessian_component(s, q, comp) -
                       fe_values.shape_hessian_component(s, q, comp))
                          .norm() /
                        10.);
                }
        }
    }

  deallog << name << ": " << tria.n_active_cells() << " cells, "
          << n_cells_with_flipped_lines
          << " with lines in non-default orientation; gradients "
          << (max_gradient_error < 1e-6 * max_gradient ? "OK" : "WRONG");
  if (hessians)
    deallog << ", Hessians "
            << (max_hessian_error < 1e-5 * std::max(max_hessian, max_gradient) ?
                  "OK" :
                  "WRONG");
  deallog << ", faces "
          << (max_face_difference < 1e-9 * max_face_size ? "OK" : "WRONG")
          << std::endl;
  if (max_gradient_error >= 1e-6 * max_gradient)
    deallog << "  gradient error " << max_gradient_error << " (largest gradient "
            << max_gradient << ")" << std::endl;
  if (hessians &&
      max_hessian_error >= 1e-5 * std::max(max_hessian, max_gradient))
    deallog << "  Hessian error " << max_hessian_error << " (largest Hessian "
            << max_hessian << ")" << std::endl;
}



int
main(int argc, char **)
{
  initlog();
  // (without arguments with Hessians; with any argument gradients only)
  const bool hessians = (argc == 1);

  const std::vector<
    std::pair<std::string, std::function<void(Triangulation<3> &)>>>
    meshes = {
      {"tetrahedra",
       [](Triangulation<3> &t) {
         GridGenerator::subdivided_hyper_cube_with_simplices(t, 2);
       }},
      {"wedges",
       [](Triangulation<3> &t) {
         GridGenerator::subdivided_hyper_cube_with_wedges(t, 2);
       }},
      {"hexahedra",
       [](Triangulation<3> &t) { GridGenerator::subdivided_hyper_cube(t, 2); }},
      {"pyramids and tetrahedra",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 2); }},
      {"hexahedra and pyramids",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 1); }},
      {"wedges and tetrahedra",
       [](Triangulation<3> &t) { make_wedge_tetrahedron_mesh(t, 2); }}};

  for (const auto &[name, make] : meshes)
    {
      Triangulation<3> tria;
      make(tria);
      // non-affine cells
      GridTools::distort_random(0.15, tria, false, 42);
      check(tria, name, hessians);
    }
}
