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


// MatrixFree with discontinuous elements on distorted 3d meshes of
// tetrahedra, wedges, pyramids, hexahedra and their mixtures, without and
// with hanging faces. The operator
//   a(u,v) = sum_K int_K (u v + grad u . grad v) + sum_{boundary F} int_F u v
//          + sum_{interior F} int_F ([u] + {grad u . n}) [v]
// (values, gradients and normal vectors on both sides of each face) applied
// with cell and face loops is compared with the matrix assembled with
// FEValues and FEInterfaceValues, with the same quadrature rules. In case of
// a difference, the differences of the parts of the operator are printed.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/util.h>

#include <functional>

#include "../tests.h"

#include "./pyramid_meshes.h"
#include "./simplex_grids.h"
#include "./wedge_meshes.h"


constexpr int dim = 3;
using VectorType  = LinearAlgebra::distributed::Vector<double>;

// the parts of the operator, to find which part differs
namespace Part
{
  constexpr unsigned int cells           = 1;
  constexpr unsigned int boundary        = 2;
  constexpr unsigned int jump            = 4;
  constexpr unsigned int normal_gradient = 8;
  constexpr unsigned int cell_gradients  = 16;
  constexpr unsigned int all             = 31;
} // namespace Part


unsigned int
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



void
refine(Triangulation<dim> &tria, const Point<dim> &center, const double radius)
{
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->center().distance(center) < radius)
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
apply_matrix_free(const MatrixFree<dim, double> &matrix_free,
                  VectorType                    &dst,
                  const VectorType              &src,
                  const unsigned int             parts)
{
  const double c         = (parts & Part::cells) ? 1. : 0.;
  const double b         = (parts & Part::boundary) ? 1. : 0.;
  const double j         = (parts & Part::jump) ? 1. : 0.;
  const double g         = (parts & Part::normal_gradient) ? 1. : 0.;
  const double k         = (parts & Part::cell_gradients) ? 1. : 0.;
  using FECellIntegrator = FEEvaluation<dim, -1, 0, 1, double>;
  using FEFaceIntegrator = FEFaceEvaluation<dim, -1, 0, 1, double>;
  matrix_free.template loop<VectorType, VectorType>(
    [c, k](const auto &data, auto &dst, const auto &src, const auto range) {
      FECellIntegrator phi(data, range);
      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::values |
                                EvaluationFlags::gradients);
          for (const unsigned int q : phi.quadrature_point_indices())
            {
              phi.submit_value(c * phi.get_value(q), q);
              phi.submit_gradient(k * phi.get_gradient(q), q);
            }
          phi.integrate_scatter(EvaluationFlags::values |
                                  EvaluationFlags::gradients,
                                dst);
        }
    },
    [j, g](const auto &data, auto &dst, const auto &src, const auto range) {
      FEFaceIntegrator phi_m(data, range, true);
      FEFaceIntegrator phi_p(data, range, false);
      for (unsigned int face = range.first; face < range.second; ++face)
        {
          phi_m.reinit(face);
          phi_p.reinit(face);
          phi_m.gather_evaluate(src,
                                EvaluationFlags::values |
                                  EvaluationFlags::gradients);
          phi_p.gather_evaluate(src,
                                EvaluationFlags::values |
                                  EvaluationFlags::gradients);
          for (const unsigned int q : phi_m.quadrature_point_indices())
            {
              // both normal derivatives with the normal of the interior side
              const auto flux = j * (phi_m.get_value(q) - phi_p.get_value(q)) +
                                g * 0.5 *
                                  (phi_m.get_normal_derivative(q) +
                                   phi_p.get_normal_derivative(q));
              phi_m.submit_value(flux, q);
              phi_p.submit_value(-flux, q);
            }
          phi_m.integrate_scatter(EvaluationFlags::values, dst);
          phi_p.integrate_scatter(EvaluationFlags::values, dst);
        }
    },
    [b](const auto &data, auto &dst, const auto &src, const auto range) {
      FEFaceIntegrator phi(data, range, true);
      for (unsigned int face = range.first; face < range.second; ++face)
        {
          phi.reinit(face);
          phi.gather_evaluate(src, EvaluationFlags::values);
          for (const unsigned int q : phi.quadrature_point_indices())
            phi.submit_value(b * phi.get_value(q), q);
          phi.integrate_scatter(EvaluationFlags::values, dst);
        }
    },
    dst,
    src,
    true);
}



void
assemble_matrix(const DoFHandler<dim>                 &dof_handler,
                const hp::MappingCollection<dim>      &mapping,
                const hp::QCollection<dim>            &quadrature,
                const std::vector<hp::QCollection<2>> &face_quadratures,
                const unsigned int                     parts,
                SparseMatrix<double>                  &matrix)
{
  const double c  = (parts & Part::cells) ? 1. : 0.;
  const double b  = (parts & Part::boundary) ? 1. : 0.;
  const double j  = (parts & Part::jump) ? 1. : 0.;
  const double g  = (parts & Part::normal_gradient) ? 1. : 0.;
  const double cg = (parts & Part::cell_gradients) ? 1. : 0.;
  matrix          = 0;
  const hp::FECollection<dim>         &fe = dof_handler.get_fe_collection();
  hp::FEValues<dim>                    hp_fe_values(mapping,
                                 fe,
                                 quadrature,
                                 update_values | update_gradients |
                                   update_JxW_values);
  hp::FEFaceValues<dim>                hp_fe_face_values(mapping,
                                          fe,
                                          face_quadratures,
                                          update_values | update_JxW_values);
  FEInterfaceValues<dim>               fe_interface_values(mapping,
                                             fe,
                                             face_quadratures,
                                             update_values | update_gradients |
                                               update_JxW_values |
                                               update_normal_vectors);
  FullMatrix<double>                   local;
  std::vector<types::global_dof_index> dofs;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      hp_fe_values.reinit(cell);
      const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
      const unsigned int   n         = fe_values.dofs_per_cell;
      local.reinit(n, n);
      for (const unsigned int q : fe_values.quadrature_point_indices())
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int k = 0; k < n; ++k)
            local(i, k) +=
              (c * fe_values.shape_value(i, q) * fe_values.shape_value(k, q) +
               cg * fe_values.shape_grad(i, q) * fe_values.shape_grad(k, q)) *
              fe_values.JxW(q);
      dofs.resize(n);
      cell->get_dof_indices(dofs);
      matrix.add(dofs, local);

      for (const unsigned int f : cell->face_indices())
        {
          if (cell->at_boundary(f))
            {
              hp_fe_face_values.reinit(cell, f);
              const FEFaceValues<dim> &fv =
                hp_fe_face_values.get_present_fe_values();
              local.reinit(n, n);
              for (const unsigned int q : fv.quadrature_point_indices())
                for (unsigned int i = 0; i < n; ++i)
                  for (unsigned int k = 0; k < n; ++k)
                    local(i, k) += b * fv.shape_value(i, q) *
                                   fv.shape_value(k, q) * fv.JxW(q);
              matrix.add(dofs, local);
              continue;
            }
          // each regular face once, each hanging face from the fine side
          const auto neighbor = cell->neighbor(f);
          if (cell->neighbor_is_coarser(f))
            {
              const auto [neighbor_face, neighbor_subface] =
                cell->neighbor_of_coarser_neighbor(f);
              fe_interface_values.reinit(cell,
                                         f,
                                         numbers::invalid_unsigned_int,
                                         neighbor,
                                         neighbor_face,
                                         neighbor_subface);
            }
          else if (!cell->face(f)->has_children() &&
                   neighbor->active_cell_index() > cell->active_cell_index())
            fe_interface_values.reinit(cell,
                                       f,
                                       numbers::invalid_unsigned_int,
                                       neighbor,
                                       cell->neighbor_of_neighbor(f),
                                       numbers::invalid_unsigned_int);
          else
            continue;

          const unsigned int m = fe_interface_values.n_current_interface_dofs();
          local.reinit(m, m);
          for (const unsigned int q :
               fe_interface_values.quadrature_point_indices())
            {
              const Tensor<1, dim> &normal =
                fe_interface_values.normal_vector(q);
              for (unsigned int i = 0; i < m; ++i)
                for (unsigned int k = 0; k < m; ++k)
                  local(i, k) +=
                    (j * fe_interface_values.jump_in_shape_values(k, q) +
                     g * fe_interface_values.average_of_shape_gradients(k, q) *
                       normal) *
                    fe_interface_values.jump_in_shape_values(i, q) *
                    fe_interface_values.JxW(q);
            }
          matrix.add(fe_interface_values.get_interface_dof_indices(), local);
        }
    }
}


void
check(Triangulation<dim> &tria, const unsigned int degree)
{
  hp::FECollection<dim>      fe{FE_SimplexDGP<dim>(degree),
                           FE_WedgeDGP<dim>(degree),
                           FE_PyramidDGP<dim>(degree),
                           FE_DGQ<dim>(degree)};
  hp::MappingCollection<dim> mapping{MappingFE<dim>(FE_SimplexP<dim>(1)),
                                     MappingFE<dim>(FE_WedgeP<dim>(1)),
                                     MappingFE<dim>(FE_PyramidP<dim>(1)),
                                     MappingQ<dim>(1)};
  hp::QCollection<dim>       quadrature{QGaussSimplex<dim>(degree + 1),
                                  QGaussWedge<dim>(degree + 1),
                                  QGaussPyramid<dim>(degree + 1),
                                  QGauss<dim>(degree + 1)};
  // the face quadratures MatrixFree derives from the cell quadratures
  std::vector<hp::QCollection<2>> face_quadratures;
  for (unsigned int i = 0; i < quadrature.size(); ++i)
    face_quadratures.push_back(
      internal::MatrixFreeFunctions::get_face_quadrature_collection(
        quadrature[i])
        .second);

  DoFHandler<dim> dof_handler(tria);
  for (const auto &cell : dof_handler.active_cell_iterators())
    cell->set_active_fe_index(fe_index(cell->reference_cell()));
  dof_handler.distribute_dofs(fe);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);
  SparseMatrix<double> matrix(sparsity);

  AffineConstraints<double> constraints;
  constraints.close();
  MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags = update_values | update_gradients;
  additional_data.mapping_update_flags_inner_faces =
    update_values | update_gradients;
  additional_data.mapping_update_flags_boundary_faces = update_values;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, double>::AdditionalData::none;
  MatrixFree<dim, double> matrix_free;
  matrix_free.reinit(
    mapping, dof_handler, constraints, quadrature, additional_data);

  VectorType src, dst, reference;
  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst);
  matrix_free.initialize_dof_vector(reference);
  for (unsigned int i = 0; i < src.size(); ++i)
    src(i) = random_value<double>(-1., 1.);

  // the relative difference of the matrix-free and the matrix-based operator
  const auto difference = [&](const unsigned int parts) {
    assemble_matrix(
      dof_handler, mapping, quadrature, face_quadratures, parts, matrix);
    apply_matrix_free(matrix_free, dst, src, parts);
    matrix.vmult(reference, src);
    dst -= reference;
    return dst.linfty_norm() / reference.linfty_norm();
  };

  const double all = difference(Part::all);
  deallog << "degree " << degree << ", " << tria.n_active_cells()
          << " cells: " << (all < 1e-12 ? "OK" : "WRONG") << std::endl;
  if (all >= 1e-12)
    {
      deallog << "  relative difference " << all << "; cells "
              << difference(Part::cells) << ", boundary faces "
              << difference(Part::boundary) << ", interior faces: jump "
              << difference(Part::jump) << ", normal derivative "
              << difference(Part::normal_gradient) << "; cell gradients "
              << difference(Part::cell_gradients) << std::endl;
    }
}



int
main()
{
  initlog();

  const std::vector<
    std::pair<std::string, std::function<void(Triangulation<dim> &)>>>
    meshes = {{"tetrahedra",
               [](Triangulation<dim> &t) {
                 GridGenerator::subdivided_hyper_cube_with_simplices(t, 2);
               }},
              {"wedges",
               [](Triangulation<dim> &t) {
                 GridGenerator::subdivided_hyper_cube_with_wedges(t, 2);
               }},
              {"hexahedra",
               [](Triangulation<dim> &t) {
                 GridGenerator::subdivided_hyper_cube(t, 2);
               }},
              {"pyramids",
               [](Triangulation<dim> &t) { make_pyramid_mesh(t, 2, 0); }},
              {"pyramids and tetrahedra",
               [](Triangulation<dim> &t) { make_pyramid_mesh(t, 2, 2); }},
              {"hexahedra and pyramids",
               [](Triangulation<dim> &t) { make_pyramid_mesh(t, 2, 1); }},
              {"wedges and tetrahedra", [](Triangulation<dim> &t) {
                 make_wedge_tetrahedron_mesh(t, 2);
               }}};

  for (const bool hanging : {false, true})
    for (const auto &[name, make] : meshes)
      for (const unsigned int degree : {1, 2})
        {
          deallog << name << (hanging ? " with hanging faces" : "") << ", ";
          Triangulation<dim> tria;
          make(tria);
          GridTools::distort_random(0.15, tria, true, 42);
          if (hanging)
            {
              const double size = GridTools::diameter(tria);
              refine(tria, Point<dim>(0.2, 0.3, 0.25) * size, 0.45 * size);
              refine(tria, Point<dim>(0.3, 0.2, 0.3) * size, 0.25 * size);
            }
          check(tria, degree);
        }
}
