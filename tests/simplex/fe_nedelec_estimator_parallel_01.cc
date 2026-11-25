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


// EdgeElementTools::estimate_residual() in parallel
// (parallel::distributed::Triangulation with t8code) on distorted hybrid
// meshes with local refinement, so that hanging faces also lie between
// cells of different processes. The vector is the same for every number of
// processes: the line integrals of a smooth field along the line objects,
// with the hanging node constraints applied. The total estimate, the
// largest indicator and the number of cells must be the same on all numbers
// of processes.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

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
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

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
    return 3;
  }
};



Tensor<1, 3>
field(const Point<3> &p)
{
  return Tensor<1, 3>({std::sin(2 * p[1]) + p[2] * p[2],
                       std::cos(p[2]) * p[0],
                       std::sin(p[0] + p[1])});
}

// some f with its gradient
class RightHandSide : public Function<3>
{
public:
  RightHandSide()
    : Function<3>(3)
  {}

  void
  vector_value(const Point<3> &p, Vector<double> &values) const override
  {
    values(0) = std::sin(p[1]);
    values(1) = p[0] * p[2];
    values(2) = std::cos(p[0]);
  }

  void
  vector_gradient(const Point<3>            &p,
                  std::vector<Tensor<1, 3>> &gradients) const override
  {
    for (auto &g : gradients)
      g = Tensor<1, 3>();
    gradients[0][1] = std::cos(p[1]);
    gradients[1][0] = p[2];
    gradients[1][2] = p[0];
    gradients[2][0] = -std::sin(p[0]);
  }
};



void
refine(parallel::distributed::Triangulation<3> &tria,
       const Point<3>                          &center,
       const double                             radius)
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
check(const std::string                             &name,
      const std::function<void(Triangulation<3> &)> &make)
{
  const MPI_Comm   comm = MPI_COMM_WORLD;
  const Collections c;

  Triangulation<3> coarse;
  make(coarse);
  GridTools::distort_random(0.1, coarse, true, 42);
  parallel::distributed::Triangulation<3> tria(comm);
  tria.copy_triangulation(coarse);
  const double size = GridTools::diameter(coarse);
  refine(tria, Point<3>(0.2, 0.3, 0.25) * size, 0.45 * size);
  refine(tria, Point<3>(0.3, 0.2, 0.3) * size, 0.25 * size);

  DoFHandler<3> dof_handler(tria);
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(Collections::fe_index(cell->reference_cell()));
  dof_handler.distribute_dofs(c.fe);
  const IndexSet &owned = dof_handler.locally_owned_dofs();
  const IndexSet  relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
  AffineConstraints<double> constraints(owned, relevant);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  // the line integrals of the field along the line objects (4-point Gauss)
  TrilinosWrappers::MPI::Vector u(owned, comm);
  const QGauss<1>               line_quadrature(4);
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (unsigned int l = 0; l < cell->n_lines(); ++l)
        {
          const auto                    line = cell->line(l);
          const types::global_dof_index dof =
            line->dof_index(0, cell->active_fe_index());
          if (!owned.is_element(dof))
            continue;
          const Tensor<1, 3> t = line->vertex(1) - line->vertex(0);
          double             integral = 0;
          for (unsigned int q = 0; q < line_quadrature.size(); ++q)
            integral +=
              line_quadrature.weight(q) *
              (field(line->vertex(0) + line_quadrature.point(q)[0] * t) * t);
          u(dof) = integral;
        }
  u.compress(VectorOperation::insert);
  constraints.distribute(u);
  TrilinosWrappers::MPI::Vector ghosted(owned, relevant, comm);
  ghosted = u;

  Vector<float> indicators(tria.n_active_cells());
  const std::array<Quadrature<2>, 2> face_quadratures = {QGaussSimplex<2>(3),
                                                         QGauss<2>(3)};
  EdgeElementTools::estimate_residual(c.mapping,
                                      dof_handler,
                                      ghosted,
                                      RightHandSide(),
                                      2.,
                                      0.5,
                                      c.quadrature,
                                      face_quadratures,
                                      indicators);
  double sum = 0, largest = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const double eta = indicators(cell->active_cell_index());
        sum += eta * eta;
        largest = std::max(largest, eta);
      }
  sum     = Utilities::MPI::sum(sum, comm);
  largest = Utilities::MPI::max(largest, comm);

  deallog << name << ": cells " << tria.n_global_active_cells() << ", DoFs "
          << dof_handler.n_dofs() << ", eta " << std::setprecision(10)
          << std::sqrt(sum) << ", largest indicator " << largest
          << std::setprecision(6) << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  MPILogInitAll                    log;

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
      {"pyramids", [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 0); }},
      {"pyramids and tetrahedra",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 2); }},
      {"hexahedra and pyramids",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 1); }},
      {"wedges and tetrahedra",
       [](Triangulation<3> &t) { make_wedge_tetrahedron_mesh(t, 2); }}};

  for (const auto &[name, make] : meshes)
    check(name, make);
}
