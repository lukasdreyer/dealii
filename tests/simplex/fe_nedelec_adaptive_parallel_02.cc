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


// Adaptive refinement for curl curl u + beta u = f with a singular solution
// on hybrid meshes in parallel (parallel::distributed::Triangulation with
// t8code, Trilinos), as in fe_nedelec_adaptive_parallel_01: the thick
// L-shape (0,2)^3 without the quarter x > 1, y < 1, with the reentrant edge
// x = y = 1, and u = grad (S chi) with S = r^{2/3} sin(2 phi / 3) in polar
// coordinates around the edge (phi in (0, 3 pi / 2) in the domain) and a
// polynomial cutoff chi that vanishes on the faces away from the edge, so
// that n x u = 0 on the whole boundary; curl u = 0 and f = beta u. u is in
// H(curl) but not in H^1 (u ~ r^{-1/3} along the edge): uniform refinement
// converges like h^{2/3} ~ N^{-2/9}, adaptive refinement should give the
// optimal rate N^{-1/3}. (With the inhomogeneous boundary values of grad S
// alone, the discrete spaces are not nested at refined boundary faces next
// to coarse ones, and near the edge this gave a spurious curl and stagnating
// errors.) The meshes are the hybrid meshes of [0,2]^3 without the cells in
// the removed quarter, randomly distorted inside, refined once globally
// before the first cycle. Doerfler marking with theta = 0.8.
// Output per cycle: cells, DoFs, multigrid levels and CG iterations, the
// H(curl) error, the effectivity index, and the rate of the error in terms
// of the number of DoFs (-log(e_k / e_{k-1}) / log(N_k / N_{k-1})). At the
// start, the closed forms of u and grad f are checked against finite
// differences of S chi and f, and the branch of phi on the boundary faces of
// the removed quarter (with the cut on such a face, mapped quadrature points
// with round-off picked the wrong branch).
//
// Optional arguments (for runs outside the test suite; theta = 1 refines
// uniformly):
//   mesh n_cycles theta     (mesh: index into the list of meshes in main())

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/grid_refinement.h>
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
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother_hiptmair.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrices.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/vector_tools.h>

#include <functional>
#include <memory>

#include "../tests.h"

#include "./pyramid_meshes.h"
#include "./simplex_grids.h"
#include "./wedge_meshes.h"


using VectorType = TrilinosWrappers::MPI::Vector;
using MatrixType = TrilinosWrappers::SparseMatrix;

constexpr double beta = 1.;


// the elements, nodal elements, mappings and quadratures, with the FE index
// of a cell given by its reference cell
struct Collections
{
  hp::FECollection<3>      fe{FE_NedelecSimplex<3>(0),
                         FE_NedelecWedge<3>(0),
                         FE_NedelecPyramid<3>(0),
                         FE_NedelecHypercube<3>(0)};
  hp::FECollection<3>      nodal_fe{FE_SimplexP<3>(1),
                               FE_WedgeP<3>(1),
                               FE_PyramidP<3>(1),
                               FE_Q<3>(1)};
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
  distribute_dofs(DoFHandler<3> &dof_handler, const bool nodal) const
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_active_fe_index(fe_index(cell->reference_cell()));
    dof_handler.distribute_dofs(nodal ? nodal_fe : fe);
  }
};



// u = grad (S chi) with the singular function S = r^{2/3} sin(2 phi / 3)
// around the edge x = y = 1 (phi in [0, 3 pi / 2] in the domain, the cut
// phi = -pi / 4 inside the removed quarter) and the cutoff
// chi = x (2 - x) y (2 - y) z (2 - z): S vanishes on the two faces at the
// edge, chi on all other faces, so S chi = 0 and n x u = 0 on the whole
// boundary. grad S = 2/3 r^{-1/3} (-sin(phi / 3), cos(phi / 3), 0), and
// its Hessian is 2/9 r^{-4/3} ((sin(4 phi / 3), -cos(4 phi / 3)),
// (-cos(4 phi / 3), -sin(4 phi / 3))) in the x-y block.
namespace LShape
{
  void
  polar(const Point<3> &x, double &r, double &phi)
  {
    const double dx = x[0] - 1., dy = x[1] - 1.;
    r = std::sqrt(dx * dx + dy * dy);
    // the cut in the middle of the removed quarter, so that points on its
    // boundary faces (up to round-off, e.g. mapped quadrature points) get
    // the right branch
    phi = std::atan2(dy, dx);
    if (phi < -numbers::PI / 4.)
      phi += 2. * numbers::PI;
  }

  double
  singular(const Point<3> &x)
  {
    double r, phi;
    polar(x, r, phi);
    return std::pow(r, 2. / 3.) * std::sin(2. * phi / 3.);
  }

  // grad S (zero on the edge itself, where only the tangential component
  // u_z = 0 is used, in the boundary values of the lines along the edge)
  Tensor<1, 3>
  grad_singular(const Point<3> &x)
  {
    double r, phi;
    polar(x, r, phi);
    if (r < 1e-12)
      return Tensor<1, 3>();
    const double a = 2. / 3. * std::pow(r, -1. / 3.);
    return Tensor<1, 3>({-a * std::sin(phi / 3.), a * std::cos(phi / 3.), 0.});
  }

  Tensor<2, 3>
  hessian_singular(const Point<3> &x)
  {
    double r, phi;
    polar(x, r, phi);
    Tensor<2, 3> h;
    if (r < 1e-12)
      return h;
    const double a = 2. / 9. * std::pow(r, -4. / 3.);
    h[0][0] = a * std::sin(4. * phi / 3.);
    h[0][1] = -a * std::cos(4. * phi / 3.);
    h[1][0] = h[0][1];
    h[1][1] = -h[0][0];
    return h;
  }

  // b(t) = t (2 - t) per coordinate
  double
  cutoff(const Point<3> &x)
  {
    return x[0] * (2 - x[0]) * x[1] * (2 - x[1]) * x[2] * (2 - x[2]);
  }

  Tensor<1, 3>
  grad_cutoff(const Point<3> &x)
  {
    std::array<double, 3> b, db;
    for (unsigned int d = 0; d < 3; ++d)
      {
        b[d]  = x[d] * (2 - x[d]);
        db[d] = 2 - 2 * x[d];
      }
    return Tensor<1, 3>(
      {db[0] * b[1] * b[2], b[0] * db[1] * b[2], b[0] * b[1] * db[2]});
  }

  Tensor<2, 3>
  hessian_cutoff(const Point<3> &x)
  {
    std::array<double, 3> b, db;
    for (unsigned int d = 0; d < 3; ++d)
      {
        b[d]  = x[d] * (2 - x[d]);
        db[d] = 2 - 2 * x[d];
      }
    Tensor<2, 3> h;
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        {
          double v = 1;
          for (unsigned int d = 0; d < 3; ++d)
            v *= (d == i && d == j) ? -2. : (d == i || d == j) ? db[d] : b[d];
          h[i][j] = v;
        }
    return h;
  }

  double
  potential(const Point<3> &x)
  {
    return singular(x) * cutoff(x);
  }

  Tensor<1, 3>
  u(const Point<3> &x)
  {
    return cutoff(x) * grad_singular(x) + singular(x) * grad_cutoff(x);
  }

  Tensor<1, 3>
  curl(const Point<3> &)
  {
    return Tensor<1, 3>();
  }

  Tensor<1, 3>
  f(const Point<3> &x)
  {
    return beta * u(x);
  }

  // grad f = beta Hess(S chi)
  Tensor<2, 3>
  grad_f(const Point<3> &x)
  {
    const Tensor<1, 3> gs = grad_singular(x), gc = grad_cutoff(x);
    return beta * (cutoff(x) * hessian_singular(x) + outer_product(gs, gc) +
                   outer_product(gc, gs) + singular(x) * hessian_cutoff(x));
  }
} // namespace LShape



// the closed forms against central differences at points in the domain
void
check_exact_solution()
{
  const double                h     = 1e-6;
  double                      max_u = 0, max_grad = 0;
  const std::vector<Point<3>> points = {Point<3>(0.3, 0.4, 0.5),
                                        Point<3>(1.7, 1.2, 1.1),
                                        Point<3>(0.8, 0.2, 0.3),
                                        Point<3>(1.05, 1.5, 0.7),
                                        Point<3>(0.6, 1.9, 1.8)};
  for (const Point<3> &x : points)
    {
      const Tensor<1, 3> u = LShape::u(x);
      const Tensor<2, 3> g = LShape::grad_f(x);
      for (unsigned int j = 0; j < 3; ++j)
        {
          Point<3> xp = x, xm = x;
          xp[j] += h;
          xm[j] -= h;
          max_u = std::max(
            max_u,
            std::abs((LShape::potential(xp) - LShape::potential(xm)) / (2 * h) -
                     u[j]));
          const Tensor<1, 3> df = (LShape::f(xp) - LShape::f(xm)) / (2 * h);
          for (unsigned int i = 0; i < 3; ++i)
            max_grad = std::max(max_grad, std::abs(df[i] - g[i][j]));
        }
    }
  // the same branch on the faces of the removed quarter and 1e-15 inside
  // it (round-off of mapped quadrature points)
  double max_jump = 0;
  for (const Point<3> &x : {Point<3>(1.5, 1., 0.3), Point<3>(1., 0.5, 1.2)})
    {
      const Point<3> inside = x + Tensor<1, 3>({1e-15, -1e-15, 0.});
      max_jump = std::max(max_jump, (LShape::u(x) - LShape::u(inside)).norm());
    }
  // n x u = 0 on the boundary
  double max_trace = 0;
  for (const auto &[x, n] : std::vector<std::pair<Point<3>, Tensor<1, 3>>>{
         {Point<3>(0., 1.3, 0.7), Tensor<1, 3>({1., 0., 0.})},
         {Point<3>(1.5, 2., 1.2), Tensor<1, 3>({0., 1., 0.})},
         {Point<3>(0.4, 0.6, 0.), Tensor<1, 3>({0., 0., 1.})},
         {Point<3>(1., 0.5, 1.1), Tensor<1, 3>({1., 0., 0.})},
         {Point<3>(1.6, 1., 0.9), Tensor<1, 3>({0., 1., 0.})}})
    max_trace =
      std::max(max_trace, cross_product_3d(n, LShape::u(x)).norm());
  deallog << "exact solution: u = grad (S chi) "
          << (max_u < 1e-6 ? "OK" : "WRONG") << ", grad f "
          << (max_grad < 1e-6 ? "OK" : "WRONG") << ", branch on the boundary "
          << (max_jump < 1e-10 ? "OK" : "WRONG") << ", n x u = 0 "
          << (max_trace < 1e-12 ? "OK" : "WRONG") << std::endl;
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
    const Tensor<1, 3> u = LShape::u(p);
    for (unsigned int d = 0; d < 3; ++d)
      values(d) = u[d];
  }
};



class RightHandSide : public Function<3>
{
public:
  RightHandSide()
    : Function<3>(3)
  {}

  void
  vector_value(const Point<3> &p, Vector<double> &values) const override
  {
    const Tensor<1, 3> f = LShape::f(p);
    for (unsigned int d = 0; d < 3; ++d)
      values(d) = f[d];
  }

  void
  vector_gradient(const Point<3>            &p,
                  std::vector<Tensor<1, 3>> &gradients) const override
  {
    const Tensor<2, 3> g = LShape::grad_f(p);
    for (unsigned int d = 0; d < 3; ++d)
      gradients[d] = g[d];
  }
};



// remove the cells of a mesh of [0,2]^3 in the quarter x > 1, y < 1
void
cut_l_shape(Triangulation<3> &tria)
{
  std::vector<Point<3>>    vertices = tria.get_vertices();
  std::vector<CellData<3>> cells;
  for (const auto &cell : tria.active_cell_iterators())
    {
      const Point<3> c = cell->center();
      if (c[0] > 1. && c[1] < 1.)
        continue;
      CellData<3> data(cell->n_vertices());
      for (const unsigned int v : cell->vertex_indices())
        data.vertices[v] = cell->vertex_index(v);
      cells.push_back(data);
    }
  SubCellData subcell_data;
  GridTools::delete_unused_vertices(vertices, cells, subcell_data);
  tria.clear();
  tria.create_triangulation(vertices, cells, subcell_data);
}



void
refine(parallel::distributed::Triangulation<3> &tria)
{
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



struct Level
{
  Level(const Triangulation<3> &tria)
    : edge_dh(tria)
    , nodal_dh(tria)
  {}

  DoFHandler<3>             edge_dh;
  DoFHandler<3>             nodal_dh;
  AffineConstraints<double> edge_constraints;
  AffineConstraints<double> nodal_constraints;
  VectorType                rhs;
};



class CoarseSolver : public MGCoarseGridBase<VectorType>
{
public:
  CoarseSolver(const MatrixType &matrix)
    : control(1, 0)
    , direct(control)
  {
    direct.initialize(matrix);
  }

  void
  operator()(const unsigned int,
             VectorType       &dst,
             const VectorType &src) const override
  {
    direct.solve(dst, src);
  }

private:
  mutable SolverControl                  control;
  mutable TrilinosWrappers::SolverDirect direct;
};



// edge operator A and right-hand side, nodal operator beta (grad p, grad q),
// discrete gradient G
void
setup_level(Level &l, MatrixType &a, MatrixType &s, MatrixType &g)
{
  const Collections                c;
  const FEValuesExtractors::Vector E(0);
  const ExactSolution              exact_solution;
  const MPI_Comm                   comm = l.edge_dh.get_mpi_communicator();
  const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

  c.distribute_dofs(l.edge_dh, false);
  c.distribute_dofs(l.nodal_dh, true);
  const IndexSet &owned = l.edge_dh.locally_owned_dofs();
  const IndexSet  relevant =
    DoFTools::extract_locally_relevant_dofs(l.edge_dh);
  const IndexSet &nodal_owned = l.nodal_dh.locally_owned_dofs();
  const IndexSet  nodal_relevant =
    DoFTools::extract_locally_relevant_dofs(l.nodal_dh);

  l.edge_constraints.reinit(owned, relevant);
  DoFTools::make_hanging_node_constraints(l.edge_dh, l.edge_constraints);
  VectorTools::interpolate_boundary_values_curl_conforming(
    c.mapping, l.edge_dh, {{0, &exact_solution}}, l.edge_constraints);
  l.edge_constraints.close();
  l.nodal_constraints.reinit(nodal_owned, nodal_relevant);
  DoFTools::make_hanging_node_constraints(l.nodal_dh, l.nodal_constraints);
  VectorTools::interpolate_boundary_values(c.mapping,
                                           l.nodal_dh,
                                           0,
                                           Functions::ZeroFunction<3>(),
                                           l.nodal_constraints);
  l.nodal_constraints.close();

  // edge operator and right-hand side
  {
    TrilinosWrappers::SparsityPattern sparsity(owned, owned, relevant, comm);
    DoFTools::make_sparsity_pattern(
      l.edge_dh, sparsity, l.edge_constraints, false, my_rank);
    sparsity.compress();
    a.reinit(sparsity);
    l.rhs.reinit(owned, comm);

    hp::FEValues<3> hp_fe_values(c.mapping,
                                 c.fe,
                                 c.quadrature,
                                 update_values | update_gradients |
                                   update_quadrature_points |
                                   update_JxW_values);
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> dof_indices;
    for (const auto &cell : l.edge_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          hp_fe_values.reinit(cell);
          const FEValues<3> &fe_values = hp_fe_values.get_present_fe_values();
          const unsigned int n_dofs    = cell->get_fe().n_dofs_per_cell();
          cell_matrix.reinit(n_dofs, n_dofs);
          cell_rhs.reinit(n_dofs);
          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
              const Tensor<1, 3> f = LShape::f(fe_values.quadrature_point(q));
              for (unsigned int i = 0; i < n_dofs; ++i)
                {
                  for (unsigned int j = 0; j < n_dofs; ++j)
                    cell_matrix(i, j) +=
                      (fe_values[E].curl(i, q) * fe_values[E].curl(j, q) +
                       beta * fe_values[E].value(i, q) *
                         fe_values[E].value(j, q)) *
                      fe_values.JxW(q);
                  cell_rhs(i) +=
                    f * fe_values[E].value(i, q) * fe_values.JxW(q);
                }
            }
          dof_indices.resize(n_dofs);
          cell->get_dof_indices(dof_indices);
          l.edge_constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, dof_indices, a, l.rhs);
        }
    a.compress(VectorOperation::add);
    l.rhs.compress(VectorOperation::add);
  }

  // nodal operator
  {
    TrilinosWrappers::SparsityPattern sparsity(nodal_owned,
                                               nodal_owned,
                                               nodal_relevant,
                                               comm);
    DoFTools::make_sparsity_pattern(
      l.nodal_dh, sparsity, l.nodal_constraints, false, my_rank);
    sparsity.compress();
    s.reinit(sparsity);

    hp::FEValues<3> hp_fe_values(c.mapping,
                                 c.nodal_fe,
                                 c.quadrature,
                                 update_gradients | update_JxW_values);
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> dof_indices;
    for (const auto &cell : l.nodal_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          hp_fe_values.reinit(cell);
          const FEValues<3> &fe_values = hp_fe_values.get_present_fe_values();
          const unsigned int n_dofs    = cell->get_fe().n_dofs_per_cell();
          cell_matrix.reinit(n_dofs, n_dofs);
          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            for (unsigned int i = 0; i < n_dofs; ++i)
              for (unsigned int j = 0; j < n_dofs; ++j)
                cell_matrix(i, j) += beta * fe_values.shape_grad(i, q) *
                                     fe_values.shape_grad(j, q) *
                                     fe_values.JxW(q);
          dof_indices.resize(n_dofs);
          cell->get_dof_indices(dof_indices);
          l.nodal_constraints.distribute_local_to_global(cell_matrix,
                                                         dof_indices,
                                                         s);
        }
    s.compress(VectorOperation::add);
  }

  // discrete gradient
  {
    TrilinosWrappers::SparsityPattern sparsity(owned, nodal_owned, comm);
    DoFTools::make_discrete_gradient_sparsity_pattern(l.edge_dh,
                                                      l.edge_constraints,
                                                      l.nodal_dh,
                                                      l.nodal_constraints,
                                                      sparsity);
    sparsity.compress();
    g.reinit(sparsity);
    DoFTools::make_discrete_gradient(l.edge_dh,
                                     l.edge_constraints,
                                     l.nodal_dh,
                                     l.nodal_constraints,
                                     g);
  }
}



void
setup_prolongation(const Level &coarse, const Level &fine, MatrixType &p)
{
  TrilinosWrappers::SparsityPattern sparsity(
    fine.edge_dh.locally_owned_dofs(),
    coarse.edge_dh.locally_owned_dofs(),
    fine.edge_dh.get_mpi_communicator());
  DoFTools::make_edge_element_prolongation_sparsity_pattern(
    coarse.edge_dh,
    coarse.edge_constraints,
    fine.edge_dh,
    fine.edge_constraints,
    sparsity);
  sparsity.compress();
  p.reinit(sparsity);
  DoFTools::make_edge_element_prolongation(coarse.edge_dh,
                                           coarse.edge_constraints,
                                           fine.edge_dh,
                                           fine.edge_constraints,
                                           p);
}



// solve on the current mesh with multigrid; returns the solution on the
// finest level (ghosted), its DoFHandler stays alive in the returned levels
unsigned int
solve(const parallel::distributed::Triangulation<3> &tria,
      std::vector<std::unique_ptr<Level>>           &levels,
      VectorType                                    &ghosted_solution,
      unsigned int                                  &n_levels)
{
  const MPI_Comm comm = tria.get_mpi_communicator();
  const auto     trias =
    MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(tria);
  n_levels = trias.size();

  // the finest level lives on tria itself, so that the indicators belong to
  // its cells
  levels.clear();
  MGLevelObject<MatrixType> a(0, n_levels - 1), s(0, n_levels - 1),
    g(0, n_levels - 1), p(0, n_levels - 1);
  for (unsigned int l = 0; l < n_levels; ++l)
    {
      levels.push_back(std::make_unique<Level>(
        l + 1 == n_levels ? static_cast<const Triangulation<3> &>(tria) :
                            *trias[l]));
      setup_level(*levels[l], a[l], s[l], g[l]);
      if (l > 0)
        setup_prolongation(*levels[l - 1], *levels[l], p[l]);
    }
  const Level &fine = *levels.back();

  mg::Matrix<VectorType>                     matrix(a);
  MGTransferMatrices<VectorType, MatrixType> transfer;
  transfer.initialize(p);
  CoarseSolver coarse_solver(a[0]);

  // Trilinos SSOR is block Jacobi across processes and needs damping
  using Relaxation = TrilinosWrappers::PreconditionSSOR;
  MGSmootherHiptmair<VectorType, MatrixType, Relaxation> hiptmair;
  hiptmair.initialize(
    a,
    g,
    s,
    MGSmootherHiptmair<VectorType, MatrixType, Relaxation>::AdditionalData(
      1, Relaxation::AdditionalData(0.8), Relaxation::AdditionalData(0.8)));
  Multigrid<VectorType> mg(
    matrix, coarse_solver, transfer, hiptmair, hiptmair, 0, n_levels - 1);
  PreconditionMG<3, VectorType, MGTransferMatrices<VectorType, MatrixType>>
    preconditioner(fine.edge_dh, mg, transfer);

  VectorType    solution(fine.edge_dh.locally_owned_dofs(), comm);
  SolverControl control(200, 1e-12 * fine.rhs.l2_norm(), false, false);
  SolverCG<VectorType> cg(control);
  cg.solve(a[n_levels - 1], solution, fine.rhs, preconditioner);
  fine.edge_constraints.distribute(solution);

  ghosted_solution.reinit(fine.edge_dh.locally_owned_dofs(),
                          DoFTools::extract_locally_relevant_dofs(
                            fine.edge_dh),
                          comm);
  ghosted_solution = solution;
  return control.last_step();
}



// the error in the H(curl) norm
double
error(const DoFHandler<3> &dof_handler, const VectorType &solution)
{
  const Collections                c;
  const FEValuesExtractors::Vector E(0);
  hp::FEValues<3>                  hp_fe_values(c.mapping,
                               c.fe,
                               c.quadrature,
                               update_values | update_gradients |
                                 update_quadrature_points | update_JxW_values);
  std::vector<types::global_dof_index> dof_indices;
  double                               error2 = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
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
            error2 += ((u - LShape::u(x)).norm_square() +
                       (curl - LShape::curl(x)).norm_square()) *
                      fe_values.JxW(q);
          }
      }
  return std::sqrt(
    Utilities::MPI::sum(error2, dof_handler.get_mpi_communicator()));
}



void
test(const std::string                             &name,
     const std::function<void(Triangulation<3> &)> &make,
     const unsigned int                             n_cycles,
     const double                                   theta,
     const bool                                     verbose)
{
  const MPI_Comm                     comm = MPI_COMM_WORLD;
  const Collections                  c;
  const RightHandSide                rhs_function;
  const std::array<Quadrature<2>, 2> face_quadratures = {QGaussSimplex<2>(3),
                                                         QGauss<2>(3)};

  Triangulation<3> coarse;
  make(coarse);
  cut_l_shape(coarse);
  GridTools::distort_random(0.1, coarse, true, 42);
  parallel::distributed::Triangulation<3> tria(comm);
  tria.copy_triangulation(coarse);
  // one global refinement
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_refine_flag();
  refine(tria);

  if (!verbose)
    deallog << name << ":" << std::endl;
  double previous_error = 0, previous_dofs = 0;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      if (cycle > 0)
        refine(tria);

      std::vector<std::unique_ptr<Level>> levels;
      VectorType                          solution;
      unsigned int                        n_levels   = 0;
      const unsigned int                  iterations =
        solve(tria, levels, solution, n_levels);
      const DoFHandler<3> &dof_handler = levels.back()->edge_dh;
      const double         err         = error(dof_handler, solution);

      Vector<float> indicators(tria.n_active_cells());
      EdgeElementTools::estimate_residual(c.mapping,
                                          dof_handler,
                                          solution,
                                          rhs_function,
                                          beta,
                                          1.,
                                          c.quadrature,
                                          face_quadratures,
                                          indicators);
      double eta2 = 0;
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          eta2 += Utilities::fixed_power<2>(
            static_cast<double>(indicators(cell->active_cell_index())));
      const double eta = std::sqrt(Utilities::MPI::sum(eta2, comm));

      if (verbose)
        {
          if (Utilities::MPI::this_mpi_process(comm) == 0)
            std::cout << name << " theta=" << theta << " cycle=" << cycle
                      << " cells=" << tria.n_global_active_cells()
                      << " dofs=" << dof_handler.n_dofs()
                      << " levels=" << n_levels
                      << " iterations=" << iterations << " error=" << err
                      << " eta=" << eta << std::endl;
        }
      else
        deallog << "  cells=" << tria.n_global_active_cells()
                << " dofs=" << dof_handler.n_dofs() << " levels=" << n_levels
                << " iterations=" << iterations << std::setprecision(3)
                << " error=" << err << std::fixed << std::setprecision(1)
                << " effectivity=" << eta / err;
      if (!verbose && cycle > 0)
        deallog << std::setprecision(2) << " rate="
                << -std::log(err / previous_error) /
                     std::log(dof_handler.n_dofs() / previous_dofs);
      if (!verbose)
        deallog << std::defaultfloat << std::setprecision(6) << std::endl;
      previous_error = err;
      previous_dofs  = dof_handler.n_dofs();

      // Doerfler marking: the smallest set of cells with theta eta^2
      if (cycle + 1 < n_cycles)
        parallel::distributed::GridRefinement::
          refine_and_coarsen_fixed_fraction(
            tria, indicators, theta, 0., VectorTools::L2_norm);
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const std::vector<
    std::pair<std::string, std::function<void(Triangulation<3> &)>>>
    meshes = {
      {"tetrahedra",
       [](Triangulation<3> &t) {
         GridGenerator::subdivided_hyper_cube_with_simplices(t, 2, 0., 2.);
       }},
      {"wedges",
       [](Triangulation<3> &t) {
         GridGenerator::subdivided_hyper_cube_with_wedges(t, 2, 0., 2.);
       }},
      {"hexahedra",
       [](Triangulation<3> &t) { GridGenerator::subdivided_hyper_cube(t, 2, 0., 2.); }},
      {"pyramids", [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 0); }},
      {"pyramids and tetrahedra",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 2); }},
      {"hexahedra and pyramids",
       [](Triangulation<3> &t) { make_pyramid_mesh(t, 2, 1); }},
      {"wedges and tetrahedra",
       [](Triangulation<3> &t) { make_wedge_tetrahedron_mesh(t, 2); }}};

  if (argc == 4)
    {
      const unsigned int mesh     = std::stoi(argv[1]);
      const unsigned int n_cycles = std::stoi(argv[2]);
      const double       theta    = std::stod(argv[3]);
      test(meshes[mesh].first, meshes[mesh].second, n_cycles, theta, true);
      return 0;
    }

  MPILogInitAll log;
  check_exact_solution();
  // the meshes that contain all four cell shapes
  for (const auto &[name, make] : meshes)
    if (name != "wedges" && name != "hexahedra" && name != "pyramids")
      test(name, make, 4, 0.8, false);
}
