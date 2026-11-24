/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * ------------------------------------------------------------------------
 *
 * Linear advection u_t + beta . grad u = 0 on the unit square with
 * discontinuous Galerkin elements on a hybrid mesh of quadrilaterals (FE_DGQ)
 * and triangles (FE_SimplexDGP), parallel::distributed::Triangulation (t8code
 * backend), explicit Runge-Kutta time stepping, and adaptive refinement and
 * coarsening every few time steps with SolutionTransfer.
 *
 * The velocity is the solid body rotation beta = 2 pi (-(y - 1/2), x - 1/2)
 * (one turn per unit time), the initial value a Gaussian hill, so that the
 * exact solution is the rotated hill (zero inflow: the hill stays away from
 * the boundary).
 *
 * Spatial discretization: upwind fluxes, M u' = A u with the operator matrix A
 * assembled once per mesh (cell terms, interior faces with FEInterfaceValues,
 * also between cells of different shapes and at hanging faces, outflow
 * boundary), and the block diagonal mass matrix inverted cell by cell. Time
 * stepping: SSP-RK3 (Shu-Osher) or the classical RK4, with the time step from
 * the CFL condition dt = cfl h_min / (|beta|_max (2p + 1)) for the cells of
 * the finest level.
 *
 * Usage: mpirun -np N ./advection_hybrid [key=value ...]
 *   mesh=hybrid       hybrid (squares alternating with pairs of triangles),
 *                     quad or tri, n x n squares
 *   n=4               coarse mesh size
 *   degree=2          polynomial degree
 *   global=3          global refinements
 *   max_delta=3       maximal number of adaptive levels above global
 *   adapt=1           0: no adaptation
 *   adapt_every=10    time steps between adaptations
 *   refine=0.1        refine where the estimate exceeds this times the largest
 *   coarsen=0.01      coarsen where it is below this times the largest
 *   scheme=ssprk3     ssprk3 or rk4
 *   cfl=0.5           CFL number (with respect to h_min / (2p + 1))
 *   t_end=1           final time (one turn)
 *   sigma=0.06        width of the hill
 *   csv=advection.csv one line per output step
 *   output=           directory for VTU output, empty for none
 *   output_every=50   time steps between outputs
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace LA
{
#if defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
  using namespace dealii::LinearAlgebraPETSc;
#endif
} // namespace LA


namespace AdvectionHybrid
{
  using namespace dealii;

  constexpr int dim = 2;

  struct Parameters
  {
    std::string  mesh         = "hybrid";
    unsigned int n            = 4;
    unsigned int degree       = 2;
    unsigned int global       = 3;
    unsigned int max_delta    = 3;
    bool         adapt        = true;
    unsigned int adapt_every  = 10;
    double       refine       = 0.1;
    double       coarsen      = 0.01;
    std::string  scheme       = "ssprk3";
    double       cfl          = 0.5;
    double       t_end        = 1.0;
    double       sigma        = 0.06;
    std::string  csv          = "advection.csv";
    std::string  output       = "";
    unsigned int output_every = 50;

    void
    parse(int argc, char *argv[])
    {
      for (int i = 1; i < argc; ++i)
        {
          const std::string arg = argv[i];
          const auto        eq  = arg.find('=');
          AssertThrow(eq != std::string::npos,
                      ExcMessage("Arguments have the form key=value: " + arg));
          const std::string key = arg.substr(0, eq), value = arg.substr(eq + 1);
          if (key == "mesh")
            mesh = value;
          else if (key == "n")
            n = std::stoi(value);
          else if (key == "degree")
            degree = std::stoi(value);
          else if (key == "global")
            global = std::stoi(value);
          else if (key == "max_delta")
            max_delta = std::stoi(value);
          else if (key == "adapt")
            adapt = std::stoi(value) != 0;
          else if (key == "adapt_every")
            adapt_every = std::stoi(value);
          else if (key == "refine")
            refine = std::stod(value);
          else if (key == "coarsen")
            coarsen = std::stod(value);
          else if (key == "scheme")
            scheme = value;
          else if (key == "cfl")
            cfl = std::stod(value);
          else if (key == "t_end")
            t_end = std::stod(value);
          else if (key == "sigma")
            sigma = std::stod(value);
          else if (key == "csv")
            csv = value;
          else if (key == "output")
            output = value;
          else if (key == "output_every")
            output_every = std::stoi(value);
          else
            AssertThrow(false, ExcMessage("Unknown argument " + key));
        }
      AssertThrow(mesh == "hybrid" || mesh == "quad" || mesh == "tri",
                  ExcMessage("mesh must be hybrid, quad or tri"));
      AssertThrow(scheme == "ssprk3" || scheme == "rk4",
                  ExcMessage("scheme must be ssprk3 or rk4"));
    }
  };



  Tensor<1, dim>
  velocity(const Point<dim> &p)
  {
    Tensor<1, dim> b;
    b[0] = -2 * numbers::PI * (p[1] - 0.5);
    b[1] = 2 * numbers::PI * (p[0] - 0.5);
    return b;
  }



  // the hill, rotated back by the angle 2 pi t
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double sigma)
      : sigma(sigma)
    {}

    double
    value(const Point<dim> &p, const unsigned int = 0) const override
    {
      const double a = -2 * numbers::PI * this->get_time();
      const double x = p[0] - 0.5, y = p[1] - 0.5;
      const Point<dim> q(0.5 + std::cos(a) * x - std::sin(a) * y,
                         0.5 + std::sin(a) * x + std::cos(a) * y);
      return std::exp(-(q - Point<dim>(0.75, 0.5)).norm_square() / (sigma * sigma));
    }

  private:
    const double sigma;
  };



  // the unit square in n x n squares: quadrilaterals, pairs of triangles, or
  // both in a checkerboard pattern (hybrid); as in heat_hybrid
  void
  make_mesh(Triangulation<dim> &tria, const unsigned int n, const std::string &type)
  {
    std::vector<Point<dim>> vertices;
    for (unsigned int y = 0; y <= n; ++y)
      for (unsigned int x = 0; x <= n; ++x)
        vertices.emplace_back(1. * x / n, 1. * y / n);
    const auto vertex = [n](const unsigned int x, const unsigned int y) {
      return x + (n + 1) * y;
    };
    const auto area = [&](const unsigned int a, const unsigned int b, const unsigned int c) {
      const Tensor<1, dim> u = vertices[b] - vertices[a], v = vertices[c] - vertices[a];
      return u[0] * v[1] - u[1] * v[0];
    };
    std::vector<CellData<dim>> cells;
    for (unsigned int y = 0; y < n; ++y)
      for (unsigned int x = 0; x < n; ++x)
        {
          const unsigned int v00 = vertex(x, y), v10 = vertex(x + 1, y),
                             v01 = vertex(x, y + 1), v11 = vertex(x + 1, y + 1);
          if (type == "quad" || (type == "hybrid" && (x + y) % 2 == 0))
            {
              CellData<dim> cell(4);
              cell.vertices = {v00, v10, v01, v11};
              cells.push_back(cell);
            }
          else
            for (std::array<unsigned int, 3> t :
                 {std::array<unsigned int, 3>{{v00, v10, v11}},
                  std::array<unsigned int, 3>{{v00, v11, v01}}})
              {
                if (area(t[0], t[1], t[2]) < 0)
                  std::swap(t[1], t[2]);
                CellData<dim> cell(3);
                cell.vertices = {t[0], t[1], t[2]};
                cells.push_back(cell);
              }
        }
    tria.create_triangulation(vertices, cells, SubCellData());
  }



  class AdvectionProblem
  {
  public:
    AdvectionProblem(const Parameters &parameters);

    void
    run();

  private:
    unsigned int
    fe_index(const ReferenceCell &r) const
    {
      return r == ReferenceCells::Quadrilateral ? 0 : 1;
    }

    void
    setup_system();
    void
    assemble_operator();
    // y = M^{-1} A u
    void
    apply(const LA::MPI::Vector &u, LA::MPI::Vector &y) const;
    void
    time_step(const double dt);
    void
    adapt_mesh();
    void
    update_ghosted();
    double
    l2_error(const double time);
    double
    mass();
    void
    output_results(const unsigned int step, const double time) const;

    const Parameters   parameters;
    MPI_Comm           mpi_communicator;
    ConditionalOStream pcout;

    parallel::distributed::Triangulation<dim> triangulation;
    hp::FECollection<dim>                     fe;
    hp::MappingCollection<dim>                mapping;
    hp::QCollection<dim>                      quadrature;
    hp::QCollection<dim - 1>                  face_quadrature;
    DoFHandler<dim>                           dof_handler;

    IndexSet                        locally_owned_dofs, locally_relevant_dofs;
    LA::MPI::SparseMatrix           system_matrix;
    std::vector<FullMatrix<double>> inverse_mass; // per active cell index
    LA::MPI::Vector                 solution;     // owned
    LA::MPI::Vector                 ghosted;      // locally relevant copy
    mutable LA::MPI::Vector         tmp;

    ExactSolution exact;

    std::map<std::string, double> times;
    template <typename F>
    void
    time(const std::string &phase, const F &f)
    {
      Timer timer;
      f();
      timer.stop();
      times[phase] += timer.wall_time();
    }
  };



  AdvectionProblem::AdvectionProblem(const Parameters &parameters)
    : parameters(parameters)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , triangulation(mpi_communicator)
    , dof_handler(triangulation)
    , exact(parameters.sigma)
  {
    const unsigned int p = parameters.degree;
    fe.push_back(FE_DGQ<dim>(p));
    fe.push_back(FE_SimplexDGP<dim>(p));
    mapping.push_back(MappingQ<dim>(1));
    mapping.push_back(MappingFE<dim>(FE_SimplexP<dim>(1)));
    quadrature.push_back(QGauss<dim>(p + 1));
    quadrature.push_back(QGaussSimplex<dim>(p + 1));
    face_quadrature.push_back(QGauss<dim - 1>(p + 1));
    face_quadrature.push_back(QGauss<dim - 1>(p + 1));
  }



  void
  AdvectionProblem::setup_system()
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_active_fe_index(fe_index(cell->reference_cell()));
    dof_handler.distribute_dofs(fe);
    locally_owned_dofs    = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);
    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
    solution.reinit(locally_owned_dofs, mpi_communicator);
    tmp.reinit(locally_owned_dofs, mpi_communicator);
    ghosted.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    assemble_operator();
  }



  // A u: cell terms int u beta . grad v, interior faces - int (beta . n) u_upwind [v],
  // outflow boundary - int (beta . n) u v; inflow values are zero. Each interior
  // face is assembled once: from the finer side at hanging faces, else from the
  // cell with the smaller id (or the smaller subdomain at process boundaries).
  void
  AdvectionProblem::assemble_operator()
  {
    this->time("assembly", [&]() {
      system_matrix = 0;
      inverse_mass.assign(triangulation.n_active_cells(), FullMatrix<double>());
      const UpdateFlags flags =
        update_values | update_gradients | update_quadrature_points | update_JxW_values;
      hp::FEValues<dim>     hp_fe_values(mapping, fe, quadrature, flags);
      hp::FEFaceValues<dim> hp_fe_face_values(mapping,
                                              fe,
                                              face_quadrature,
                                              update_values | update_quadrature_points |
                                                update_normal_vectors | update_JxW_values);
      FEInterfaceValues<dim> fe_iv(mapping,
                                   fe,
                                   face_quadrature,
                                   update_values | update_quadrature_points |
                                     update_normal_vectors | update_JxW_values);
      const AffineConstraints<double> no_constraints;

      FullMatrix<double>                   cell_matrix, cell_mass;
      std::vector<types::global_dof_index> dof_indices;

      const auto assemble_face = [&]() {
        const unsigned int n = fe_iv.n_current_interface_dofs();
        FullMatrix<double> face_matrix(n, n);
        const auto        &q_points = fe_iv.get_quadrature_points();
        const auto        &normals  = fe_iv.get_normal_vectors();
        for (unsigned int q = 0; q < q_points.size(); ++q)
          {
            const double beta_n = velocity(q_points[q]) * normals[q];
            for (unsigned int i = 0; i < n; ++i)
              for (unsigned int j = 0; j < n; ++j)
                face_matrix(i, j) -= fe_iv.jump_in_shape_values(i, q) * beta_n *
                                     fe_iv.shape_value(beta_n > 0, j, q) * fe_iv.JxW(q);
          }
        const std::vector<types::global_dof_index> indices = fe_iv.get_interface_dof_indices();
        no_constraints.distribute_local_to_global(face_matrix, indices, system_matrix);
      };

      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            hp_fe_values.reinit(cell);
            const FEValues<dim> &fv = hp_fe_values.get_present_fe_values();
            const unsigned int   n  = cell->get_fe().n_dofs_per_cell();
            cell_matrix.reinit(n, n);
            cell_mass.reinit(n, n);
            for (const unsigned int q : fv.quadrature_point_indices())
              {
                const Tensor<1, dim> beta = velocity(fv.quadrature_point(q));
                for (unsigned int i = 0; i < n; ++i)
                  for (unsigned int j = 0; j < n; ++j)
                    {
                      cell_matrix(i, j) +=
                        fv.shape_value(j, q) * (beta * fv.shape_grad(i, q)) * fv.JxW(q);
                      cell_mass(i, j) += fv.shape_value(i, q) * fv.shape_value(j, q) * fv.JxW(q);
                    }
              }
            // outflow boundary
            for (const unsigned int f : cell->face_indices())
              if (cell->at_boundary(f))
                {
                  hp_fe_face_values.reinit(cell, f);
                  const FEFaceValues<dim> &ffv = hp_fe_face_values.get_present_fe_values();
                  for (const unsigned int q : ffv.quadrature_point_indices())
                    {
                      const double beta_n =
                        velocity(ffv.quadrature_point(q)) * ffv.normal_vector(q);
                      if (beta_n > 0)
                        for (unsigned int i = 0; i < n; ++i)
                          for (unsigned int j = 0; j < n; ++j)
                            cell_matrix(i, j) -= beta_n * ffv.shape_value(i, q) *
                                                 ffv.shape_value(j, q) * ffv.JxW(q);
                    }
                }
            dof_indices.resize(n);
            cell->get_dof_indices(dof_indices);
            no_constraints.distribute_local_to_global(cell_matrix, dof_indices, system_matrix);
            cell_mass.gauss_jordan();
            inverse_mass[cell->active_cell_index()] = cell_mass;

            // interior faces
            for (const unsigned int f : cell->face_indices())
              {
                if (cell->at_boundary(f))
                  continue;
                const auto neighbor = cell->neighbor(f);
                if (neighbor->has_children())
                  continue; // assembled from the finer cells
                if (cell->neighbor_is_coarser(f))
                  {
                    const auto [nf, nsf] = cell->neighbor_of_coarser_neighbor(f);
                    fe_iv.reinit(cell, f, numbers::invalid_unsigned_int, neighbor, nf, nsf);
                    assemble_face();
                  }
                else
                  {
                    const bool mine = neighbor->is_locally_owned() ?
                                        cell->id() < neighbor->id() :
                                        cell->subdomain_id() < neighbor->subdomain_id();
                    if (!mine)
                      continue;
                    fe_iv.reinit(cell,
                                 f,
                                 numbers::invalid_unsigned_int,
                                 neighbor,
                                 cell->neighbor_of_neighbor(f),
                                 numbers::invalid_unsigned_int);
                    assemble_face();
                  }
              }
          }
      system_matrix.compress(VectorOperation::add);
    });
  }



  void
  AdvectionProblem::apply(const LA::MPI::Vector &u, LA::MPI::Vector &y) const
  {
    system_matrix.vmult(tmp, u);
    Vector<double>                       local_in, local_out;
    std::vector<types::global_dof_index> dof_indices;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const unsigned int n = cell->get_fe().n_dofs_per_cell();
          dof_indices.resize(n);
          cell->get_dof_indices(dof_indices);
          local_in.reinit(n);
          local_out.reinit(n);
          for (unsigned int i = 0; i < n; ++i)
            local_in(i) = tmp(dof_indices[i]);
          inverse_mass[cell->active_cell_index()].vmult(local_out, local_in);
          for (unsigned int i = 0; i < n; ++i)
            y(dof_indices[i]) = local_out(i);
        }
    y.compress(VectorOperation::insert);
  }



  void
  AdvectionProblem::time_step(const double dt)
  {
    this->time("time step", [&]() {
      LA::MPI::Vector k(locally_owned_dofs, mpi_communicator);
      if (parameters.scheme == "ssprk3")
        {
          // Shu-Osher form
          LA::MPI::Vector u1(solution), u2(solution);
          apply(solution, k);
          u1.add(dt, k);
          apply(u1, k);
          u2.equ(0.75, solution);
          u2.add(0.25, u1, 0.25 * dt, k);
          apply(u2, k);
          solution *= 1. / 3.;
          solution.add(2. / 3., u2, 2. / 3. * dt, k);
        }
      else
        {
          LA::MPI::Vector stage(solution), result(solution);
          const double    a[4] = {0, 0.5, 0.5, 1}, b[4] = {1. / 6, 1. / 3, 1. / 3, 1. / 6};
          for (unsigned int s = 0; s < 4; ++s)
            {
              if (s > 0)
                {
                  stage = solution;
                  stage.add(a[s] * dt, k);
                }
              apply(stage, k);
              result.add(b[s] * dt, k);
            }
          solution = result;
        }
    });
  }



  void
  AdvectionProblem::update_ghosted()
  {
    ghosted = solution;
  }



  void
  AdvectionProblem::adapt_mesh()
  {
    this->time("adapt", [&]() {
      update_ghosted();
      Vector<float> estimate(triangulation.n_active_cells());
      KellyErrorEstimator<dim>::estimate(mapping,
                                         dof_handler,
                                         face_quadrature,
                                         std::map<types::boundary_id, const Function<dim> *>(),
                                         ghosted,
                                         estimate);
      // the jump estimate scaled by h: cells are refined where the solution
      // varies, independent of the mesh size so far
      const double max_estimate =
        Utilities::MPI::max(estimate.size() ? *std::max_element(estimate.begin(), estimate.end()) : 0.f,
                            mpi_communicator);
      const unsigned int max_level = parameters.global + parameters.max_delta;
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const double e = estimate(cell->active_cell_index());
            if (e > parameters.refine * max_estimate && cell->level() < static_cast<int>(max_level))
              cell->set_refine_flag();
            else if (e < parameters.coarsen * max_estimate &&
                     cell->level() > static_cast<int>(parameters.global))
              cell->set_coarsen_flag();
          }

      SolutionTransfer<dim, LA::MPI::Vector> transfer(dof_handler);
      triangulation.prepare_coarsening_and_refinement();
      transfer.prepare_for_coarsening_and_refinement(ghosted);
      triangulation.execute_coarsening_and_refinement();
      // the parts of the refinement in the backend (t8code): adapt, balance, partition, copy
      {
        const auto &rt = triangulation.get_refinement_times();
        times["refine: adapt"] += rt.adapt;
        times["refine: balance"] += rt.balance;
        times["refine: partition"] += rt.partition;
        times["refine: copy"] += rt.copy;
      }
      setup_system();
      transfer.interpolate(solution);
    });
  }



  double
  AdvectionProblem::l2_error(const double time)
  {
    update_ghosted();
    exact.set_time(time);
    const unsigned int   p = parameters.degree;
    hp::QCollection<dim> error_quadrature(QGauss<dim>(p + 3),
                                          QGaussSimplex<dim>(std::min(p + 3, 5u)));
    Vector<float>        cell_errors(triangulation.n_active_cells());
    VectorTools::integrate_difference(
      mapping, dof_handler, ghosted, exact, cell_errors, error_quadrature, VectorTools::L2_norm);
    return VectorTools::compute_global_error(triangulation, cell_errors, VectorTools::L2_norm);
  }



  // the integral of the solution, conserved by the scheme (no inflow) and by the
  // solution transfer
  double
  AdvectionProblem::mass()
  {
    update_ghosted();
    hp::FEValues<dim> hp_fe_values(mapping, fe, quadrature, update_values | update_JxW_values);
    double              m = 0;
    std::vector<double> values;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          hp_fe_values.reinit(cell);
          const FEValues<dim> &fv = hp_fe_values.get_present_fe_values();
          values.resize(fv.n_quadrature_points);
          fv.get_function_values(ghosted, values);
          for (const unsigned int q : fv.quadrature_point_indices())
            m += values[q] * fv.JxW(q);
        }
    return Utilities::MPI::sum(m, mpi_communicator);
  }



  void
  AdvectionProblem::output_results(const unsigned int step, const double time) const
  {
    if (parameters.output.empty())
      return;
    Vector<float> subdomain(triangulation.n_active_cells());
    Vector<float> cell_type(triangulation.n_active_cells());
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          subdomain(cell->active_cell_index()) = triangulation.locally_owned_subdomain();
          cell_type(cell->active_cell_index()) = fe_index(cell->reference_cell());
        }
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(ghosted, "u");
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.add_data_vector(cell_type, "cell_type");
    data_out.build_patches(mapping, parameters.degree);
    DataOutBase::VtkFlags flags;
    flags.time = time;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record(parameters.output + "/", "advection", step, mpi_communicator, 4);
  }



  void
  AdvectionProblem::run()
  {
    pcout << "advection_hybrid: " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " processes, mesh " << parameters.mesh << ", n " << parameters.n << ", degree "
          << parameters.degree << ", scheme " << parameters.scheme << ", cfl " << parameters.cfl
          << ", adapt " << parameters.adapt << std::endl;

    make_mesh(triangulation, parameters.n, parameters.mesh);
    triangulation.refine_global(parameters.global);
    setup_system();

    const auto interpolate_initial = [&]() {
      exact.set_time(0.0);
      VectorTools::interpolate(mapping, dof_handler, exact, solution);
    };
    interpolate_initial();
    if (parameters.adapt)
      for (unsigned int l = 0; l < parameters.max_delta; ++l)
        {
          adapt_mesh();
          interpolate_initial();
        }

    // the time step from the smallest cell that the adaptation can create:
    // the shortest edge of the mesh after global refinement, halved per level
    double h_min = std::numeric_limits<double>::max();
    {
      Triangulation<dim> coarse;
      make_mesh(coarse, parameters.n, parameters.mesh);
      for (const auto &cell : coarse.active_cell_iterators())
        h_min = std::min(h_min, cell->minimum_vertex_distance());
    }
    h_min *= std::ldexp(1.0, -static_cast<int>(parameters.global + (parameters.adapt ? parameters.max_delta : 0)));
    const double beta_max = 2 * numbers::PI * std::sqrt(0.5);
    const unsigned int n_steps = static_cast<unsigned int>(
      std::ceil(parameters.t_end * beta_max * (2 * parameters.degree + 1) / (parameters.cfl * h_min)));
    const double dt = parameters.t_end / n_steps;
    pcout << "h_min " << h_min << ", " << n_steps << " steps of " << dt << std::endl;

    std::ofstream csv;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        csv.open(parameters.csv);
        csv << "step,time,cells,dofs,cells_min,cells_max,l2_error,mass,time_step,adapt,assembly,"
               "refine_adapt,refine_balance,refine_partition,refine_copy\n";
      }
    const double mass0 = mass();
    update_ghosted();
    output_results(0, 0.0);
    // the phase times summed over the output intervals (maxima over the processes per interval)
    std::map<std::string, double> total_times;
    double                        sum_cells  = 0;
    const double                  wall_start = MPI_Wtime();
    for (unsigned int step = 1; step <= n_steps; ++step)
      {
        const double t = step * dt;
        time_step(dt);
        sum_cells += triangulation.n_global_active_cells();
        if (step % parameters.output_every == 0 || step == n_steps)
          {
            const double error = l2_error(t);
            const double m     = mass();
            const auto   cells = Utilities::MPI::min_max_avg(
              1.0 * triangulation.n_locally_owned_active_cells(), mpi_communicator);
            std::map<std::string, double> t_max;
            for (const std::string phase : {"time step",
                                            "adapt",
                                            "assembly",
                                            "refine: adapt",
                                            "refine: balance",
                                            "refine: partition",
                                            "refine: copy"})
              {
                t_max[phase] = Utilities::MPI::max(times[phase], mpi_communicator);
                total_times[phase] += t_max[phase];
              }
            times.clear();
            if (csv.is_open())
              csv << step << ',' << t << ',' << triangulation.n_global_active_cells() << ','
                  << dof_handler.n_dofs() << ',' << cells.min << ',' << cells.max << ','
                  << std::setprecision(10) << error << ',' << m << ',' << std::setprecision(6)
                  << t_max["time step"] << ',' << t_max["adapt"] << ',' << t_max["assembly"]
                  << ',' << t_max["refine: adapt"] << ',' << t_max["refine: balance"] << ','
                  << t_max["refine: partition"] << ',' << t_max["refine: copy"] << '\n';
            pcout << "step " << step << ", t " << t << ": " << triangulation.n_global_active_cells()
                  << " cells, " << dof_handler.n_dofs() << " dofs, L2 error " << error
                  << ", mass change " << (m - mass0) / mass0 << std::endl;
            output_results(step, t);
          }
        if (parameters.adapt && step % parameters.adapt_every == 0 && step < n_steps)
          adapt_mesh();
      }
    const double wall = MPI_Wtime() - wall_start;
    pcout << "final L2 error " << l2_error(n_steps * dt) << ", mass change "
          << (mass() - mass0) / mass0 << std::endl;
    pcout << "summary: steps " << n_steps << ", sum cells " << sum_cells << ", wall time " << wall
          << " s (time steps " << total_times["time step"] << ", adapt " << total_times["adapt"]
          << ", assembly " << total_times["assembly"] << "; in adapt: refine adapt "
          << total_times["refine: adapt"] << ", balance " << total_times["refine: balance"]
          << ", partition " << total_times["refine: partition"] << ", copy "
          << total_times["refine: copy"] << ")" << std::endl;
  }
} // namespace AdvectionHybrid



int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      AdvectionHybrid::Parameters      parameters;
      parameters.parse(argc, argv);
      AdvectionHybrid::AdvectionProblem problem(parameters);
      problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << "Exception: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
