/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * ------------------------------------------------------------------------
 *
 * heat_hybrid with PETSc TS (as step-86): the heat equation u_t - Laplace u = f
 * with the moving Gaussian as exact solution, on a mesh of quadrilaterals and
 * triangles with parallel::distributed::Triangulation (t8code), adaptive
 * refinement and coarsening every few time steps, and the time integration by
 * PETScWrappers::TimeStepper: any PETSc TS type (beuler, cn, bdf, arkimex,
 * rosw, ...) with a fixed or an adaptive time step. PETSc options can also be
 * given on the command line (arguments starting with -, e.g. -ts_bdf_order 2).
 *
 * The residual R(t, y, y') = M y' + A y - F(t) and the Jacobian alpha M + A
 * are assembled with the constraints of time t (hanging nodes and Dirichlet
 * values); the constrained entries are treated as algebraic components.
 *
 * Usage: mpirun -np N ./heat_hybrid_ts [key=value ...] [-petsc_option ...]
 *   mesh=hybrid, n=4, degree=2, global=3, max_delta=3, adapt=1, adapt_every=5,
 *   refine=0.1, coarsen=0.01, sigma=0.1, radius=0.25: as heat_hybrid
 *   ts=bdf           PETSc TS type
 *   dt=0.005         (initial) time step
 *   t_end=1          final time
 *   ts_adapt=none    none or basic (adaptive time step with the tolerances)
 *   atol=1e-6        absolute tolerance of the adaptive time step
 *   rtol=1e-6        relative tolerance of the adaptive time step
 *   csv=heat_ts.csv  one line per time step
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

#include <deal.II/fe/fe_q.h>
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
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_ts.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace HeatHybridTS
{
  using namespace dealii;

  constexpr int dim = 2;
  using VectorType  = PETScWrappers::MPI::Vector;
  using MatrixType  = PETScWrappers::MPI::SparseMatrix;

  struct Parameters
  {
    std::string  mesh        = "hybrid";
    unsigned int n           = 4;
    unsigned int degree      = 2;
    unsigned int global      = 3;
    unsigned int max_delta   = 3;
    bool         adapt       = true;
    unsigned int adapt_every = 5;
    double       refine      = 0.1;
    double       coarsen     = 0.01;
    double       sigma       = 0.1;
    double       radius      = 0.25;
    std::string  ts          = "bdf";
    double       dt          = 0.005;
    double       t_end       = 1.0;
    std::string  ts_adapt    = "none";
    double       atol        = 1e-6;
    double       rtol        = 1e-6;
    std::string  csv         = "heat_ts.csv";

    void
    parse(int argc, char *argv[])
    {
      for (int i = 1; i < argc; ++i)
        {
          const std::string arg = argv[i];
          if (arg[0] == '-')
            {
              // a PETSc option and maybe its value
              if (i + 1 < argc && argv[i + 1][0] != '-' &&
                  std::string(argv[i + 1]).find('=') == std::string::npos)
                ++i;
              continue;
            }
          const auto eq = arg.find('=');
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
          else if (key == "sigma")
            sigma = std::stod(value);
          else if (key == "radius")
            radius = std::stod(value);
          else if (key == "ts")
            ts = value;
          else if (key == "dt")
            dt = std::stod(value);
          else if (key == "t_end")
            t_end = std::stod(value);
          else if (key == "ts_adapt")
            ts_adapt = value;
          else if (key == "atol")
            atol = std::stod(value);
          else if (key == "rtol")
            rtol = std::stod(value);
          else if (key == "csv")
            csv = value;
          else
            AssertThrow(false, ExcMessage("Unknown argument " + key));
        }
    }
  };



  // the moving Gaussian and its source (as in heat_hybrid)
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double sigma, const double radius)
      : sigma(sigma)
      , radius(radius)
    {}

    Point<dim>
    center() const
    {
      const double t = this->get_time();
      return Point<dim>(0.5 + radius * std::cos(2 * numbers::PI * t),
                        0.5 + radius * std::sin(2 * numbers::PI * t));
    }

    Tensor<1, dim>
    velocity() const
    {
      const double   t = this->get_time();
      Tensor<1, dim> v;
      v[0] = -2 * numbers::PI * radius * std::sin(2 * numbers::PI * t);
      v[1] = 2 * numbers::PI * radius * std::cos(2 * numbers::PI * t);
      return v;
    }

    double
    value(const Point<dim> &p, const unsigned int = 0) const override
    {
      return std::exp(-(p - center()).norm_square() / (sigma * sigma));
    }

    double
    source(const Point<dim> &p) const
    {
      const Tensor<1, dim> r   = p - center();
      const double         s2  = sigma * sigma;
      const double         g   = std::exp(-r.norm_square() / s2);
      const double         u_t = 2 * (r * velocity()) / s2 * g;
      const double lap = (4 * r.norm_square() / (s2 * s2) - 2. * dim / s2) * g;
      return u_t - lap;
    }

  private:
    const double sigma, radius;
  };



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



  class HeatEquation
  {
  public:
    HeatEquation(const Parameters &parameters);
    void
    run();

  private:
    unsigned int
    fe_index(const ReferenceCell &r) const
    {
      return r == ReferenceCells::Quadrilateral ? 0 : 1;
    }

    void
    setup_system(const double time);
    void
    update_current_constraints(const double time);
    void
    implicit_function(const double time, const VectorType &y, const VectorType &y_dot, VectorType &residual);
    void
    assemble_jacobian(const double alpha);
    void
    solve_with_jacobian(const VectorType &src, VectorType &dst);
    void
    mark_cells(const VectorType &y);
    void
    transfer(const std::vector<VectorType> &all_in, std::vector<VectorType> &all_out, const double time);
    double
    l2_error(const VectorType &y, const double time);

    const Parameters   parameters;
    MPI_Comm           mpi_communicator;
    ConditionalOStream pcout;

    parallel::distributed::Triangulation<dim> triangulation;
    hp::FECollection<dim>                     fe;
    hp::MappingCollection<dim>                mapping;
    hp::QCollection<dim>                      quadrature;
    hp::QCollection<dim - 1>                  face_quadrature;
    DoFHandler<dim>                           dof_handler;

    IndexSet                  locally_owned_dofs, locally_relevant_dofs;
    AffineConstraints<double> hanging_node_constraints, homogeneous_constraints, current_constraints;
    double                    constraints_time = -1;
    MatrixType                jacobian_matrix;

    ExactSolution exact;

    // statistics
    unsigned int                  n_solves = 0, n_iterations = 0;
    double                        sum_dof_iterations = 0;
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



  HeatEquation::HeatEquation(const Parameters &parameters)
    : parameters(parameters)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , triangulation(mpi_communicator)
    , dof_handler(triangulation)
    , exact(parameters.sigma, parameters.radius)
  {
    const unsigned int p = parameters.degree;
    fe.push_back(FE_Q<dim>(QIterated<1>(QTrapezoid<1>(), p)));
    fe.push_back(FE_SimplexP<dim>(p));
    mapping.push_back(MappingQ<dim>(1));
    mapping.push_back(MappingFE<dim>(FE_SimplexP<dim>(1)));
    quadrature.push_back(QGauss<dim>(p + 1));
    quadrature.push_back(QGaussSimplex<dim>(p + 1));
    face_quadrature.push_back(QGauss<dim - 1>(p + 1));
    face_quadrature.push_back(QGauss<dim - 1>(p + 1));
  }



  void
  HeatEquation::setup_system(const double time)
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_active_fe_index(fe_index(cell->reference_cell()));
    dof_handler.distribute_dofs(fe);
    locally_owned_dofs    = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    hanging_node_constraints.clear();
    hanging_node_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
    hanging_node_constraints.make_consistent_in_parallel(locally_owned_dofs,
                                                         locally_relevant_dofs,
                                                         mpi_communicator);
    hanging_node_constraints.close();

    homogeneous_constraints.clear();
    homogeneous_constraints.reinit(locally_owned_dofs, hanging_node_constraints.get_local_lines());
    homogeneous_constraints.merge(hanging_node_constraints);
    VectorTools::interpolate_boundary_values(
      mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), homogeneous_constraints);
    homogeneous_constraints.make_consistent_in_parallel(locally_owned_dofs,
                                                        hanging_node_constraints.get_local_lines(),
                                                        mpi_communicator);
    homogeneous_constraints.close();

    constraints_time = -1;
    update_current_constraints(time);

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, homogeneous_constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);
    jacobian_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  }



  void
  HeatEquation::update_current_constraints(const double time)
  {
    if (time == constraints_time)
      return;
    constraints_time = time;
    exact.set_time(time);
    current_constraints.clear();
    current_constraints.reinit(locally_owned_dofs, hanging_node_constraints.get_local_lines());
    current_constraints.merge(hanging_node_constraints);
    VectorTools::interpolate_boundary_values(mapping, dof_handler, 0, exact, current_constraints);
    current_constraints.make_consistent_in_parallel(locally_owned_dofs,
                                                    locally_relevant_dofs,
                                                    mpi_communicator);
    current_constraints.close();
  }



  void
  HeatEquation::implicit_function(const double      time,
                                  const VectorType &y,
                                  const VectorType &y_dot,
                                  VectorType       &residual)
  {
    this->time("residual", [&]() {
      VectorType tmp(locally_owned_dofs, mpi_communicator), tmp_dot(locally_owned_dofs, mpi_communicator);
      tmp     = y;
      tmp_dot = y_dot;
      update_current_constraints(time);
      current_constraints.distribute(tmp);
      homogeneous_constraints.distribute(tmp_dot);
      VectorType u(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
      VectorType u_dot(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
      u     = tmp;
      u_dot = tmp_dot;

      hp::FEValues<dim> hp_fe_values(mapping,
                                     fe,
                                     quadrature,
                                     update_values | update_gradients |
                                       update_quadrature_points | update_JxW_values);
      exact.set_time(time);
      std::vector<types::global_dof_index> dof_indices;
      std::vector<Tensor<1, dim>>          gradients;
      std::vector<double>                  dot_values;
      Vector<double>                       cell_residual;
      residual = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            hp_fe_values.reinit(cell);
            const FEValues<dim> &fv = hp_fe_values.get_present_fe_values();
            const unsigned int   n  = cell->get_fe().n_dofs_per_cell();
            gradients.resize(fv.n_quadrature_points);
            dot_values.resize(fv.n_quadrature_points);
            fv.get_function_gradients(u, gradients);
            fv.get_function_values(u_dot, dot_values);
            cell_residual.reinit(n);
            for (const unsigned int q : fv.quadrature_point_indices())
              {
                const double f = exact.source(fv.quadrature_point(q));
                for (unsigned int i = 0; i < n; ++i)
                  cell_residual(i) += (fv.shape_value(i, q) * (dot_values[q] - f) +
                                       fv.shape_grad(i, q) * gradients[q]) *
                                      fv.JxW(q);
              }
            dof_indices.resize(n);
            cell->get_dof_indices(dof_indices);
            current_constraints.distribute_local_to_global(cell_residual, dof_indices, residual);
          }
      residual.compress(VectorOperation::add);
      // the algebraic components (see step-86)
      for (const auto &c : current_constraints.get_lines())
        if (locally_owned_dofs.is_element(c.index))
          {
            if (c.entries.empty())
              residual[c.index] = y[c.index] - tmp[c.index];
            else
              residual[c.index] = y[c.index];
          }
      residual.compress(VectorOperation::insert);
    });
  }



  void
  HeatEquation::assemble_jacobian(const double alpha)
  {
    this->time("jacobian", [&]() {
      hp::FEValues<dim> hp_fe_values(mapping,
                                     fe,
                                     quadrature,
                                     update_values | update_gradients | update_JxW_values);
      std::vector<types::global_dof_index> dof_indices;
      FullMatrix<double>                   cell_matrix;
      jacobian_matrix = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            hp_fe_values.reinit(cell);
            const FEValues<dim> &fv = hp_fe_values.get_present_fe_values();
            const unsigned int   n  = cell->get_fe().n_dofs_per_cell();
            cell_matrix.reinit(n, n);
            for (const unsigned int q : fv.quadrature_point_indices())
              for (unsigned int i = 0; i < n; ++i)
                for (unsigned int j = 0; j < n; ++j)
                  cell_matrix(i, j) += (fv.shape_grad(i, q) * fv.shape_grad(j, q) +
                                        alpha * fv.shape_value(i, q) * fv.shape_value(j, q)) *
                                       fv.JxW(q);
            dof_indices.resize(n);
            cell->get_dof_indices(dof_indices);
            current_constraints.distribute_local_to_global(cell_matrix, dof_indices, jacobian_matrix);
          }
      jacobian_matrix.compress(VectorOperation::add);
      for (const auto &c : current_constraints.get_lines())
        if (locally_owned_dofs.is_element(c.index))
          jacobian_matrix.set(c.index, c.index, 1.0);
      jacobian_matrix.compress(VectorOperation::insert);
    });
  }



  void
  HeatEquation::solve_with_jacobian(const VectorType &src, VectorType &dst)
  {
    this->time("solve", [&]() {
      PETScWrappers::PreconditionBoomerAMG preconditioner;
      PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
      data.symmetric_operator = true;
      preconditioner.initialize(jacobian_matrix, data);
      SolverControl           solver_control(1000, 1e-10 * src.l2_norm());
      PETScWrappers::SolverCG cg(solver_control);
      cg.set_prefix("user_");
      cg.solve(jacobian_matrix, dst, src, preconditioner);
      ++n_solves;
      n_iterations += solver_control.last_step();
      sum_dof_iterations += 1.0 * dof_handler.n_dofs() * solver_control.last_step();
    });
  }



  void
  HeatEquation::mark_cells(const VectorType &y)
  {
    VectorType u(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    u = y;
    Vector<float> estimate(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(mapping,
                                       dof_handler,
                                       face_quadrature,
                                       std::map<types::boundary_id, const Function<dim> *>(),
                                       u,
                                       estimate);
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
  }



  void
  HeatEquation::transfer(const std::vector<VectorType> &all_in,
                         std::vector<VectorType>       &all_out,
                         const double                   time)
  {
    this->time("adapt", [&]() {
      SolutionTransfer<dim, VectorType> solution_transfer(dof_handler);
      std::vector<VectorType>           ghosted(all_in.size());
      std::vector<const VectorType *>   in_ptr(all_in.size());
      std::vector<VectorType *>         out_ptr(all_in.size());
      for (unsigned int i = 0; i < all_in.size(); ++i)
        {
          ghosted[i].reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
          ghosted[i] = all_in[i];
          in_ptr[i]  = &ghosted[i];
        }
      triangulation.prepare_coarsening_and_refinement();
      solution_transfer.prepare_for_coarsening_and_refinement(in_ptr);
      triangulation.execute_coarsening_and_refinement();
      const auto &rt = triangulation.get_refinement_times();
      times["refine: adapt"] += rt.adapt;
      times["refine: balance"] += rt.balance;
      times["refine: partition"] += rt.partition;
      times["refine: copy"] += rt.copy;
      setup_system(time);
      all_out.resize(all_in.size());
      for (unsigned int i = 0; i < all_in.size(); ++i)
        {
          all_out[i].reinit(locally_owned_dofs, mpi_communicator);
          out_ptr[i] = &all_out[i];
        }
      solution_transfer.interpolate(out_ptr);
      for (VectorType &v : all_out)
        hanging_node_constraints.distribute(v);
    });
  }



  double
  HeatEquation::l2_error(const VectorType &y, const double time)
  {
    VectorType u(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    u = y;
    exact.set_time(time);
    const unsigned int   p = parameters.degree;
    hp::QCollection<dim> error_quadrature(QGauss<dim>(p + 3), QGaussSimplex<dim>(std::min(p + 3, 5u)));
    Vector<float>        cell_errors(triangulation.n_active_cells());
    VectorTools::integrate_difference(
      mapping, dof_handler, u, exact, cell_errors, error_quadrature, VectorTools::L2_norm);
    return VectorTools::compute_global_error(triangulation, cell_errors, VectorTools::L2_norm);
  }



  void
  HeatEquation::run()
  {
    pcout << "heat_hybrid_ts: " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " processes, mesh " << parameters.mesh << ", degree " << parameters.degree << ", ts "
          << parameters.ts << ", dt " << parameters.dt << ", ts_adapt " << parameters.ts_adapt
          << ", adapt " << parameters.adapt << std::endl;

    make_mesh(triangulation, parameters.n, parameters.mesh);
    triangulation.refine_global(parameters.global);
    setup_system(0.0);

    VectorType solution(locally_owned_dofs, mpi_communicator);
    const auto interpolate_initial = [&]() {
      solution.reinit(locally_owned_dofs, mpi_communicator);
      exact.set_time(0.0);
      VectorTools::interpolate(mapping, dof_handler, exact, solution);
      hanging_node_constraints.distribute(solution);
    };
    interpolate_initial();
    if (parameters.adapt)
      for (unsigned int l = 0; l < parameters.max_delta; ++l)
        {
          mark_cells(solution);
          std::vector<VectorType> in(1, solution), out;
          transfer(in, out, 0.0);
          interpolate_initial();
        }
    times.clear();

    PETScWrappers::TimeStepperData data("",
                                        parameters.ts,
                                        0.0,
                                        parameters.t_end,
                                        parameters.dt,
                                        -1,
                                        /* match_step */ true,
                                        /* restart_if_remesh */ false,
                                        parameters.ts_adapt,
                                        -1.0,
                                        -1.0,
                                        parameters.atol,
                                        parameters.rtol,
                                        /* ignore_algebraic_lte */ true);
    PETScWrappers::TimeStepper<VectorType, MatrixType> ts(data);
    ts.set_matrices(jacobian_matrix, jacobian_matrix);
    ts.implicit_function = [&](const double t, const VectorType &y, const VectorType &y_dot, VectorType &r) {
      implicit_function(t, y, y_dot, r);
    };
    ts.setup_jacobian = [&](const double, const VectorType &, const VectorType &, const double alpha) {
      assemble_jacobian(alpha);
    };
    ts.solve_with_jacobian = [&](const VectorType &src, VectorType &dst) { solve_with_jacobian(src, dst); };
    // the algebraic components: boundary DoFs and hanging node DoFs. The latter
    // are taken from the hanging node constraints (DoFTools::
    // extract_hanging_node_dofs() returned invalid indices in parallel on
    // hybrid meshes); with HEAT_TS_CHECK, both are compared.
    ts.algebraic_components = [&]() {
      IndexSet algebraic(dof_handler.n_dofs());
      algebraic.add_indices(DoFTools::extract_boundary_dofs(dof_handler));
      IndexSet hanging(dof_handler.n_dofs());
      for (const auto &line : hanging_node_constraints.get_lines())
        hanging.add_index(line.index);
      hanging.compress();
      algebraic.add_indices(hanging);
      if (std::getenv("HEAT_TS_CHECK"))
        {
          const IndexSet tools = DoFTools::extract_hanging_node_dofs(dof_handler);
          types::global_dof_index max_tools = 0;
          for (const auto i : tools)
            max_tools = std::max(max_tools, i);
          // compare on the locally owned DoFs
          IndexSet a = hanging & locally_owned_dofs, b = tools & locally_owned_dofs;
          IndexSet only_constraints = a, only_tools = b;
          only_constraints.subtract_set(b);
          only_tools.subtract_set(a);
          std::cout << "  rank " << Utilities::MPI::this_mpi_process(mpi_communicator)
                    << ": hanging DoFs from constraints " << hanging.n_elements()
                    << ", from DoFTools " << tools.n_elements() << " (max " << max_tools << " of "
                    << dof_handler.n_dofs() << "); owned only in constraints "
                    << only_constraints.n_elements() << ", only in DoFTools " << only_tools.n_elements()
                    << std::endl;
        }
      return algebraic;
    };
    ts.update_constrained_components = [&](const double t, VectorType &y) {
      update_current_constraints(t);
      current_constraints.distribute(y);
    };
    ts.decide_and_prepare_for_remeshing = [&](const double, const unsigned int step, const VectorType &y) -> bool {
      if (parameters.adapt && step > 0 && step % parameters.adapt_every == 0)
        {
          mark_cells(y);
          return true;
        }
      return false;
    };
    ts.transfer_solution_vectors_to_new_mesh =
      [&](const double t, const std::vector<VectorType> &in, std::vector<VectorType> &out) {
        transfer(in, out, t);
      };

    std::ofstream csv;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        csv.open(parameters.csv);
        csv << "step,time,cells,dofs,l2_error\n";
      }
    double       max_error = 0, last_time = 0, last_error = 0, sum_cells = 0;
    unsigned int last_step = 0;
    ts.monitor = [&](const double t, const VectorType &y, const unsigned int step) {
      const double error = l2_error(y, t);
      max_error          = std::max(max_error, error);
      last_time = t, last_error = error, last_step = step;
      sum_cells += triangulation.n_global_active_cells();
      if (csv.is_open())
        csv << step << ',' << t << ',' << triangulation.n_global_active_cells() << ','
            << dof_handler.n_dofs() << ',' << std::setprecision(10) << error << std::setprecision(6)
            << '\n';
    };

    const double wall_start = MPI_Wtime();
    ts.solve(solution);
    const double wall = MPI_Wtime() - wall_start;
    std::map<std::string, double> t_max;
    for (const std::string phase : {"residual",
                                    "jacobian",
                                    "solve",
                                    "adapt",
                                    "refine: adapt",
                                    "refine: balance",
                                    "refine: partition",
                                    "refine: copy"})
      t_max[phase] = Utilities::MPI::max(times[phase], mpi_communicator);
    pcout << "final L2 error " << last_error << " at t " << last_time << ", max over the steps "
          << max_error << std::endl;
    pcout << "summary: steps " << last_step << ", sum cells " << sum_cells << ", solves " << n_solves
          << ", CG iterations " << n_iterations << ", sum dofs*cg_iterations " << sum_dof_iterations
          << ", wall time " << wall << " s (residual " << t_max["residual"] << ", jacobian "
          << t_max["jacobian"] << ", solve " << t_max["solve"] << ", adapt " << t_max["adapt"]
          << "; in adapt: refine adapt " << t_max["refine: adapt"] << ", balance "
          << t_max["refine: balance"] << ", partition " << t_max["refine: partition"] << ", copy "
          << t_max["refine: copy"] << ")" << std::endl;
  }
} // namespace HeatHybridTS



int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      HeatHybridTS::Parameters         parameters;
      parameters.parse(argc, argv);
      HeatHybridTS::HeatEquation heat(parameters);
      heat.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << "Exception: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
