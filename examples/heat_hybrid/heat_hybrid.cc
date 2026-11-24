/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * ------------------------------------------------------------------------
 *
 * The heat equation u_t - Laplace u = f on the unit square with a moving
 * source, on a hybrid mesh of quadrilaterals and triangles with
 * parallel::distributed::Triangulation (t8code backend), adaptive refinement
 * and coarsening every few time steps with SolutionTransfer.
 *
 * The exact solution is a Gaussian moving on a circle,
 *   u(x, t) = exp(-|x - c(t)|^2 / s^2),  c(t) = (0.5, 0.5) + R (cos 2 pi t, sin 2 pi t),
 * f = u_t - Laplace u and the Dirichlet values come from it, so that the
 * error in space and time can be measured.
 *
 * Time stepping: diagonally implicit Runge-Kutta methods for M u' = F - A u,
 * each stage
 *   (M + k a_ii A) U_i = M u^n + sum_{j<i} k a_ij (F(t_j) - A U_j) + k a_ii F(t_i),
 * assembled cell by cell with the hanging node and boundary constraints of the
 * stage time: the theta scheme (theta = 1 backward Euler, 1/2 Crank-Nicolson)
 * and the L-stable SDIRK2. Crank-Nicolson does not damp the high-frequency
 * errors that the solution transfer of every adaptation introduces; the error
 * estimator then keeps the trail of the source refined. The AMG preconditioner
 * is rebuilt only after a change of the mesh (the matrix does not change
 * otherwise).
 *
 * Usage: mpirun -np N ./heat_hybrid [key=value ...]
 *   mesh=hybrid       hybrid (squares alternating with pairs of triangles,
 *                     checkerboard), quad or tri, n x n squares
 *   n=4               coarse mesh size
 *   degree=2          polynomial degree
 *   global=3          global refinements
 *   max_delta=3       maximal number of adaptive levels above global
 *   adapt=1           0: no adaptation (fixed mesh, for the order in time)
 *   adapt_every=5     time steps between adaptations
 *   marking=threshold threshold: refine the cells whose estimate exceeds
 *                     refine times the largest estimate, coarsen those below
 *                     coarsen times it (bounded mesh size for a moving
 *                     feature); number: the given fractions of the cells with
 *                     the largest/smallest estimates; fraction: the cells that
 *                     hold the given fractions of the total estimate
 *   refine=0.1        threshold or fraction for refinement
 *   coarsen=0.01      threshold or fraction for coarsening
 *   scheme=theta      theta: the theta scheme; sdirk2: the two-stage SDIRK
 *                     method of Alexander (order 2, L-stable, stiffly accurate)
 *   theta=0.5         theta of the theta scheme
 *   dt=0.01           time step
 *   t_end=1           final time (one turn of the source)
 *   sigma=0.1         width of the Gaussian
 *   radius=0.25       radius of the circle of the source
 *   smoothing=none    mesh smoothing of the Triangulation: none (2:1 balance
 *                     across faces only) or step26 (smoothing_on_refinement |
 *                     smoothing_on_coarsening, which blocks much coarsening)
 *   csv=heat.csv      one line per time step
 *   output=           directory for VTU output, empty for none
 *   output_every=10   time steps between outputs
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
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA


namespace HeatHybrid
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
    unsigned int adapt_every  = 5;
    std::string  marking      = "threshold";
    double       refine       = 0.1;
    double       coarsen      = 0.01;
    std::string  scheme       = "theta";
    double       theta        = 0.5;
    double       dt           = 0.01;
    double       t_end        = 1.0;
    double       sigma        = 0.1;
    double       radius       = 0.25;
    std::string  smoothing    = "none";
    std::string  csv          = "heat.csv";
    std::string  output       = "";
    unsigned int output_every = 10;

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
          else if (key == "marking")
            marking = value;
          else if (key == "refine")
            refine = std::stod(value);
          else if (key == "coarsen")
            coarsen = std::stod(value);
          else if (key == "scheme")
            scheme = value;
          else if (key == "theta")
            theta = std::stod(value);
          else if (key == "dt")
            dt = std::stod(value);
          else if (key == "t_end")
            t_end = std::stod(value);
          else if (key == "sigma")
            sigma = std::stod(value);
          else if (key == "radius")
            radius = std::stod(value);
          else if (key == "smoothing")
            smoothing = value;
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
      AssertThrow(theta >= 0 && theta <= 1, ExcMessage("theta must be in [0,1]"));
      AssertThrow(scheme == "theta" || scheme == "sdirk2",
                  ExcMessage("scheme must be theta or sdirk2"));
      AssertThrow(marking == "threshold" || marking == "number" || marking == "fraction",
                  ExcMessage("marking must be threshold, number or fraction"));
    }
  };



  // the moving Gaussian and its source
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
      const double t = this->get_time();
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

    // f = u_t - Laplace u
    double
    source(const Point<dim> &p) const
    {
      const Tensor<1, dim> r  = p - center();
      const double         s2 = sigma * sigma;
      const double         g  = std::exp(-r.norm_square() / s2);
      const double         u_t = 2 * (r * velocity()) / s2 * g;
      const double lap = (4 * r.norm_square() / (s2 * s2) - 2. * dim / s2) * g;
      return u_t - lap;
    }

  private:
    const double sigma, radius;
  };



  // The unit square in n x n squares: quadrilaterals, pairs of triangles, or
  // both in a checkerboard pattern (hybrid).
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
          const bool quad = type == "quad" || (type == "hybrid" && (x + y) % 2 == 0);
          if (quad)
            {
              CellData<dim> cell(4);
              cell.vertices = {v00, v10, v01, v11}; // lexicographic
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
    setup_system();
    void
    make_constraints(const double time);
    void
    stage(const double                                                          diag,
          const double                                                          t_stage,
          const std::vector<std::tuple<double, const LA::MPI::Vector *, double>> &terms,
          LA::MPI::Vector                                                      &result);
    void
    time_step(const double time);
    void
    adapt_mesh();
    double
    l2_error(const double time);
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

    IndexSet                  locally_owned_dofs, locally_relevant_dofs;
    AffineConstraints<double> hanging_constraints, constraints;
    LA::MPI::SparseMatrix     system_matrix;
    LA::MPI::Vector           system_rhs;
    LA::MPI::Vector           solution, old_solution, stage_solution; // ghosted

    std::unique_ptr<LA::MPI::PreconditionAMG> preconditioner;
    unsigned int                              last_iterations = 0;

    ExactSolution exact;

    // the wall times of the phases since the last output line
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
    , triangulation(mpi_communicator,
                    parameters.smoothing == "step26" ?
                      Triangulation<dim>::MeshSmoothing(
                        Triangulation<dim>::smoothing_on_refinement |
                        Triangulation<dim>::smoothing_on_coarsening) :
                      Triangulation<dim>::none)
    , dof_handler(triangulation)
    , exact(parameters.sigma, parameters.radius)
  {
    const unsigned int p = parameters.degree;
    // equidistant FE_Q, so that the DoFs on the edges coincide with those of
    // FE_SimplexP
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
  HeatEquation::setup_system()
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_active_fe_index(fe_index(cell->reference_cell()));
    dof_handler.distribute_dofs(fe);
    locally_owned_dofs    = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    hanging_constraints.clear();
    hanging_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, hanging_constraints);
    hanging_constraints.close();

    // the sparsity pattern with the structure of the constraints of every
    // time step (hanging nodes and boundary values)
    make_constraints(0.0);
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);
    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    old_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    stage_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    preconditioner.reset();
  }



  void
  HeatEquation::make_constraints(const double time)
  {
    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    constraints.merge(hanging_constraints);
    exact.set_time(time);
    VectorTools::interpolate_boundary_values(mapping, dof_handler, 0, exact, constraints);
    constraints.close();
  }



  // One stage of a diagonally implicit Runge-Kutta method for M u' = F(t) - A u:
  //   (M + diag A) U = M u^n + sum_j c_j (F(t_j) - A U_j) + diag F(t_stage),
  // with the constraints (hanging nodes, Dirichlet values) of t_stage. The
  // result is written to `result` (ghosted). `terms` are (c_j, U_j, t_j) with
  // ghosted U_j. The matrix M + diag A is the same for all stages and steps of
  // a scheme with constant diag, so the AMG preconditioner is only built after
  // a change of the mesh.
  void
  HeatEquation::stage(const double                                                    diag,
                      const double                                                    t_stage,
                      const std::vector<std::tuple<double, const LA::MPI::Vector *, double>> &terms,
                      LA::MPI::Vector                                                &result)
  {
    this->time("assembly", [&]() {
      make_constraints(t_stage);
      system_matrix = 0;
      system_rhs    = 0;
      hp::FEValues<dim> hp_fe_values(mapping,
                                     fe,
                                     quadrature,
                                     update_values | update_gradients |
                                       update_quadrature_points | update_JxW_values);
      std::vector<ExactSolution> term_exact(terms.size(), exact);
      for (unsigned int j = 0; j < terms.size(); ++j)
        term_exact[j].set_time(std::get<2>(terms[j]));
      exact.set_time(t_stage);

      FullMatrix<double>                   cell_matrix, cell_mass, cell_laplace;
      Vector<double>                       cell_rhs, values, tmp;
      std::vector<types::global_dof_index> dof_indices;
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            hp_fe_values.reinit(cell);
            const FEValues<dim> &fv = hp_fe_values.get_present_fe_values();
            const unsigned int   n  = cell->get_fe().n_dofs_per_cell();
            cell_mass.reinit(n, n);
            cell_laplace.reinit(n, n);
            cell_matrix.reinit(n, n);
            cell_rhs.reinit(n);
            values.reinit(n);
            tmp.reinit(n);
            for (const unsigned int q : fv.quadrature_point_indices())
              {
                double f = diag * exact.source(fv.quadrature_point(q));
                for (unsigned int j = 0; j < terms.size(); ++j)
                  f += std::get<0>(terms[j]) * term_exact[j].source(fv.quadrature_point(q));
                for (unsigned int i = 0; i < n; ++i)
                  {
                    for (unsigned int j = 0; j < n; ++j)
                      {
                        cell_mass(i, j) +=
                          fv.shape_value(i, q) * fv.shape_value(j, q) * fv.JxW(q);
                        cell_laplace(i, j) +=
                          fv.shape_grad(i, q) * fv.shape_grad(j, q) * fv.JxW(q);
                      }
                    cell_rhs(i) += f * fv.shape_value(i, q) * fv.JxW(q);
                  }
              }
            cell_matrix.add(1.0, cell_mass, diag, cell_laplace);
            dof_indices.resize(n);
            cell->get_dof_indices(dof_indices);
            // M u^n
            for (unsigned int i = 0; i < n; ++i)
              values(i) = old_solution(dof_indices[i]);
            cell_mass.vmult(tmp, values);
            cell_rhs += tmp;
            // - c_j A U_j
            for (const auto &[c, U, t_j] : terms)
              {
                for (unsigned int i = 0; i < n; ++i)
                  values(i) = (*U)(dof_indices[i]);
                cell_laplace.vmult(tmp, values);
                cell_rhs.add(-c, tmp);
              }
            constraints.distribute_local_to_global(
              cell_matrix, cell_rhs, dof_indices, system_matrix, system_rhs);
          }
      system_matrix.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);
    });

    if (!preconditioner)
      this->time("amg setup", [&]() {
        preconditioner = std::make_unique<LA::MPI::PreconditionAMG>();
        LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
        data.symmetric_operator = true;
#endif
        preconditioner->initialize(system_matrix, data);
      });

    this->time("solve", [&]() {
      LA::MPI::Vector distributed(locally_owned_dofs, mpi_communicator);
      distributed = old_solution; // start from the old solution
      constraints.set_zero(distributed);
      SolverControl solver_control(1000, 1e-10 * system_rhs.l2_norm());
      LA::SolverCG  cg(solver_control);
      cg.solve(system_matrix, distributed, system_rhs, *preconditioner);
      constraints.distribute(distributed);
      result = distributed;
      last_iterations += solver_control.last_step();
    });
  }



  // one step from time - dt to time
  void
  HeatEquation::time_step(const double time)
  {
    const double k = parameters.dt;
    last_iterations = 0;
    if (parameters.scheme == "theta")
      {
        // (M + theta k A) u = M u^n + (1 - theta) k (F^n - A u^n) + theta k F^{n+1}
        const double theta = parameters.theta;
        stage(theta * k, time, {{(1 - theta) * k, &old_solution, time - k}}, solution);
      }
    else
      {
        // SDIRK2 (Alexander): gamma = 1 - 1/sqrt(2), c = (gamma, 1),
        // a = [[gamma, 0], [1 - gamma, gamma]], stiffly accurate and L-stable
        const double gamma = 1 - 1 / std::sqrt(2.);
        stage(gamma * k, time - k + gamma * k, {}, stage_solution);
        stage(gamma * k, time, {{(1 - gamma) * k, &stage_solution, time - k + gamma * k}}, solution);
      }
  }



  void
  HeatEquation::adapt_mesh()
  {
    this->time("adapt", [&]() {
      Vector<float> estimate(triangulation.n_active_cells());
      KellyErrorEstimator<dim>::estimate(mapping,
                                         dof_handler,
                                         face_quadrature,
                                         std::map<types::boundary_id, const Function<dim> *>(),
                                         solution,
                                         estimate);
      if (parameters.marking == "threshold")
        {
          const double max_estimate =
            Utilities::MPI::max(estimate.size() ? *std::max_element(estimate.begin(), estimate.end()) : 0.f,
                                mpi_communicator);
          for (const auto &cell : triangulation.active_cell_iterators())
            if (cell->is_locally_owned())
              {
                const double e = estimate(cell->active_cell_index());
                if (e > parameters.refine * max_estimate)
                  cell->set_refine_flag();
                else if (e < parameters.coarsen * max_estimate)
                  cell->set_coarsen_flag();
              }
        }
      else if (parameters.marking == "number")
        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
          triangulation, estimate, parameters.refine, parameters.coarsen);
      else
        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
          triangulation, estimate, parameters.refine, parameters.coarsen);
      const unsigned int max_level = parameters.global + parameters.max_delta;
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (cell->level() >= static_cast<int>(max_level))
              cell->clear_refine_flag();
            if (cell->level() <= static_cast<int>(parameters.global))
              cell->clear_coarsen_flag();
          }

      // the flags before and after the smoothing of prepare_coarsening_and_refinement()
      const auto count_flags = [&]() {
        unsigned int n[2] = {0, 0};
        for (const auto &cell : triangulation.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              n[0] += cell->refine_flag_set() ? 1 : 0;
              n[1] += cell->coarsen_flag_set() ? 1 : 0;
            }
        return std::make_pair(Utilities::MPI::sum(n[0], mpi_communicator),
                              Utilities::MPI::sum(n[1], mpi_communicator));
      };
      const auto         flags_marked = count_flags();
      // families whose children are all active, locally owned and flagged for
      // coarsening (these can be coarsened), per reference cell of the parent
      unsigned int complete[2] = {0, 0};
      for (const auto &cell : triangulation.cell_iterators())
        if (cell->has_children())
          {
            bool all = true;
            for (unsigned int c = 0; c < cell->n_children(); ++c)
              all = all && cell->child(c)->is_active() && cell->child(c)->is_locally_owned() &&
                    cell->child(c)->coarsen_flag_set();
            if (all)
              ++complete[fe_index(cell->reference_cell())];
          }
      for (unsigned int &c : complete)
        c = Utilities::MPI::sum(c, mpi_communicator);
      const unsigned int cells_before = triangulation.n_global_active_cells();
      SolutionTransfer<dim, LA::MPI::Vector> transfer(dof_handler);
      triangulation.prepare_coarsening_and_refinement();
      const auto flags_prepared = count_flags();
      transfer.prepare_for_coarsening_and_refinement(solution);
      triangulation.execute_coarsening_and_refinement();
      if (std::getenv("HEAT_ADAPT_LOG"))
        pcout << "  adapt: " << cells_before << " -> " << triangulation.n_global_active_cells()
              << " cells; flags refine/coarsen marked " << flags_marked.first << "/"
              << flags_marked.second << ", after smoothing " << flags_prepared.first << "/"
              << flags_prepared.second << "; complete families (quad/tri) " << complete[0] << "/"
              << complete[1] << std::endl;
      setup_system();
      LA::MPI::Vector transferred(locally_owned_dofs, mpi_communicator);
      transfer.interpolate(transferred);
      hanging_constraints.distribute(transferred);
      solution = transferred;
    });
  }



  double
  HeatEquation::l2_error(const double time)
  {
    exact.set_time(time);
    const unsigned int   p = parameters.degree;
    hp::QCollection<dim> error_quadrature(QGauss<dim>(p + 3),
                                          QGaussSimplex<dim>(std::min(p + 3, 5u)));
    Vector<float>        cell_errors(triangulation.n_active_cells());
    VectorTools::integrate_difference(
      mapping, dof_handler, solution, exact, cell_errors, error_quadrature, VectorTools::L2_norm);
    return VectorTools::compute_global_error(triangulation, cell_errors, VectorTools::L2_norm);
  }



  void
  HeatEquation::output_results(const unsigned int step, const double time) const
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
    data_out.add_data_vector(solution, "u");
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.add_data_vector(cell_type, "cell_type");
    data_out.build_patches(mapping, parameters.degree);
    DataOutBase::VtkFlags flags;
    flags.time = time;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record(parameters.output + "/", "heat", step, mpi_communicator, 4);
  }



  void
  HeatEquation::run()
  {
    pcout << "heat_hybrid: " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " processes, mesh " << parameters.mesh << ", n " << parameters.n << ", degree "
          << parameters.degree << ", scheme " << parameters.scheme << ", theta " << parameters.theta << ", dt " << parameters.dt
          << ", adapt " << parameters.adapt << std::endl;

    make_mesh(triangulation, parameters.n, parameters.mesh);
    triangulation.refine_global(parameters.global);
    setup_system();

    // the initial value, and as many adaptations to it as there are levels
    const auto interpolate_initial = [&]() {
      LA::MPI::Vector distributed(locally_owned_dofs, mpi_communicator);
      exact.set_time(0.0);
      VectorTools::interpolate(mapping, dof_handler, exact, distributed);
      hanging_constraints.distribute(distributed);
      solution = distributed;
    };
    interpolate_initial();
    if (parameters.adapt)
      for (unsigned int l = 0; l < parameters.max_delta; ++l)
        {
          adapt_mesh();
          interpolate_initial();
        }

    std::ofstream csv;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        csv.open(parameters.csv);
        csv << "step,time,cells,dofs,cells_min,cells_max,cg_iterations,l2_error,"
               "assembly,amg_setup,solve,adapt\n";
      }

    const unsigned int n_steps =
      static_cast<unsigned int>(std::round(parameters.t_end / parameters.dt));
    double max_error = 0;
    output_results(0, 0.0);
    for (unsigned int step = 1; step <= n_steps; ++step)
      {
        const double t = step * parameters.dt;
        old_solution   = solution;
        time_step(t);
        const double error = l2_error(t);
        max_error          = std::max(max_error, error);

        const unsigned int n_local = triangulation.n_locally_owned_active_cells();
        const auto         cells   = Utilities::MPI::min_max_avg(1.0 * n_local, mpi_communicator);
        std::map<std::string, double> t_max;
        for (const std::string phase : {"assembly", "amg setup", "solve", "adapt"})
          t_max[phase] = Utilities::MPI::max(times[phase], mpi_communicator);
        if (csv.is_open())
          csv << step << ',' << t << ',' << triangulation.n_global_active_cells() << ','
              << dof_handler.n_dofs() << ',' << cells.min << ',' << cells.max << ','
              << last_iterations << ',' << std::setprecision(10) << error << ','
              << t_max["assembly"] << ',' << t_max["amg setup"] << ',' << t_max["solve"] << ','
              << t_max["adapt"] << std::setprecision(6) << '\n';
        times.clear();
        if (step % parameters.output_every == 0 || step == n_steps)
          {
            pcout << "step " << step << ", t " << t << ": "
                  << triangulation.n_global_active_cells() << " cells, " << dof_handler.n_dofs()
                  << " dofs, " << last_iterations << " CG iterations, L2 error " << error
                  << std::endl;
            output_results(step, t);
          }

        if (parameters.adapt && step % parameters.adapt_every == 0 && step < n_steps)
          adapt_mesh();
      }
    pcout << "final L2 error " << l2_error(n_steps * parameters.dt) << ", max over the steps "
          << max_error << std::endl;
    // the centers and levels of the final cells (with HEAT_CELLS=prefix), for analysis
    if (const char *prefix = std::getenv("HEAT_CELLS"))
      {
        std::ofstream out(std::string(prefix) + "_r" +
                          std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) + ".txt");
        Vector<float> estimate(triangulation.n_active_cells());
        KellyErrorEstimator<dim>::estimate(mapping,
                                           dof_handler,
                                           face_quadrature,
                                           std::map<types::boundary_id, const Function<dim> *>(),
                                           solution,
                                           estimate);
        for (const auto &cell : triangulation.active_cell_iterators())
          if (cell->is_locally_owned())
            out << cell->center()[0] << ' ' << cell->center()[1] << ' ' << cell->level() << ' '
                << estimate(cell->active_cell_index()) << '\n';
      }
  }
} // namespace HeatHybrid



int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      HeatHybrid::Parameters           parameters;
      parameters.parse(argc, argv);
      HeatHybrid::HeatEquation heat(parameters);
      heat.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << "Exception: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
