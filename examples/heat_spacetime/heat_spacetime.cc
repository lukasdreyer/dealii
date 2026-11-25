/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * ------------------------------------------------------------------------
 *
 * Space-time finite elements for the heat equation u_t - Laplace u = f in 2+1
 * dimensions: the third coordinate of a hexahedral mesh is the time. The
 * time interval is cut into slabs; every slab is a
 * parallel::distributed::Triangulation<3> (t8code backend) that is refined
 * adaptively in space and time at once, so the local time step follows the
 * local mesh size.
 *
 * The problem is the one of heat_hybrid: a Gaussian moving on a circle,
 *   u(x, t) = exp(-|x - c(t)|^2 / s^2),  c(t) = (0.5, 0.5) + R (cos 2 pi t, sin 2 pi t),
 * on the unit square, f = u_t - Laplace u, Dirichlet values from u.
 *
 * Two discretizations (method=):
 *
 * stabilized  continuous FE_Q in space and time, tested with v + theta h_K v_t
 *             (Langer, Neumueller, Schafelner 2016):
 *               sum_K (u_t, v + theta h_K v_t)_K + (grad_x u, grad_x v)_K
 *                     - theta h_K (Laplace_x u, v_t)_K = sum_K (f, v + theta h_K v_t)_K,
 *             initial values strongly (Dirichlet on the bottom of the slab).
 *             Nonsymmetric: GMRES with AMG.
 *
 * fosls       first-order system least squares (Fuehrer, Karkulik 2021) with
 *             sigma = -grad_x u, u and sigma in FE_Q:
 *               min |u_t + div_x sigma - f|^2_Q + |sigma + grad_x u|^2_Q
 *                   + |u(t_n) - u_prev|^2_{bottom},
 *             initial values weakly. Symmetric positive definite: CG with AMG.
 *             The local contributions of the functional are a reliable and
 *             efficient error estimator (estimator=residual).
 *
 * dg          upwind discontinuous Galerkin (FE_DGQ) in space and time with the
 *             flux along a = (b, 1) (b = 0 except for problem=rotating) and
 *             SIPG for the diffusion on the faces with a spatial normal; the
 *             initial values (the trace of the previous slab at the quadrature
 *             points of the bottom) and the lateral values weakly. GMRES with
 *             AMG or with a block Gauss-Seidel sweep over the cells in upwind
 *             order (precondition=sweep): for pure transport the exact inverse
 *             on one process.
 *
 * Coupling of the slabs (bottom=): every slab has the same coarse mesh,
 * shifted in time. freeze: before the adaptive loop of a slab its bottom is
 * refined like the top of the previous slab (the levels of the cells above the
 * same point), and the cells at the bottom are never coarsened, so the traces
 * match and the transfer is exact. free: the slab starts from the global mesh
 * and the old trace is evaluated at the new points (interpolation for
 * stabilized, quadrature points for fosls), which loses what the new bottom
 * does not resolve.
 *
 * Every slab runs an adaptive loop: solve, estimate, mark, refine (and
 * coarsen); `cycles` times, then a last solve. Every solve starts from scratch,
 * so there is no solution transfer between the cycles.
 *
 * Usage: mpirun -np N ./heat_spacetime [key=value ...]
 *   method=stabilized stabilized, fosls or dg
 *   precondition=amg  amg; for dg also sweep (block Gauss-Seidel in upwind
 *                     order, couplings between processes dropped) or jacobi
 *                     (block Jacobi)
 *   penalty=2         dg: SIPG penalty sigma = penalty (p + 1)^2 / h
 *   problem=gaussian  gaussian (the moving source), smooth
 *                     (u = sin(pi x) sin(pi y) sin(pi t + 1), for the rates) or
 *                     pulses (several time scales: n_pulses Gaussians in space
 *                     and time, sum_k exp(-|x - x_k|^2 / s^2 - (t - t_k)^2 / tau^2),
 *                     t_k = (k + 1/2) / n_pulses, quiet in between) or
 *                     rotating (advection-diffusion u_t + b . grad u
 *                     - epsilon Laplace u = 0 with the solid body rotation
 *                     b = 2 pi (-(y - 1/2), x - 1/2) of advection_hybrid: the
 *                     Gaussian starts at (1/2 + radius, 1/2) and spreads,
 *                     u = s^2 / s(t)^2 exp(-|x - c(t)|^2 / s(t)^2),
 *                     s(t)^2 = s^2 + 4 epsilon t; stabilized only, the
 *                     stabilization is then along a = (b, 1):
 *                     v + tau v_a, v_a = v_t + b . grad_x v)
 *   n_pulses=8, pulse_sigma=0.05 (s), pulse_tau=0.02 (tau)
 *   epsilon=1         diffusion (problem=rotating only; 0: pure transport)
 *   n=4               coarse cells per space direction
 *   nt=1              coarse cells in time per slab
 *   slab=0.25         slab length in time (coarse cell: 1/n x 1/n x slab/nt)
 *   t_end=1           final time (one turn of the source)
 *   degree=1          polynomial degree (fosls: of u and of sigma)
 *   global=3          global refinements
 *   max_delta=3       maximal number of adaptive levels above global
 *   cycles=3          adaptations per slab (0: fixed mesh, for convergence)
 *   marking=fraction  threshold, number or fraction (as in heat_hybrid)
 *   refine=0.6        threshold or fraction for refinement
 *   coarsen=0.02      threshold or fraction for coarsening
 *   estimator=residual kelly (on u, time as a space direction) or residual
 *                     (fosls: the least-squares functional; stabilized: the
 *                     cell residual and the jumps of the normal derivative
 *                     across the faces with a spatial normal; dg: the cell
 *                     residual, the jumps of u and of epsilon d_n u)
 *   bottom=freeze     freeze or free (see above)
 *   theta=0.01        stabilization parameter of stabilized (theta h_K); the
 *                     error grows about linearly with theta (for degree 1
 *                     Laplace_x u_h = 0, so theta h_K (Laplace_x u, v_t) is not
 *                     approximated), and degree 2 needs it small for
 *                     coercivity (theta = 1 does not converge). Degree 1 had
 *                     the optimal L2 rate 2 only with theta = 0.001.
 *   tau=theta         theta (tau_K = theta h_K) or supg (from the Peclet
 *                     number of a = (b, 1) and epsilon, see stabilization())
 *   sigma=0.1         width of the Gaussian
 *   radius=0.25       radius of the circle of the source
 *   tol=1e-8          relative tolerance of the linear solver (1e-6 gave the
 *                     same errors as 1e-10 in the tests)
 *   amg_elliptic=1    ML settings for elliptic (1) or nonsymmetric (0) problems
 *   smoother_sweeps=2 smoothing sweeps of the AMG
 *   renumber=         empty or downstream (cells in upwind order along
 *                     a = (b, 1), by t for b = 0; for Gauss-Seidel/ILU)
 *   smoother=         ML smoother type (empty: deal.II's default, Chebyshev),
 *                     e.g. "ILU": 3.6x fewer GMRES iterations than Chebyshev on
 *                     a uniform mesh (163 vs 589 at h = 1/128), but on adaptive
 *                     meshes up to 9x more (4300 vs 495 at 700k DoFs). Also
 *                     "symmetric Gauss-Seidel"
 *   csv=spacetime.csv one line per slab and cycle
 *   output=           directory for VTU output of the final mesh of each slab
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <vector>


namespace HeatSpaceTime
{
  using namespace dealii;

  // x, y, t
  constexpr int dim = 3;

  // boundary ids of the colorized hyper rectangle
  // (lateral: 0 to 3)
  constexpr types::boundary_id bottom_id = 4, top_id = 5;

  using VectorType = TrilinosWrappers::MPI::Vector;
  using MatrixType = TrilinosWrappers::SparseMatrix;

  // problem=pulses: short pulses at different places and times, a solution
  // with several time scales (quiet in between):
  //   u = sum_k exp(-|x - x_k|^2 / s^2) exp(-(t - t_k)^2 / tau^2),
  // t_k = (k + 1/2) / K on [0, 1], x_k on a spiral around the center (golden
  // angle, radii 0.1, 0.2, 0.3, so at least 0.2 from the boundary).
  struct Pulses
  {
    unsigned int n   = 8;
    double       s   = 0.05;
    double       tau = 0.02;

    Point<2>
    center(const unsigned int k) const
    {
      const double angle = 2.399963229728653 * k, r = 0.1 + 0.1 * (k % 3);
      return Point<2>(0.5 + r * std::cos(angle), 0.5 + r * std::sin(angle));
    }

    // u, u_t, grad_x u and Laplace_x u at (x, t); pulses further than 8 widths
    // away (factor < exp(-64)) are skipped
    void
    evaluate(const Point<2> &x,
             const double    t,
             double         &u,
             double         &u_t,
             Tensor<1, 2>   &grad,
             double         &lap) const
    {
      u = u_t = lap = 0;
      grad          = Tensor<1, 2>();
      const double s2 = s * s, tau2 = tau * tau;
      for (unsigned int k = 0; k < n; ++k)
        {
          const double dt = t - (k + 0.5) / n;
          if (dt * dt > 64 * tau2)
            continue;
          const Tensor<1, 2> r  = x - center(k);
          const double       r2 = r.norm_square();
          if (r2 > 64 * s2)
            continue;
          const double g = std::exp(-r2 / s2 - dt * dt / tau2);
          u += g;
          u_t += -2 * dt / tau2 * g;
          grad += -2 / s2 * g * r;
          lap += (4 * r2 / (s2 * s2) - 4 / s2) * g;
        }
    }
  };



  struct Parameters
  {
    std::string  method    = "stabilized";
    std::string  problem   = "gaussian";
    Pulses       pulses;
    unsigned int n         = 4;
    unsigned int nt        = 1;
    double       slab      = 0.25;
    double       t_end     = 1.0;
    unsigned int degree    = 1;
    unsigned int global    = 3;
    unsigned int max_delta = 3;
    unsigned int cycles    = 3;
    std::string  marking   = "fraction";
    double       refine    = 0.6;
    double       coarsen   = 0.02;
    std::string  estimator = "residual";
    std::string  bottom    = "freeze";
    double       theta     = 0.01;
    std::string  tau       = "theta";
    double       epsilon   = 1.0;
    double       sigma     = 0.1;
    double       radius    = 0.25;
    double       tol       = 1e-8;
    bool         amg_elliptic    = true;
    unsigned int smoother_sweeps = 2;
    std::string  smoother        = "";
    std::string  renumber        = "";
    std::string  precondition    = "amg";
    double       penalty         = 2.0;
    std::string  csv       = "spacetime.csv";
    std::string  output    = "";

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
          if (key == "method")
            method = value;
          else if (key == "problem")
            problem = value;
          else if (key == "n_pulses")
            pulses.n = std::stoi(value);
          else if (key == "pulse_sigma")
            pulses.s = std::stod(value);
          else if (key == "pulse_tau")
            pulses.tau = std::stod(value);
          else if (key == "n")
            n = std::stoi(value);
          else if (key == "nt")
            nt = std::stoi(value);
          else if (key == "slab")
            slab = std::stod(value);
          else if (key == "t_end")
            t_end = std::stod(value);
          else if (key == "degree")
            degree = std::stoi(value);
          else if (key == "global")
            global = std::stoi(value);
          else if (key == "max_delta")
            max_delta = std::stoi(value);
          else if (key == "cycles")
            cycles = std::stoi(value);
          else if (key == "marking")
            marking = value;
          else if (key == "refine")
            refine = std::stod(value);
          else if (key == "coarsen")
            coarsen = std::stod(value);
          else if (key == "estimator")
            estimator = value;
          else if (key == "bottom")
            bottom = value;
          else if (key == "theta")
            theta = std::stod(value);
          else if (key == "tau")
            tau = value;
          else if (key == "epsilon")
            epsilon = std::stod(value);
          else if (key == "sigma")
            sigma = std::stod(value);
          else if (key == "radius")
            radius = std::stod(value);
          else if (key == "amg_elliptic")
            amg_elliptic = std::stoi(value) != 0;
          else if (key == "smoother_sweeps")
            smoother_sweeps = std::stoi(value);
          else if (key == "smoother")
            smoother = value;
          else if (key == "renumber")
            renumber = value;
          else if (key == "precondition")
            precondition = value;
          else if (key == "penalty")
            penalty = std::stod(value);
          else if (key == "tol")
            tol = std::stod(value);
          else if (key == "csv")
            csv = value;
          else if (key == "output")
            output = value;
          else
            AssertThrow(false, ExcMessage("Unknown argument " + key));
        }
      AssertThrow(method == "stabilized" || method == "fosls" || method == "dg",
                  ExcMessage("method must be stabilized, fosls or dg"));
      AssertThrow(precondition == "amg" || (method == "dg" && (precondition == "sweep" || precondition == "jacobi")),
                  ExcMessage("precondition must be amg, or sweep or jacobi for method=dg"));
      AssertThrow(problem == "gaussian" || problem == "smooth" || problem == "pulses" ||
                    problem == "rotating",
                  ExcMessage("problem must be gaussian, smooth, pulses or rotating"));
      AssertThrow(problem == "rotating" || epsilon == 1.0,
                  ExcMessage("epsilon != 1 only for problem=rotating"));
      AssertThrow(problem != "rotating" || method != "fosls",
                  ExcMessage("problem=rotating only for method=stabilized or dg"));
      AssertThrow(epsilon >= 0, ExcMessage("epsilon must be >= 0"));
      AssertThrow(tau == "theta" || tau == "supg", ExcMessage("tau must be theta or supg"));
      AssertThrow(marking == "threshold" || marking == "number" || marking == "fraction",
                  ExcMessage("marking must be threshold, number or fraction"));
      AssertThrow(estimator == "kelly" || estimator == "residual",
                  ExcMessage("estimator must be kelly or residual"));
      AssertThrow(bottom == "freeze" || bottom == "free",
                  ExcMessage("bottom must be freeze or free"));
      AssertThrow(slab > 0 && t_end > 0 && n > 0 && nt > 0, ExcMessage("bad sizes"));
    }
  };



  // The moving Gaussian on space-time points (x, y, t). Component 0 is u, the
  // others (fosls) sigma = -grad_x u.
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double       sigma,
                  const double       radius,
                  const unsigned int n_components,
                  const std::string &problem,
                  const Pulses      &pulses,
                  const double       epsilon = 1.0)
      : Function<dim>(n_components)
      , sigma(sigma)
      , radius(radius)
      , epsilon(epsilon)
      , smooth(problem == "smooth")
      , use_pulses(problem == "pulses")
      , rotating(problem == "rotating")
      , pulses(pulses)
    {}

    // the velocity b of u_t + b . grad_x u - epsilon Laplace_x u = f: the solid
    // body rotation 2 pi (-(y - 1/2), x - 1/2) for problem=rotating, else 0
    Tensor<1, 2>
    velocity(const Point<dim> &p) const
    {
      Tensor<1, 2> b;
      if (rotating)
        {
          b[0] = -2 * numbers::PI * (p[1] - 0.5);
          b[1] = 2 * numbers::PI * (p[0] - 0.5);
        }
      return b;
    }

    double
    diffusion() const
    {
      return epsilon;
    }

    // problem=rotating: the Gaussian of width sigma at (1/2 + R, 1/2) at t = 0,
    // rotated by b and spread by the diffusion (b commutes with the Laplacian):
    //   u = s^2 / s(t)^2 exp(-|x - c(t)|^2 / s(t)^2),  s(t)^2 = s^2 + 4 epsilon t,
    // f = 0
    double
    width2(const Point<dim> &p) const
    {
      return sigma * sigma + 4 * epsilon * p[2];
    }

    // problem=pulses: u, u_t, grad_x u, Laplace_x u
    void
    evaluate_pulses(const Point<dim> &p, double &u, double &u_t, Tensor<1, 2> &grad, double &lap) const
    {
      pulses.evaluate(Point<2>(p[0], p[1]), p[2], u, u_t, grad, lap);
    }

    // problem=smooth: u = sin(pi x) sin(pi y) sin(pi t + 1), for convergence
    // rates
    double
    u_smooth(const Point<dim> &p) const
    {
      return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]) *
             std::sin(numbers::PI * p[2] + 1);
    }

    Tensor<1, 2>
    r(const Point<dim> &p) const
    {
      const double t = p[2];
      Tensor<1, 2> d;
      d[0] = p[0] - 0.5 - radius * std::cos(2 * numbers::PI * t);
      d[1] = p[1] - 0.5 - radius * std::sin(2 * numbers::PI * t);
      return d;
    }

    double
    u(const Point<dim> &p) const
    {
      if (smooth)
        return u_smooth(p);
      if (use_pulses)
        {
          double       v, v_t, lap;
          Tensor<1, 2> grad;
          evaluate_pulses(p, v, v_t, grad, lap);
          return v;
        }
      if (rotating)
        {
          const double w2 = width2(p);
          return sigma * sigma / w2 * std::exp(-r(p).norm_square() / w2);
        }
      return std::exp(-r(p).norm_square() / (sigma * sigma));
    }

    Tensor<1, 2>
    grad_x(const Point<dim> &p) const
    {
      if (smooth)
        {
          const double pi = numbers::PI, s = std::sin(pi * p[2] + 1);
          Tensor<1, 2> g;
          g[0] = pi * std::cos(pi * p[0]) * std::sin(pi * p[1]) * s;
          g[1] = pi * std::sin(pi * p[0]) * std::cos(pi * p[1]) * s;
          return g;
        }
      if (use_pulses)
        {
          double       v, v_t, lap;
          Tensor<1, 2> grad;
          evaluate_pulses(p, v, v_t, grad, lap);
          return grad;
        }
      if (rotating)
        return -2. / width2(p) * u(p) * r(p);
      return -2. / (sigma * sigma) * u(p) * r(p);
    }

    // f = u_t + b . grad_x u - epsilon Laplace_x u
    double
    source(const Point<dim> &p) const
    {
      if (rotating)
        return 0;
      if (smooth)
        {
          const double pi = numbers::PI;
          return std::sin(pi * p[0]) * std::sin(pi * p[1]) *
                 (pi * std::cos(pi * p[2] + 1) + 2 * pi * pi * std::sin(pi * p[2] + 1));
        }
      if (use_pulses)
        {
          double       v, v_t, lap;
          Tensor<1, 2> grad;
          evaluate_pulses(p, v, v_t, grad, lap);
          return v_t - lap;
        }
      const double       t = p[2];
      const Tensor<1, 2> d = r(p);
      Tensor<1, 2>       v; // velocity of the center
      v[0]            = -2 * numbers::PI * radius * std::sin(2 * numbers::PI * t);
      v[1]            = 2 * numbers::PI * radius * std::cos(2 * numbers::PI * t);
      const double s2 = sigma * sigma;
      const double g  = u(p);
      const double u_t = 2 * (d * v) / s2 * g;
      const double lap = (4 * d.norm_square() / (s2 * s2) - 4. / s2) * g;
      return u_t - lap;
    }

    double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      if (component == 0)
        return u(p);
      return -grad_x(p)[component - 1];
    }

  private:
    const double sigma, radius, epsilon;
    const bool   smooth, use_pulses, rotating;
    const Pulses pulses;
  };



  // the solution of a slab, kept for the coupling to the next one
  struct Slab
  {
    std::unique_ptr<parallel::distributed::Triangulation<dim>> triangulation;
    std::unique_ptr<DoFHandler<dim>>                           dof_handler;
    VectorType                                                 solution; // ghosted
  };



  // dg: block Gauss-Seidel with the cells as blocks, one forward sweep in the
  // given order of the cells (upwind along a = (b, 1)) from a zero initial
  // guess, x_K = A_KK^{-1} (b_K - sum_{J before K} A_KJ x_J). For pure transport
  // and an upwind order it is the exact inverse on one process; the couplings
  // to other processes are dropped (block Jacobi between the processes), so
  // GMRES has to carry the information across process boundaries. jacobi:
  // block Jacobi (no off-diagonal blocks), for comparison.
  class BlockSweep
  {
  public:
    void
    initialize(const MatrixType                                        &matrix,
               const IndexSet                                          &owned,
               const std::vector<std::vector<types::global_dof_index>> &blocks,
               const bool                                               jacobi)
    {
      block_dofs.clear();
      inverses.clear();
      row_start.assign(1, 0);
      columns.clear();
      values.clear();
      for (const auto &dofs : blocks)
        {
          const unsigned int        n = dofs.size();
          std::vector<unsigned int> local(n);
          for (unsigned int i = 0; i < n; ++i)
            local[i] = owned.index_within_set(dofs[i]);
          FullMatrix<double> block(n, n);
          for (unsigned int i = 0; i < n; ++i)
            {
              for (auto it = matrix.begin(dofs[i]); it != matrix.end(dofs[i]); ++it)
                {
                  const types::global_dof_index col = it->column();
                  const auto pos = std::find(dofs.begin(), dofs.end(), col);
                  if (pos != dofs.end())
                    block(i, pos - dofs.begin()) = it->value();
                  else if (!jacobi && owned.is_element(col) && it->value() != 0)
                    {
                      columns.push_back(owned.index_within_set(col));
                      values.push_back(it->value());
                    }
                }
              row_start.push_back(columns.size());
            }
          block.gauss_jordan();
          inverses.push_back(std::move(block));
          block_dofs.push_back(std::move(local));
        }
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      dst = 0;
      double       *x = dst.begin();
      const double *b = src.begin();
      Vector<double> r, y;
      unsigned int   row = 0;
      for (unsigned int k = 0; k < block_dofs.size(); ++k)
        {
          const auto &dofs = block_dofs[k];
          r.reinit(dofs.size());
          y.reinit(dofs.size());
          for (unsigned int i = 0; i < dofs.size(); ++i, ++row)
            {
              double v = b[dofs[i]];
              for (unsigned int e = row_start[row]; e < row_start[row + 1]; ++e)
                v -= values[e] * x[columns[e]];
              r(i) = v;
            }
          inverses[k].vmult(y, r);
          for (unsigned int i = 0; i < dofs.size(); ++i)
            x[dofs[i]] = y(i);
        }
    }

  private:
    std::vector<std::vector<unsigned int>> block_dofs; // local indices
    std::vector<FullMatrix<double>>        inverses;
    // the couplings to the other blocks of this process, row by row in the
    // order of the blocks
    std::vector<unsigned int> row_start, columns;
    std::vector<double>       values;
  };



  class SpaceTimeHeat
  {
  public:
    SpaceTimeHeat(const Parameters &parameters);

    void
    run();

  private:
    bool
    fosls() const
    {
      return parameters.method == "fosls";
    }

    bool
    dg() const
    {
      return parameters.method == "dg";
    }

    // the stabilization parameter tau_K of the test function v + tau_K v_a,
    // v_a = v_t + b . grad_x v. theta: theta h_K. supg: the SUPG parameter for
    // the direction a = (b, 1) with the diffusion epsilon in space,
    //   tau = h / (2 |a|) (coth(Pe) - 1 / Pe),  Pe = |a| h / (2 epsilon),  h = h_K / p,
    // that is h / (2 |a|) for pure transport and about h^2 / (12 epsilon) in the
    // diffusive limit (the heat equation: close to theta = 0.001 at h = 1/128).
    double
    stabilization(const double h_K, const Tensor<1, 2> &b) const
    {
      if (parameters.tau == "theta")
        return parameters.theta * h_K;
      const double h   = h_K / parameters.degree;
      const double a   = std::sqrt(1 + b.norm_square());
      const double eps = exact.diffusion();
      if (eps == 0)
        return h / (2 * a);
      const double pe = a * h / (2 * eps);
      const double xi = pe < 1e-3 ? pe / 3 : (pe > 20 ? 1 - 1 / pe : 1 / std::tanh(pe) - 1 / pe);
      return h / (2 * a) * xi;
    }

    // renumber=downstream: the locally owned cells in an upwind order along
    // a = (b, 1) (a topological sort by the inflow faces, earliest t first)
    std::vector<typename DoFHandler<3>::active_cell_iterator>
    downstream_cells() const;

    void
    make_slab(const double t0, const double t1);
    void
    mirror_bottom_refinement();
    void
    setup_system();
    void
    compute_bottom_values();
    void
    assemble();
    void
    solve();
    void
    estimate(Vector<float> &estimate);
    void
    residual_estimate_stabilized(Vector<float> &estimate) const;
    void
    assemble_dg();
    void
    residual_estimate_dg(Vector<float> &estimate) const;
    void
    adapt(const Vector<float> &estimate);
    void
    compute_errors(double &l2_q2, double &h1x_q2, double &l2_top2) const;
    void
    output_results(const unsigned int slab) const;

    // the active cells of the current slab with a face at the bottom
    template <typename F>
    void
    for_bottom_faces(const F &f) const
    {
      for (const auto &cell : dof_handler->active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int face : cell->face_indices())
            if (cell->face(face)->at_boundary() && cell->face(face)->boundary_id() == bottom_id)
              f(cell, face);
    }

    const Parameters   parameters;
    MPI_Comm           mpi_communicator;
    ConditionalOStream pcout;

    FESystem<dim>  fe;
    MappingQ<dim>  mapping;
    ExactSolution  exact;
    unsigned int   slab_index = 0;
    double         t0 = 0, t1 = 0;

    std::unique_ptr<parallel::distributed::Triangulation<dim>> triangulation;
    std::unique_ptr<DoFHandler<dim>>                           dof_handler;
    Slab                                                       previous;

    IndexSet                  locally_owned_dofs, locally_relevant_dofs;
    AffineConstraints<double> constraints;
    MatrixType                system_matrix;
    VectorType                system_rhs;
    VectorType                solution; // ghosted

    // fosls: the initial values at the quadrature points of the bottom faces,
    // in the order of for_bottom_faces()
    std::vector<double> bottom_values;

    unsigned int last_iterations = 0;

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



  SpaceTimeHeat::SpaceTimeHeat(const Parameters &parameters)
    : parameters(parameters)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , fe(parameters.method == "dg" ? static_cast<const FiniteElement<dim> &>(FE_DGQ<dim>(parameters.degree)) :
                                      static_cast<const FiniteElement<dim> &>(FE_Q<dim>(parameters.degree)),
         parameters.method == "fosls" ? 3 : 1)
    , mapping(1)
    , exact(parameters.sigma,
            parameters.radius,
            parameters.method == "fosls" ? 3 : 1,
            parameters.problem,
            parameters.pulses,
            parameters.epsilon)
  {}



  void
  SpaceTimeHeat::make_slab(const double t0_, const double t1_)
  {
    t0            = t0_;
    t1            = t1_;
    triangulation = std::make_unique<parallel::distributed::Triangulation<dim>>(
      mpi_communicator, Triangulation<dim>::none);
    GridGenerator::subdivided_hyper_rectangle(*triangulation,
                                              {parameters.n, parameters.n, parameters.nt},
                                              Point<dim>(0, 0, t0),
                                              Point<dim>(1, 1, t1),
                                              /*colorize*/ true);
    triangulation->refine_global(parameters.global);
    dof_handler = std::make_unique<DoFHandler<dim>>(*triangulation);
  }



  // Refine the bottom of the new slab like the top of the previous one. Both
  // slabs have the same coarse mesh (shifted in time), so a new cell at the
  // bottom on level l has the footprint of the old cells at the top on level
  // l: refine it while the old cell above a point inside it is finer. The
  // point lies inside one quarter of the face, away from the edges of the old
  // cells.
  void
  SpaceTimeHeat::mirror_bottom_refinement()
  {
    for (unsigned int iteration = 0;; ++iteration)
      {
        std::vector<Point<dim>>                                    points;
        std::vector<typename Triangulation<dim>::active_cell_iterator> cells;
        for (const auto &cell : triangulation->active_cell_iterators())
          if (cell->is_locally_owned())
            for (const unsigned int face : cell->face_indices())
              if (cell->face(face)->at_boundary() &&
                  cell->face(face)->boundary_id() == bottom_id)
                {
                  const auto f = cell->face(face);
                  points.push_back(0.75 * f->vertex(0) + 0.25 * f->vertex(3));
                  cells.push_back(cell);
                }

        Utilities::MPI::RemotePointEvaluation<dim> rpe;
        rpe.reinit(points, *previous.triangulation, mapping);
        AssertThrow(rpe.all_points_found(),
                    ExcMessage("Points at the bottom not found in the previous slab"));
        const std::vector<double> found = rpe.evaluate_and_process<double>(
          [](const ArrayView<double> &values, const auto &cell_data) {
            for (const auto c : cell_data.cell_indices())
              {
                const auto view  = cell_data.get_data_view(c, values);
                const int  level = cell_data.get_active_cell_iterator(c)->level();
                for (double &v : view)
                  v = level;
              }
          });
        const auto &ptrs = rpe.get_point_ptrs();

        unsigned int n_flagged = 0;
        for (unsigned int i = 0; i < cells.size(); ++i)
          {
            int old_level = 0;
            for (unsigned int j = ptrs[i]; j < ptrs[i + 1]; ++j)
              old_level = std::max(old_level, static_cast<int>(found[j]));
            if (old_level > cells[i]->level() && !cells[i]->refine_flag_set())
              {
                cells[i]->set_refine_flag();
                ++n_flagged;
              }
          }
        if (Utilities::MPI::sum(n_flagged, mpi_communicator) == 0)
          break;
        AssertThrow(iteration < 64, ExcInternalError());
        triangulation->execute_coarsening_and_refinement();
      }
  }



  // The upwind order along a = (b, 1): cell K comes after its locally owned
  // neighbors across the faces where a . n_K < 0 (at the center of the face,
  // or of the subface at a hanging face, so that both sides see the same
  // sign). Kahn's algorithm, among the ready cells the earliest t first; a
  // cycle (possible for curved b on coarse cells) is broken by taking the
  // earliest remaining cell. For b = 0 this orders by t.
  std::vector<typename DoFHandler<3>::active_cell_iterator>
  SpaceTimeHeat::downstream_cells() const
  {
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
    std::vector<Iterator>     cells;
    std::vector<unsigned int> local(triangulation->n_active_cells(), numbers::invalid_unsigned_int);
    for (const auto &cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
        {
          local[cell->active_cell_index()] = cells.size();
          cells.push_back(cell);
        }
    const unsigned int                     n = cells.size();
    std::vector<std::vector<unsigned int>> downstream(n);
    std::vector<unsigned int>              n_upwind(n, 0);
    const auto                             inflow = [&](const unsigned int i,
                                    const Iterator    &other,
                                    const Point<dim>  &center,
                                    const unsigned int face) {
      if (!other->is_locally_owned())
        return;
      const Tensor<1, 2>   b = exact.velocity(center);
      const Tensor<1, dim> normal = GeometryInfo<dim>::unit_normal_vector[face]; // axis-aligned cells
      const double         an     = normal[2] + b[0] * normal[0] + b[1] * normal[1];
      if (an < -1e-12)
        {
          const unsigned int j = local[other->active_cell_index()];
          downstream[j].push_back(i);
          ++n_upwind[i];
        }
    };
    for (unsigned int i = 0; i < n; ++i)
      {
        const auto &cell = cells[i];
        for (const unsigned int f : cell->face_indices())
          {
            if (cell->at_boundary(f))
              continue;
            const auto neighbor = cell->neighbor(f);
            if (neighbor->has_children())
              for (unsigned int sf = 0; sf < cell->face(f)->n_children(); ++sf)
                inflow(i, cell->neighbor_child_on_subface(f, sf), cell->face(f)->child(sf)->center(), f);
            else
              inflow(i, neighbor, cell->face(f)->center(), f);
          }
      }

    const auto later = [&](const unsigned int a, const unsigned int b) {
      return cells[a]->center()[2] > cells[b]->center()[2];
    };
    std::priority_queue<unsigned int, std::vector<unsigned int>, decltype(later)> ready(later);
    for (unsigned int i = 0; i < n; ++i)
      if (n_upwind[i] == 0)
        ready.push(i);
    std::vector<unsigned int> by_t(n);
    std::iota(by_t.begin(), by_t.end(), 0u);
    std::sort(by_t.begin(), by_t.end(), [&](const unsigned int a, const unsigned int b) {
      return later(b, a);
    });
    std::vector<bool>     done(n, false);
    std::vector<Iterator> order;
    order.reserve(n);
    unsigned int next_by_t = 0, n_cycles = 0;
    while (order.size() < n)
      {
        unsigned int i;
        if (!ready.empty())
          {
            i = ready.top();
            ready.pop();
            if (done[i])
              continue;
          }
        else
          {
            while (done[by_t[next_by_t]])
              ++next_by_t;
            i = by_t[next_by_t];
            ++n_cycles;
          }
        done[i] = true;
        order.push_back(cells[i]);
        for (const unsigned int j : downstream[i])
          if (!done[j] && --n_upwind[j] == 0)
            ready.push(j);
      }
    const unsigned int total_cycles = Utilities::MPI::sum(n_cycles, mpi_communicator);
    if (total_cycles > 0)
      pcout << "  downstream order: " << total_cycles << " cycles broken" << std::endl;
    return order;
  }



  void
  SpaceTimeHeat::setup_system()
  {
    dof_handler->distribute_dofs(fe);
    // downstream: locally owned DoFs in upwind order along a = (b, 1), so that
    // Gauss-Seidel and ILU sweep along the transport (DoFRenumbering::downstream
    // is not implemented for parallel meshes and takes a constant direction;
    // cell_wise works in parallel)
    if (parameters.renumber == "downstream")
      DoFRenumbering::cell_wise(*dof_handler, downstream_cells());
    else
      AssertThrow(parameters.renumber.empty(), ExcMessage("renumber must be empty or downstream"));
    locally_owned_dofs    = dof_handler->locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(*dof_handler);

    const ComponentMask u_mask = fe.component_mask(FEValuesExtractors::Scalar(0));
    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    // dg: no constraints (boundary and initial values weakly)
    if (!dg())
      {
        DoFTools::make_hanging_node_constraints(*dof_handler, constraints);
        for (const types::boundary_id b : {0, 1, 2, 3})
          VectorTools::interpolate_boundary_values(mapping, *dof_handler, b, exact, constraints, u_mask);
      }

    // stabilized: the initial values strongly
    if (!fosls() && !dg())
      {
        if (slab_index == 0)
          VectorTools::interpolate_boundary_values(
            mapping, *dof_handler, bottom_id, exact, constraints, u_mask);
        else
          this->time("transfer", [&]() {
            // the DoFs at the bottom (on the unit face z = 0 of the cells),
            // evaluated in the previous slab
            const std::vector<Point<dim>> &unit_points = fe.get_unit_support_points();
            std::map<types::global_dof_index, Point<dim>> dof_points;
            std::vector<types::global_dof_index>           dof_indices(fe.n_dofs_per_cell());
            for_bottom_faces([&](const auto &cell, const unsigned int) {
              cell->get_dof_indices(dof_indices);
              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                if (std::abs(unit_points[i][2]) < 1e-12 && !constraints.is_constrained(dof_indices[i]))
                  dof_points[dof_indices[i]] =
                    mapping.transform_unit_to_real_cell(cell, unit_points[i]);
            });
            std::vector<Point<dim>> points;
            for (const auto &[dof, p] : dof_points)
              {
                (void)dof;
                points.push_back(p);
              }
            Utilities::MPI::RemotePointEvaluation<dim> rpe;
            const std::vector<double> values = VectorTools::point_values<1>(
              mapping, *previous.dof_handler, previous.solution, points, rpe);
            AssertThrow(rpe.all_points_found(), ExcMessage("Bottom DoFs not found in the previous slab"));
            unsigned int i = 0;
            for (const auto &[dof, p] : dof_points)
              {
                (void)p;
                constraints.add_constraint(dof, {}, values[i++]);
              }
          });
      }
    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    if (dg())
      DoFTools::make_flux_sparsity_pattern(*dof_handler, dsp);
    else
      DoFTools::make_sparsity_pattern(*dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);
    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    if (fosls() || dg())
      compute_bottom_values();
  }



  // fosls: the initial values at the quadrature points of the bottom faces:
  // the exact ones in the first slab, otherwise u of the previous slab
  void
  SpaceTimeHeat::compute_bottom_values()
  {
    this->time("transfer", [&]() {
      const QGauss<dim - 1>   face_quadrature(parameters.degree + 2);
      FEFaceValues<dim>       fe_face_values(mapping, fe, face_quadrature, update_quadrature_points);
      std::vector<Point<dim>> points;
      for_bottom_faces([&](const auto &cell, const unsigned int face) {
        fe_face_values.reinit(cell, face);
        for (const unsigned int q : fe_face_values.quadrature_point_indices())
          points.push_back(fe_face_values.quadrature_point(q));
      });
      if (slab_index == 0)
        {
          bottom_values.resize(points.size());
          for (unsigned int i = 0; i < points.size(); ++i)
            bottom_values[i] = exact.u(points[i]);
        }
      else
        {
          Utilities::MPI::RemotePointEvaluation<dim> rpe;
          bottom_values = VectorTools::point_values<1>(
            mapping, *previous.dof_handler, previous.solution, points, rpe, VectorTools::EvaluationFlags::avg, 0);
          AssertThrow(rpe.all_points_found(), ExcMessage("Bottom points not found in the previous slab"));
        }
    });
  }



  void
  SpaceTimeHeat::assemble()
  {
    if (dg())
      {
        this->time("assembly", [&]() { assemble_dg(); });
        return;
      }
    this->time("assembly", [&]() {
      system_matrix = 0;
      system_rhs    = 0;
      const QGauss<dim>     quadrature(parameters.degree + 2);
      const QGauss<dim - 1> face_quadrature(parameters.degree + 2);
      FEValues<dim>         fe_values(mapping,
                              fe,
                              quadrature,
                              update_values | update_gradients |
                                (parameters.degree > 1 ? update_hessians : update_default) |
                                update_quadrature_points | update_JxW_values);
      FEFaceValues<dim>     fe_face_values(mapping, fe, face_quadrature, update_values | update_JxW_values);
      const unsigned int    n = fe.n_dofs_per_cell();
      FullMatrix<double>    cell_matrix(n, n);
      Vector<double>        cell_rhs(n);
      std::vector<types::global_dof_index> dof_indices(n);

      // fosls: the shape functions in the operators of the functional,
      // L = u_t + div_x sigma and M = sigma + grad_x u
      std::vector<double>       L(n);
      std::vector<Tensor<1, 2>> M(n);
      // stabilized: test function, its time derivative, the Laplacian
      std::vector<double> w(n), v_t(n), lap(n);

      unsigned int bottom_q = 0; // position in bottom_values
      for (const auto &cell : dof_handler->active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell_matrix = 0;
            cell_rhs    = 0;
            const double h = std::cbrt(cell->measure());
            for (const unsigned int q : fe_values.quadrature_point_indices())
              {
                const double f   = exact.source(fe_values.quadrature_point(q));
                const double JxW = fe_values.JxW(q);
                if (fosls())
                  {
                    for (unsigned int i = 0; i < n; ++i)
                      {
                        const unsigned int   c    = fe.system_to_component_index(i).first;
                        const Tensor<1, dim> grad = fe_values.shape_grad_component(i, q, c);
                        const double         phi  = fe_values.shape_value_component(i, q, c);
                        M[i]                      = Tensor<1, 2>();
                        if (c == 0)
                          {
                            L[i]    = grad[2];
                            M[i][0] = grad[0];
                            M[i][1] = grad[1];
                          }
                        else
                          {
                            L[i]        = grad[c - 1];
                            M[i][c - 1] = phi;
                          }
                      }
                    for (unsigned int i = 0; i < n; ++i)
                      {
                        for (unsigned int j = 0; j < n; ++j)
                          cell_matrix(i, j) += (L[i] * L[j] + M[i] * M[j]) * JxW;
                        cell_rhs(i) += f * L[i] * JxW;
                      }
                  }
                else
                  {
                    // the derivative along a = (b, 1): v_a = v_t + b . grad_x v
                    const Tensor<1, 2> b   = exact.velocity(fe_values.quadrature_point(q));
                    const double       eps = exact.diffusion();
                    const double       tau = stabilization(h, b);
                    for (unsigned int i = 0; i < n; ++i)
                      {
                        const Tensor<1, dim> g = fe_values.shape_grad(i, q);
                        v_t[i] = g[2] + b[0] * g[0] + b[1] * g[1];
                        w[i]   = fe_values.shape_value(i, q) + tau * v_t[i];
                        if (parameters.degree > 1)
                          {
                            const Tensor<2, dim> hess = fe_values.shape_hessian(i, q);
                            lap[i]                    = hess[0][0] + hess[1][1];
                          }
                        else
                          lap[i] = 0;
                      }
                    for (unsigned int i = 0; i < n; ++i)
                      {
                        const Tensor<1, dim> gi = fe_values.shape_grad(i, q);
                        for (unsigned int j = 0; j < n; ++j)
                          {
                            const Tensor<1, dim> gj = fe_values.shape_grad(j, q);
                            cell_matrix(i, j) += (v_t[j] * w[i] + eps * (gi[0] * gj[0] + gi[1] * gj[1]) -
                                                  eps * tau * lap[j] * v_t[i]) *
                                                 JxW;
                          }
                        cell_rhs(i) += f * w[i] * JxW;
                      }
                  }
              }

            // fosls: |u - u_prev|^2 on the bottom
            if (fosls())
              for (const unsigned int face : cell->face_indices())
                if (cell->face(face)->at_boundary() && cell->face(face)->boundary_id() == bottom_id)
                  {
                    fe_face_values.reinit(cell, face);
                    for (const unsigned int q : fe_face_values.quadrature_point_indices())
                      {
                        const double g = bottom_values[bottom_q++];
                        for (unsigned int i = 0; i < n; ++i)
                          if (fe.system_to_component_index(i).first == 0)
                            {
                              const double ui = fe_face_values.shape_value_component(i, q, 0);
                              for (unsigned int j = 0; j < n; ++j)
                                if (fe.system_to_component_index(j).first == 0)
                                  cell_matrix(i, j) +=
                                    ui * fe_face_values.shape_value_component(j, q, 0) *
                                    fe_face_values.JxW(q);
                              cell_rhs(i) += g * ui * fe_face_values.JxW(q);
                            }
                      }
                  }

            cell->get_dof_indices(dof_indices);
            constraints.distribute_local_to_global(
              cell_matrix, cell_rhs, dof_indices, system_matrix, system_rhs);
          }
      AssertThrow(!fosls() || bottom_q == bottom_values.size(), ExcInternalError());
      system_matrix.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);
    });
  }



  // dg: upwind DG in space-time for u_t + b . grad_x u - epsilon Laplace_x u = f
  // with a = (b, 1) (div a = 0):
  //   sum_K [ -(u, a . grad v)_K + epsilon (grad_x u, grad_x v)_K ]
  //   + sum_F int_F (a . n) u^upwind [v]                        (transport)
  //   + SIPG in x on the faces with a spatial normal n_x:        (diffusion)
  //     epsilon int_F (-{grad_x u} . n_x [v] - {grad_x v} . n_x [u] + sigma / h [u][v]),
  // inflow data g weakly: the exact solution on the lateral boundary, the trace
  // of the previous slab (bottom_values) at the bottom; Nitsche on the lateral
  // boundary for the diffusion. Every process assembles the rows of its own
  // cells, so the interior faces are visited from both sides.
  void
  SpaceTimeHeat::assemble_dg()
  {
    system_matrix = 0;
    system_rhs    = 0;
    const unsigned int    p = parameters.degree;
    const QGauss<dim>     quadrature(p + 2);
    const QGauss<dim - 1> face_quadrature(p + 2); // as compute_bottom_values()
    const UpdateFlags     flags = update_values | update_gradients | update_quadrature_points | update_JxW_values;
    FEValues<dim>         fe_values(mapping, fe, quadrature, flags);
    FEFaceValues<dim>     face_this(mapping, fe, face_quadrature, flags | update_normal_vectors);
    FEFaceValues<dim>     face_other(mapping, fe, face_quadrature, flags);
    FESubfaceValues<dim>  subface_this(mapping, fe, face_quadrature, flags | update_normal_vectors);
    FESubfaceValues<dim>  subface_other(mapping, fe, face_quadrature, flags);
    const unsigned int    n       = fe.n_dofs_per_cell();
    const double          eps     = exact.diffusion();
    const double          penalty = parameters.penalty * (p + 1) * (p + 1);
    FullMatrix<double>    cell_matrix(n, n), coupling(n, n);
    Vector<double>        cell_rhs(n);
    std::vector<types::global_dof_index> dof_indices(n), neighbor_indices(n);

    const auto a_dot = [&](const Point<dim> &x, const Tensor<1, dim> &v) {
      const Tensor<1, 2> b = exact.velocity(x);
      return v[2] + b[0] * v[0] + b[1] * v[1];
    };
    const auto x_dot = [](const Tensor<1, dim> &u, const Tensor<1, dim> &v) {
      return u[0] * v[0] + u[1] * v[1];
    };

    // an interior face (or subface) seen from this cell: cell_matrix gets the
    // terms with u of this cell, coupling those with u of the other cell
    const auto interior = [&](const FEFaceValuesBase<dim> &here, const FEFaceValuesBase<dim> &there, const double h) {
      coupling = 0;
      for (const unsigned int q : here.quadrature_point_indices())
        {
          Assert(here.quadrature_point(q).distance(there.quadrature_point(q)) < 1e-10,
                 ExcMessage("The quadrature points of the two sides of a face differ"));
          const Tensor<1, dim> &normal  = here.normal_vector(q);
          const double          an      = a_dot(here.quadrature_point(q), normal);
          const bool            spatial = std::abs(normal[2]) < 1e-8;
          const double          JxW     = here.JxW(q);
          for (unsigned int i = 0; i < n; ++i)
            {
              const double          vi  = here.shape_value(i, q);
              const Tensor<1, dim> &gvi = here.shape_grad(i, q);
              for (unsigned int j = 0; j < n; ++j)
                {
                  const double uj = here.shape_value(j, q), wj = there.shape_value(j, q);
                  double       own   = an >= 0 ? an * uj * vi : 0;
                  double       other = an < 0 ? an * wj * vi : 0;
                  if (spatial && eps > 0)
                    {
                      const Tensor<1, dim> &guj = here.shape_grad(j, q), &gwj = there.shape_grad(j, q);
                      own += eps * (-0.5 * x_dot(guj, normal) * vi - 0.5 * x_dot(gvi, normal) * uj +
                                    penalty / h * uj * vi);
                      other += eps * (-0.5 * x_dot(gwj, normal) * vi + 0.5 * x_dot(gvi, normal) * wj -
                                      penalty / h * wj * vi);
                    }
                  cell_matrix(i, j) += own * JxW;
                  coupling(i, j) += other * JxW;
                }
            }
        }
    };

    unsigned int bottom_q = 0; // position in bottom_values
    for (const auto &cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(dof_indices);
          fe_values.reinit(cell);
          cell_matrix = 0;
          cell_rhs    = 0;
          const double h_cell = std::cbrt(cell->measure());
          for (const unsigned int q : fe_values.quadrature_point_indices())
            {
              const Point<dim> &x   = fe_values.quadrature_point(q);
              const double      f   = exact.source(x);
              const double      JxW = fe_values.JxW(q);
              for (unsigned int i = 0; i < n; ++i)
                {
                  const Tensor<1, dim> &gvi = fe_values.shape_grad(i, q);
                  const double          avi = a_dot(x, gvi);
                  for (unsigned int j = 0; j < n; ++j)
                    cell_matrix(i, j) +=
                      (-fe_values.shape_value(j, q) * avi + eps * x_dot(fe_values.shape_grad(j, q), gvi)) * JxW;
                  cell_rhs(i) += f * fe_values.shape_value(i, q) * JxW;
                }
            }

          for (const unsigned int f : cell->face_indices())
            {
              if (cell->at_boundary(f))
                {
                  face_this.reinit(cell, f);
                  const types::boundary_id id      = cell->face(f)->boundary_id();
                  const bool               lateral = id != bottom_id && id != top_id;
                  for (const unsigned int q : face_this.quadrature_point_indices())
                    {
                      const Point<dim>     &x      = face_this.quadrature_point(q);
                      const Tensor<1, dim> &normal = face_this.normal_vector(q);
                      const double          an     = a_dot(x, normal);
                      const double          JxW    = face_this.JxW(q);
                      const double          g      = id == bottom_id ? bottom_values[bottom_q++] : exact.u(x);
                      for (unsigned int i = 0; i < n; ++i)
                        {
                          const double          vi  = face_this.shape_value(i, q);
                          const Tensor<1, dim> &gvi = face_this.shape_grad(i, q);
                          if (an >= 0)
                            for (unsigned int j = 0; j < n; ++j)
                              cell_matrix(i, j) += an * face_this.shape_value(j, q) * vi * JxW;
                          else
                            cell_rhs(i) -= an * g * vi * JxW;
                          if (lateral && eps > 0)
                            {
                              // Nitsche: -eps (d_n u, v) - eps (d_n v, u - g) + eps sigma / h (u - g, v)
                              for (unsigned int j = 0; j < n; ++j)
                                {
                                  const double uj = face_this.shape_value(j, q);
                                  cell_matrix(i, j) += eps *
                                                       (-x_dot(face_this.shape_grad(j, q), normal) * vi -
                                                        x_dot(gvi, normal) * uj + penalty / h_cell * uj * vi) *
                                                       JxW;
                                }
                              cell_rhs(i) += eps * (-x_dot(gvi, normal) + penalty / h_cell * vi) * g * JxW;
                            }
                        }
                    }
                  continue;
                }
              const auto neighbor = cell->neighbor(f);
              if (neighbor->has_children())
                for (unsigned int sf = 0; sf < cell->face(f)->n_children(); ++sf)
                  {
                    const auto child = cell->neighbor_child_on_subface(f, sf);
                    subface_this.reinit(cell, f, sf);
                    face_other.reinit(child, GeometryInfo<dim>::opposite_face[f]);
                    interior(subface_this, face_other, std::min(h_cell, std::cbrt(child->measure())));
                    child->get_dof_indices(neighbor_indices);
                    constraints.distribute_local_to_global(coupling, dof_indices, neighbor_indices, system_matrix);
                  }
              else
                {
                  face_this.reinit(cell, f);
                  if (cell->neighbor_is_coarser(f))
                    {
                      const auto [nf, nsf] = cell->neighbor_of_coarser_neighbor(f);
                      subface_other.reinit(neighbor, nf, nsf);
                      interior(face_this, subface_other, std::min(h_cell, std::cbrt(neighbor->measure())));
                    }
                  else
                    {
                      face_other.reinit(neighbor, cell->neighbor_of_neighbor(f));
                      interior(face_this, face_other, std::min(h_cell, std::cbrt(neighbor->measure())));
                    }
                  neighbor->get_dof_indices(neighbor_indices);
                  constraints.distribute_local_to_global(coupling, dof_indices, neighbor_indices, system_matrix);
                }
            }
          constraints.distribute_local_to_global(cell_matrix, cell_rhs, dof_indices, system_matrix, system_rhs);
        }
    AssertThrow(bottom_q == bottom_values.size(), ExcInternalError());
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }



  // dg: eta_K^2 = h_K^2 |f - u_a + epsilon Laplace_x u_h|^2_K
  //   + sum over the faces of K: 1/2 h_F |[u_h]|^2_F (interior; on the
  //     boundary the full jump to the inflow data where a . n < 0)
  //   + 1/2 epsilon^2 h_F |[d_n u_h]|^2_F on the interior faces with a spatial normal
  void
  SpaceTimeHeat::residual_estimate_dg(Vector<float> &estimate) const
  {
    const QGauss<dim>                quadrature(parameters.degree + 2);
    const QGauss<dim - 1>            face_quadrature(parameters.degree + 2);
    const UpdateFlags                face_flags = update_values | update_gradients | update_normal_vectors |
                                   update_quadrature_points | update_JxW_values;
    FEValues<dim>                    fe_values(mapping,
                            fe,
                            quadrature,
                            update_gradients | update_hessians | update_quadrature_points | update_JxW_values);
    FEFaceValues<dim>                face_this(mapping, fe, face_quadrature, face_flags);
    FEFaceValues<dim>                face_other(mapping, fe, face_quadrature, face_flags);
    FESubfaceValues<dim>             subface_this(mapping, fe, face_quadrature, face_flags);
    FESubfaceValues<dim>             subface_other(mapping, fe, face_quadrature, face_flags);
    const FEValuesExtractors::Scalar u(0);
    std::vector<Tensor<1, dim>>      grads(quadrature.size()), g_this(face_quadrature.size()),
      g_other(face_quadrature.size());
    std::vector<double>         v_this(face_quadrature.size()), v_other(face_quadrature.size());
    std::vector<Tensor<2, dim>> hessians(quadrature.size());
    const double                eps = exact.diffusion();

    const auto jump = [&](const FEFaceValuesBase<dim> &here, const FEFaceValuesBase<dim> &there) {
      here[u].get_function_values(solution, v_this);
      there[u].get_function_values(solution, v_other);
      here[u].get_function_gradients(solution, g_this);
      there[u].get_function_gradients(solution, g_other);
      const bool spatial = std::abs(here.normal_vector(0)[2]) < 1e-8;
      double     j0 = 0, j1 = 0, area = 0;
      for (const unsigned int q : here.quadrature_point_indices())
        {
          const double d  = v_this[q] - v_other[q];
          const double dn = (g_this[q] - g_other[q]) * here.normal_vector(q);
          j0 += d * d * here.JxW(q);
          if (spatial)
            j1 += dn * dn * here.JxW(q);
          area += here.JxW(q);
        }
      return 0.5 * std::sqrt(area) * (j0 + eps * eps * j1);
    };

    unsigned int bottom_q = 0;
    for (const auto &cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[u].get_function_gradients(solution, grads);
          if (parameters.degree > 1)
            fe_values[u].get_function_hessians(solution, hessians);
          const double h    = std::cbrt(cell->measure());
          double       eta2 = 0;
          for (const unsigned int q : fe_values.quadrature_point_indices())
            {
              const Point<dim>  &x = fe_values.quadrature_point(q);
              const Tensor<1, 2> b = exact.velocity(x);
              double             r = exact.source(x) - grads[q][2] - b[0] * grads[q][0] - b[1] * grads[q][1];
              if (parameters.degree > 1)
                r += eps * (hessians[q][0][0] + hessians[q][1][1]);
              eta2 += h * h * r * r * fe_values.JxW(q);
            }
          for (const unsigned int f : cell->face_indices())
            {
              if (cell->at_boundary(f))
                {
                  // the jump to the inflow data
                  face_this.reinit(cell, f);
                  face_this[u].get_function_values(solution, v_this);
                  const types::boundary_id id = cell->face(f)->boundary_id();
                  double                   j0 = 0, area = 0;
                  for (const unsigned int q : face_this.quadrature_point_indices())
                    {
                      const Point<dim>     &x      = face_this.quadrature_point(q);
                      const Tensor<1, 2>    b      = exact.velocity(x);
                      const Tensor<1, dim> &normal = face_this.normal_vector(q);
                      const double          g      = id == bottom_id ? bottom_values[bottom_q++] : exact.u(x);
                      if (normal[2] + b[0] * normal[0] + b[1] * normal[1] < 0)
                        j0 += (v_this[q] - g) * (v_this[q] - g) * face_this.JxW(q);
                      area += face_this.JxW(q);
                    }
                  eta2 += std::sqrt(area) * j0;
                  continue;
                }
              const auto neighbor = cell->neighbor(f);
              if (neighbor->has_children())
                for (unsigned int sf = 0; sf < cell->face(f)->n_children(); ++sf)
                  {
                    subface_this.reinit(cell, f, sf);
                    face_other.reinit(cell->neighbor_child_on_subface(f, sf), GeometryInfo<dim>::opposite_face[f]);
                    eta2 += jump(subface_this, face_other);
                  }
              else if (cell->neighbor_is_coarser(f))
                {
                  const auto [nf, nsf] = cell->neighbor_of_coarser_neighbor(f);
                  face_this.reinit(cell, f);
                  subface_other.reinit(neighbor, nf, nsf);
                  eta2 += jump(face_this, subface_other);
                }
              else
                {
                  face_this.reinit(cell, f);
                  face_other.reinit(neighbor, cell->neighbor_of_neighbor(f));
                  eta2 += jump(face_this, face_other);
                }
            }
          estimate(cell->active_cell_index()) = std::sqrt(eta2);
        }
  }



  void
  SpaceTimeHeat::solve()
  {
    TrilinosWrappers::PreconditionAMG preconditioner;
    BlockSweep                        sweep;
    const bool                        use_sweep = parameters.precondition != "amg";
    this->time("amg setup", [&]() {
      if (use_sweep)
        {
          // dg: the cells (blocks) in upwind order along a = (b, 1)
          const auto cells = downstream_cells();
          std::vector<std::vector<types::global_dof_index>> blocks(cells.size());
          for (unsigned int k = 0; k < cells.size(); ++k)
            {
              blocks[k].resize(fe.n_dofs_per_cell());
              cells[k]->get_dof_indices(blocks[k]);
            }
          sweep.initialize(system_matrix, locally_owned_dofs, blocks, parameters.precondition == "jacobi");
          return;
        }
      TrilinosWrappers::PreconditionAMG::AdditionalData data;
      data.elliptic              = parameters.amg_elliptic;
      data.higher_order_elements = parameters.degree > 1;
      data.smoother_sweeps       = parameters.smoother_sweeps;
      if (!parameters.smoother.empty())
        data.smoother_type = parameters.smoother.c_str();
      if (fosls())
        DoFTools::extract_constant_modes(*dof_handler, ComponentMask(3, true), data.constant_modes);
      preconditioner.initialize(system_matrix, data);
    });

    this->time("solve", [&]() {
      VectorType distributed(locally_owned_dofs, mpi_communicator);
      SolverControl control(5000, parameters.tol * system_rhs.l2_norm());
      if (fosls())
        {
          SolverCG<VectorType> cg(control);
          cg.solve(system_matrix, distributed, system_rhs, preconditioner);
        }
      else
        {
          // right preconditioning: the stopping criterion is the true residual
          // (left preconditioning stopped on the preconditioned residual and
          // accepted wrong solutions with bad preconditioners)
          SolverGMRES<VectorType> gmres(control,
                                        SolverGMRES<VectorType>::AdditionalData(100, true));
          if (use_sweep)
            gmres.solve(system_matrix, distributed, system_rhs, sweep);
          else
            gmres.solve(system_matrix, distributed, system_rhs, preconditioner);
        }
      constraints.distribute(distributed);
      solution        = distributed;
      last_iterations = control.last_step();
    });
  }



  // stabilized: a residual estimator for the space-time mesh,
  //   eta_K^2 = h_K^2 |f - u_t + Laplace_x u_h|^2_K
  //             + 1/2 sum over the faces of K with a spatial normal h_F |[d_n u_h]|^2_F.
  // Unlike Kelly it has no jumps of u_t across the faces with normal in time:
  // the equation has no flux in time. The meshes are axis-parallel, so every
  // face has a spatial or a temporal normal, and every cell has the standard
  // orientation, so the quadrature points of both sides coincide (checked).
  void
  SpaceTimeHeat::residual_estimate_stabilized(Vector<float> &estimate) const
  {
    const QGauss<dim>          quadrature(parameters.degree + 2);
    const QGauss<dim - 1>      face_quadrature(parameters.degree + 2);
    const UpdateFlags          face_flags = update_gradients | update_normal_vectors |
                                   update_quadrature_points | update_JxW_values;
    FEValues<dim>              fe_values(mapping,
                            fe,
                            quadrature,
                            update_gradients | update_hessians | update_quadrature_points |
                              update_JxW_values);
    FEFaceValues<dim>          face_this(mapping, fe, face_quadrature, face_flags);
    FEFaceValues<dim>          face_other(mapping, fe, face_quadrature, face_flags);
    FESubfaceValues<dim>       subface_this(mapping, fe, face_quadrature, face_flags);
    FESubfaceValues<dim>       subface_other(mapping, fe, face_quadrature, face_flags);
    const FEValuesExtractors::Scalar u(0);
    std::vector<Tensor<1, dim>> grads(quadrature.size()), g_this(face_quadrature.size()),
      g_other(face_quadrature.size());
    std::vector<Tensor<2, dim>> hessians(quadrature.size());

    // 1/2 h_F |[d_n u]|^2 on one face (or part of a face) seen from this side
    const auto jump = [&](const FEFaceValuesBase<dim> &here, const FEFaceValuesBase<dim> &there) {
      if (std::abs(here.normal_vector(0)[2]) > 1e-8)
        return 0.; // normal in time
      here[u].get_function_gradients(solution, g_this);
      there[u].get_function_gradients(solution, g_other);
      double j2 = 0, area = 0;
      for (const unsigned int q : here.quadrature_point_indices())
        {
          Assert(here.quadrature_point(q).distance(there.quadrature_point(q)) < 1e-10,
                 ExcMessage("The quadrature points of the two sides of a face differ"));
          const double d = (g_this[q] - g_other[q]) * here.normal_vector(q);
          j2 += d * d * here.JxW(q);
          area += here.JxW(q);
        }
      // the jump of the diffusive flux epsilon d_n u
      const double eps = exact.diffusion();
      return 0.5 * std::sqrt(area) * eps * eps * j2;
    };

    for (const auto &cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[u].get_function_gradients(solution, grads);
          if (parameters.degree > 1)
            fe_values[u].get_function_hessians(solution, hessians);
          const double h    = std::cbrt(cell->measure());
          double       eta2 = 0;
          for (const unsigned int q : fe_values.quadrature_point_indices())
            {
              const Point<dim>  &x = fe_values.quadrature_point(q);
              const Tensor<1, 2> b = exact.velocity(x);
              double r = exact.source(x) - grads[q][2] - b[0] * grads[q][0] - b[1] * grads[q][1];
              if (parameters.degree > 1)
                r += exact.diffusion() * (hessians[q][0][0] + hessians[q][1][1]);
              eta2 += h * h * r * r * fe_values.JxW(q);
            }
          for (const unsigned int f : cell->face_indices())
            {
              if (cell->at_boundary(f))
                continue;
              const auto neighbor = cell->neighbor(f);
              if (neighbor->has_children())
                // finer neighbors: one subface per child
                for (unsigned int sf = 0; sf < cell->face(f)->n_children(); ++sf)
                  {
                    const auto child = cell->neighbor_child_on_subface(f, sf);
                    subface_this.reinit(cell, f, sf);
                    face_other.reinit(child, GeometryInfo<dim>::opposite_face[f]);
                    eta2 += jump(subface_this, face_other);
                  }
              else if (cell->neighbor_is_coarser(f))
                {
                  const auto [nf, nsf] = cell->neighbor_of_coarser_neighbor(f);
                  face_this.reinit(cell, f);
                  subface_other.reinit(neighbor, nf, nsf);
                  eta2 += jump(face_this, subface_other);
                }
              else
                {
                  face_this.reinit(cell, f);
                  face_other.reinit(neighbor, cell->neighbor_of_neighbor(f));
                  eta2 += jump(face_this, face_other);
                }
            }
          estimate(cell->active_cell_index()) = std::sqrt(eta2);
        }
  }



  void
  SpaceTimeHeat::estimate(Vector<float> &estimate)
  {
    this->time("estimate", [&]() {
      estimate.reinit(triangulation->n_active_cells());
      if (parameters.estimator == "kelly")
        {
          KellyErrorEstimator<dim>::estimate(mapping,
                                             *dof_handler,
                                             QGauss<dim - 1>(parameters.degree + 1),
                                             std::map<types::boundary_id, const Function<dim> *>(),
                                             solution,
                                             estimate,
                                             fe.component_mask(FEValuesExtractors::Scalar(0)));
          return;
        }
      if (dg())
        {
          residual_estimate_dg(estimate);
          return;
        }
      if (!fosls())
        {
          residual_estimate_stabilized(estimate);
          return;
        }

      // fosls: the least-squares functional, cell by cell
      const QGauss<dim>     quadrature(parameters.degree + 2);
      const QGauss<dim - 1> face_quadrature(parameters.degree + 2);
      FEValues<dim>         fe_values(mapping,
                              fe,
                              quadrature,
                              update_gradients | update_values | update_quadrature_points |
                                update_JxW_values);
      FEFaceValues<dim>     fe_face_values(mapping, fe, face_quadrature, update_values | update_JxW_values);
      const FEValuesExtractors::Scalar        u(0), s0(1), s1(2);
      std::vector<Tensor<1, dim>>             grad_u(quadrature.size()), grad_s0(quadrature.size()),
        grad_s1(quadrature.size());
      std::vector<double> s0_values(quadrature.size()), s1_values(quadrature.size()),
        u_face(face_quadrature.size());
      unsigned int bottom_q = 0;
      for (const auto &cell : dof_handler->active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values[u].get_function_gradients(solution, grad_u);
            fe_values[s0].get_function_gradients(solution, grad_s0);
            fe_values[s1].get_function_gradients(solution, grad_s1);
            fe_values[s0].get_function_values(solution, s0_values);
            fe_values[s1].get_function_values(solution, s1_values);
            double eta2 = 0;
            for (const unsigned int q : fe_values.quadrature_point_indices())
              {
                const double r = grad_u[q][2] + grad_s0[q][0] + grad_s1[q][1] -
                                 exact.source(fe_values.quadrature_point(q));
                const double m0 = s0_values[q] + grad_u[q][0];
                const double m1 = s1_values[q] + grad_u[q][1];
                eta2 += (r * r + m0 * m0 + m1 * m1) * fe_values.JxW(q);
              }
            for (const unsigned int face : cell->face_indices())
              if (cell->face(face)->at_boundary() && cell->face(face)->boundary_id() == bottom_id)
                {
                  fe_face_values.reinit(cell, face);
                  fe_face_values[u].get_function_values(solution, u_face);
                  for (const unsigned int q : fe_face_values.quadrature_point_indices())
                    {
                      const double d = u_face[q] - bottom_values[bottom_q++];
                      eta2 += d * d * fe_face_values.JxW(q);
                    }
                }
            estimate(cell->active_cell_index()) = std::sqrt(eta2);
          }
    });
  }



  void
  SpaceTimeHeat::adapt(const Vector<float> &estimate)
  {
    this->time("adapt", [&]() {
      if (parameters.marking == "threshold")
        {
          const double max_estimate = Utilities::MPI::max(
            estimate.size() ? *std::max_element(estimate.begin(), estimate.end()) : 0.f,
            mpi_communicator);
          for (const auto &cell : triangulation->active_cell_iterators())
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
          *triangulation, estimate, parameters.refine, parameters.coarsen);
      else
        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
          *triangulation, estimate, parameters.refine, parameters.coarsen);

      const bool         freeze    = slab_index > 0 && parameters.bottom == "freeze";
      const unsigned int max_level = parameters.global + parameters.max_delta;
      for (const auto &cell : triangulation->active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (cell->level() >= static_cast<int>(max_level))
              cell->clear_refine_flag();
            if (cell->level() <= static_cast<int>(parameters.global))
              cell->clear_coarsen_flag();
            // keep the trace of the previous slab: a family with a child at
            // the bottom keeps that child, so it cannot be coarsened
            if (freeze)
              for (const unsigned int face : cell->face_indices())
                if (cell->face(face)->at_boundary() && cell->face(face)->boundary_id() == bottom_id)
                  cell->clear_coarsen_flag();
          }
      triangulation->execute_coarsening_and_refinement();
    });
  }



  // squares of |u - u_h|_L2(slab), |grad_x(u - u_h)|_L2(slab), |u - u_h|_L2(top)
  void
  SpaceTimeHeat::compute_errors(double &l2_q2, double &h1x_q2, double &l2_top2) const
  {
    const QGauss<dim>                quadrature(parameters.degree + 3);
    const QGauss<dim - 1>            face_quadrature(parameters.degree + 3);
    FEValues<dim>                    fe_values(mapping,
                            fe,
                            quadrature,
                            update_values | update_gradients | update_quadrature_points |
                              update_JxW_values);
    FEFaceValues<dim>                fe_face_values(mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points | update_JxW_values);
    const FEValuesExtractors::Scalar u(0);
    std::vector<double>              values(quadrature.size()), face_values(face_quadrature.size());
    std::vector<Tensor<1, dim>>      grads(quadrature.size());
    double                           e[3] = {0, 0, 0};
    for (const auto &cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[u].get_function_values(solution, values);
          fe_values[u].get_function_gradients(solution, grads);
          for (const unsigned int q : fe_values.quadrature_point_indices())
            {
              const Point<dim>  &p  = fe_values.quadrature_point(q);
              const double       d  = values[q] - exact.u(p);
              const Tensor<1, 2> gx = exact.grad_x(p);
              const double       d0 = grads[q][0] - gx[0], d1 = grads[q][1] - gx[1];
              e[0] += d * d * fe_values.JxW(q);
              e[1] += (d0 * d0 + d1 * d1) * fe_values.JxW(q);
            }
          for (const unsigned int face : cell->face_indices())
            if (cell->face(face)->at_boundary() && cell->face(face)->boundary_id() == top_id)
              {
                fe_face_values.reinit(cell, face);
                fe_face_values[u].get_function_values(solution, face_values);
                for (const unsigned int q : fe_face_values.quadrature_point_indices())
                  {
                    const double d = face_values[q] - exact.u(fe_face_values.quadrature_point(q));
                    e[2] += d * d * fe_face_values.JxW(q);
                  }
              }
        }
    Utilities::MPI::sum(ArrayView<const double>(e, 3), mpi_communicator, ArrayView<double>(e, 3));
    l2_q2   = e[0];
    h1x_q2  = e[1];
    l2_top2 = e[2];
  }



  void
  SpaceTimeHeat::output_results(const unsigned int slab) const
  {
    if (parameters.output.empty())
      return;
    Vector<float> subdomain(triangulation->n_active_cells()), level(triangulation->n_active_cells());
    for (const auto &cell : triangulation->active_cell_iterators())
      if (cell->is_locally_owned())
        {
          subdomain(cell->active_cell_index()) = triangulation->locally_owned_subdomain();
          level(cell->active_cell_index())     = cell->level();
        }
    DataOut<dim>             data_out;
    std::vector<std::string> names = {"u"};
    if (fosls())
      names = {"u", "sigma_x", "sigma_y"};
    data_out.attach_dof_handler(*dof_handler);
    data_out.add_data_vector(solution, names);
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.add_data_vector(level, "level");
    data_out.build_patches(mapping, parameters.degree);
    data_out.write_vtu_with_pvtu_record(parameters.output + "/", "slab", slab, mpi_communicator, 4);
  }



  void
  SpaceTimeHeat::run()
  {
    const unsigned int n_slabs =
      static_cast<unsigned int>(std::ceil(parameters.t_end / parameters.slab - 1e-12));
    pcout << "heat_spacetime: " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " processes, method " << parameters.method << ", degree " << parameters.degree
          << ", n " << parameters.n << ", nt " << parameters.nt << ", " << n_slabs
          << " slabs of " << parameters.slab << ", global " << parameters.global << ", max_delta "
          << parameters.max_delta << ", cycles " << parameters.cycles << ", estimator "
          << parameters.estimator << ", bottom " << parameters.bottom << std::endl;

    std::ofstream csv;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        csv.open(parameters.csv);
        csv << "slab,t0,t1,cycle,cells,dofs,cells_min,cells_max,iterations,estimate,"
               "l2_slab,h1x_slab,l2_top,mirror,setup,transfer,assembly,amg_setup,solve,estimate_time,adapt\n";
      }

    double       l2_q2 = 0, h1x_q2 = 0, l2_end = 0;
    double       dofs_final = 0, dofs_all = 0, cells_final = 0;
    Timer        total_timer;
    for (slab_index = 0; slab_index < n_slabs; ++slab_index)
      {
        const double a = slab_index * parameters.slab;
        const double b = std::min(a + parameters.slab, parameters.t_end);
        make_slab(a, b);
        if (slab_index > 0 && parameters.bottom == "freeze")
          this->time("mirror", [&]() { mirror_bottom_refinement(); });

        for (unsigned int cycle = 0; cycle <= parameters.cycles; ++cycle)
          {
            this->time("setup", [&]() { setup_system(); });
            assemble();
            solve();
            Vector<float> eta;
            estimate(eta);
            const double eta_total =
              std::sqrt(Utilities::MPI::sum(eta.norm_sqr(), mpi_communicator));
            double e_q2, e_h1x2, e_top2;
            compute_errors(e_q2, e_h1x2, e_top2);
            dofs_all += dof_handler->n_dofs();

            const auto cells = Utilities::MPI::min_max_avg(
              1.0 * triangulation->n_locally_owned_active_cells(), mpi_communicator);
            std::map<std::string, double> t_max;
            for (const std::string phase :
                 {"mirror", "setup", "transfer", "assembly", "amg setup", "solve", "estimate", "adapt"})
              t_max[phase] = Utilities::MPI::max(times[phase], mpi_communicator);
            times.clear();
            if (csv.is_open())
              csv << slab_index << ',' << t0 << ',' << t1 << ',' << cycle << ','
                  << triangulation->n_global_active_cells() << ',' << dof_handler->n_dofs() << ','
                  << cells.min << ',' << cells.max << ',' << last_iterations << ','
                  << std::setprecision(10) << eta_total << ',' << std::sqrt(e_q2) << ','
                  << std::sqrt(e_h1x2) << ',' << std::sqrt(e_top2) << std::setprecision(6) << ','
                  << t_max["mirror"] << ',' << t_max["setup"] << ',' << t_max["transfer"] << ','
                  << t_max["assembly"] << ',' << t_max["amg setup"] << ',' << t_max["solve"] << ','
                  << t_max["estimate"] << ',' << t_max["adapt"] << std::endl;

            if (cycle == parameters.cycles)
              {
                l2_q2 += e_q2;
                h1x_q2 += e_h1x2;
                l2_end = std::sqrt(e_top2);
                dofs_final += dof_handler->n_dofs();
                cells_final += triangulation->n_global_active_cells();
                pcout << "slab " << slab_index << " [" << t0 << ", " << t1 << "]: "
                      << triangulation->n_global_active_cells() << " cells, " << dof_handler->n_dofs()
                      << " dofs, " << last_iterations << " iterations, L2 error slab "
                      << std::sqrt(e_q2) << ", at t1 " << std::sqrt(e_top2) << ", estimate "
                      << eta_total << std::endl;
                output_results(slab_index);
              }
            else
              adapt(eta);
          }

        // keep this slab for the coupling of the next one
        previous.solution      = VectorType();
        previous.dof_handler   = std::move(dof_handler);
        previous.triangulation = std::move(triangulation);
        previous.solution      = solution;
      }
    pcout << "total: L2(Q) error " << std::sqrt(l2_q2) << ", L2(H1_x) error " << std::sqrt(h1x_q2)
          << ", L2 error at t_end " << l2_end << ", final cells " << cells_final << ", final dofs "
          << dofs_final << ", dofs of all solves " << dofs_all << ", wall time "
          << total_timer.wall_time() << " s" << std::endl;
  }
} // namespace HeatSpaceTime



int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      HeatSpaceTime::Parameters        parameters;
      parameters.parse(argc, argv);
      HeatSpaceTime::SpaceTimeHeat problem(parameters);
      problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << "Exception: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
