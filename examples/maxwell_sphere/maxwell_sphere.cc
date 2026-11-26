/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * ------------------------------------------------------------------------
 *
 * Time-harmonic Maxwell for a dielectric sphere, as a test problem for
 * preconditioners of the indefinite system.
 *
 * A sphere of radius a and refractive index n sits in the cube (-R,R)^3:
 *
 *   curl curl E - k^2 eps E = 0          in the cube, eps = n^2 for |x| < a,
 *                                                    eps = 1   elsewhere,
 *   n x curl E + i k E_T = g             on the boundary of the cube
 *
 * (time dependence exp(-i omega t), E_T the tangential part). The boundary
 * data g is that of an exact solution, a transverse electric multipole of
 * degree l with azimuthal number 0 around a given axis:
 *
 *   E = f(r) P_l^1(cos theta) e_phi,   f(r) = A j_l(n k r)             r < a,
 *                                      f(r) = j_l(k r) + B h_l^(1)(k r) r > a.
 *
 * A and B follow from the continuity of the tangential components of E and
 * curl E at r = a. Since the data is exact, the truncation of the domain
 * introduces no error, and the solution is known for every k. Close to a
 * Mie resonance of the sphere (a whispering gallery mode), |A| is large and
 * the system is nearly singular; away from the resonances this is an
 * ordinary scattering problem.
 *
 * Discretization: FE_NedelecSimplex(0) on tetrahedra in a
 * parallel::distributed::Triangulation. The interface |x| = a is resolved
 * by faces whose vertices lie on the sphere (the faces themselves are
 * flat). Cells at the interface can be refined further, which gives hanging
 * nodes.
 *
 * Solver: with E = E_r + i E_i the real block system
 *   [A  -B] [E_r]   [b_r]      A = K - k^2 M,
 *   [B   A] [E_i] = [b_i],     B = -k M_bdry,
 * is solved by FGMRES. Preconditioners (precond=...):
 *   ams      diag(P, P)^-1, P^-1 approximated by ams_cycles steps of the
 *            stationary iteration with Hypre's AMS, for the positive
 *            definite P = K + k^2 M + k M_bdry
 *   exact    the same with P inverted by CG (and AMS) to a tight tolerance
 *   shifted  the shifted operator K - (1 + i alpha) k^2 M - i k M_bdry,
 *            inverted by an inner GMRES (preconditioned as "ams") to the
 *            relative tolerance inner_tol
 *   ras      one-level restricted additive Schwarz: every rank solves the
 *            problem on its owned and ghost cells with MUMPS, with n x E = 0
 *            on the subdomain boundary, and keeps the values of its owned
 *            DoFs
 *   oras     the same with the Robin condition n x curl E + p E_T = 0 on
 *            the subdomain boundary as transmission condition, with
 *            p = robin_real / h - i k_loc (k_loc = k sqrt(eps), h the
 *            diameter of the face). robin_real = 0 is the impedance
 *            condition (Despres); a positive real part also damps
 *            evanescent modes (optimized Schwarz).
 * ras_serial and oras_serial emulate the Schwarz methods on one process,
 * with control over the overlap: the active cells, sorted by CellId (an
 * SFC-like order), are split into `subdomains` contiguous parts, and each
 * part is extended by overlap_layers layers of cells sharing a vertex
 * (overlap_mode=fine), or by overlap_layers layers of level-overlap_level
 * cells sharing a vertex, taking all active cells inside them
 * (overlap_mode=level, overlap width at least overlap_layers times the
 * size of the level-overlap_level cells). A DoF belongs to the part with
 * the smallest number among its cells; each subdomain solve keeps the
 * values of the DoFs of its part. overlap_mode=fine with one layer is
 * about the vertex ghost layer of the parallel version.
 * With ghost_overlap_layers > 0, the ghost layer of the triangulation is an
 * overlap of that many layers of cells of level ghost_overlap_level
 * (Triangulation::set_ghost_overlap, t8code), so ras/oras use the same
 * level-based overlap as overlap_mode=level of the serial emulation.
 * ras and oras take a coarse correction (coarse=additive|hybrid): the
 * problem on the coarse mesh (the mesh before the global refinement,
 * H = 2h), with the mass term k^2 (1 + i coarse_shift) eps M, solved by
 * MUMPS on every rank. coarse_op=algebraic (default) computes P^T A P
 * with the global interpolation P (PETSc MatPtAP); coarse_op=galerkin
 * assembles P_c^T A_c P_c cell by cell with the interpolation from each
 * fine cell's coarse ancestor; coarse_op=rediscretize assembles the
 * problem on the coarse mesh. Near the curved interface the fine cells are
 * not contained in their coarse cells (the vertices moved onto the sphere),
 * so the spaces are not nested and only "algebraic" gives the exact
 * Galerkin operator of P (E = I - P A_H^-1 P^T A is then a projection).
 * coarse_solver=distributed factorizes the coarse problem with one MUMPS
 * over all ranks, each providing its rows of P^T A P, instead of a
 * sequential MUMPS on every rank (redundant); the coarse right hand side is
 * summed over the ranks, solved, and the solution broadcast. MUMPS runs
 * on coarse_procs processes (every m-th rank; 0 = all), the other ranks send
 * their rows to them.
 * ras and oras also take a correction in the space of gradients
 * (gradient_correction=amg|exact): on gradient fields G phi the operator is
 * about -k^2 L with L = G^T M G, so x0 = -G L^-1 G^T r / k^2, with L^-1 one
 * BoomerAMG cycle (amg) or CG with BoomerAMG (exact). It is applied before
 * and after the Schwarz step (multiplicatively). The impedance
 * transmission condition acts as a Dirichlet condition on the potentials,
 * which the one-level method handles badly. The coarse space is mapped to the fine mesh by
 * interpolation (line integrals along the fine edges). "additive" adds the
 * coarse correction to the Schwarz correction, "hybrid" applies the
 * Schwarz method to the residual after the coarse correction.
 *
 * Usage: maxwell_sphere [key=value ...], see Parameters below. Examples:
 *   maxwell_sphere mode=selftest
 *   maxwell_sphere mode=resonances n=3 l=3 k_min=0.5 k_max=4
 *   maxwell_sphere n=3 l=3 k_min=1 k_max=2 k_steps=11 refine=3
 * ------------------------------------------------------------------------ */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/edge_element_tools.h>

#include <deal.II/fe/fe_nedelec_simplex.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <sc.h>
#include <t8.h>

#include <petscksp.h>
#include <petscpc.h>

#include <dmumps_c.h>

#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <string>

using namespace dealii;

using Complex = std::complex<double>;
const Complex I(0., 1.);


struct Parameters
{
  std::string  mode    = "solve"; // solve, selftest, resonances
  double       n       = 2.;      // refractive index of the sphere
  double       a       = 1.;      // radius of the sphere
  double       R       = 2.;      // half width of the cube
  unsigned int l       = 1;       // degree of the multipole
  double       k_min   = 1.;      // wave numbers k_min ... k_max (vacuum)
  double       k_max   = 1.;
  unsigned int k_steps = 1;
  unsigned int hex_refine = 2;    // refinements of the 13 hexahedral blocks
  unsigned int refine  = 1;       // global refinements of the tetrahedra
  unsigned int interface_refine = 0; // further refinements at the interface
  std::string  precond          = "ams"; // ams, exact, shifted
  unsigned int ams_cycles       = 1;     // AMS cycles per block application
  double       robin_real       = 0.;    // oras: c in p = c/h - i k_loc
  std::string  coarse           = "none"; // ras/oras: none, additive, hybrid
  double       coarse_shift     = 0.;    // absorption of the coarse problem
  std::string  coarse_op        = "algebraic"; // algebraic, galerkin, rediscretize
  // the coarse solve: MUMPS on every rank (redundant) or one distributed
  // MUMPS over all ranks (distributed, only with coarse_op=algebraic)
  std::string  coarse_solver    = "redundant";
  // coarse_solver=distributed: the number of processes running MUMPS (0: all)
  unsigned int coarse_procs     = 0;
  std::string  gradient_correction = "none"; // ras/oras: none, amg, exact
  bool         checks           = false; // tests of the Schwarz components
  // oras_serial / ras_serial: subdomains emulated on one process
  unsigned int subdomains       = 8;
  std::string  overlap_mode     = "fine"; // fine, level
  unsigned int overlap_layers   = 1;
  unsigned int overlap_level    = 0;
  // ras/oras (parallel): the ghost layer as an overlap of this many layers of
  // cells of level ghost_overlap_level (t8code); 0 = the vertex ghost layer
  unsigned int ghost_overlap_level  = 0;
  unsigned int ghost_overlap_layers = 0;
  // > 0: before the solve, run this many unrestarted GMRES iterations and
  // report the Ritz values of the preconditioned operator
  unsigned int ritz             = 0;
  // Schwarz local problems only (the preconditioner, not the problem):
  // absorption k^2 (1 + i local_shift) eps M, and impedance_scale * k_loc
  // as the imaginary Robin coefficient on artificial boundaries
  double       local_shift      = 0.;
  double       impedance_scale  = 1.;
  // the polyhedral sphere (default since 2026-11-24): no vertices are moved
  // onto the sphere, so the meshes are nested; 0 = the curved sphere
  bool         flat_interface   = true;
  double       alpha            = 0.5;   // shift of "shifted"
  double       inner_tol        = 0.1;   // inner tolerance of "shifted"
  double       tol              = 1e-8;  // relative residual of FGMRES
  unsigned int max_iterations   = 3000;
  unsigned int restart          = 200;
  std::string  csv              = "";    // append one line per solve

  void
  parse(int argc, char *argv[])
  {
    for (int i = 1; i < argc; ++i)
      {
        const std::string arg(argv[i]);
        const auto        pos = arg.find('=');
        AssertThrow(pos != std::string::npos,
                    ExcMessage("Arguments have the form key=value: " + arg));
        const std::string key = arg.substr(0, pos), value = arg.substr(pos + 1);
        if (key == "mode")
          mode = value;
        else if (key == "n")
          n = std::stod(value);
        else if (key == "a")
          a = std::stod(value);
        else if (key == "R")
          R = std::stod(value);
        else if (key == "l")
          l = std::stoi(value);
        else if (key == "k")
          k_min = k_max = std::stod(value);
        else if (key == "k_min")
          k_min = std::stod(value);
        else if (key == "k_max")
          k_max = std::stod(value);
        else if (key == "k_steps")
          k_steps = std::stoi(value);
        else if (key == "hex_refine")
          hex_refine = std::stoi(value);
        else if (key == "refine")
          refine = std::stoi(value);
        else if (key == "interface_refine")
          interface_refine = std::stoi(value);
        else if (key == "precond")
          precond = value;
        else if (key == "alpha")
          alpha = std::stod(value);
        else if (key == "coarse")
          coarse = value;
        else if (key == "subdomains")
          subdomains = std::stoi(value);
        else if (key == "overlap_mode")
          overlap_mode = value;
        else if (key == "overlap_layers")
          overlap_layers = std::stoi(value);
        else if (key == "local_shift")
          local_shift = std::stod(value);
        else if (key == "impedance_scale")
          impedance_scale = std::stod(value);
        else if (key == "ritz")
          ritz = std::stoi(value);
        else if (key == "ghost_overlap_level")
          ghost_overlap_level = std::stoi(value);
        else if (key == "ghost_overlap_layers")
          ghost_overlap_layers = std::stoi(value);
        else if (key == "overlap_level")
          overlap_level = std::stoi(value);
        else if (key == "flat_interface")
          flat_interface = (value == "1" || value == "true");
        else if (key == "checks")
          checks = (value == "1" || value == "true");
        else if (key == "gradient_correction")
          gradient_correction = value;
        else if (key == "coarse_solver")
          coarse_solver = value;
        else if (key == "coarse_procs")
          coarse_procs = std::stoi(value);
        else if (key == "coarse_op")
          coarse_op = value;
        else if (key == "coarse_shift")
          coarse_shift = std::stod(value);
        else if (key == "robin_real")
          robin_real = std::stod(value);
        else if (key == "ams_cycles")
          ams_cycles = std::stoi(value);
        else if (key == "inner_tol")
          inner_tol = std::stod(value);
        else if (key == "tol")
          tol = std::stod(value);
        else if (key == "max_iterations")
          max_iterations = std::stoi(value);
        else if (key == "restart")
          restart = std::stoi(value);
        else if (key == "csv")
          csv = value;
        else
          AssertThrow(false, ExcMessage("Unknown argument " + key));
      }
    AssertThrow(l >= 1, ExcMessage("The multipole degree l must be >= 1."));
    AssertThrow(R > a, ExcMessage("The cube must contain the sphere."));
    AssertThrow(coarse == "none" || coarse == "additive" || coarse == "hybrid",
                ExcMessage("Unknown coarse correction " + coarse));
    AssertThrow(coarse_solver == "redundant" ||
                  (coarse_solver == "distributed" && coarse_op == "algebraic"),
                ExcMessage("coarse_solver must be redundant, or distributed "
                           "with coarse_op=algebraic."));
    AssertThrow(coarse_op == "algebraic" || coarse_op == "galerkin" ||
                  coarse_op == "rediscretize",
                ExcMessage("Unknown coarse operator " + coarse_op));
    AssertThrow(overlap_mode == "fine" || overlap_mode == "level",
                ExcMessage("Unknown overlap mode " + overlap_mode));
    AssertThrow(precond == "ams" || precond == "exact" ||
                  precond == "shifted" || precond == "ras" ||
                  precond == "oras" || precond == "ras_serial" ||
                  precond == "oras_serial",
                ExcMessage("Unknown preconditioner " + precond));
  }

  double
  wave_number(const unsigned int step) const
  {
    return k_steps <= 1 ? k_min :
                          k_min + (k_max - k_min) * step / (k_steps - 1.);
  }
};



/**
 * The transverse electric multipole solution of the dielectric sphere.
 */
class MieTE
{
public:
  MieTE(const unsigned int l,
        const double       n,
        const double       a,
        const Tensor<1, 3> &axis)
    : l(l)
    , n(n)
    , a(a)
    , axis(axis / axis.norm())
    , k(0.)
  {}

  void
  set_wave_number(const double k_)
  {
    k              = k_;
    const double x = k * a, y = n * k * a;
    // continuity of f and of (r f)' at r = a
    const Complex jx = j(x), hx = h(x), pjx = psi_j(x), phx = psi_h(x);
    const double  jy = j(y), pjy = psi_j(y);
    const Complex D  = jy * phx - hx * pjy;
    A                = (jx * phx - hx * pjx) / D;
    B                = (pjy * jx - jy * pjx) / D;
  }

  /// Amplitude of the field inside the sphere; large at a resonance.
  Complex
  interior_coefficient() const
  {
    return A;
  }

  Complex
  scattering_coefficient() const
  {
    return B;
  }

  /// Real and imaginary parts of E and curl E.
  void
  evaluate(const Point<3> &p,
           Tensor<1, 3>   &E_r,
           Tensor<1, 3>   &E_i,
           Tensor<1, 3>   &curl_r,
           Tensor<1, 3>   &curl_i) const
  {
    const double       r     = std::max(p.norm(), 1e-12 * a);
    const Tensor<1, 3> e_r   = (p.norm() > 0. ? Tensor<1, 3>(p) / p.norm() :
                                                axis);
    const double       cos_t = std::min(1., std::max(-1., e_r * axis));
    const double       sin_t = std::sqrt(1. - cos_t * cos_t);
    Tensor<1, 3>       e_phi, e_theta;
    if (sin_t > 1e-14)
      {
        e_phi   = cross_product_3d(axis, e_r) / sin_t;
        e_theta = cross_product_3d(e_phi, e_r);
      }

    // f(r) and (r f)'(r)
    Complex f, rf_prime;
    if (r < a)
      {
        const double rho = n * k * r;
        f                = A * j(rho);
        rf_prime         = A * psi_j(rho);
      }
    else
      {
        const double rho = k * r;
        f                = j(rho) + B * h(rho);
        rf_prime         = psi_j(rho) + B * psi_h(rho);
      }

    // P_l^1 = sin(theta) P_l'(cos theta) = -d/dtheta P_l(cos theta)
    const double  P1    = std::assoc_legendre(l, 1, cos_t);
    const double  P0    = std::legendre(l, cos_t);
    const Complex e     = f * P1;
    const Complex c_r   = double(l * (l + 1)) * P0 * f / r;
    const Complex c_th  = -rf_prime / r * P1;
    E_r                 = e.real() * e_phi;
    E_i                 = e.imag() * e_phi;
    curl_r              = c_r.real() * e_r + c_th.real() * e_theta;
    curl_i              = c_r.imag() * e_r + c_th.imag() * e_theta;
  }

private:
  const unsigned int l;
  const double       n, a;
  const Tensor<1, 3> axis;
  double             k;
  Complex            A, B;

  double
  j(const double rho) const
  {
    return std::sph_bessel(l, rho);
  }

  Complex
  h(const double rho) const
  {
    return Complex(std::sph_bessel(l, rho), std::sph_neumann(l, rho));
  }

  // (rho z_l(rho))' = rho z_{l-1}(rho) - l z_l(rho)
  double
  psi_j(const double rho) const
  {
    return rho * std::sph_bessel(l - 1, rho) - l * std::sph_bessel(l, rho);
  }

  Complex
  psi_h(const double rho) const
  {
    return rho *
             Complex(std::sph_bessel(l - 1, rho), std::sph_neumann(l - 1, rho)) -
           double(l) * h(rho);
  }
};



/**
 * Checks of the exact solution with finite differences.
 */
void
self_test(const Parameters &prm)
{
  MieTE mie(prm.l, prm.n, prm.a, Tensor<1, 3>({0.3, -0.5, 0.8}));
  mie.set_wave_number(prm.k_min);
  const double eps = 1e-5;
  const auto   E   = [&](const Point<3> &p) {
    Tensor<1, 3> e_r, e_i, c_r, c_i;
    mie.evaluate(p, e_r, e_i, c_r, c_i);
    return std::array<Tensor<1, 3>, 4>{{e_r, e_i, c_r, c_i}};
  };
  // curl of the component `part` (0: E, 2: curl E) by central differences
  const auto fd_curl = [&](const Point<3> &p, const unsigned int part) {
    std::array<Tensor<1, 3>, 2> result;
    for (unsigned int c = 0; c < 2; ++c)
      {
        Tensor<2, 3> grad;
        for (unsigned int d = 0; d < 3; ++d)
          {
            Point<3> plus = p, minus = p;
            plus[d] += eps;
            minus[d] -= eps;
            const Tensor<1, 3> diff =
              (E(plus)[part + c] - E(minus)[part + c]) / (2 * eps);
            for (unsigned int i = 0; i < 3; ++i)
              grad[i][d] = diff[i];
          }
        result[c][0] = grad[2][1] - grad[1][2];
        result[c][1] = grad[0][2] - grad[2][0];
        result[c][2] = grad[1][0] - grad[0][1];
      }
    return result;
  };

  double err_curl = 0, err_curlcurl = 0;
  for (const double radius : {0.37 * prm.a, 0.9 * prm.a, 1.2 * prm.a, prm.R})
    for (const Point<3> &dir : {Point<3>(1, 0.2, 0.1),
                                Point<3>(-0.3, 0.7, 0.4),
                                Point<3>(0.2, -0.1, -1.)})
      {
        const Point<3> p        = Point<3>(radius * dir / dir.norm());
        const auto     values   = E(p);
        const auto     curl     = fd_curl(p, 0);
        const auto     curlcurl = fd_curl(p, 2);
        const double   eps_r    = (radius < prm.a ? prm.n * prm.n : 1.);
        const double   scale    = values[0].norm() + values[1].norm();
        for (unsigned int c = 0; c < 2; ++c)
          {
            err_curl = std::max(err_curl,
                                (curl[c] - values[2 + c]).norm() /
                                  (values[2].norm() + values[3].norm()));
            err_curlcurl = std::max(
              err_curlcurl,
              (curlcurl[c] - prm.k_min * prm.k_min * eps_r * values[c]).norm() /
                (prm.k_min * prm.k_min * eps_r * scale));
          }
      }

  // jumps of the tangential components at the interface
  double jump_E = 0, jump_curl = 0;
  for (const Point<3> &dir : {Point<3>(1, 0.2, 0.1), Point<3>(-0.3, 0.7, 0.4)})
    {
      const Tensor<1, 3> e_r = dir / dir.norm();
      const auto in  = E(Point<3>((1 - 1e-9) * prm.a * e_r));
      const auto out = E(Point<3>((1 + 1e-9) * prm.a * e_r));
      for (unsigned int c = 0; c < 2; ++c)
        {
          jump_E = std::max(jump_E,
                            cross_product_3d(e_r, in[c] - out[c]).norm() /
                              (in[0].norm() + in[1].norm()));
          jump_curl =
            std::max(jump_curl,
                     cross_product_3d(e_r, in[2 + c] - out[2 + c]).norm() /
                       (in[2].norm() + in[3].norm()));
        }
    }
  std::cout << "self test l=" << prm.l << " n=" << prm.n << " k=" << prm.k_min
            << "\n  curl E against finite differences:        " << err_curl
            << "\n  curl curl E - k^2 eps E (finite diff.):   " << err_curlcurl
            << "\n  jump of n x E at the interface:           " << jump_E
            << "\n  jump of n x curl E at the interface:      " << jump_curl
            << std::endl;
}



/**
 * Scan |A(k)| for local maxima and estimate their quality factors from the
 * full width at half maximum of |A|^2.
 */
void
find_resonances(const Parameters &prm)
{
  MieTE      mie(prm.l, prm.n, prm.a, Tensor<1, 3>({0., 0., 1.}));
  const auto amplitude = [&](const double k) {
    mie.set_wave_number(k);
    return std::abs(mie.interior_coefficient());
  };
  const unsigned int steps = 200000;
  const double       dk    = (prm.k_max - prm.k_min) / steps;
  std::cout << "TE resonances for l=" << prm.l << " n=" << prm.n
            << " a=" << prm.a << " in k = " << prm.k_min << " ... " << prm.k_max
            << "\n        k         k*a      |A|          Q" << std::endl;
  double previous = amplitude(prm.k_min), current = amplitude(prm.k_min + dk);
  for (unsigned int i = 2; i <= steps; ++i)
    {
      const double next = amplitude(prm.k_min + i * dk);
      if (current > previous && current >= next)
        {
          // refine the maximum by golden section
          double lo = prm.k_min + (i - 2) * dk, hi = prm.k_min + i * dk;
          const double g = 0.5 * (std::sqrt(5.) - 1.);
          for (unsigned int it = 0; it < 80; ++it)
            {
              const double x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo);
              if (amplitude(x1) > amplitude(x2))
                hi = x2;
              else
                lo = x1;
            }
          const double k0 = 0.5 * (lo + hi), peak = amplitude(k0);
          // half width: |A|^2 = peak^2 / 2, by bisection on both sides
          const auto half = [&](const double sign) {
            double step = 1e-12 * k0;
            while (amplitude(k0 + sign * step) > peak / std::sqrt(2.) &&
                   step < k0)
              step *= 2;
            double inner = 0.5 * step, outer = step;
            for (unsigned int it = 0; it < 60; ++it)
              {
                const double mid = 0.5 * (inner + outer);
                (amplitude(k0 + sign * mid) > peak / std::sqrt(2.) ? inner :
                                                                     outer) =
                  mid;
              }
            return outer;
          };
          const double width = half(1.) + half(-1.);
          std::cout << std::setw(12) << std::setprecision(9) << k0
                    << std::setw(11) << std::setprecision(6) << k0 * prm.a
                    << std::setw(12) << std::setprecision(4) << peak
                    << std::setw(12) << std::setprecision(4) << k0 / width
                    << std::endl;
        }
      previous = current;
      current  = next;
    }
}



/**
 * AMS through PETSc's interface to Hypre.
 */
class PreconditionAMS : public PETScWrappers::PreconditionBase
{
public:
  void
  initialize(const PETScWrappers::MPI::SparseMatrix        &matrix,
             const PETScWrappers::MPI::SparseMatrix        &gradient,
             const std::vector<PETScWrappers::MPI::Vector> &constant_fields)
  {
    create_pc_with_mat(matrix);
    PetscErrorCode ierr = PCSetType(pc, PCHYPRE);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    ierr = PCHYPRESetType(pc, "ams");
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    ierr = PCHYPRESetDiscreteGradient(pc, gradient);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    ierr = PCHYPRESetEdgeConstantVectors(pc,
                                         constant_fields[0],
                                         constant_fields[1],
                                         constant_fields[2]);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    ierr = PCSetUp(pc);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
  }
};



/**
 * The real block form [A -B; B A] of A + i B.
 */
struct ComplexBlockMatrix
{
  const PETScWrappers::MPI::SparseMatrix &A;
  const PETScWrappers::MPI::SparseMatrix &B;

  void
  vmult(PETScWrappers::MPI::BlockVector       &dst,
        const PETScWrappers::MPI::BlockVector &src) const
  {
    PETScWrappers::MPI::Vector tmp(dst.block(0));
    A.vmult(dst.block(0), src.block(0));
    B.vmult(tmp, src.block(1));
    dst.block(0) -= tmp;
    B.vmult(dst.block(1), src.block(0));
    A.vmult_add(dst.block(1), src.block(1));
  }
};



/**
 * diag(P^-1, P^-1) with one application of AMS or a CG solve with AMS.
 */
struct BlockDiagonalPreconditioner
{
  const PETScWrappers::MPI::SparseMatrix &P;
  const PreconditionAMS                  &ams;
  const bool                              inner_solve;
  const unsigned int                      n_cycles = 1;
  mutable unsigned int                    n_ams    = 0;

  void
  apply(PETScWrappers::MPI::Vector       &dst,
        const PETScWrappers::MPI::Vector &src) const
  {
    if (inner_solve)
      {
        SolverControl control(1000, 1e-10 * src.l2_norm());
        SolverCG<PETScWrappers::MPI::Vector> cg(control);
        dst = 0;
        cg.solve(P, dst, src, ams);
        n_ams += control.last_step() + 1;
      }
    else
      {
        // n_cycles steps of the stationary iteration x += AMS (src - P x),
        // starting from 0: a fixed linear operator
        ams.vmult(dst, src);
        ++n_ams;
        if (n_cycles > 1)
          {
            PETScWrappers::MPI::Vector residual(src), correction(src);
            for (unsigned int c = 1; c < n_cycles; ++c)
              {
                P.vmult(residual, dst);
                residual.sadd(-1., 1., src);
                ams.vmult(correction, residual);
                dst += correction;
                ++n_ams;
              }
          }
      }
  }

  void
  vmult(PETScWrappers::MPI::BlockVector       &dst,
        const PETScWrappers::MPI::BlockVector &src) const
  {
    apply(dst.block(0), src.block(0));
    apply(dst.block(1), src.block(1));
  }
};



/**
 * A sequential sparse LU factorization by MUMPS (real arithmetic).
 */
class SequentialMUMPS
{
public:
  SequentialMUMPS()
  {
    id.comm_fortran = static_cast<MUMPS_INT>(MPI_Comm_c2f(MPI_COMM_SELF));
    id.par          = 1;
    id.sym          = 0;
    id.job          = -1;
    dmumps_c(&id);
    check("initialization");
    // no output
    id.icntl[0] = -1;
    id.icntl[1] = -1;
    id.icntl[2] = -1;
    id.icntl[3] = 0;
    // more working space than estimated
    id.icntl[13] = 50;
  }

  ~SequentialMUMPS()
  {
    id.job = -2;
    dmumps_c(&id);
  }

  SequentialMUMPS(const SequentialMUMPS &) = delete;
  SequentialMUMPS &
  operator=(const SequentialMUMPS &) = delete;

  /// Analysis and factorization of the matrix given by 0-based triplets.
  void
  factorize(const unsigned int                 n,
            const std::vector<MUMPS_INT>      &rows,
            const std::vector<MUMPS_INT>      &cols,
            const std::vector<double>         &values)
  {
    irn.resize(rows.size());
    jcn.resize(cols.size());
    for (std::size_t i = 0; i < rows.size(); ++i)
      {
        irn[i] = rows[i] + 1;
        jcn[i] = cols[i] + 1;
      }
    a      = values;
    id.n   = n;
    id.nnz = static_cast<MUMPS_INT8>(a.size());
    id.irn = irn.data();
    id.jcn = jcn.data();
    id.a   = a.data();
    id.job = 4;
    dmumps_c(&id);
    check("factorization");
  }

  /// Solves in place.
  void
  solve(std::vector<double> &rhs)
  {
    AssertDimension(rhs.size(), static_cast<std::size_t>(id.n));
    id.rhs  = rhs.data();
    id.nrhs = 1;
    id.lrhs = id.n;
    id.job  = 3;
    dmumps_c(&id);
    check("solve");
  }

  /// Solves in place with MUMPS' error analysis (ICNTL(11) = 1) and
  /// returns the condition number estimates COND1 and COND2 (RINFOG(10),
  /// RINFOG(11)) and the scaled residual (RINFOG(6)).
  std::array<double, 3>
  solve_with_error_analysis(std::vector<double> &rhs)
  {
    id.icntl[10] = 1;
    solve(rhs);
    id.icntl[10] = 0;
    return {{id.rinfog[9], id.rinfog[10], id.rinfog[5]}};
  }

  /// The number of entries of the factors.
  double
  factor_entries() const
  {
    return id.infog[28];
  }

private:
  DMUMPS_STRUC_C         id;
  std::vector<MUMPS_INT> irn, jcn;
  std::vector<double>    a;

  void
  check(const char *phase) const
  {
    AssertThrow(id.infog[0] >= 0,
                ExcMessage(std::string("MUMPS ") + phase + " failed: INFOG(1)=" +
                           std::to_string(id.infog[0]) + " INFOG(2)=" +
                           std::to_string(id.infog[1])));
  }
};



/**
 * A sparse LU factorization by MUMPS distributed over a subset of the
 * processes of a communicator (real arithmetic): every m-th process runs
 * MUMPS (m = ceil(P / n_procs)), the other processes send their entries of
 * the matrix to the MUMPS process of their group (MUMPS' distributed
 * assembled input on the MUMPS processes). The right hand side and the
 * solution are replicated on all processes.
 */
class DistributedMUMPS
{
public:
  /// n_procs = 0: all processes run MUMPS.
  DistributedMUMPS(const MPI_Comm comm, const unsigned int n_procs)
    : comm(comm)
  {
    const unsigned int rank   = Utilities::MPI::this_mpi_process(comm);
    const unsigned int size   = Utilities::MPI::n_mpi_processes(comm);
    const unsigned int procs  = (n_procs == 0 || n_procs > size) ? size : n_procs;
    stride                    = (size + procs - 1) / procs;
    member                    = (rank % stride == 0);
    // the group of a MUMPS process: itself and the next stride - 1 processes
    MPI_Comm_split(comm, rank / stride, rank, &group_comm);
    MPI_Comm_split(comm, member ? 0 : MPI_UNDEFINED, rank, &mumps_comm);
    n_mumps_procs = (size + stride - 1) / stride;
    if (!member)
      return;
    id.comm_fortran = static_cast<MUMPS_INT>(MPI_Comm_c2f(mumps_comm));
    id.par          = 1;
    id.sym          = 0;
    id.job          = -1;
    dmumps_c(&id);
    check("initialization");
    id.icntl[0]  = -1;
    id.icntl[1]  = -1;
    id.icntl[2]  = -1;
    id.icntl[3]  = 0;
    id.icntl[13] = 50;
    // distributed assembled matrix
    id.icntl[17] = 3;
  }

  ~DistributedMUMPS()
  {
    if (member)
      {
        id.job = -2;
        dmumps_c(&id);
        MPI_Comm_free(&mumps_comm);
      }
    MPI_Comm_free(&group_comm);
  }

  DistributedMUMPS(const DistributedMUMPS &) = delete;
  DistributedMUMPS &
  operator=(const DistributedMUMPS &) = delete;

  /// The number of processes running MUMPS.
  unsigned int
  n_processes() const
  {
    return n_mumps_procs;
  }

  /// Analysis and factorization; the triplets (0-based) of this process.
  void
  factorize(const unsigned int             n,
            const std::vector<MUMPS_INT>  &rows,
            const std::vector<MUMPS_INT>  &cols,
            const std::vector<double>     &values)
  {
    static_assert(sizeof(MUMPS_INT) == sizeof(int));
    // collect the triplets of the group on its MUMPS process
    const int           count = static_cast<int>(values.size());
    const unsigned int  group_size = Utilities::MPI::n_mpi_processes(group_comm);
    std::vector<int>    counts(member ? group_size : 0), offsets;
    MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, group_comm);
    std::size_t total = 0;
    for (const int c : counts)
      {
        offsets.push_back(static_cast<int>(total));
        total += c;
      }
    std::vector<MUMPS_INT> all_rows(total), all_cols(total);
    a.resize(member ? total : 0);
    MPI_Gatherv(rows.data(), count, MPI_INT, all_rows.data(), counts.data(),
                offsets.data(), MPI_INT, 0, group_comm);
    MPI_Gatherv(cols.data(), count, MPI_INT, all_cols.data(), counts.data(),
                offsets.data(), MPI_INT, 0, group_comm);
    MPI_Gatherv(values.data(), count, MPI_DOUBLE, a.data(), counts.data(),
                offsets.data(), MPI_DOUBLE, 0, group_comm);
    if (!member)
      return;
    irn.resize(total);
    jcn.resize(total);
    for (std::size_t i = 0; i < total; ++i)
      {
        irn[i] = all_rows[i] + 1;
        jcn[i] = all_cols[i] + 1;
      }
    id.n       = n;
    id.nnz_loc = static_cast<MUMPS_INT8>(a.size());
    id.irn_loc = irn.data();
    id.jcn_loc = jcn.data();
    id.a_loc   = a.data();
    id.job     = 4;
    dmumps_c(&id);
    check("factorization");
  }

  /// Solves in place; rhs is replicated on all processes (on input the
  /// right hand side of the first process is used).
  void
  solve(std::vector<double> &rhs)
  {
    if (member)
      {
        AssertDimension(rhs.size(), static_cast<std::size_t>(id.n));
        id.rhs  = rhs.data();
        id.nrhs = 1;
        id.lrhs = id.n;
        id.job  = 3;
        dmumps_c(&id);
        check("solve");
      }
    // process 0 of comm is the MUMPS host
    MPI_Bcast(rhs.data(), rhs.size(), MPI_DOUBLE, 0, comm);
  }

private:
  const MPI_Comm         comm;
  MPI_Comm               group_comm = MPI_COMM_NULL;
  MPI_Comm               mumps_comm = MPI_COMM_NULL;
  unsigned int           stride        = 1;
  unsigned int           n_mumps_procs = 1;
  bool                   member        = true;
  DMUMPS_STRUC_C         id;
  std::vector<MUMPS_INT> irn, jcn;
  std::vector<double>    a;

  void
  check(const char *phase) const
  {
    // INFOG is the same on all MUMPS processes
    AssertThrow(id.infog[0] >= 0,
                ExcMessage(std::string("distributed MUMPS ") + phase +
                           " failed: INFOG(1)=" + std::to_string(id.infog[0]) +
                           " INFOG(2)=" + std::to_string(id.infog[1])));
  }
};



/**
 * Accumulates the complex matrix A + i B of a local problem as triplets and
 * factorizes its real block form [A -B; B A].
 */
class ComplexTriplets
{
public:
  void
  add(const unsigned int i,
      const unsigned int j,
      const double       re,
      const double       im)
  {
    auto &entry = entries[{i, j}];
    entry[0] += re;
    entry[1] += im;
  }

  void
  set_identity_row_and_column(const unsigned int i)
  {
    dirichlet.insert(i);
  }

  /// y = [A -B; B A] x for the accumulated A + i B (without the Dirichlet
  /// rows and columns; for tests).
  std::vector<double>
  vmult(const unsigned int n, const std::vector<double> &x) const
  {
    AssertDimension(x.size(), 2 * n);
    std::vector<double> y(2 * n, 0.);
    for (const auto &[ij, v] : entries)
      {
        const auto [i, j] = ij;
        y[i] += v[0] * x[j] - v[1] * x[j + n];
        y[i + n] += v[1] * x[j] + v[0] * x[j + n];
      }
    return y;
  }

  void
  factorize(const unsigned int n, SequentialMUMPS &mumps) const
  {
    std::vector<MUMPS_INT> rows, cols;
    std::vector<double>    values;
    rows.reserve(4 * entries.size());
    cols.reserve(4 * entries.size());
    values.reserve(4 * entries.size());
    const auto push = [&](const unsigned int i,
                          const unsigned int j,
                          const double       v) {
      rows.push_back(i);
      cols.push_back(j);
      values.push_back(v);
    };
    for (const auto &[ij, v] : entries)
      {
        const auto [i, j] = ij;
        if (dirichlet.count(i) || dirichlet.count(j))
          continue;
        push(i, j, v[0]);
        push(i + n, j + n, v[0]);
        if (v[1] != 0.)
          {
            push(i, j + n, -v[1]);
            push(i + n, j, v[1]);
          }
      }
    for (const unsigned int i : dirichlet)
      {
        push(i, i, 1.);
        push(i + n, i + n, 1.);
      }
    mumps.factorize(2 * n, rows, cols, values);
  }

private:
  std::map<std::pair<unsigned int, unsigned int>, std::array<double, 2>>
                         entries;
  std::set<unsigned int> dirichlet;
};



/**
 * The solver of the coarse problem: a sequential MUMPS on every rank, or a
 * distributed one. The coarse vectors are replicated on all ranks.
 */
struct CoarseSolver
{
  SequentialMUMPS                   redundant;
  std::unique_ptr<DistributedMUMPS> distributed;

  unsigned int n_solves = 0;

  void
  solve(std::vector<double> &coarse)
  {
    Timer timer;
    if (distributed)
      distributed->solve(coarse);
    else
      redundant.solve(coarse);
    // the wall time of the first solves, to see slow coarse solves early
    if (n_solves++ < 3 &&
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "coarse solve " << n_solves << ": " << timer.wall_time()
                << " s" << std::endl;
  }
};



/**
 * The interface of the one-level Schwarz preconditioners.
 */
struct SchwarzPreconditioner
{
  virtual ~SchwarzPreconditioner() = default;

  virtual void
  vmult(PETScWrappers::MPI::BlockVector       &dst,
        const PETScWrappers::MPI::BlockVector &src) const = 0;

  mutable unsigned int n_applications = 0;
};



/**
 * Restricted additive Schwarz: the local problem of every rank lives on the
 * DoFs of its owned and ghost cells (the locally relevant DoFs), and only
 * the values of the owned DoFs are kept.
 */
struct RestrictedSchwarz : public SchwarzPreconditioner
{
  RestrictedSchwarz(const IndexSet  &owned,
                    const IndexSet  &relevant,
                    SequentialMUMPS &local_solver,
                    const MPI_Comm   comm)
    : owned(owned)
    , relevant(relevant)
    , local_solver(local_solver)
    , comm(comm)
  {}

  const IndexSet  &owned;
  const IndexSet  &relevant;
  SequentialMUMPS &local_solver;
  const MPI_Comm   comm;

  void
  vmult(PETScWrappers::MPI::BlockVector       &dst,
        const PETScWrappers::MPI::BlockVector &src) const override
  {
    const unsigned int         n = relevant.n_elements();
    PETScWrappers::MPI::Vector ghosted(owned, relevant, comm);
    std::vector<double>        x(2 * n);
    for (unsigned int b = 0; b < 2; ++b)
      {
        ghosted = src.block(b);
        unsigned int i = 0;
        for (const types::global_dof_index g : relevant)
          x[b * n + i++] = ghosted(g);
      }
    local_solver.solve(x);
    for (unsigned int b = 0; b < 2; ++b)
      {
        for (const types::global_dof_index g : owned)
          dst.block(b)(g) = x[b * n + relevant.index_within_set(g)];
        dst.block(b).compress(VectorOperation::insert);
      }
    ++n_applications;
  }
};



/**
 * Restricted additive Schwarz with all subdomains on one process.
 */
struct SerialSchwarz : public SchwarzPreconditioner
{
  struct Subdomain
  {
    // the global DoFs of the subdomain (sorted), and the local indices of
    // those whose values are kept
    std::vector<types::global_dof_index> dofs;
    std::vector<unsigned int>            kept;
    std::unique_ptr<SequentialMUMPS>     solver;
  };

  std::vector<Subdomain> subdomains;
  unsigned int           n_global = 0;

  void
  vmult(PETScWrappers::MPI::BlockVector       &dst,
        const PETScWrappers::MPI::BlockVector &src) const override
  {
    std::vector<types::global_dof_index> all(n_global);
    std::iota(all.begin(), all.end(), 0);
    std::array<std::vector<double>, 2> in, out;
    for (unsigned int b = 0; b < 2; ++b)
      {
        in[b].resize(n_global);
        out[b].assign(n_global, 0.);
        src.block(b).extract_subvector_to(all.begin(), all.end(), in[b].begin());
      }
    for (const Subdomain &sub : subdomains)
      {
        const unsigned int  n = sub.dofs.size();
        std::vector<double> x(2 * n);
        for (unsigned int b = 0; b < 2; ++b)
          for (unsigned int i = 0; i < n; ++i)
            x[b * n + i] = in[b][sub.dofs[i]];
        sub.solver->solve(x);
        for (unsigned int b = 0; b < 2; ++b)
          for (const unsigned int i : sub.kept)
            out[b][sub.dofs[i]] = x[b * n + i];
      }
    for (unsigned int b = 0; b < 2; ++b)
      {
        dst.block(b).set(all, out[b]);
        dst.block(b).compress(VectorOperation::insert);
      }
    ++n_applications;
  }
};



/**
 * The correction in the space of gradients: x = -G L^-1 G^T r / k^2 with
 * L = G^T M G (the eps-weighted nodal Laplacian), inverted by BoomerAMG.
 */
class GradientCorrection
{
public:
  GradientCorrection(const PETScWrappers::MPI::SparseMatrix &gradient,
                     const PETScWrappers::MPI::SparseMatrix &M,
                     const IndexSet                         &nodal_owned,
                     const bool                              exact,
                     const MPI_Comm                          comm)
    : gradient(gradient)
    , nodal_owned(nodal_owned)
    , comm(comm)
  {
    PetscErrorCode ierr = MatPtAP(static_cast<Mat>(M),
                                  static_cast<Mat>(gradient),
                                  MAT_INITIAL_MATRIX,
                                  PETSC_DETERMINE,
                                  &L);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    ierr = KSPCreate(comm, &ksp);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    KSPSetOperators(ksp, L, L);
    if (exact)
      {
        KSPSetType(ksp, KSPCG);
        KSPSetTolerances(ksp, 1e-10, 1e-50, PETSC_DEFAULT, 500);
      }
    else
      {
        KSPSetType(ksp, KSPPREONLY);
      }
    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCHYPRE);
    PCHYPRESetType(pc, "boomeramg");
    ierr = KSPSetUp(ksp);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
  }

  ~GradientCorrection()
  {
    KSPDestroy(&ksp);
    MatDestroy(&L);
  }

  GradientCorrection(const GradientCorrection &) = delete;
  GradientCorrection &
  operator=(const GradientCorrection &) = delete;

  /// dst = -G L^-1 G^T src / k^2 for one real block
  void
  apply(PETScWrappers::MPI::Vector       &dst,
        const PETScWrappers::MPI::Vector &src,
        const double                      k) const
  {
    PETScWrappers::MPI::Vector rhs(nodal_owned, comm), phi(nodal_owned, comm);
    gradient.Tvmult(rhs, src);
    const PetscErrorCode ierr =
      KSPSolve(ksp, static_cast<Vec>(rhs), static_cast<Vec>(phi));
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    phi *= -1. / (k * k);
    gradient.vmult(dst, phi);
  }

private:
  const PETScWrappers::MPI::SparseMatrix &gradient;
  const IndexSet                         &nodal_owned;
  const MPI_Comm                          comm;
  Mat                                     L   = nullptr;
  KSP                                     ksp = nullptr;
};



/**
 * Wraps a preconditioner S with the gradient correction C:
 * x = C r, x += S (r - A x), x += C (r - A x).
 */
template <typename Inner>
struct GradientCorrected
{
  const Inner              &inner;
  const ComplexBlockMatrix &system;
  const GradientCorrection &correction;
  const double              k;

  void
  correct(PETScWrappers::MPI::BlockVector       &x,
          const PETScWrappers::MPI::BlockVector &r) const
  {
    PETScWrappers::MPI::BlockVector residual(x), update(x);
    system.vmult(residual, x);
    residual.sadd(-1., 1., r);
    for (unsigned int b = 0; b < 2; ++b)
      correction.apply(update.block(b), residual.block(b), k);
    x += update;
  }

  void
  vmult(PETScWrappers::MPI::BlockVector       &dst,
        const PETScWrappers::MPI::BlockVector &src) const
  {
    dst = 0;
    correct(dst, src);
    PETScWrappers::MPI::BlockVector residual(dst), update(dst);
    system.vmult(residual, dst);
    residual.sadd(-1., 1., src);
    inner.vmult(update, residual);
    dst += update;
    correct(dst, src);
  }
};



/**
 * A coarse correction on top of restricted additive Schwarz. The
 * interpolation from the coarse space is stored by rows for the owned fine
 * DoFs; the coarse problem is solved redundantly on every rank.
 */
struct TwoLevelSchwarz
{
  using Row = std::vector<std::pair<unsigned int, double>>;

  const SchwarzPreconditioner                &schwarz;
  const ComplexBlockMatrix                   &system;
  const std::vector<types::global_dof_index> &owned_indices;
  const std::vector<Row>                     &prolongation;
  CoarseSolver                               &coarse_solver;
  const unsigned int                          n_coarse;
  const bool                                  hybrid;
  const MPI_Comm                              comm;

  /// P^T src, summed over the ranks
  std::vector<double>
  restrict_to_coarse(const PETScWrappers::MPI::BlockVector &src) const
  {
    std::vector<double> coarse(2 * n_coarse, 0.), values(owned_indices.size());
    for (unsigned int b = 0; b < 2; ++b)
      {
        src.block(b).extract_subvector_to(owned_indices.begin(),
                                          owned_indices.end(),
                                          values.begin());
        for (unsigned int i = 0; i < owned_indices.size(); ++i)
          for (const auto &[j, w] : prolongation[i])
            coarse[b * n_coarse + j] += w * values[i];
      }
    MPI_Allreduce(MPI_IN_PLACE,
                  coarse.data(),
                  coarse.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);
    return coarse;
  }

  /// dst = P coarse
  void
  prolongate(PETScWrappers::MPI::BlockVector &dst,
             const std::vector<double>       &coarse) const
  {
    std::vector<double> values(owned_indices.size());
    for (unsigned int b = 0; b < 2; ++b)
      {
        for (unsigned int i = 0; i < owned_indices.size(); ++i)
          {
            double v = 0;
            for (const auto &[j, w] : prolongation[i])
              v += w * coarse[b * n_coarse + j];
            values[i] = v;
          }
        dst.block(b).set(owned_indices, values);
        dst.block(b).compress(VectorOperation::insert);
      }
  }

  /// dst = P A_H^-1 P^T src
  void
  coarse_correction(PETScWrappers::MPI::BlockVector       &dst,
                    const PETScWrappers::MPI::BlockVector &src) const
  {
    std::vector<double> coarse = restrict_to_coarse(src);
    coarse_solver.solve(coarse);
    prolongate(dst, coarse);
  }

  void
  vmult(PETScWrappers::MPI::BlockVector       &dst,
        const PETScWrappers::MPI::BlockVector &src) const
  {
    PETScWrappers::MPI::BlockVector x0(dst);
    coarse_correction(x0, src);

    if (hybrid)
      {
        // Schwarz on the residual after the coarse correction
        PETScWrappers::MPI::BlockVector residual(dst);
        system.vmult(residual, x0);
        residual.sadd(-1., 1., src);
        schwarz.vmult(dst, residual);
      }
    else
      schwarz.vmult(dst, src);
    dst += x0;
  }
};



/**
 * The inverse of the shifted operator by an inner GMRES.
 */
struct ShiftedPreconditioner
{
  const ComplexBlockMatrix          &shifted;
  const BlockDiagonalPreconditioner &block_ams;
  const double                       tolerance;
  mutable unsigned int               n_inner = 0;

  void
  vmult(PETScWrappers::MPI::BlockVector       &dst,
        const PETScWrappers::MPI::BlockVector &src) const
  {
    SolverControl control(500, tolerance * src.l2_norm());
    SolverGMRES<PETScWrappers::MPI::BlockVector>::AdditionalData data;
    data.max_basis_size        = 100;
    data.right_preconditioning = true;
    SolverGMRES<PETScWrappers::MPI::BlockVector> gmres(control, data);
    dst = 0;
    try
      {
        gmres.solve(shifted, dst, src, block_ams);
      }
    catch (const SolverControl::NoConvergence &)
      {}
    n_inner += control.last_step();
  }
};



/**
 * The cube (-R,R)^3 around the ball of radius a as a coarse mesh of
 * tetrahedra. The 7 hexahedra of the ball and the 6 between the sphere and
 * the cube are refined @p hex_refine times as hexahedra, with the faces on
 * the sphere following it and a transfinite interpolation in between. Then
 * every hexahedron is split into 6 pyramids around its center and every
 * pyramid into two tetrahedra, along the diagonal through the vertex with
 * the smallest index (which makes the mesh conforming).
 *
 * The hexahedral refinement is needed for the geometry: later, only the
 * new vertices on the interface are moved to the sphere, by h^2/(8a) for an
 * edge of length h. This must be small compared to h, or the cells next to
 * the interface degenerate.
 *
 * Returns for every tetrahedron whether it is inside the sphere.
 */
std::vector<bool>
create_coarse_mesh(const double       a,
                   const double       R,
                   const unsigned int hex_refine,
                   Triangulation<3>  &tria)
{
  Triangulation<3> ball, shell, hexes;
  GridGenerator::hyper_ball(ball, Point<3>(), a);
  GridGenerator::hyper_shell(shell, Point<3>(), a, R * std::sqrt(3.), 6);
  ball.reset_all_manifolds();
  shell.reset_all_manifolds();
  GridGenerator::merge_triangulations(ball, shell, hexes, 1e-10 * a);
  AssertThrow(hexes.n_cells() == 13 && hexes.n_vertices() == 24,
              ExcInternalError());
  // the cells of the ball come first
  for (const auto &hex : hexes.active_cell_iterators())
    AssertThrow((hex->center().norm() < a) == (hex->index() < 7),
                ExcInternalError());

  const auto on_sphere = [a](const Point<3> &p) {
    return std::abs(p.norm() - a) < 1e-8 * a;
  };
  hexes.set_all_manifold_ids(1);
  for (const auto &hex : hexes.active_cell_iterators())
    for (const unsigned int f : hex->face_indices())
      {
        bool all = true;
        for (const unsigned int v : hex->face(f)->vertex_indices())
          all = all && on_sphere(hex->face(f)->vertex(v));
        if (all)
          hex->face(f)->set_all_manifold_ids(0);
      }
  hexes.set_manifold(0, SphericalManifold<3>(Point<3>()));
  TransfiniteInterpolationManifold<3> transfinite;
  transfinite.initialize(hexes);
  hexes.set_manifold(1, transfinite);
  hexes.refine_global(hex_refine);

  std::vector<Point<3>>    vertices = hexes.get_vertices();
  std::vector<CellData<3>> cells;
  std::vector<bool>        in_sphere;
  for (const auto &hex : hexes.active_cell_iterators())
    {
      auto coarse = Triangulation<3>::cell_iterator(hex);
      while (coarse->level() > 0)
        coarse = coarse->parent();
      const unsigned int center = vertices.size();
      vertices.push_back(hex->center());
      for (const unsigned int f : hex->face_indices())
        {
          // the vertices of the face in cyclic order, smallest index first
          std::array<unsigned int, 4>       quad;
          const std::array<unsigned int, 4> cyclic{{0, 1, 3, 2}};
          for (unsigned int i = 0; i < 4; ++i)
            quad[i] = hex->face(f)->vertex_index(cyclic[i]);
          std::rotate(quad.begin(),
                      std::min_element(quad.begin(), quad.end()),
                      quad.end());
          for (unsigned int t = 0; t < 2; ++t)
            {
              CellData<3> tet(4);
              tet.vertices = {center, quad[0], quad[1 + t], quad[2 + t]};
              const Tensor<1, 3> e1 = vertices[tet.vertices[1]] -
                                      vertices[center],
                                 e2 = vertices[tet.vertices[2]] -
                                      vertices[center],
                                 e3 = vertices[tet.vertices[3]] -
                                      vertices[center];
              if (cross_product_3d(e1, e2) * e3 < 0)
                std::swap(tet.vertices[2], tet.vertices[3]);
              cells.push_back(tet);
              in_sphere.push_back(coarse->index() < 7);
            }
        }
    }
  tria.create_triangulation(vertices, cells, SubCellData());
  return in_sphere;
}



class SphereProblem
{
public:
  SphereProblem(const Parameters &prm)
    : prm(prm)
    , comm(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
    , tria(comm)
    , fe(0)
    , fe_nodal(1)
    , mapping(FE_SimplexP<3>(1))
    , dof_handler(tria)
    , nodal_dh(tria)
    , mie(prm.l, prm.n, prm.a, Tensor<1, 3>({0.3, -0.5, 0.8}))
  {}

  void
  run()
  {
    make_mesh();
    setup_and_assemble();
    for (unsigned int step = 0; step < std::max(1u, prm.k_steps); ++step)
      solve(prm.wave_number(step));
  }

private:
  const Parameters   prm;
  const MPI_Comm     comm;
  ConditionalOStream pcout;

  parallel::distributed::Triangulation<3> tria;
  const FE_NedelecSimplex<3>              fe;
  const FE_SimplexP<3>                    fe_nodal;
  const MappingFE<3>                      mapping;
  DoFHandler<3>                           dof_handler, nodal_dh;
  MieTE                                   mie;

  IndexSet                  owned, relevant;
  AffineConstraints<double> constraints;
  // stiffness, mass with eps, tangential boundary mass, discrete gradient
  PETScWrappers::MPI::SparseMatrix        K, M, M_bdry, gradient;
  std::vector<PETScWrappers::MPI::Vector> constant_fields;
  double                                  interface_error = 0;
  std::vector<bool>                       coarse_cell_in_sphere;

  // the coarse mesh on every rank, its DoFs, and the interpolation to the
  // owned fine DoFs (rows in the order of owned_indices)
  Triangulation<3>                                 coarse_tria;
  DoFHandler<3>                                    coarse_dh;
  std::vector<types::global_dof_index>             owned_indices;
  std::vector<TwoLevelSchwarz::Row>                prolongation;
  // a copy of the coarse matrix for the tests (checks=1)
  mutable std::unique_ptr<ComplexTriplets> coarse_triplets;

  bool
  on_interface(const Point<3> &p) const
  {
    return std::abs(p.norm() - prm.a) < 1e-8 * prm.a;
  }

  // A cell belongs to the sphere if its coarse cell does. (Material ids
  // are not used, since the triangulation does not keep them.)
  bool
  in_sphere(Triangulation<3>::cell_iterator cell) const
  {
    while (cell->level() > 0)
      cell = cell->parent();
    return coarse_cell_in_sphere[cell->index()];
  }

  void
  make_mesh()
  {
    Triangulation<3> coarse;
    coarse_cell_in_sphere =
      create_coarse_mesh(prm.a, prm.R, prm.hex_refine, coarse);
    coarse_tria.copy_triangulation(coarse);
    // the subdomains of ras/oras are the owned and the ghost cells
    if (prm.ghost_overlap_layers > 0)
      tria.set_ghost_overlap(prm.ghost_overlap_level, prm.ghost_overlap_layers);
    tria.copy_triangulation(coarse);

    // New vertices on the interface are placed on the sphere: these are the
    // midpoints of the lines of faces on the interface. Everything else is
    // flat.
    // With flat_interface (for tests), they stay in the plane of the coarse
    // face, so the refined meshes are nested.
    if (!prm.flat_interface)
      {
        tria.set_manifold(1, SphericalManifold<3>(Point<3>()));
        for (const auto &cell : tria.cell_iterators_on_level(0))
          for (const unsigned int f : cell->face_indices())
            {
              bool all = true;
              for (const unsigned int v : cell->face(f)->vertex_indices())
                all = all && on_interface(cell->face(f)->vertex(v));
              if (all)
                cell->face(f)->set_all_manifold_ids(1);
            }
      }

    tria.refine_global(prm.refine);
    for (unsigned int i = 0; i < prm.interface_refine; ++i)
      {
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            for (const unsigned int v : cell->vertex_indices())
              if (on_interface(cell->vertex(v)))
                {
                  cell->set_refine_flag();
                  break;
                }
        tria.execute_coarsening_and_refinement();
      }

    // Check the geometry: the faces between the materials have their
    // vertices on the sphere.
    unsigned int n_interface_faces = 0;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned() && in_sphere(cell))
        for (const unsigned int f : cell->face_indices())
          if (!cell->at_boundary(f) && !in_sphere(cell->neighbor(f)))
            {
              ++n_interface_faces;
              for (const unsigned int v : cell->face(f)->vertex_indices())
                interface_error =
                  std::max(interface_error,
                           std::abs(cell->face(f)->vertex(v).norm() - prm.a));
            }
    // All cells are flat tetrahedra (the manifold only places vertices):
    // check their orientation, since Release builds do not.
    double min_quality = std::numeric_limits<double>::max();
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const Tensor<1, 3> e1 = cell->vertex(1) - cell->vertex(0),
                             e2 = cell->vertex(2) - cell->vertex(0),
                             e3 = cell->vertex(3) - cell->vertex(0);
          min_quality =
            std::min(min_quality,
                     cross_product_3d(e1, e2) * e3 /
                       Utilities::fixed_power<3>(cell->diameter()));
        }
    min_quality = Utilities::MPI::min(min_quality, comm);
    AssertThrow(min_quality > 0,
                ExcMessage("The mesh has inverted cells; use hex_refine >= 1."));
    interface_error   = Utilities::MPI::max(interface_error, comm);
    n_interface_faces = Utilities::MPI::sum(n_interface_faces, comm);
    pcout << "mesh: " << tria.n_global_active_cells() << " cells, "
          << n_interface_faces
          << " interface faces, largest distance of their vertices from the "
             "sphere "
          << interface_error << ", smallest det/diameter^3 " << min_quality
          << std::endl;
  }

  void
  setup_and_assemble()
  {
    dof_handler.distribute_dofs(fe);
    nodal_dh.distribute_dofs(fe_nodal);
    owned    = dof_handler.locally_owned_dofs();
    relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
    const IndexSet &nodal_owned = nodal_dh.locally_owned_dofs();
    const IndexSet  nodal_relevant =
      DoFTools::extract_locally_relevant_dofs(nodal_dh);

    constraints.reinit(owned, relevant);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();
    AffineConstraints<double> nodal_constraints(nodal_owned, nodal_relevant);
    DoFTools::make_hanging_node_constraints(nodal_dh, nodal_constraints);
    nodal_constraints.close();

    DynamicSparsityPattern dsp(relevant);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp, owned, comm, relevant);
    K.reinit(owned, owned, dsp, comm);
    M.reinit(owned, owned, dsp, comm);
    M_bdry.reinit(owned, owned, dsp, comm);

    const QGaussSimplex<3> q_cell(3);
    const QGaussSimplex<2> q_face(3);
    FEValues<3>            fe_values(mapping,
                          fe,
                          q_cell,
                          update_values | update_gradients | update_JxW_values);
    FEFaceValues<3>        fe_face_values(mapping,
                                   fe,
                                   q_face,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);
    const FEValuesExtractors::Vector E(0);
    const unsigned int               n_dofs = fe.n_dofs_per_cell();
    FullMatrix<double> stiffness(n_dofs, n_dofs), mass(n_dofs, n_dofs),
      boundary_mass(n_dofs, n_dofs);
    std::vector<types::global_dof_index> dof_indices(n_dofs);
    std::vector<Tensor<1, 3>>            tangential(n_dofs);
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          stiffness     = 0;
          mass          = 0;
          boundary_mass = 0;
          const double eps = (in_sphere(cell) ? prm.n * prm.n : 1.);
          for (unsigned int q = 0; q < q_cell.size(); ++q)
            for (unsigned int i = 0; i < n_dofs; ++i)
              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  stiffness(i, j) += fe_values[E].curl(i, q) *
                                     fe_values[E].curl(j, q) * fe_values.JxW(q);
                  mass(i, j) += eps * fe_values[E].value(i, q) *
                                fe_values[E].value(j, q) * fe_values.JxW(q);
                }
          for (const unsigned int f : cell->face_indices())
            if (cell->at_boundary(f))
              {
                fe_face_values.reinit(cell, f);
                for (unsigned int q = 0; q < q_face.size(); ++q)
                  {
                    const Tensor<1, 3> normal = fe_face_values.normal_vector(q);
                    for (unsigned int i = 0; i < n_dofs; ++i)
                      {
                        const Tensor<1, 3> v = fe_face_values[E].value(i, q);
                        tangential[i]        = v - (v * normal) * normal;
                      }
                    for (unsigned int i = 0; i < n_dofs; ++i)
                      for (unsigned int j = 0; j < n_dofs; ++j)
                        boundary_mass(i, j) += tangential[i] * tangential[j] *
                                               fe_face_values.JxW(q);
                  }
              }
          cell->get_dof_indices(dof_indices);
          // Only K gets diagonal entries for the constrained DoFs, so that
          // they stay positive in every combination of the matrices.
          constraints.distribute_local_to_global(stiffness, dof_indices, K);
          constraints.distribute_local_to_global(mass,
                                                 dof_indices,
                                                 dof_indices,
                                                 M);
          constraints.distribute_local_to_global(boundary_mass,
                                                 dof_indices,
                                                 dof_indices,
                                                 M_bdry);
        }
    K.compress(VectorOperation::add);
    M.compress(VectorOperation::add);
    M_bdry.compress(VectorOperation::add);

    // discrete gradient and the edge vectors of the constant fields for AMS
    DynamicSparsityPattern gradient_dsp(dof_handler.n_dofs(),
                                        nodal_dh.n_dofs(),
                                        owned);
    DoFTools::make_discrete_gradient_sparsity_pattern(
      dof_handler, constraints, nodal_dh, nodal_constraints, gradient_dsp);
    gradient.reinit(owned, nodal_owned, gradient_dsp, comm);
    DoFTools::make_discrete_gradient(
      dof_handler, constraints, nodal_dh, nodal_constraints, gradient);
    const std::map<types::global_dof_index, Point<3>> support_points =
      DoFTools::map_dofs_to_support_points(mapping, nodal_dh);
    constant_fields.resize(3);
    for (unsigned int d = 0; d < 3; ++d)
      {
        PETScWrappers::MPI::Vector coordinate(nodal_owned, comm);
        for (const types::global_dof_index dof : nodal_owned)
          coordinate(dof) = support_points.at(dof)[d];
        coordinate.compress(VectorOperation::insert);
        constant_fields[d].reinit(owned, comm);
        gradient.vmult(constant_fields[d], coordinate);
      }
    {
      unsigned int n_ghost = 0;
      for (const auto &cell : tria.active_cell_iterators())
        n_ghost += cell->is_ghost();
      pcout << "ghost cells: " << Utilities::MPI::sum(n_ghost, comm)
            << " (sum over the processes), owned and ghost DoFs per process "
            << Utilities::MPI::sum<unsigned long>(relevant.n_elements(), comm) /
                 Utilities::MPI::n_mpi_processes(comm)
            << " (mean)" << std::endl;
    }
    pcout << "dofs: " << 2 * dof_handler.n_dofs() << " (real and imaginary), "
          << constraints.n_constraints() << " local hanging node constraints"
          << std::endl;

    owned_indices.assign(owned.begin(), owned.end());
    if (prm.coarse != "none")
      build_prolongation();
  }

  /**
   * The rows of the interpolation from the coarse Nedelec space to the
   * owned fine DoFs: the DoF of a fine line is the line integral of the
   * tangential component, which for the coarse functions (linear on every
   * coarse cell) is the value at the midpoint times (v1 - v0). Every fine
   * line lies in a coarse cell; on coarse faces both sides give the same
   * tangential component. Lines on the interface moved onto the sphere lie
   * slightly outside their coarse cell, where the affine coarse functions
   * are extrapolated.
   */
  void
  build_prolongation()
  {
    coarse_dh.reinit(coarse_tria);
    coarse_dh.distribute_dofs(fe);
    prolongation.assign(owned_indices.size(), {});
    std::vector<bool> done(owned_indices.size(), false);

    const FEValuesExtractors::Vector     E(0);
    const unsigned int                   n_dofs = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> dof_indices(n_dofs),
      coarse_indices(n_dofs);
    double max_center_distance = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() || cell->is_ghost())
        {
          cell->get_dof_indices(dof_indices);
          std::vector<unsigned int> lines;
          for (unsigned int i = 0; i < n_dofs; ++i)
            if (owned.is_element(dof_indices[i]) &&
                !done[owned.index_within_set(dof_indices[i])])
              lines.push_back(i);
          if (lines.empty())
            continue;

          Triangulation<3>::cell_iterator ancestor = cell;
          while (ancestor->level() > 0)
            ancestor = ancestor->parent();
          const DoFHandler<3>::active_cell_iterator coarse_cell(
            &coarse_tria, 0, ancestor->index(), &coarse_dh);
          max_center_distance =
            std::max(max_center_distance,
                     (coarse_cell->center() - ancestor->center()).norm());

          // the line i of the cell carries its DoF i
          std::vector<Point<3>>     points;
          std::vector<Tensor<1, 3>> tangents;
          for (const unsigned int i : lines)
            {
              const Point<3> v0 = cell->line(i)->vertex(0),
                             v1 = cell->line(i)->vertex(1);
              points.push_back(mapping.transform_real_to_unit_cell(
                coarse_cell, Point<3>(0.5 * (v0 + v1))));
              tangents.push_back(v1 - v0);
            }
          FEValues<3> fe_values(mapping,
                                fe,
                                Quadrature<3>(points),
                                update_values);
          fe_values.reinit(coarse_cell);
          coarse_cell->get_dof_indices(coarse_indices);
          for (unsigned int p = 0; p < lines.size(); ++p)
            {
              const unsigned int row =
                owned.index_within_set(dof_indices[lines[p]]);
              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  const double w =
                    fe_values[E].value(j, p) * tangents[p];
                  if (std::abs(w) > 1e-14)
                    prolongation[row].emplace_back(coarse_indices[j], w);
                }
              done[row] = true;
            }
        }
    AssertThrow(std::all_of(done.begin(), done.end(), [](bool d) { return d; }),
                ExcInternalError());
    max_center_distance = Utilities::MPI::max(max_center_distance, comm);
    AssertThrow(max_center_distance < 1e-10 * prm.a,
                ExcMessage("The coarse cells do not match the level-0 cells."));
    // Check: a linear field F(x) = c + d x x lies in both Nedelec spaces,
    // so the interpolation of its coarse DoFs gives its fine DoFs.
    {
      const Tensor<1, 3> c({1., 2., 3.}), d({0.3, -0.2, 0.5});
      const auto F = [&](const Point<3> &x) {
        return c + cross_product_3d(d, Tensor<1, 3>(x));
      };
      std::vector<double> coarse_values(coarse_dh.n_dofs());
      for (const auto &cell : coarse_dh.active_cell_iterators())
        {
          cell->get_dof_indices(coarse_indices);
          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const Point<3> v0 = cell->line(i)->vertex(0),
                             v1 = cell->line(i)->vertex(1);
              coarse_values[coarse_indices[i]] =
                F(Point<3>(0.5 * (v0 + v1))) * (v1 - v0);
            }
        }
      std::vector<double> fine_values(owned_indices.size());
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() || cell->is_ghost())
          {
            cell->get_dof_indices(dof_indices);
            for (unsigned int i = 0; i < n_dofs; ++i)
              if (owned.is_element(dof_indices[i]))
                {
                  const Point<3> v0 = cell->line(i)->vertex(0),
                                 v1 = cell->line(i)->vertex(1);
                  fine_values[owned.index_within_set(dof_indices[i])] =
                    F(Point<3>(0.5 * (v0 + v1))) * (v1 - v0);
                }
          }
      double error = 0, scale = 0;
      for (unsigned int i = 0; i < owned_indices.size(); ++i)
        {
          double v = 0;
          for (const auto &[j, w] : prolongation[i])
            v += w * coarse_values[j];
          error = std::max(error, std::abs(v - fine_values[i]));
          scale = std::max(scale, std::abs(fine_values[i]));
        }
      error = Utilities::MPI::max(error, comm) / Utilities::MPI::max(scale, comm);
      pcout << "coarse space: " << 2 * coarse_dh.n_dofs()
            << " unknowns (real and imaginary), interpolation error of a "
               "linear field "
            << error << std::endl;
    }
  }

  /**
   * The Galerkin coarse operator P^T (A + i B) P with the global
   * interpolation P, computed by PETSc (MatPtAP) and copied to every rank.
   * coarse_shift adds -i coarse_shift k^2 M to B before the product.
   */
  void
  factorize_algebraic_coarse_problem(const double                            k,
                                     const PETScWrappers::MPI::SparseMatrix &A,
                                     const PETScWrappers::MPI::SparseMatrix &B,
                                     CoarseSolver &coarse_solver) const
  {
    const PetscInt n_coarse = coarse_dh.n_dofs();
    Mat            P_mat;
    PetscErrorCode ierr = MatCreateAIJ(comm,
                                       owned_indices.size(),
                                       PETSC_DECIDE,
                                       PETSC_DETERMINE,
                                       n_coarse,
                                       12,
                                       nullptr,
                                       12,
                                       nullptr,
                                       &P_mat);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    MatSetOption(P_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    for (unsigned int i = 0; i < owned_indices.size(); ++i)
      {
        const PetscInt        row = owned_indices[i];
        std::vector<PetscInt> cols;
        std::vector<PetscScalar> vals;
        for (const auto &[j, w] : prolongation[i])
          {
            cols.push_back(j);
            vals.push_back(w);
          }
        MatSetValues(P_mat,
                     1,
                     &row,
                     cols.size(),
                     cols.data(),
                     vals.data(),
                     INSERT_VALUES);
      }
    MatAssemblyBegin(P_mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(P_mat, MAT_FINAL_ASSEMBLY);

    PETScWrappers::MPI::SparseMatrix B_coarse;
    B_coarse.reinit(K);
    B_coarse.copy_from(B);
    if (prm.coarse_shift != 0.)
      B_coarse.add(-prm.coarse_shift * k * k, M);

    // the real and the imaginary part, distributed by rows
    const bool         distributed = (prm.coarse_solver == "distributed");
    std::array<Mat, 2> products, copies = {{nullptr, nullptr}};
    const std::array<Mat, 2> parts = {{static_cast<Mat>(A),
                                       static_cast<Mat>(B_coarse)}};
    for (unsigned int part = 0; part < 2; ++part)
      {
        ierr = MatPtAP(parts[part],
                       P_mat,
                       MAT_INITIAL_MATRIX,
                       PETSC_DETERMINE,
                       &products[part]);
        AssertThrow(ierr == 0, ExcPETScError(ierr));
      }
    if (distributed)
      {
        // every rank provides the entries of its rows of the real block
        // form [A_H -B_H; B_H A_H]
        std::vector<MUMPS_INT> rows, cols;
        std::vector<double>    values;
        for (unsigned int part = 0; part < 2; ++part)
          {
            PetscInt row_begin, row_end;
            MatGetOwnershipRange(products[part], &row_begin, &row_end);
            for (PetscInt row = row_begin; row < row_end; ++row)
              {
                PetscInt           n_cols;
                const PetscInt    *cols_row;
                const PetscScalar *vals;
                MatGetRow(products[part], row, &n_cols, &cols_row, &vals);
                for (PetscInt e = 0; e < n_cols; ++e)
                  {
                    const PetscInt col = cols_row[e];
                    const double   v   = vals[e];
                    if (part == 0)
                      {
                        rows.push_back(row);
                        cols.push_back(col);
                        values.push_back(v);
                        rows.push_back(row + n_coarse);
                        cols.push_back(col + n_coarse);
                        values.push_back(v);
                      }
                    else
                      {
                        rows.push_back(row);
                        cols.push_back(col + n_coarse);
                        values.push_back(-v);
                        rows.push_back(row + n_coarse);
                        cols.push_back(col);
                        values.push_back(v);
                      }
                  }
                MatRestoreRow(products[part], row, &n_cols, &cols_row, &vals);
              }
          }
        Timer timer;
        coarse_solver.distributed =
          std::make_unique<DistributedMUMPS>(comm, prm.coarse_procs);
        coarse_solver.distributed->factorize(2 * n_coarse, rows, cols, values);
        MPI_Barrier(comm);
        pcout << "coarse factorization: distributed MUMPS on "
              << coarse_solver.distributed->n_processes() << " processes, "
              << timer.wall_time() << " s" << std::endl;
      }
    // the redundant copies: for the redundant solve, and for the tests
    if (!distributed || prm.checks)
      for (unsigned int part = 0; part < 2; ++part)
      {
        ierr = MatCreateRedundantMatrix(products[part],
                                        Utilities::MPI::n_mpi_processes(comm),
                                        MPI_COMM_NULL,
                                        MAT_INITIAL_MATRIX,
                                        &copies[part]);
        AssertThrow(ierr == 0, ExcPETScError(ierr));
      }
    ComplexTriplets triplets;
    if (!distributed || prm.checks)
    for (unsigned int part = 0; part < 2; ++part)
      for (PetscInt row = 0; row < n_coarse; ++row)
        {
          PetscInt           n_cols;
          const PetscInt    *cols;
          const PetscScalar *vals;
          MatGetRow(copies[part], row, &n_cols, &cols, &vals);
          for (PetscInt e = 0; e < n_cols; ++e)
            if (part == 0)
              triplets.add(row, cols[e], vals[e], 0.);
            else
              triplets.add(row, cols[e], 0., vals[e]);
          MatRestoreRow(copies[part], row, &n_cols, &cols, &vals);
        }
    for (unsigned int part = 0; part < 2; ++part)
      {
        MatDestroy(&products[part]);
        if (copies[part] != nullptr)
          MatDestroy(&copies[part]);
      }
    MatDestroy(&P_mat);
    if (!distributed)
      triplets.factorize(n_coarse, coarse_solver.redundant);
    if (prm.checks)
      coarse_triplets = std::make_unique<ComplexTriplets>(triplets);
  }

  /**
   * The Galerkin coarse operator P^T A P with A = K - k^2 (1 + i
   * coarse_shift) eps M - i k M_bdry, assembled from the fine cell
   * matrices and the interpolation from the coarse ancestor of each fine
   * cell, and summed over the ranks.
   */
  void
  factorize_galerkin_coarse_problem(const double     k,
                                    SequentialMUMPS &mumps) const
  {
    DynamicSparsityPattern coarse_dsp(coarse_dh.n_dofs());
    DoFTools::make_sparsity_pattern(coarse_dh, coarse_dsp);
    SparsityPattern sparsity;
    sparsity.copy_from(coarse_dsp);
    std::vector<double> values(2 * sparsity.n_nonzero_elements(), 0.);
    const std::size_t   nnz = sparsity.n_nonzero_elements();

    const QGaussSimplex<3> q_cell(3);
    const QGaussSimplex<2> q_face(3);
    FEValues<3>            fe_values(mapping,
                          fe,
                          q_cell,
                          update_values | update_gradients | update_JxW_values);
    FEFaceValues<3>        fe_face_values(mapping,
                                   fe,
                                   q_face,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);
    const FEValuesExtractors::Vector     E(0);
    const unsigned int                   n_dofs = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> coarse_indices(n_dofs);
    std::vector<Tensor<1, 3>>            tangential(n_dofs);
    FullMatrix<double> A_re(n_dofs, n_dofs), A_im(n_dofs, n_dofs),
      P_c(n_dofs, n_dofs), tmp(n_dofs, n_dofs), G(n_dofs, n_dofs);
    unsigned int n_rows_checked = 0, n_rows_differ = 0;
    double       max_row_difference = 0, min_distance_differing = 1e300,
           max_distance_differing = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const double eps = (in_sphere(cell) ? prm.n * prm.n : 1.);
          A_re             = 0;
          A_im             = 0;
          fe_values.reinit(cell);
          for (unsigned int q = 0; q < q_cell.size(); ++q)
            for (unsigned int i = 0; i < n_dofs; ++i)
              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  const double m = fe_values[E].value(i, q) *
                                   fe_values[E].value(j, q) * fe_values.JxW(q);
                  A_re(i, j) += fe_values[E].curl(i, q) *
                                  fe_values[E].curl(j, q) * fe_values.JxW(q) -
                                k * k * eps * m;
                  A_im(i, j) -= prm.coarse_shift * k * k * eps * m;
                }
          for (const unsigned int f : cell->face_indices())
            if (cell->at_boundary(f))
              {
                fe_face_values.reinit(cell, f);
                for (unsigned int q = 0; q < q_face.size(); ++q)
                  {
                    const Tensor<1, 3> normal = fe_face_values.normal_vector(q);
                    for (unsigned int i = 0; i < n_dofs; ++i)
                      {
                        const Tensor<1, 3> v = fe_face_values[E].value(i, q);
                        tangential[i]        = v - (v * normal) * normal;
                      }
                    for (unsigned int i = 0; i < n_dofs; ++i)
                      for (unsigned int j = 0; j < n_dofs; ++j)
                        A_im(i, j) -= k * tangential[i] * tangential[j] *
                                      fe_face_values.JxW(q);
                  }
              }

          // P_c(i, j): fine DoF i from coarse DoF j of the coarse ancestor
          Triangulation<3>::cell_iterator ancestor = cell;
          while (ancestor->level() > 0)
            ancestor = ancestor->parent();
          const DoFHandler<3>::active_cell_iterator coarse_cell(
            &coarse_tria, 0, ancestor->index(), &coarse_dh);
          std::vector<Point<3>>     points(n_dofs);
          std::vector<Tensor<1, 3>> tangents(n_dofs);
          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const Point<3> v0 = cell->line(i)->vertex(0),
                             v1 = cell->line(i)->vertex(1);
              points[i] = mapping.transform_real_to_unit_cell(
                coarse_cell, Point<3>(0.5 * (v0 + v1)));
              tangents[i] = v1 - v0;
            }
          FEValues<3> coarse_values(mapping,
                                    fe,
                                    Quadrature<3>(points),
                                    update_values);
          coarse_values.reinit(coarse_cell);
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              P_c(i, j) = coarse_values[E].value(j, i) * tangents[i];
          coarse_cell->get_dof_indices(coarse_indices);

          // test: the rows of P_c equal the rows of the global
          // interpolation for the owned fine DoFs
          if (prm.checks)
            {
              std::vector<types::global_dof_index> fine_indices(n_dofs);
              cell->get_dof_indices(fine_indices);
              for (unsigned int i = 0; i < n_dofs; ++i)
                if (owned.is_element(fine_indices[i]))
                  {
                    const auto &row =
                      prolongation[owned.index_within_set(fine_indices[i])];
                    std::map<unsigned int, double> global_row(row.begin(),
                                                              row.end());
                    double difference = 0;
                    for (unsigned int j = 0; j < n_dofs; ++j)
                      {
                        const auto it = global_row.find(coarse_indices[j]);
                        const double g = (it == global_row.end() ? 0. : it->second);
                        difference = std::max(difference, std::abs(P_c(i, j) - g));
                        if (it != global_row.end())
                          global_row.erase(it);
                      }
                    // entries of the global row outside this coarse cell
                    for (const auto &[jj, g] : global_row)
                      difference = std::max(difference, std::abs(g));
                    if (difference > 1e-10)
                      {
                        ++n_rows_differ;
                        const Point<3> mid = cell->line(i)->center();
                        max_row_difference =
                          std::max(max_row_difference, difference);
                        min_distance_differing =
                          std::min(min_distance_differing,
                                   std::abs(mid.norm() - prm.a));
                        max_distance_differing =
                          std::max(max_distance_differing,
                                   std::abs(mid.norm() - prm.a));
                      }
                    ++n_rows_checked;
                  }
            }

          for (unsigned int part = 0; part < 2; ++part)
            {
              (part == 0 ? A_re : A_im).mmult(tmp, P_c);
              P_c.Tmmult(G, tmp);
              for (unsigned int a = 0; a < n_dofs; ++a)
                for (unsigned int b = 0; b < n_dofs; ++b)
                  values[part * nnz +
                         sparsity(coarse_indices[a], coarse_indices[b])] +=
                    G(a, b);
            }
        }
    if (prm.checks)
      {
        n_rows_checked = Utilities::MPI::sum(n_rows_checked, comm);
        n_rows_differ  = Utilities::MPI::sum(n_rows_differ, comm);
        max_row_difference = Utilities::MPI::max(max_row_difference, comm);
        min_distance_differing =
          Utilities::MPI::min(min_distance_differing, comm);
        max_distance_differing =
          Utilities::MPI::max(max_distance_differing, comm);
        pcout << "check: rows of P_c = rows of P: " << n_rows_differ << " of "
              << n_rows_checked << " (cell, row) pairs differ"
              << (n_rows_differ == 0 ? " OK" : " FAILED");
        if (n_rows_differ > 0)
          pcout << ", max difference " << max_row_difference
                << ", distance of their edge midpoints from the sphere "
                << min_distance_differing << " to " << max_distance_differing;
        pcout << std::endl;
      }
    MPI_Allreduce(MPI_IN_PLACE,
                  values.data(),
                  values.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);
    ComplexTriplets triplets;
    for (unsigned int row = 0; row < sparsity.n_rows(); ++row)
      for (auto it = sparsity.begin(row); it != sparsity.end(row); ++it)
        {
          const std::size_t e = sparsity(row, it->column());
          triplets.add(row, it->column(), values[e], values[nnz + e]);
        }
    triplets.factorize(coarse_dh.n_dofs(), mumps);
    if (prm.checks)
      coarse_triplets = std::make_unique<ComplexTriplets>(triplets);
  }

  /**
   * The coarse problem with the mass term k^2 (1 + i coarse_shift) eps M
   * and the impedance condition on the outer boundary.
   */
  void
  factorize_coarse_problem(const double k, SequentialMUMPS &mumps) const
  {
    if (prm.coarse_op == "galerkin")
      {
        factorize_galerkin_coarse_problem(k, mumps);
        return;
      }
    const QGaussSimplex<3> q_cell(3);
    const QGaussSimplex<2> q_face(3);
    FEValues<3>            fe_values(mapping,
                          fe,
                          q_cell,
                          update_values | update_gradients | update_JxW_values);
    FEFaceValues<3>        fe_face_values(mapping,
                                   fe,
                                   q_face,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);
    const FEValuesExtractors::Vector     E(0);
    const unsigned int                   n_dofs = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> dof_indices(n_dofs);
    std::vector<Tensor<1, 3>>            tangential(n_dofs);
    ComplexTriplets                      triplets;
    for (const auto &cell : coarse_dh.active_cell_iterators())
      {
        cell->get_dof_indices(dof_indices);
        const double eps =
          (coarse_cell_in_sphere[cell->index()] ? prm.n * prm.n : 1.);
        fe_values.reinit(cell);
        for (unsigned int q = 0; q < q_cell.size(); ++q)
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              {
                const double m = fe_values[E].value(i, q) *
                                 fe_values[E].value(j, q) * fe_values.JxW(q);
                triplets.add(dof_indices[i],
                             dof_indices[j],
                             fe_values[E].curl(i, q) *
                                 fe_values[E].curl(j, q) * fe_values.JxW(q) -
                               k * k * eps * m,
                             -prm.coarse_shift * k * k * eps * m);
              }
        for (const unsigned int f : cell->face_indices())
          if (cell->at_boundary(f))
            {
              fe_face_values.reinit(cell, f);
              for (unsigned int q = 0; q < q_face.size(); ++q)
                {
                  const Tensor<1, 3> normal = fe_face_values.normal_vector(q);
                  for (unsigned int i = 0; i < n_dofs; ++i)
                    {
                      const Tensor<1, 3> v = fe_face_values[E].value(i, q);
                      tangential[i]        = v - (v * normal) * normal;
                    }
                  for (unsigned int i = 0; i < n_dofs; ++i)
                    for (unsigned int j = 0; j < n_dofs; ++j)
                      triplets.add(dof_indices[i],
                                   dof_indices[j],
                                   0.,
                                   -k * tangential[i] * tangential[j] *
                                     fe_face_values.JxW(q));
                }
            }
      }
    triplets.factorize(coarse_dh.n_dofs(), mumps);
    if (prm.checks)
      coarse_triplets = std::make_unique<ComplexTriplets>(triplets);
  }

  /**
   * The local problem of this rank on its owned and ghost cells, with
   * n x E = 0 (ras) or the impedance condition (oras) on the faces to
   * artificial cells.
   */
  void
  factorize_subdomain_problem(const double k, SequentialMUMPS &mumps) const
  {
    ComplexTriplets triplets;
    assemble_subdomain_problem(
      k,
      [](const DoFHandler<3>::active_cell_iterator &cell) {
        return cell->is_locally_owned() || cell->is_ghost();
      },
      [this](const types::global_dof_index g) {
        return relevant.index_within_set(g);
      },
      triplets);
    triplets.factorize(relevant.n_elements(), mumps);
    if (prm.checks)
      check_subdomain_matrix(triplets, k);
  }

  /**
   * The local problem on the cells for which in_subdomain is true, with
   * n x E = 0 (ras) or the Robin condition (oras) on the faces to cells
   * outside; local_index maps the global DoFs of these cells to the local
   * numbering.
   */
  template <typename InSubdomain, typename LocalIndex>
  void
  assemble_subdomain_problem(const double       k,
                             const InSubdomain &in_subdomain,
                             const LocalIndex  &local_index,
                             ComplexTriplets   &triplets) const
  {
    AssertThrow(constraints.n_constraints() == 0,
                ExcMessage("The Schwarz preconditioners are only implemented "
                           "for meshes without hanging nodes."));
    const bool             impedance =
      (prm.precond == "oras" || prm.precond == "oras_serial");
    const QGaussSimplex<3> q_cell(3);
    const QGaussSimplex<2> q_face(3);
    FEValues<3>            fe_values(mapping,
                          fe,
                          q_cell,
                          update_values | update_gradients | update_JxW_values);
    FEFaceValues<3>        fe_face_values(mapping,
                                   fe,
                                   q_face,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);
    const FEValuesExtractors::Vector     E(0);
    const unsigned int                   n_dofs = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> dof_indices(n_dofs);
    std::vector<unsigned int>            local(n_dofs);
    std::vector<Tensor<1, 3>>            tangential(n_dofs);
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (in_subdomain(cell))
        {
          cell->get_dof_indices(dof_indices);
          for (unsigned int i = 0; i < n_dofs; ++i)
            local[i] = local_index(dof_indices[i]);
          const double eps = (in_sphere(cell) ? prm.n * prm.n : 1.);
          fe_values.reinit(cell);
          for (unsigned int q = 0; q < q_cell.size(); ++q)
            for (unsigned int i = 0; i < n_dofs; ++i)
              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  const double m = fe_values[E].value(i, q) *
                                   fe_values[E].value(j, q) * fe_values.JxW(q);
                  triplets.add(local[i],
                               local[j],
                               fe_values[E].curl(i, q) *
                                   fe_values[E].curl(j, q) * fe_values.JxW(q) -
                                 k * k * eps * m,
                               -prm.local_shift * k * k * eps * m);
                }
          for (const unsigned int f : cell->face_indices())
            {
              // the outer boundary, or the boundary of the subdomain
              const bool outer = cell->at_boundary(f);
              const bool artificial =
                !outer && !in_subdomain(cell->neighbor(f));
              if (!outer && !artificial)
                continue;
              if (artificial && !impedance)
                {
                  for (unsigned int i = 0; i < n_dofs; ++i)
                    if (fe.has_support_on_face(i, f))
                      triplets.set_identity_row_and_column(local[i]);
                  continue;
                }
              // p int E_T v_T with p = q - i k_loc (q = 0 on the outer
              // boundary)
              const double k_loc =
                (outer ? k : prm.impedance_scale * k * std::sqrt(eps));
              const double q =
                (outer ? 0. : prm.robin_real / cell->face(f)->diameter());
              fe_face_values.reinit(cell, f);
              for (unsigned int qp = 0; qp < q_face.size(); ++qp)
                {
                  const Tensor<1, 3> normal = fe_face_values.normal_vector(qp);
                  for (unsigned int i = 0; i < n_dofs; ++i)
                    {
                      const Tensor<1, 3> v = fe_face_values[E].value(i, qp);
                      tangential[i]        = v - (v * normal) * normal;
                    }
                  for (unsigned int i = 0; i < n_dofs; ++i)
                    for (unsigned int j = 0; j < n_dofs; ++j)
                      {
                        const double m = tangential[i] * tangential[j] *
                                         fe_face_values.JxW(qp);
                        triplets.add(local[i], local[j], q * m, -k_loc * m);
                      }
                }
            }
        }
  }

  /**
   * The Ritz values of the preconditioned operator A M from prm.ritz steps
   * of unrestarted, right-preconditioned GMRES: a summary on the screen,
   * all values in ritz_<parameters>.txt next to the CSV file (or in the
   * working directory).
   */
  template <typename Preconditioner>
  void
  analyze_ritz_values(const ComplexBlockMatrix              &system,
                      const Preconditioner                  &preconditioner,
                      const PETScWrappers::MPI::BlockVector &rhs,
                      const double                           k) const
  {
    PETScWrappers::MPI::BlockVector x(rhs);
    x = 0;
    SolverControl control(prm.ritz, 1e-14 * rhs.l2_norm());
    SolverGMRES<PETScWrappers::MPI::BlockVector>::AdditionalData data;
    data.max_basis_size        = prm.ritz + 2;
    data.right_preconditioning = true;
    SolverGMRES<PETScWrappers::MPI::BlockVector> gmres(control, data);
    std::vector<std::complex<double>>              ritz_values;
    gmres.connect_eigenvalues_slot(
      [&ritz_values](const std::vector<std::complex<double>> &values) {
        ritz_values = values;
      });
    try
      {
        gmres.solve(system, x, rhs, preconditioner);
      }
    catch (const SolverControl::NoConvergence &)
      {}
    double       min_abs = 1e300, max_abs = 0;
    unsigned int negative = 0, small = 0;
    for (const auto &v : ritz_values)
      {
        min_abs = std::min(min_abs, std::abs(v));
        max_abs = std::max(max_abs, std::abs(v));
        negative += (v.real() < 0);
        small += (std::abs(v) < 0.1);
      }
    pcout << "ritz: " << ritz_values.size() << " Ritz values after "
          << control.last_step() << " GMRES steps (residual "
          << control.last_value() / rhs.l2_norm() << "): |lambda| in ["
          << min_abs << ", " << max_abs << "], " << negative
          << " with Re < 0, " << small << " with |lambda| < 0.1" << std::endl;
    if (Utilities::MPI::this_mpi_process(comm) == 0)
      {
        std::string directory = ".";
        if (const auto slash = prm.csv.rfind('/'); slash != std::string::npos)
          directory = prm.csv.substr(0, slash);
        const std::string name =
          directory + "/ritz_" + prm.precond + "_k" + std::to_string(k) +
          "_N" + std::to_string(prm.subdomains) + "_shift" +
          std::to_string(prm.local_shift) + "_imp" +
          std::to_string(prm.impedance_scale) + "_" + prm.overlap_mode +
          std::to_string(prm.overlap_layers) + "_l" +
          std::to_string(prm.overlap_level) + "_r" +
          std::to_string(prm.refine) + "_coarse-" + prm.coarse + ".txt";
        std::ofstream out(name);
        out << "# real imag\n" << std::setprecision(10);
        for (const auto &v : ritz_values)
          out << v.real() << ' ' << v.imag() << '\n';
      }
  }

  /**
   * Test of the one-level iteration x <- x + M (b - A x): estimates the
   * spectral radius of I - M A by power iteration from a random vector,
   * and the contraction of one step for the interpolant of the exact
   * solution (a smooth error).
   */
  void
  check_schwarz_iteration(const SchwarzPreconditioner &schwarz,
                          const ComplexBlockMatrix    &system) const
  {
    PETScWrappers::MPI::BlockVector e(std::vector<IndexSet>{owned, owned},
                                      comm),
      Ae(e), MAe(e);
    const auto step = [&](PETScWrappers::MPI::BlockVector &v) {
      system.vmult(Ae, v);
      schwarz.vmult(MAe, Ae);
      v -= MAe;
    };
    for (unsigned int b = 0; b < 2; ++b)
      {
        for (const types::global_dof_index g : owned)
          e.block(b)(g) = std::sin(2.7 * g + 0.4 * b);
        e.block(b).compress(VectorOperation::insert);
      }
    double ratio = 0;
    for (unsigned int it = 0; it < 30; ++it)
      {
        const double before = e.l2_norm();
        step(e);
        ratio = e.l2_norm() / before;
        e /= e.l2_norm();
      }
    pcout << "check: spectral radius of I - M A (power iteration, 30 steps): "
          << ratio << (ratio < 1 ? " (stationary iteration converges)" :
                                   " (stationary iteration diverges)")
          << std::endl;

    // the interpolant of the exact solution
    const unsigned int                   n_dofs = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> dof_indices(n_dofs);
    Tensor<1, 3>                         e_r, e_i, c_r, c_i;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(dof_indices);
          for (unsigned int i = 0; i < n_dofs; ++i)
            if (owned.is_element(dof_indices[i]))
              {
                const Point<3> v0 = cell->line(i)->vertex(0),
                               v1 = cell->line(i)->vertex(1);
                mie.evaluate(Point<3>(0.5 * (v0 + v1)), e_r, e_i, c_r, c_i);
                e.block(0)(dof_indices[i]) = e_r * (v1 - v0);
                e.block(1)(dof_indices[i]) = e_i * (v1 - v0);
              }
        }
    e.compress(VectorOperation::insert);
    const double before = e.l2_norm();
    step(e);
    pcout << "check: one Schwarz step on the interpolant of the exact "
             "solution: |e1| / |e0| = "
          << e.l2_norm() / before << std::endl;
  }

  /**
   * The subdomains of oras_serial / ras_serial (see the description at the
   * top), their local problems, and the test that the rows of the kept
   * DoFs equal the global rows (checks=1).
   */
  void
  setup_serial_schwarz(const double k, SerialSchwarz &schwarz) const
  {
    AssertThrow(Utilities::MPI::n_mpi_processes(comm) == 1,
                ExcMessage("oras_serial and ras_serial run on one process."));
    const unsigned int n_parts = prm.subdomains;
    const unsigned int n_cells = tria.n_active_cells();

    // the parts: contiguous in CellId order
    std::vector<DoFHandler<3>::active_cell_iterator> cells;
    for (const auto &cell : dof_handler.active_cell_iterators())
      cells.push_back(cell);
    std::sort(cells.begin(), cells.end(), [](const auto &a, const auto &b) {
      return a->id() < b->id();
    });
    std::vector<unsigned int> part(n_cells);
    for (unsigned int c = 0; c < n_cells; ++c)
      part[cells[c]->active_cell_index()] =
        static_cast<unsigned long>(c) * n_parts / n_cells;

    // vertex -> active cells, and vertex -> cells of the overlap level
    std::vector<std::vector<unsigned int>> vertex_to_active(tria.n_vertices());
    for (const auto &cell : dof_handler.active_cell_iterators())
      for (const unsigned int v : cell->vertex_indices())
        vertex_to_active[cell->vertex_index(v)].push_back(
          cell->active_cell_index());
    const unsigned int level = prm.overlap_level;
    AssertThrow(prm.overlap_mode == "fine" || level < tria.n_levels(),
                ExcMessage("overlap_level is above the finest level."));
    std::vector<Triangulation<3>::cell_iterator> level_cells;
    std::vector<std::vector<unsigned int>>       vertex_to_level(
      tria.n_vertices());
    // the level-`level` ancestor of every active cell (itself if coarser)
    std::vector<int> ancestor(n_cells, -1);
    if (prm.overlap_mode == "level")
      {
        for (const auto &cell : tria.cell_iterators_on_level(level))
          {
            for (const unsigned int v : cell->vertex_indices())
              vertex_to_level[cell->vertex_index(v)].push_back(
                level_cells.size());
            level_cells.push_back(cell);
          }
        std::map<std::pair<int, int>, unsigned int> level_index;
        for (unsigned int i = 0; i < level_cells.size(); ++i)
          level_index[{level_cells[i]->level(), level_cells[i]->index()}] = i;
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            AssertThrow(static_cast<unsigned int>(cell->level()) >= level,
                        ExcMessage("Active cells coarser than overlap_level "
                                   "are not supported."));
            Triangulation<3>::cell_iterator a = cell;
            while (static_cast<unsigned int>(a->level()) > level)
              a = a->parent();
            ancestor[cell->active_cell_index()] =
              level_index.at({a->level(), a->index()});
          }
      }

    const unsigned int                   n_dofs = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> dof_indices(n_dofs);
    // the part a DoF belongs to: the smallest part among its cells
    std::vector<unsigned int> dof_part(dof_handler.n_dofs(), n_parts);
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell->get_dof_indices(dof_indices);
        for (const types::global_dof_index g : dof_indices)
          dof_part[g] = std::min(dof_part[g], part[cell->active_cell_index()]);
      }

    schwarz.n_global = dof_handler.n_dofs();
    schwarz.subdomains.resize(n_parts);
    std::size_t total_dofs = 0, max_dofs = 0, rows_differ = 0, rows = 0;
    std::size_t total_cells = 0;
    double      max_forward_error = 0, max_cond1 = 0, max_cond2 = 0;
    for (unsigned int p = 0; p < n_parts; ++p)
      {
        // the cells of the subdomain
        std::vector<bool> in(n_cells, false);
        if (prm.overlap_mode == "fine")
          {
            for (unsigned int c = 0; c < n_cells; ++c)
              in[c] = (part[c] == p);
            for (unsigned int layer = 0; layer < prm.overlap_layers; ++layer)
              {
                std::vector<unsigned int> next;
                for (const auto &cell : dof_handler.active_cell_iterators())
                  if (in[cell->active_cell_index()])
                    for (const unsigned int v : cell->vertex_indices())
                      for (const unsigned int c :
                           vertex_to_active[cell->vertex_index(v)])
                        if (!in[c])
                          next.push_back(c);
                for (const unsigned int c : next)
                  in[c] = true;
              }
          }
        else
          {
            std::vector<bool> region(level_cells.size(), false);
            for (unsigned int c = 0; c < n_cells; ++c)
              if (part[c] == p)
                region[ancestor[c]] = true;
            for (unsigned int layer = 0; layer < prm.overlap_layers; ++layer)
              {
                std::vector<unsigned int> next;
                for (unsigned int i = 0; i < level_cells.size(); ++i)
                  if (region[i])
                    for (const unsigned int v : level_cells[i]->vertex_indices())
                      for (const unsigned int j :
                           vertex_to_level[level_cells[i]->vertex_index(v)])
                        if (!region[j])
                          next.push_back(j);
                for (const unsigned int j : next)
                  region[j] = true;
              }
            for (unsigned int c = 0; c < n_cells; ++c)
              in[c] = region[ancestor[c]];
          }

        // its DoFs, and those whose values it keeps
        SerialSchwarz::Subdomain &sub = schwarz.subdomains[p];
        std::vector<bool>         dof_in(dof_handler.n_dofs(), false);
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (in[cell->active_cell_index()])
            {
              ++total_cells;
              cell->get_dof_indices(dof_indices);
              for (const types::global_dof_index g : dof_indices)
                dof_in[g] = true;
            }
        for (types::global_dof_index g = 0; g < dof_handler.n_dofs(); ++g)
          if (dof_in[g])
            {
              if (dof_part[g] == p)
                sub.kept.push_back(sub.dofs.size());
              sub.dofs.push_back(g);
            }
        const auto local_index = [&sub](const types::global_dof_index g) {
          return static_cast<unsigned int>(
            std::lower_bound(sub.dofs.begin(), sub.dofs.end(), g) -
            sub.dofs.begin());
        };
        ComplexTriplets triplets;
        assemble_subdomain_problem(
          k,
          [&in](const DoFHandler<3>::active_cell_iterator &cell) {
            return static_cast<bool>(in[cell->active_cell_index()]);
          },
          local_index,
          triplets);
        sub.solver = std::make_unique<SequentialMUMPS>();
        triplets.factorize(sub.dofs.size(), *sub.solver);
        if (prm.checks)
          {
            // accuracy of the local solve and MUMPS' condition estimates
            const unsigned int  n = sub.dofs.size();
            std::vector<double> x_true(2 * n);
            for (unsigned int i = 0; i < 2 * n; ++i)
              x_true[i] = std::sin(0.91 * i + 0.3 * p);
            std::vector<double> x = triplets.vmult(n, x_true);
            const auto          analysis =
              sub.solver->solve_with_error_analysis(x);
            double e = 0, sc = 0;
            for (unsigned int i = 0; i < 2 * n; ++i)
              {
                e  = std::max(e, std::abs(x[i] - x_true[i]));
                sc = std::max(sc, std::abs(x_true[i]));
              }
            max_forward_error = std::max(max_forward_error, e / sc);
            max_cond1         = std::max(max_cond1, analysis[0]);
            max_cond2         = std::max(max_cond2, analysis[1]);
          }
        total_dofs += sub.dofs.size();
        max_dofs = std::max<std::size_t>(max_dofs, sub.dofs.size());

        // test: the rows of the kept DoFs equal the global rows (they do
        // when all cells around a kept DoF are in the subdomain)
        if (prm.checks)
          {
            PETScWrappers::MPI::SparseMatrix A_global;
            A_global.reinit(K);
            A_global.copy_from(K);
            A_global.add(-k * k, M);
            PETScWrappers::MPI::SparseMatrix B_global;
            B_global.reinit(K);
            B_global.copy_from(M_bdry);
            B_global *= -k;
            const unsigned int  n = sub.dofs.size();
            std::vector<double> x(2 * n);
            PETScWrappers::MPI::BlockVector xg(
              std::vector<IndexSet>{owned, owned}, comm),
              yg(xg);
            for (unsigned int b = 0; b < 2; ++b)
              {
                for (unsigned int i = 0; i < n; ++i)
                  {
                    x[b * n + i] = std::sin(1.3 * sub.dofs[i] + 0.7 * b);
                    xg.block(b)(sub.dofs[i]) = x[b * n + i];
                  }
                xg.block(b).compress(VectorOperation::insert);
              }
            ComplexBlockMatrix{A_global, B_global}.vmult(yg, xg);
            const std::vector<double> y = triplets.vmult(n, x);
            for (const unsigned int i : sub.kept)
              {
                ++rows;
                double difference = 0, scale = 0;
                for (unsigned int b = 0; b < 2; ++b)
                  {
                    const double global = yg.block(b)(sub.dofs[i]);
                    difference =
                      std::max(difference, std::abs(global - y[b * n + i]));
                    scale = std::max(scale, std::abs(global));
                  }
                if (difference > 1e-12 * std::max(1., scale))
                  ++rows_differ;
              }
          }
      }
    pcout << "serial Schwarz: " << n_parts << " subdomains, overlap "
          << prm.overlap_mode << " layers=" << prm.overlap_layers
          << (prm.overlap_mode == "level" ?
                " level=" + std::to_string(prm.overlap_level) :
                std::string())
          << ", cells per subdomain " << total_cells / n_parts << " (mean), DoFs per subdomain "
          << total_dofs / n_parts << " (mean), " << max_dofs
          << " (max), sum over subdomains / global DoFs "
          << double(total_dofs) / dof_handler.n_dofs() << std::endl;
    if (prm.checks)
      pcout << "check: local solves: max relative error "
            << max_forward_error << ", max MUMPS condition estimates COND1 "
            << max_cond1 << " COND2 " << max_cond2 << std::endl;
    if (prm.checks)
      pcout << "check: kept rows of the subdomain matrices = global rows: "
            << rows_differ << " of " << rows << " differ"
            << (rows_differ == 0 ? " OK" :
                                   " (expected without overlap layers)")
            << std::endl;
  }

  /**
   * Test: for every owned DoF, all cells around its edge are owned or
   * ghost cells, so the row of the subdomain matrix equals the row of the
   * global matrix (the transmission terms only touch DoFs on faces to
   * artificial cells). Compares A_loc x_loc with A x on the owned DoFs for
   * a random x.
   */
  void
  check_subdomain_matrix(const ComplexTriplets &triplets, const double k) const
  {
    PETScWrappers::MPI::SparseMatrix A, B;
    A.reinit(K);
    A.copy_from(K);
    A.add(-k * k, M);
    B.reinit(K);
    B.copy_from(M_bdry);
    B *= -k;
    const ComplexBlockMatrix        system{A, B};
    PETScWrappers::MPI::BlockVector x(std::vector<IndexSet>{owned, owned},
                                      comm),
      y(x);
    for (unsigned int b = 0; b < 2; ++b)
      {
        for (const types::global_dof_index g : owned)
          x.block(b)(g) = std::sin(1.3 * g + 0.7 * b) + 0.1 * b;
        x.block(b).compress(VectorOperation::insert);
      }
    system.vmult(y, x);

    const unsigned int         n = relevant.n_elements();
    std::vector<double>        x_local(2 * n);
    PETScWrappers::MPI::Vector ghosted(owned, relevant, comm);
    for (unsigned int b = 0; b < 2; ++b)
      {
        ghosted = x.block(b);
        unsigned int i = 0;
        for (const types::global_dof_index g : relevant)
          x_local[b * n + i++] = ghosted(g);
      }
    const std::vector<double> y_local = triplets.vmult(n, x_local);
    double error = 0, scale = 0;
    for (unsigned int b = 0; b < 2; ++b)
      for (const types::global_dof_index g : owned)
        {
          const double global = y.block(b)(g);
          error = std::max(error,
                           std::abs(global -
                                    y_local[b * n + relevant.index_within_set(g)]));
          scale = std::max(scale, std::abs(global));
        }
    error = Utilities::MPI::max(error, comm) / Utilities::MPI::max(scale, comm);
    pcout << "check: owned rows of the subdomain matrices = global rows: "
             "relative error "
          << error << (error < 1e-12 ? " OK" : " FAILED") << std::endl;
  }

  /**
   * Tests of the coarse correction E = I - P A_H^-1 P^T A:
   * - A_H^-1 P^T A P c = c for a random c (Galerkin: exact),
   * - E is a projection: E (E x) = E x,
   * - E applied to the interpolant of the exact solution leaves a small
   *   remainder (the coarse space approximates smooth fields).
   */
  void
  check_coarse_correction(const TwoLevelSchwarz    &two_level,
                          const ComplexBlockMatrix &system,
                          CoarseSolver             &coarse_solver) const
  {
    const unsigned int  n_coarse = coarse_dh.n_dofs();
    std::vector<double> c(2 * n_coarse);
    for (unsigned int j = 0; j < c.size(); ++j)
      c[j] = std::sin(0.37 * j + 0.2);
    PETScWrappers::MPI::BlockVector x(std::vector<IndexSet>{owned, owned},
                                      comm),
      y(x), z(x), w(x);

    // the coarse solve alone: A_H^-1 (A_H c) = c
    {
      std::vector<double> v = coarse_triplets->vmult(n_coarse, c);
      coarse_solver.solve(v);
      double e = 0, sc = 0;
      for (unsigned int j = 0; j < c.size(); ++j)
        {
          e  = std::max(e, std::abs(v[j] - c[j]));
          sc = std::max(sc, std::abs(c[j]));
        }
      pcout << "check: coarse solve A_H^-1 (A_H c) = c: relative error "
            << e / sc << (e / sc < 1e-8 ? " OK" : " FAILED") << std::endl;
    }

    two_level.prolongate(x, c);
    system.vmult(y, x);
    std::vector<double> d = two_level.restrict_to_coarse(y);
    // the coarse matrix against the Galerkin product: P^T A P c = A_H c
    {
      const std::vector<double> v = coarse_triplets->vmult(n_coarse, c);
      double                    e = 0, sc = 0;
      for (unsigned int j = 0; j < c.size(); ++j)
        {
          e  = std::max(e, std::abs(v[j] - d[j]));
          sc = std::max(sc, std::abs(d[j]));
        }
      pcout << "check: P^T A P c = A_H c: relative error " << e / sc
            << (prm.coarse_op == "algebraic" || prm.flat_interface ?
                  (e / sc < 1e-10 ? " OK" : " FAILED") :
                  " (not exact: non-nested near the sphere)")
            << std::endl;
      // the same per block, and the norms, to see which part differs
      for (unsigned int b = 0; b < 2; ++b)
        {
          double eb = 0, nb_v = 0, nb_d = 0;
          for (unsigned int j = 0; j < n_coarse; ++j)
            {
              eb = std::max(eb, std::abs(v[b * n_coarse + j] - d[b * n_coarse + j]));
              nb_v = std::max(nb_v, std::abs(v[b * n_coarse + j]));
              nb_d = std::max(nb_d, std::abs(d[b * n_coarse + j]));
            }
          pcout << "       block " << b << ": max |A_H c| " << nb_v
                << ", max |P^T A P c| " << nb_d << ", max difference " << eb
                << std::endl;
        }
    }
    coarse_solver.solve(d);
    double error = 0, scale = 0;
    for (unsigned int j = 0; j < c.size(); ++j)
      {
        error = std::max(error, std::abs(d[j] - c[j]));
        scale = std::max(scale, std::abs(c[j]));
      }
    pcout << "check: A_H^-1 P^T A P c = c: relative error " << error / scale
          << (prm.coarse_op == "algebraic" || prm.flat_interface ?
                (error / scale < 1e-8 ? " OK" : " FAILED") :
                " (not exact: non-nested near the sphere)")
          << std::endl;

    const auto apply_E = [&](PETScWrappers::MPI::BlockVector       &dst,
                             const PETScWrappers::MPI::BlockVector &src) {
      PETScWrappers::MPI::BlockVector Asrc(src), correction(src);
      system.vmult(Asrc, src);
      two_level.coarse_correction(correction, Asrc);
      dst = src;
      dst -= correction;
    };

    // a random fine vector: E (E x) = E x
    for (unsigned int b = 0; b < 2; ++b)
      {
        for (const types::global_dof_index g : owned)
          x.block(b)(g) = std::sin(2.1 * g + 0.3 * b);
        x.block(b).compress(VectorOperation::insert);
      }
    apply_E(y, x);
    apply_E(z, y);
    z -= y;
    pcout << "check: E^2 = E: |E E x - E x| / |E x| = "
          << z.l2_norm() / y.l2_norm()
          << (prm.coarse_op == "algebraic" || prm.flat_interface ?
                (z.l2_norm() / y.l2_norm() < 1e-8 ? " OK" : " FAILED") :
                "")
          << std::endl;

    // the interpolant of the exact solution (DoF = line integral, midpoint
    // rule)
    const unsigned int                   n_dofs = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> dof_indices(n_dofs);
    Tensor<1, 3>                         e_r, e_i, c_r, c_i;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(dof_indices);
          for (unsigned int i = 0; i < n_dofs; ++i)
            if (owned.is_element(dof_indices[i]))
              {
                const Point<3> v0 = cell->line(i)->vertex(0),
                               v1 = cell->line(i)->vertex(1);
                mie.evaluate(Point<3>(0.5 * (v0 + v1)), e_r, e_i, c_r, c_i);
                x.block(0)(dof_indices[i]) = e_r * (v1 - v0);
                x.block(1)(dof_indices[i]) = e_i * (v1 - v0);
              }
        }
    x.compress(VectorOperation::insert);
    apply_E(y, x);
    pcout << "check: |E x| / |x| for the interpolant x of the exact "
             "solution: "
          << y.l2_norm() / x.l2_norm() << std::endl;
  }

  void
  solve(const double k)
  {
    Timer timer;
    mie.set_wave_number(k);

    // A = K - k^2 M, B = -k M_bdry, P = K + k^2 M + k M_bdry
    PETScWrappers::MPI::SparseMatrix A, B, P, B_shifted;
    A.reinit(K);
    A.copy_from(K);
    A.add(-k * k, M);
    B.reinit(K);
    B.copy_from(M_bdry);
    B *= -k;
    P.reinit(K);
    P.copy_from(K);
    P.add(k * k, M);
    P.add(k, M_bdry);

    // right hand side -<g, v>, g = n x curl E + i k E_T
    PETScWrappers::MPI::BlockVector rhs(std::vector<IndexSet>{owned, owned},
                                        comm),
      solution(std::vector<IndexSet>{owned, owned}, comm);
    const QGaussSimplex<3> q_cell(4);
    const QGaussSimplex<2> q_face(4);
    FEFaceValues<3>        fe_face_values(mapping,
                                   fe,
                                   q_face,
                                   update_values | update_normal_vectors |
                                     update_quadrature_points |
                                     update_JxW_values);
    const FEValuesExtractors::Vector     E(0);
    const unsigned int                   n_dofs = fe.n_dofs_per_cell();
    Vector<double>                       rhs_r(n_dofs), rhs_i(n_dofs);
    std::vector<types::global_dof_index> dof_indices(n_dofs);
    Tensor<1, 3>                         e_r, e_i, c_r, c_i;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() && cell->at_boundary())
        {
          rhs_r = 0;
          rhs_i = 0;
          for (const unsigned int f : cell->face_indices())
            if (cell->at_boundary(f))
              {
                fe_face_values.reinit(cell, f);
                for (unsigned int q = 0; q < q_face.size(); ++q)
                  {
                    const Tensor<1, 3> normal = fe_face_values.normal_vector(q);
                    mie.evaluate(fe_face_values.quadrature_point(q),
                                 e_r,
                                 e_i,
                                 c_r,
                                 c_i);
                    const Tensor<1, 3> et_r = e_r - (e_r * normal) * normal,
                                       et_i = e_i - (e_i * normal) * normal;
                    // g = n x curl E + i k E_T
                    const Tensor<1, 3> g_r =
                      cross_product_3d(normal, c_r) - k * et_i,
                                       g_i =
                                         cross_product_3d(normal, c_i) + k * et_r;
                    for (unsigned int i = 0; i < n_dofs; ++i)
                      {
                        const Tensor<1, 3> v = fe_face_values[E].value(i, q);
                        rhs_r(i) -= g_r * v * fe_face_values.JxW(q);
                        rhs_i(i) -= g_i * v * fe_face_values.JxW(q);
                      }
                  }
              }
          cell->get_dof_indices(dof_indices);
          constraints.distribute_local_to_global(rhs_r,
                                                 dof_indices,
                                                 rhs.block(0));
          constraints.distribute_local_to_global(rhs_i,
                                                 dof_indices,
                                                 rhs.block(1));
        }
    rhs.compress(VectorOperation::add);

    const bool serial_schwarz =
      (prm.precond == "ras_serial" || prm.precond == "oras_serial");
    const bool schwarz =
      (prm.precond == "ras" || prm.precond == "oras" || serial_schwarz);
    PreconditionAMS ams;
    if (!schwarz)
      ams.initialize(P, gradient, constant_fields);
    SequentialMUMPS mumps;
    CoarseSolver    coarse_mumps;
    SerialSchwarz   serial;
    if (serial_schwarz)
      setup_serial_schwarz(k, serial);
    else if (schwarz)
      factorize_subdomain_problem(k, mumps);
    if (schwarz && prm.coarse != "none")
      {
        if (prm.coarse_op == "algebraic")
          factorize_algebraic_coarse_problem(k, A, B, coarse_mumps);
        else
          factorize_coarse_problem(k, coarse_mumps.redundant);
      }
    const double setup_time = timer.wall_time();
    timer.restart();

    const ComplexBlockMatrix          system{A, B};
    const BlockDiagonalPreconditioner block_ams{P,
                                                ams,
                                                prm.precond == "exact",
                                                prm.ams_cycles};
    SolverControl solver_control(prm.max_iterations, prm.tol * rhs.l2_norm());
    SolverFGMRES<PETScWrappers::MPI::BlockVector>::AdditionalData data;
    data.max_basis_size = prm.restart;
    SolverFGMRES<PETScWrappers::MPI::BlockVector> solver(solver_control, data);
    unsigned int n_inner   = 0;
    bool         converged = true;
    try
      {
        if (prm.precond == "shifted")
          {
            // K - (1 + i alpha) k^2 M - i k M_bdry
            B_shifted.reinit(K);
            B_shifted.copy_from(B);
            B_shifted.add(-prm.alpha * k * k, M);
            const ComplexBlockMatrix    shifted{A, B_shifted};
            const ShiftedPreconditioner preconditioner{shifted,
                                                       block_ams,
                                                       prm.inner_tol};
            solver.solve(system, solution, rhs, preconditioner);
            n_inner = preconditioner.n_inner;
          }
        else if (schwarz)
          {
            const RestrictedSchwarz parallel_preconditioner(owned,
                                                            relevant,
                                                            mumps,
                                                            comm);
            const SchwarzPreconditioner &schwarz_preconditioner =
              serial_schwarz ?
                static_cast<const SchwarzPreconditioner &>(serial) :
                static_cast<const SchwarzPreconditioner &>(
                  parallel_preconditioner);
            const TwoLevelSchwarz two_level{schwarz_preconditioner,
                                            system,
                                            owned_indices,
                                            prolongation,
                                            coarse_mumps,
                                            coarse_dh.n_dofs(),
                                            prm.coarse == "hybrid",
                                            comm};
            if (prm.checks && prm.coarse != "none")
              check_coarse_correction(two_level, system, coarse_mumps);
            if (prm.checks)
              check_schwarz_iteration(schwarz_preconditioner, system);
            std::unique_ptr<GradientCorrection> correction;
            if (prm.gradient_correction != "none")
              correction = std::make_unique<GradientCorrection>(
                gradient,
                M,
                nodal_dh.locally_owned_dofs(),
                prm.gradient_correction == "exact",
                comm);
            if (prm.ritz > 0 && prm.coarse == "none" && !correction)
              analyze_ritz_values(system, schwarz_preconditioner, rhs, k);
            if (prm.ritz > 0 && prm.coarse != "none" && !correction)
              analyze_ritz_values(system, two_level, rhs, k);
            if (prm.coarse == "none" && !correction)
              solver.solve(system, solution, rhs, schwarz_preconditioner);
            else if (prm.coarse == "none")
              solver.solve(system,
                           solution,
                           rhs,
                           GradientCorrected<SchwarzPreconditioner>{
                             schwarz_preconditioner, system, *correction, k});
            else if (!correction)
              solver.solve(system, solution, rhs, two_level);
            else
              solver.solve(system,
                           solution,
                           rhs,
                           GradientCorrected<TwoLevelSchwarz>{two_level,
                                                              system,
                                                              *correction,
                                                              k});
            n_inner = schwarz_preconditioner.n_applications;
          }
        else
          solver.solve(system, solution, rhs, block_ams);
      }
    catch (const SolverControl::NoConvergence &)
      {
        converged = false;
      }
    constraints.distribute(solution.block(0));
    constraints.distribute(solution.block(1));
    const double solve_time = timer.wall_time();

    // relative L2 error
    PETScWrappers::MPI::Vector ghosted_r(owned, relevant, comm),
      ghosted_i(owned, relevant, comm);
    ghosted_r = solution.block(0);
    ghosted_i = solution.block(1);
    FEValues<3> fe_values(mapping,
                          fe,
                          q_cell,
                          update_values | update_quadrature_points |
                            update_JxW_values);
    double      error = 0, norm = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(dof_indices);
          for (unsigned int q = 0; q < q_cell.size(); ++q)
            {
              Tensor<1, 3> eh_r, eh_i;
              for (unsigned int i = 0; i < n_dofs; ++i)
                {
                  eh_r += static_cast<double>(ghosted_r(dof_indices[i])) *
                          fe_values[E].value(i, q);
                  eh_i += static_cast<double>(ghosted_i(dof_indices[i])) *
                          fe_values[E].value(i, q);
                }
              mie.evaluate(fe_values.quadrature_point(q), e_r, e_i, c_r, c_i);
              error += ((eh_r - e_r).norm_square() +
                        (eh_i - e_i).norm_square()) *
                       fe_values.JxW(q);
              norm +=
                (e_r.norm_square() + e_i.norm_square()) * fe_values.JxW(q);
            }
        }
    error = std::sqrt(Utilities::MPI::sum(error, comm) /
                      Utilities::MPI::sum(norm, comm));

    const double amplitude = std::abs(mie.interior_coefficient());
    pcout << "k=" << std::setprecision(8) << k << " |A|="
          << std::setprecision(4) << amplitude << " precond=" << prm.precond
          << " iterations=" << solver_control.last_step()
          << (converged ? "" : " (not converged)")
          << " cycles=" << prm.ams_cycles << " ams=" << block_ams.n_ams << " inner=" << n_inner
          << " rel. L2 error=" << error << " setup=" << setup_time
          << "s solve=" << solve_time << "s" << std::endl;

    if (!prm.csv.empty() && Utilities::MPI::this_mpi_process(comm) == 0)
      {
        const bool    exists = std::ifstream(prm.csv).good();
        std::ofstream out(prm.csv, std::ios::app);
        if (!exists)
          out << "n,l,a,R,hex_refine,refine,interface_refine,procs,cells,dofs,precond,"
                 "ams_cycles,robin_real,coarse,coarse_op,coarse_shift,"
                 "gradient_correction,subdomains,overlap_mode,"
                 "overlap_layers,overlap_level,ghost_overlap_level,"
                 "ghost_overlap_layers,flat_interface,local_shift,"
                 "impedance_scale,alpha,inner_tol,k,amplitude,converged,iterations,"
                 "ams_applications,inner_iterations,rel_l2_error,setup_s,"
                 "solve_s\n";
        out << std::setprecision(12) << prm.n << ',' << prm.l << ',' << prm.a
            << ',' << prm.R << ',' << prm.hex_refine << ',' << prm.refine << ',' << prm.interface_refine
            << ',' << Utilities::MPI::n_mpi_processes(comm) << ','
            << tria.n_global_active_cells() << ',' << 2 * dof_handler.n_dofs()
            << ',' << prm.precond << ',' << prm.ams_cycles << ','
            << prm.robin_real << ',' << prm.coarse << ',' << prm.coarse_op
            << ',' << prm.coarse_shift << ',' << prm.gradient_correction
            << ',' << prm.subdomains << ',' << prm.overlap_mode << ','
            << prm.overlap_layers << ',' << prm.overlap_level << ','
            << prm.ghost_overlap_level << ',' << prm.ghost_overlap_layers << ','
            << prm.flat_interface << ',' << prm.local_shift << ','
            << prm.impedance_scale
            << ',' << prm.alpha
            << ',' << prm.inner_tol
            << ',' << k << ',' << amplitude << ',' << converged << ','
            << solver_control.last_step() << ',' << block_ams.n_ams << ','
            << n_inner << ',' << error << ',' << setup_time << ','
            << solve_time << '\n';
      }
  }
};



int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
      // deal.II initializes libsc and t8code silently, which also hides the
      // messages of aborts
      sc_set_log_defaults(nullptr, nullptr, SC_LP_ERROR);
      sc_package_set_verbosity(t8_get_package_id(), SC_LP_ERROR);
      Parameters                       prm;
      prm.parse(argc, argv);
      if (prm.mode == "selftest")
        self_test(prm);
      else if (prm.mode == "resonances")
        find_resonances(prm);
      else
        {
          SphereProblem problem(prm);
          problem.run();
        }
    }
  catch (const std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
