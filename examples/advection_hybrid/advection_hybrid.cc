/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * ------------------------------------------------------------------------
 *
 * Linear advection u_t + beta . grad u = 0 on the unit square (ADV_DIM 2,
 * advection_hybrid) or the unit cube (ADV_DIM 3, advection_hybrid_3d) with
 * discontinuous Galerkin elements on hybrid meshes: quadrilaterals (FE_DGQ) and
 * triangles (FE_SimplexDGP) in 2D; hexahedra, tetrahedra, wedges and pyramids
 * (FE_DGQ, FE_SimplexDGP, FE_WedgeDGP, FE_PyramidDGP) in 3D;
 * parallel::distributed::Triangulation (t8code backend), explicit Runge-Kutta
 * time stepping, and adaptive refinement and coarsening every few time steps
 * with SolutionTransfer.
 *
 * The velocity is the solid body rotation beta = 2 pi (-(y - 1/2), x - 1/2)
 * (in 3D about the axis x = y = 1/2, with beta_z = 0; one turn per unit
 * time), the initial value a Gaussian hill, so that the exact solution is the
 * rotated hill (zero inflow: the hill stays away from the boundary).
 *
 * Spatial discretization: upwind fluxes, M u' = A u with the operator matrix A
 * assembled once per mesh (cell terms, interior faces also between cells of
 * different shapes and at hanging faces, outflow boundary), and the block
 * diagonal mass matrix inverted cell by cell. The interior faces: with
 * FEInterfaceValues (faces=interface; in 3D with one face quadrature per face
 * of each reference cell), or by hand (faces=manual): FEFaceValues with one
 * quadrature per face of the reference cell (the faces of wedges and pyramids
 * are triangles and quadrilaterals) on both sides of conforming faces; at
 * hanging faces the quadrature of the fine face, and the shape functions of the
 * coarse cell at its points through the inverse mapping (FESubfaceValues is not
 * implemented for wedges and pyramids). Time stepping: SSP-RK3 (Shu-Osher) or
 * the classical RK4, with the time step from the CFL condition dt = cfl h_min /
 * (|beta|_max (2p + 1)) for the cells of the finest level.
 *
 * Usage: mpirun -np N ./advection_hybrid[_3d] [key=value ...]
 *   mesh=hybrid       2D: hybrid (squares alternating with pairs of
 *                     triangles), quad or tri, n x n squares. 3D: hybrid21
 *                     (all four shapes), hybrid16 (no pyramids) or hex
 *   n=4               coarse mesh size (2D)
 *   faces=interface   interface (FEInterfaceValues), manual (by hand, the
 *                     default in 3D), or compare (both, and the difference of
 *                     the matrices; the interface path also checks that both
 *                     sides of each face have the same points, weights, and
 *                     opposite normals)
 *   estimator=kelly   kelly (jumps of the normal derivative over the faces,
 *                     default in 2D; in 3D with one face quadrature per face
 *                     of each reference cell) or gradient (h_K
 *                     |grad u_h|_{L2(K)}, default in 3D)
 *   partition=nnz     the weights of the cells when t8code partitions the
 *                     mesh: cells (equal), dofs (the DoFs of the cell), nnz
 *                     (an estimate of its share of the operator, n^2 (1 +
 *                     number of faces) for n DoFs of the cell; balances the
 *                     time steps best on hybrid meshes)
 *   degree=2         polynomial degree
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

#include <algorithm>
#include <chrono>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_wedge_p.h>
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
#include <memory>
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

#ifndef ADV_DIM
#  define ADV_DIM 2
#endif
  constexpr int dim = ADV_DIM;

  struct Parameters
  {
    std::string  mesh         = dim == 2 ? "hybrid" : "hybrid21";
    unsigned int n            = 4;
    std::string  faces        = dim == 2 ? "interface" : "manual";
    std::string  estimator    = dim == 2 ? "kelly" : "gradient";
    std::string  partition    = "nnz";
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

    void parse(int argc, char *argv[])
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
          else if (key == "faces")
            faces = value;
          else if (key == "estimator")
            estimator = value;
          else if (key == "partition")
            partition = value;
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
      if (dim == 2)
        AssertThrow(mesh == "hybrid" || mesh == "quad" || mesh == "tri",
                    ExcMessage("mesh must be hybrid, quad or tri"));
      else
        AssertThrow(mesh == "hybrid21" || mesh == "hybrid16" || mesh == "hex",
                    ExcMessage("mesh must be hybrid21, hybrid16 or hex"));
      AssertThrow(faces == "manual" || faces == "interface" ||
                    faces == "compare",
                  ExcMessage("faces must be manual, interface, or compare"));
      AssertThrow(partition == "cells" || partition == "dofs" ||
                    partition == "nnz",
                  ExcMessage("partition must be cells, dofs or nnz"));
      AssertThrow(estimator == "gradient" || estimator == "kelly",
                  ExcMessage("estimator must be kelly or gradient"));
      AssertThrow(scheme == "ssprk3" || scheme == "rk4",
                  ExcMessage("scheme must be ssprk3 or rk4"));
    }
  };



  Tensor<1, dim> velocity(const Point<dim> &p)
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

    double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      const double a = -2 * numbers::PI * this->get_time();
      const double x = p[0] - 0.5, y = p[1] - 0.5;
      Point<dim>   q = p, center;
      q[0]           = 0.5 + std::cos(a) * x - std::sin(a) * y;
      q[1]           = 0.5 + std::sin(a) * x + std::cos(a) * y;
      for (unsigned int d = 0; d < dim; ++d)
        center[d] = d == 0 ? 0.75 : 0.5;
      return std::exp(-(q - center).norm_square() / (sigma * sigma));
    }

  private:
    const double sigma;
  };



#if ADV_DIM == 3
  // The unit cube in octants of side 0.5 (as in heat_hybrid): hybrid21 with 6
  // tetrahedra at the origin, 2 wedges in each face neighbor, 6 pyramids in
  // [0.5,1]^3 and hexahedra elsewhere; hybrid16 with a hexahedron instead of
  // the pyramids; hex with hexahedra only.
  void make_mesh(Triangulation<dim> &tria,
                 const unsigned int,
                 const std::string &type)
  {
    const bool              pyramids = type == "hybrid21";
    const bool              hex_only = type == "hex";
    std::vector<Point<dim>> vertices;
    for (unsigned int z = 0; z <= 2; ++z)
      for (unsigned int y = 0; y <= 2; ++y)
        for (unsigned int x = 0; x <= 2; ++x)
          vertices.emplace_back(0.5 * x, 0.5 * y, 0.5 * z);
    const unsigned int apex = vertices.size();
    if (pyramids)
      vertices.emplace_back(0.75, 0.75, 0.75);
    const auto volume =
      [&](unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
        return cross_product_3d(vertices[b] - vertices[a],
                                vertices[c] - vertices[a]) *
               (vertices[d] - vertices[a]);
      };
    std::vector<CellData<dim>> cells;
    const auto add_cell = [&](const std::vector<unsigned int> &v) {
      CellData<dim> cell(v.size());
      for (unsigned int i = 0; i < v.size(); ++i)
        cell.vertices[i] = v[i];
      cells.push_back(cell);
    };
    static const unsigned int cube_faces[6][4] = {{0, 2, 4, 6},
                                                  {1, 3, 5, 7},
                                                  {0, 4, 1, 5},
                                                  {2, 6, 3, 7},
                                                  {0, 1, 2, 3},
                                                  {4, 5, 6, 7}};
    for (unsigned int oz = 0; oz < 2; ++oz)
      for (unsigned int oy = 0; oy < 2; ++oy)
        for (unsigned int ox = 0; ox < 2; ++ox)
          {
            const auto corner =
              [&](unsigned int i, unsigned int j, unsigned int k) {
                return (ox + i) + 3 * ((oy + j) + 3 * (oz + k));
              };
            std::array<unsigned int, 8> cube;
            for (unsigned int v = 0; v < 8; ++v)
              cube[v] = corner(v & 1, (v >> 1) & 1, v >> 2);
            const unsigned int n_offsets = ox + oy + oz;
            if (hex_only)
              add_cell({cube.begin(), cube.end()});
            else if (n_offsets == 0)
              {
                std::array<unsigned int, 3> directions = {{0, 1, 2}};
                do
                  {
                    const unsigned int v1       = 1u << directions[0],
                                       v2       = v1 | (1u << directions[1]);
                    std::vector<unsigned int> t = {cube[0],
                                                   cube[v1],
                                                   cube[v2],
                                                   cube[7]};
                    if (volume(t[0], t[1], t[2], t[3]) < 0)
                      std::swap(t[2], t[3]);
                    add_cell(t);
                  }
                while (
                  std::next_permutation(directions.begin(), directions.end()));
              }
            else if (n_offsets == 1)
              {
                const unsigned int a = ox ? 0 : (oy ? 1 : 2), b = (a + 1) % 3,
                                   c = (a + 2) % 3;
                const auto local =
                  [&](unsigned int ia, unsigned int ib, unsigned int ic) {
                    unsigned int idx[3];
                    idx[a] = ia, idx[b] = ib, idx[c] = ic;
                    return corner(idx[0], idx[1], idx[2]);
                  };
                const unsigned int triangles[2][3][2] = {
                  {{0, 0}, {1, 0}, {1, 1}}, {{0, 0}, {1, 1}, {0, 1}}};
                for (const auto &t : triangles)
                  {
                    std::vector<unsigned int> w(6);
                    for (unsigned int i = 0; i < 3; ++i)
                      w[i]     = local(0, t[i][0], t[i][1]),
                      w[i + 3] = local(1, t[i][0], t[i][1]);
                    if (volume(w[0], w[1], w[2], w[3]) < 0)
                      std::swap(w[1], w[2]), std::swap(w[4], w[5]);
                    add_cell(w);
                  }
              }
            else if (n_offsets == 3 && pyramids)
              for (unsigned int f = 0; f < 6; ++f)
                {
                  std::vector<unsigned int> base = {cube[cube_faces[f][0]],
                                                    cube[cube_faces[f][1]],
                                                    cube[cube_faces[f][2]],
                                                    cube[cube_faces[f][3]],
                                                    apex};
                  if (volume(base[0], base[1], base[2], apex) < 0)
                    std::swap(base[1], base[2]);
                  add_cell(base);
                }
            else
              add_cell({cube.begin(), cube.end()});
          }
    tria.create_triangulation(vertices, cells, SubCellData());
  }
#else
  // the unit square in n x n squares: quadrilaterals, pairs of triangles, or
  // both in a checkerboard pattern (hybrid); as in heat_hybrid
  void make_mesh(Triangulation<dim> &tria,
                 const unsigned int  n,
                 const std::string  &type)
  {
    std::vector<Point<dim>> vertices;
    for (unsigned int y = 0; y <= n; ++y)
      for (unsigned int x = 0; x <= n; ++x)
        vertices.emplace_back(1. * x / n, 1. * y / n);
    const auto vertex = [n](const unsigned int x, const unsigned int y) {
      return x + (n + 1) * y;
    };
    const auto area =
      [&](const unsigned int a, const unsigned int b, const unsigned int c) {
        const Tensor<1, dim> u = vertices[b] - vertices[a],
                             v = vertices[c] - vertices[a];
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
#endif



  class AdvectionProblem
  {
  public:
    AdvectionProblem(const Parameters &parameters);

    void run();

  private:
    unsigned int fe_index(const ReferenceCell &r) const
    {
      return r == ReferenceCells::Quadrilateral ||
                 r == ReferenceCells::Hexahedron ?
               0 :
             r == ReferenceCells::Triangle || r == ReferenceCells::Tetrahedron ?
               1 :
             r == ReferenceCells::Wedge ? 2 :
                                          3;
    }

    // the refinement indicator per active cell (Kelly or gradient)
    void compute_indicator(Vector<float> &indicator) const;

    void setup_system();
    void assemble_operator();
    // y = M^{-1} A u
    void   apply(const LA::MPI::Vector &u, LA::MPI::Vector &y) const;
    void   time_step(const double dt);
    void   adapt_mesh();
    void   update_ghosted();
    double l2_error(const double time);
    double mass();
    void   output_results(const unsigned int step, const double time) const;

    const Parameters   parameters;
    MPI_Comm           mpi_communicator;
    ConditionalOStream pcout;

    parallel::distributed::Triangulation<dim> triangulation;
    hp::FECollection<dim>                     fe;
    hp::MappingCollection<dim>                mapping;
    hp::QCollection<dim>                      quadrature;
    hp::QCollection<dim - 1> face_quadrature; // one per element (2D)
    // per element, one quadrature per face of its reference cell
    std::vector<hp::QCollection<dim - 1>> face_quadratures;
    DoFHandler<dim>                       dof_handler;

    IndexSet                        locally_owned_dofs, locally_relevant_dofs;
    LA::MPI::SparseMatrix           system_matrix;
    LA::MPI::SparseMatrix           reference_matrix;          // faces=compare
    double                          max_matrix_difference = 0; // faces=compare
    std::vector<FullMatrix<double>> inverse_mass; // per active cell index
    LA::MPI::Vector                 solution;     // owned
    LA::MPI::Vector                 ghosted;      // locally relevant copy
    mutable LA::MPI::Vector         tmp;

    ExactSolution exact;

    std::map<std::string, double> times;
    template <typename F>
    void time(const std::string &phase, const F &f)
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
#if ADV_DIM == 3
    fe.push_back(FE_WedgeDGP<dim>(p));
    fe.push_back(FE_PyramidDGP<dim>(p));
    mapping.push_back(MappingFE<dim>(FE_WedgeP<dim>(1)));
    mapping.push_back(MappingFE<dim>(FE_PyramidP<dim>(1)));
    quadrature.push_back(QGaussWedge<dim>(p + 1));
    quadrature.push_back(QGaussPyramid<dim>(p + 1));
    const QGauss<dim - 1>        qq(p + 1);
    const QGaussSimplex<dim - 1> qt(p + 1);
    // the faces of the reference cells: hexahedron 6 quadrilaterals,
    // tetrahedron 4 triangles, wedge 2 triangles and 3 quadrilaterals,
    // pyramid 1 quadrilateral and 4 triangles
    face_quadratures = {hp::QCollection<dim - 1>(qq),
                        hp::QCollection<dim - 1>(qt),
                        hp::QCollection<dim - 1>(qt, qt, qq, qq, qq),
                        hp::QCollection<dim - 1>(qq, qt, qt, qt, qt)};
#else
    face_quadratures = {hp::QCollection<dim - 1>(QGauss<dim - 1>(p + 1)),
                        hp::QCollection<dim - 1>(QGauss<dim - 1>(p + 1))};
#endif

    // weighted partitioning: t8code asks for the weight of every cell of the
    // new mesh. For a refined cell it passes the parent (one call per child):
    // the children of a pyramid are 6 pyramids and 4 tetrahedra (t8code), so
    // the mean over the children; all other shapes keep their shape.
    if (parameters.partition != "cells")
      {
        const auto shape_weight = [this](const ReferenceCell &r) {
          const double n = fe[fe_index(r)].n_dofs_per_cell();
          return this->parameters.partition == "dofs" ?
                   n :
                   n * n * (1 + r.n_faces());
        };
        triangulation.signals.weight.connect(
          [this,
           shape_weight](const typename Triangulation<dim>::cell_iterator &cell,
                         const CellStatus status) -> unsigned int {
            double w = shape_weight(cell->reference_cell());
            if (status == CellStatus::cell_will_be_refined &&
                cell->reference_cell() == ReferenceCells::Pyramid)
              w = (6 * shape_weight(ReferenceCells::Pyramid) +
                   4 * shape_weight(ReferenceCells::Tetrahedron)) /
                  10;
            return static_cast<unsigned int>(std::round(w));
          });
      }
  }



  // Kelly: jumps of the normal derivative over the faces, with one face
  // quadrature per face of each reference cell (the faces of wedges and
  // pyramids are triangles and quadrilaterals). gradient: h_K
  // |grad u_h|_{L2(K)}, which needs no face quadrature.
  void AdvectionProblem::compute_indicator(Vector<float> &indicator) const
  {
    indicator.reinit(triangulation.n_active_cells());
    if (parameters.estimator == "kelly")
      {
        KellyErrorEstimator<dim>::estimate(
          mapping,
          dof_handler,
          face_quadratures,
          std::map<types::boundary_id, const Function<dim> *>(),
          ghosted,
          indicator);
        return;
      }
    hp::FEValues<dim>           hp_fe_values(mapping,
                                   fe,
                                   quadrature,
                                   update_gradients | update_JxW_values);
    std::vector<Tensor<1, dim>> gradients;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          hp_fe_values.reinit(cell);
          const FEValues<dim> &fv = hp_fe_values.get_present_fe_values();
          gradients.resize(fv.n_quadrature_points);
          fv.get_function_gradients(ghosted, gradients);
          double s = 0;
          for (const unsigned int q : fv.quadrature_point_indices())
            s += gradients[q].norm_square() * fv.JxW(q);
          indicator(cell->active_cell_index()) =
            cell->diameter() * std::sqrt(s);
        }
  }



  void AdvectionProblem::setup_system()
  {
    this->time("setup: dofs", [&]() {
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          cell->set_active_fe_index(fe_index(cell->reference_cell()));
      dof_handler.distribute_dofs(fe);
      locally_owned_dofs = dof_handler.locally_owned_dofs();
      locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);
    });

    this->time("setup: sparsity", [&]() {
      DynamicSparsityPattern dsp(locally_relevant_dofs);
      this->time("sparsity: flux", [&]() {
        DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
      });
      this->time("sparsity: distribute", [&]() {
        SparsityTools::distribute_sparsity_pattern(dsp,
                                                   locally_owned_dofs,
                                                   mpi_communicator,
                                                   locally_relevant_dofs);
      });
      this->time("sparsity: matrix", [&]() {
        system_matrix.reinit(locally_owned_dofs,
                             locally_owned_dofs,
                             dsp,
                             mpi_communicator);
      });
      solution.reinit(locally_owned_dofs, mpi_communicator);
      tmp.reinit(locally_owned_dofs, mpi_communicator);
      ghosted.reinit(locally_owned_dofs,
                     locally_relevant_dofs,
                     mpi_communicator);
    });
    assemble_operator();
  }



  // A u: cell terms int u beta . grad v, interior faces - int (beta . n)
  // u_upwind [v], outflow boundary - int (beta . n) u v; inflow values are
  // zero. Each interior face is assembled once: from the finer side at hanging
  // faces, else from the cell with the smaller id (or the smaller subdomain at
  // process boundaries).
  void AdvectionProblem::assemble_operator()
  {
    this->time("assembly", [&]() {
      // faces=compare: by hand first (the reference), then with
      // FEInterfaceValues
      const std::vector<bool> passes =
        parameters.faces == "compare" ?
          std::vector<bool>{false, true} :
          std::vector<bool>{parameters.faces == "interface"};
      for (unsigned int pass = 0; pass < passes.size(); ++pass)
        {
          system_matrix = 0;
          inverse_mass.assign(triangulation.n_active_cells(),
                              FullMatrix<double>());
          const UpdateFlags flags = update_values | update_gradients |
                                    update_quadrature_points |
                                    update_JxW_values;
          hp::FEValues<dim> hp_fe_values(mapping, fe, quadrature, flags);
          const UpdateFlags face_flags =
            update_values | update_quadrature_points | update_normal_vectors |
            update_JxW_values;
          hp::FEFaceValues<dim> hp_fe_face_values(mapping,
                                                  fe,
                                                  face_quadratures,
                                                  face_flags);
          hp::FEFaceValues<dim> hp_fe_face_values_neighbor(mapping,
                                                           fe,
                                                           face_quadratures,
                                                           face_flags);
          const bool            use_interface = passes[pass];
          std::unique_ptr<FEInterfaceValues<dim>> fe_iv;
          if (use_interface)
            fe_iv = dim == 2 ? std::make_unique<FEInterfaceValues<dim>>(
                                 mapping, fe, face_quadrature, face_flags) :
                               std::make_unique<FEInterfaceValues<dim>>(
                                 mapping, fe, face_quadratures, face_flags);
          const AffineConstraints<double> no_constraints;

          FullMatrix<double>                   cell_matrix, cell_mass;
          std::vector<types::global_dof_index> dof_indices,
            neighbor_dof_indices, face_dof_indices;

          const auto assemble_interface_face = [&]() {
            // both sides: the same points and weights, opposite normals
            {
              const FEFaceValuesBase<dim> &side0 = fe_iv->get_fe_face_values(0);
              const FEFaceValuesBase<dim> &side1 = fe_iv->get_fe_face_values(1);
              AssertThrow(
                side0.n_quadrature_points == side1.n_quadrature_points &&
                  side0.n_quadrature_points == fe_iv->n_quadrature_points,
                ExcMessage("different numbers of points on the two sides"));
              const double h = side0.get_cell()->diameter();
              for (unsigned int q = 0; q < fe_iv->n_quadrature_points; ++q)
                AssertThrow(
                  side0.quadrature_point(q).distance(
                    side1.quadrature_point(q)) < 1e-10 * h &&
                    std::abs(side0.JxW(q) - side1.JxW(q)) <
                      1e-10 * std::abs(side0.JxW(q)) &&
                    (side0.normal_vector(q) + side1.normal_vector(q)).norm() <
                      1e-10,
                  ExcMessage(
                    "the two sides of a face (" +
                    side0.get_cell()->reference_cell().to_string() + " face " +
                    std::to_string(side0.get_face_number()) + ", " +
                    side1.get_cell()->reference_cell().to_string() + " face " +
                    std::to_string(side1.get_face_number()) +
                    ") do not match at point " + std::to_string(q)));
            }
            const unsigned int n = fe_iv->n_current_interface_dofs();
            FullMatrix<double> face_matrix(n, n);
            const auto        &q_points = fe_iv->get_quadrature_points();
            const auto        &normals  = fe_iv->get_normal_vectors();
            for (unsigned int q = 0; q < fe_iv->n_quadrature_points; ++q)
              {
                const double beta_n = velocity(q_points[q]) * normals[q];
                for (unsigned int i = 0; i < n; ++i)
                  for (unsigned int j = 0; j < n; ++j)
                    face_matrix(i, j) -=
                      fe_iv->jump_in_shape_values(i, q) * beta_n *
                      fe_iv->shape_value(beta_n > 0, j, q) * fe_iv->JxW(q);
              }
            const std::vector<types::global_dof_index> indices =
              fe_iv->get_interface_dof_indices();
            no_constraints.distribute_local_to_global(face_matrix,
                                                      indices,
                                                      system_matrix);
          };

          // the face term by hand: the quadrature of the face of `ffv` (this
          // cell, outer normal), the shape values phi_n of the neighbor at its
          // points; the rows and columns of this cell first, then those of the
          // neighbor
          FullMatrix<double> phi_n, face_matrix;
          const auto assemble_manual_face = [&](const FEFaceValues<dim> &ffv) {
            const unsigned int nc = dof_indices.size(),
                               nn = neighbor_dof_indices.size();
            face_matrix.reinit(nc + nn, nc + nn);
            for (unsigned int q = 0; q < ffv.n_quadrature_points; ++q)
              {
                const double beta_n =
                  velocity(ffv.quadrature_point(q)) * ffv.normal_vector(q);
                const double w = beta_n * ffv.JxW(q);
                for (unsigned int i = 0; i < nc + nn; ++i)
                  {
                    const double jump =
                      i < nc ? ffv.shape_value(i, q) : -phi_n(i - nc, q);
                    if (beta_n > 0)
                      for (unsigned int j = 0; j < nc; ++j)
                        face_matrix(i, j) -= jump * w * ffv.shape_value(j, q);
                    else
                      for (unsigned int j = 0; j < nn; ++j)
                        face_matrix(i, nc + j) -= jump * w * phi_n(j, q);
                  }
              }
            face_dof_indices = dof_indices;
            face_dof_indices.insert(face_dof_indices.end(),
                                    neighbor_dof_indices.begin(),
                                    neighbor_dof_indices.end());
            no_constraints.distribute_local_to_global(face_matrix,
                                                      face_dof_indices,
                                                      system_matrix);
          };
          // the shape values of a cell at the points of a face of a finer
          // neighbor, through the inverse mapping (the shape functions of the
          // DG elements are not transformed)
          double     hanging_time = 0; // the inverse mapping at hanging faces
          const auto shape_values_at = [&](const auto              &c,
                                           const FEFaceValues<dim> &ffv) {
            const auto start = std::chrono::steady_clock::now();
            const auto stop  = [&]() {
              hanging_time += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - start)
                                .count();
            };
            const unsigned int fi = fe_index(c->reference_cell());
            phi_n.reinit(fe[fi].n_dofs_per_cell(), ffv.n_quadrature_points);
            for (unsigned int q = 0; q < ffv.n_quadrature_points; ++q)
              {
                const Point<dim> x_unit =
                  mapping[fi].transform_real_to_unit_cell(
                    c, ffv.quadrature_point(q));
                for (unsigned int i = 0; i < phi_n.m(); ++i)
                  phi_n(i, q) = fe[fi].shape_value(i, x_unit);
              }
            stop();
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
                    const Tensor<1, dim> beta =
                      velocity(fv.quadrature_point(q));
                    for (unsigned int i = 0; i < n; ++i)
                      for (unsigned int j = 0; j < n; ++j)
                        {
                          cell_matrix(i, j) += fv.shape_value(j, q) *
                                               (beta * fv.shape_grad(i, q)) *
                                               fv.JxW(q);
                          cell_mass(i, j) += fv.shape_value(i, q) *
                                             fv.shape_value(j, q) * fv.JxW(q);
                        }
                  }
                // outflow boundary
                for (const unsigned int f : cell->face_indices())
                  if (cell->at_boundary(f))
                    {
                      hp_fe_face_values.reinit(cell, f);
                      const FEFaceValues<dim> &ffv =
                        hp_fe_face_values.get_present_fe_values();
                      for (const unsigned int q :
                           ffv.quadrature_point_indices())
                        {
                          const double beta_n =
                            velocity(ffv.quadrature_point(q)) *
                            ffv.normal_vector(q);
                          if (beta_n > 0)
                            for (unsigned int i = 0; i < n; ++i)
                              for (unsigned int j = 0; j < n; ++j)
                                cell_matrix(i, j) -=
                                  beta_n * ffv.shape_value(i, q) *
                                  ffv.shape_value(j, q) * ffv.JxW(q);
                        }
                    }
                dof_indices.resize(n);
                cell->get_dof_indices(dof_indices);
                no_constraints.distribute_local_to_global(cell_matrix,
                                                          dof_indices,
                                                          system_matrix);
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
                    const bool coarser = cell->neighbor_is_coarser(f);
                    if (!coarser)
                      {
                        const bool mine =
                          neighbor->is_locally_owned() ?
                            cell->id() < neighbor->id() :
                            cell->subdomain_id() < neighbor->subdomain_id();
                        if (!mine)
                          continue;
                      }
                    if (use_interface)
                      {
                        if (coarser)
                          {
                            const auto [nf, nsf] =
                              cell->neighbor_of_coarser_neighbor(f);
                            fe_iv->reinit(cell,
                                          f,
                                          numbers::invalid_unsigned_int,
                                          neighbor,
                                          nf,
                                          nsf);
                          }
                        else
                          fe_iv->reinit(cell,
                                        f,
                                        numbers::invalid_unsigned_int,
                                        neighbor,
                                        cell->neighbor_of_neighbor(f),
                                        numbers::invalid_unsigned_int);
                        assemble_interface_face();
                        continue;
                      }
                    hp_fe_face_values.reinit(cell, f);
                    const FEFaceValues<dim> &ffv =
                      hp_fe_face_values.get_present_fe_values();
                    neighbor_dof_indices.resize(
                      neighbor->get_fe().n_dofs_per_cell());
                    neighbor->get_dof_indices(neighbor_dof_indices);
                    if (coarser)
                      shape_values_at(neighbor, ffv);
                    else
                      {
                        // conforming: the face quadrature of the neighbor gives
                        // the same points
                        hp_fe_face_values_neighbor.reinit(
                          neighbor, cell->neighbor_of_neighbor(f));
                        const FEFaceValues<dim> &nfv =
                          hp_fe_face_values_neighbor.get_present_fe_values();
                        AssertThrow(nfv.n_quadrature_points ==
                                      ffv.n_quadrature_points,
                                    ExcInternalError());
                        phi_n.reinit(neighbor_dof_indices.size(),
                                     ffv.n_quadrature_points);
                        for (unsigned int q = 0; q < ffv.n_quadrature_points;
                             ++q)
                          {
                            AssertThrow(
                              nfv.quadrature_point(q).distance(
                                ffv.quadrature_point(q)) <
                                1e-10 * cell->diameter(),
                              ExcMessage(
                                "face quadrature points do not match"));
                            for (unsigned int i = 0; i < phi_n.m(); ++i)
                              phi_n(i, q) = nfv.shape_value(i, q);
                          }
                      }
                    assemble_manual_face(ffv);
                  }
              }
          times["assembly: hanging"] += hanging_time;
          // the time until all processes have assembled (imbalance), then the
          // compress itself
          const auto barrier_start = std::chrono::steady_clock::now();
          MPI_Barrier(mpi_communicator);
          const auto compress_start = std::chrono::steady_clock::now();
          times["assembly: wait"] +=
            std::chrono::duration<double>(compress_start - barrier_start)
              .count();
          system_matrix.compress(VectorOperation::add);
          times["assembly: compress"] +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          compress_start)
              .count();
          if (passes.size() == 2 && pass == 0)
            reference_matrix.copy_from(system_matrix);
        }
      if (passes.size() == 2)
        {
          reference_matrix.add(-1.0, system_matrix);
          const double difference =
            reference_matrix.frobenius_norm() / system_matrix.frobenius_norm();
          max_matrix_difference = std::max(max_matrix_difference, difference);
        }
    });
  }



  void AdvectionProblem::apply(const LA::MPI::Vector &u,
                               LA::MPI::Vector       &y) const
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



  void AdvectionProblem::time_step(const double dt)
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
          const double    a[4] = {0, 0.5, 0.5, 1},
                       b[4]    = {1. / 6, 1. / 3, 1. / 3, 1. / 6};
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



  void AdvectionProblem::update_ghosted()
  {
    ghosted = solution;
  }



  void AdvectionProblem::adapt_mesh()
  {
    this->time("adapt", [&]() {
      Vector<float> estimate;
      this->time("adapt: indicator", [&]() {
        update_ghosted();
        compute_indicator(estimate);
      });
      this->time("adapt: flags", [&]() {
        const double max_estimate =
          Utilities::MPI::max(estimate.size() ?
                                *std::max_element(estimate.begin(),
                                                  estimate.end()) :
                                0.f,
                              mpi_communicator);
        const unsigned int max_level = parameters.global + parameters.max_delta;
        for (const auto &cell : triangulation.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const double e = estimate(cell->active_cell_index());
              if (e > parameters.refine * max_estimate &&
                  cell->level() < static_cast<int>(max_level))
                cell->set_refine_flag();
              else if (e < parameters.coarsen * max_estimate &&
                       cell->level() > static_cast<int>(parameters.global))
                cell->set_coarsen_flag();
            }
      });

      SolutionTransfer<dim, LA::MPI::Vector> transfer(dof_handler);
      this->time("adapt: prepare", [&]() {
        triangulation.prepare_coarsening_and_refinement();
        transfer.prepare_for_coarsening_and_refinement(ghosted);
      });
      // t8code (adapt, balance, partition, copy) and the deal.II side:
      // rebuilding the levels of the triangulation and packing/unpacking the
      // transfer data
      this->time("adapt: execute",
                 [&]() { triangulation.execute_coarsening_and_refinement(); });
      // the parts of the refinement in the backend (t8code): adapt, balance,
      // partition, copy
      {
        const auto &rt = triangulation.get_refinement_times();
        times["refine: adapt"] += rt.adapt;
        times["refine: balance"] += rt.balance;
        times["refine: partition"] += rt.partition;
        times["refine: copy"] += rt.copy;
      }
      setup_system();
      this->time("adapt: interpolate",
                 [&]() { transfer.interpolate(solution); });
    });
  }



  double AdvectionProblem::l2_error(const double time)
  {
    update_ghosted();
    exact.set_time(time);
    const unsigned int   p = parameters.degree;
    hp::QCollection<dim> error_quadrature(
      QGauss<dim>(p + 3), QGaussSimplex<dim>(std::min(p + 3, 5u)));
#if ADV_DIM == 3
    error_quadrature.push_back(QGaussWedge<dim>(p + 3));
    error_quadrature.push_back(QGaussPyramid<dim>(p + 3));
#endif
    Vector<float> cell_errors(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      ghosted,
                                      exact,
                                      cell_errors,
                                      error_quadrature,
                                      VectorTools::L2_norm);
    return VectorTools::compute_global_error(triangulation,
                                             cell_errors,
                                             VectorTools::L2_norm);
  }



  // the integral of the solution, conserved by the scheme (no inflow) and by
  // the solution transfer
  double AdvectionProblem::mass()
  {
    update_ghosted();
    hp::FEValues<dim>   hp_fe_values(mapping,
                                   fe,
                                   quadrature,
                                   update_values | update_JxW_values);
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



  void AdvectionProblem::output_results(const unsigned int step,
                                        const double       time) const
  {
    if (parameters.output.empty())
      return;
    Vector<float> subdomain(triangulation.n_active_cells());
    Vector<float> cell_type(triangulation.n_active_cells());
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          subdomain(cell->active_cell_index()) =
            triangulation.locally_owned_subdomain();
          cell_type(cell->active_cell_index()) =
            fe_index(cell->reference_cell());
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
    data_out.write_vtu_with_pvtu_record(
      parameters.output + "/", "advection", step, mpi_communicator, 4);
  }



  void AdvectionProblem::run()
  {
    pcout << "advection_hybrid: " << dim << "D, "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " processes, mesh " << parameters.mesh << ", n " << parameters.n
          << ", faces " << parameters.faces << ", estimator "
          << parameters.estimator << ", partition " << parameters.partition
          << ", degree " << parameters.degree << ", scheme "
          << parameters.scheme << ", cfl " << parameters.cfl << ", adapt "
          << parameters.adapt << std::endl;

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
    h_min *= std::ldexp(1.0,
                        -static_cast<int>(
                          parameters.global +
                          (parameters.adapt ? parameters.max_delta : 0)));
    const double       beta_max = 2 * numbers::PI * std::sqrt(0.5);
    const unsigned int n_steps  = static_cast<unsigned int>(
      std::ceil(parameters.t_end * beta_max * (2 * parameters.degree + 1) /
                (parameters.cfl * h_min)));
    const double dt = parameters.t_end / n_steps;
    pcout << "h_min " << h_min << ", " << n_steps << " steps of " << dt
          << std::endl;

    // the parts of the mesh adaptation (adapt = indicator + flags + prepare +
    // execute + setup + assembly + interpolate)
    const std::vector<std::string> detail_phases = {"adapt: indicator",
                                                    "adapt: flags",
                                                    "adapt: prepare",
                                                    "adapt: execute",
                                                    "setup: dofs",
                                                    "setup: sparsity",
                                                    "assembly: hanging",
                                                    "assembly: compress",
                                                    "adapt: interpolate",
                                                    "sparsity: flux",
                                                    "sparsity: distribute",
                                                    "sparsity: matrix",
                                                    "assembly: wait"};
    std::ofstream                  csv;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        csv.open(parameters.csv);
        csv
          << "step,time,cells,dofs,cells_min,cells_max,l2_error,mass,time_step,adapt,assembly,"
             "refine_adapt,refine_balance,refine_partition,refine_copy,dofs_min,dofs_max,"
             "time_step_min";
        for (const std::string &phase : detail_phases)
          {
            std::string column = phase;
            std::replace(column.begin(), column.end(), ':', '_');
            column.erase(std::remove(column.begin(), column.end(), ' '),
                         column.end());
            csv << ',' << column;
          }
        csv << '\n';
      }
    const double mass0 = mass();
    update_ghosted();
    output_results(0, 0.0);
    // the phase times summed over the output intervals (maxima over the
    // processes per interval)
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
              1.0 * triangulation.n_locally_owned_active_cells(),
              mpi_communicator);
            const auto dofs = Utilities::MPI::min_max_avg(
              1.0 * dof_handler.n_locally_owned_dofs(), mpi_communicator);
            // the load balance of the time steps: the fastest process
            const double time_step_min =
              Utilities::MPI::min(times["time step"], mpi_communicator);
            total_times["time step (min)"] += time_step_min;
            std::map<std::string, double> t_max;
            for (const std::string phase : {"time step",
                                            "adapt",
                                            "assembly",
                                            "refine: adapt",
                                            "refine: balance",
                                            "refine: partition",
                                            "refine: copy"})
              {
                t_max[phase] =
                  Utilities::MPI::max(times[phase], mpi_communicator);
                total_times[phase] += t_max[phase];
              }
            for (const std::string &phase : detail_phases)
              {
                t_max[phase] =
                  Utilities::MPI::max(times[phase], mpi_communicator);
                total_times[phase] += t_max[phase];
                total_times[phase + " (min)"] +=
                  Utilities::MPI::min(times[phase], mpi_communicator);
              }
            total_times["assembly (min)"] +=
              Utilities::MPI::min(times["assembly"], mpi_communicator);
            times.clear();
            if (csv.is_open())
              csv << step << ',' << t << ','
                  << triangulation.n_global_active_cells() << ','
                  << dof_handler.n_dofs() << ',' << cells.min << ','
                  << cells.max << ',' << std::setprecision(10) << error << ','
                  << m << ',' << std::setprecision(6) << t_max["time step"]
                  << ',' << t_max["adapt"] << ',' << t_max["assembly"] << ','
                  << t_max["refine: adapt"] << ',' << t_max["refine: balance"]
                  << ',' << t_max["refine: partition"] << ','
                  << t_max["refine: copy"] << ',' << dofs.min << ',' << dofs.max
                  << ',' << time_step_min;
            if (csv.is_open())
              {
                for (const std::string &phase : detail_phases)
                  csv << ',' << t_max[phase];
                csv << '\n';
              }
            pcout << "step " << step << ", t " << t << ": "
                  << triangulation.n_global_active_cells() << " cells, "
                  << dof_handler.n_dofs() << " dofs, L2 error " << error
                  << ", mass change " << (m - mass0) / mass0 << std::endl;
            output_results(step, t);
          }
        if (parameters.adapt && step % parameters.adapt_every == 0 &&
            step < n_steps)
          adapt_mesh();
      }
    const double wall = MPI_Wtime() - wall_start;
    pcout << "final L2 error " << l2_error(n_steps * dt) << ", mass change "
          << (mass() - mass0) / mass0 << std::endl;
    pcout << "summary: steps " << n_steps << ", sum cells " << sum_cells
          << ", wall time " << wall << " s (time steps "
          << total_times["time step"] << " (fastest process "
          << total_times["time step (min)"] << "), adapt "
          << total_times["adapt"] << ", assembly " << total_times["assembly"]
          << "; in adapt: refine adapt " << total_times["refine: adapt"]
          << ", balance " << total_times["refine: balance"] << ", partition "
          << total_times["refine: partition"] << ", copy "
          << total_times["refine: copy"] << ")" << std::endl;
    pcout << "summary of adapt (maxima over the processes):";
    for (const std::string &phase : detail_phases)
      pcout << ' ' << phase << ' ' << total_times[phase] << ';';
    pcout << std::endl << "summary of adapt (minima over the processes):";
    for (const std::string &phase : detail_phases)
      pcout << ' ' << phase << ' ' << total_times[phase + " (min)"] << ';';
    pcout << " assembly " << total_times["assembly (min)"] << ';' << std::endl;
    if (parameters.faces == "compare")
      pcout
        << "compare faces: max relative difference of the matrices (manual - interface) "
        << max_matrix_difference << std::endl;
  }
} // namespace AdvectionHybrid



int main(int argc, char *argv[])
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
