/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * ------------------------------------------------------------------------
 *
 * A benchmark of adaptive refinement with parallel::distributed::Triangulation
 * (with p4est or t8code as backend, whichever deal.II was configured with):
 * a few global refinements of a coarse mesh, then cycles of refinement around
 * a point moving through the domain and coarsening behind it, each with
 * SolutionTransfer, DoF distribution, constraints, assembly and the solution
 * of a Poisson problem with CG and AMG. Every phase is timed separately, and
 * the refinement also in its parts: the adaptation, balance and partition of
 * the forest and the copy into the deal.II triangulation (refine: ...); the
 * minimum, average and maximum over the processes of each phase and cycle
 * are written to a CSV file, and a summary table to the screen.
 *
 * Usage: mpirun -np N ./amr_benchmark [key=value ...]
 *   mesh=hex|hybrid|brick|hybrid4|hybrid16
 *                     hexahedra (n^3 cells), the hybrid mesh of step-40_hybrid
 *                     (n x n columns, 5 layers; t8code only), or the meshes of
 *                     the PASC benchmark (t8_time_amr_cycle): the brick of
 *                     2 x 2 x 2 hexahedra, or the hybrid cube of 21 cells in the
 *                     octants of side 0.5 (6 tetrahedra at the origin, 2 prisms
 *                     in each of its face neighbors, 6 pyramids in the octant
 *                     [0.5,1]^3, hexahedra elsewhere; t8code only), or
 *                     hybrid16: hybrid4 with one hexahedron instead of the 6
 *                     pyramids (no pyramids; t8code only)
 *   refinement=point  point: refine around a moving point (path=...);
 *                     shell: the cycle of the PASC benchmark, the cells whose
 *                     center is closer to a sphere of radius 0.25 than
 *                     sqrt(3)/2 h are refined up to max_level, the others
 *                     coarsened down to global, the center starts at
 *                     (0.45, 0.47, 0.49) and moves in every coordinate by the
 *                     size of a cell of max_level per cycle; the setup refines
 *                     globally to the level global and then level by level at
 *                     the shell up to max_level
 *   n=4               coarse mesh size
 *   degree=2          polynomial degree
 *   global=2          global refinements before the adaptive cycles
 *   cycles=6          adaptive cycles
 *   max_level=5       maximal refinement level
 *   solve=1           0 to skip the linear solver
 *   coarsen=1         0 to only refine
 *   smoothing=vertices  mesh smoothing: vertices (limit_level_difference_at_
 *                     vertices, balance across vertices) or none (balance
 *                     across faces and edges only)
 *   csv=timings.csv   output file
 *   dump=             directory for the locally owned active cells of every
 *                     cycle, one file per process (c<cycle>_r<rank>.txt: shape
 *                     0 hexahedron, 1 tetrahedron, 2 wedge, 3 pyramid, level,
 *                     number of vertices and their coordinates, in the vertex
 *                     order of MFEM), for the partition metrics; empty for none
 *   dump_every=1      dump only the cycles that are multiples of this
 *   output=           directory for VTU output of every cycle (cell type,
 *                     subdomain, level, solution; not timed), empty for none
 *   path=helix        path of the refinement point: helix (half a turn around
 *                     the vertical axis, rising) or cut (a full circle in the
 *                     plane y = 0.5 through all layers, for the video, which
 *                     shows this plane)
 *   rhs=constant      right hand side: constant (1) or gaussian (a source at
 *                     the refinement point, so that the solution moves with it)
 *   period=0          number of cycles for the whole path, 0 for cycles; with
 *                     path=cut, e.g. cycles=43 period=40 gives a closed loop
 *                     of the cycles 3 to 42 after the mesh has been refined
 *                     up to max_level (for the video)
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
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
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

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


namespace AMRBenchmark
{
  using namespace dealii;

  constexpr int dim = 3;

  struct Parameters
  {
    std::string  mesh      = "hex";
    std::string  refinement = "point";
    unsigned int n         = 4;
    unsigned int degree    = 2;
    unsigned int global    = 2;
    unsigned int cycles    = 6;
    unsigned int max_level = 5;
    bool         solve     = true;
    bool         coarsen   = true;
    std::string  smoothing = "vertices";
    std::string  csv       = "timings.csv";
    std::string  output    = "";
    std::string  dump      = "";
    unsigned int dump_every = 1;
    std::string  path      = "helix";
    std::string  rhs       = "constant";
    unsigned int period    = 0;

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
          else if (key == "refinement")
            refinement = value;
          else if (key == "n")
            n = std::stoi(value);
          else if (key == "degree")
            degree = std::stoi(value);
          else if (key == "global")
            global = std::stoi(value);
          else if (key == "cycles")
            cycles = std::stoi(value);
          else if (key == "max_level")
            max_level = std::stoi(value);
          else if (key == "solve")
            solve = std::stoi(value) != 0;
          else if (key == "coarsen")
            coarsen = std::stoi(value) != 0;
          else if (key == "smoothing")
            smoothing = value;
          else if (key == "csv")
            csv = value;
          else if (key == "output")
            output = value;
          else if (key == "dump")
            dump = value;
          else if (key == "dump_every")
            dump_every = std::stoi(value);
          else if (key == "path")
            path = value;
          else if (key == "rhs")
            rhs = value;
          else if (key == "period")
            period = std::stoi(value);
          else
            AssertThrow(false, ExcMessage("Unknown argument " + key));
        }
      AssertThrow(mesh == "hex" || mesh == "hybrid" || mesh == "brick" ||
                    mesh == "hybrid4" || mesh == "hybrid16",
                  ExcMessage("mesh must be hex, hybrid, brick, hybrid4 or hybrid16"));
      AssertThrow(refinement == "point" || refinement == "shell",
                  ExcMessage("refinement must be point or shell"));
      AssertThrow(smoothing == "vertices" || smoothing == "none",
                  ExcMessage("smoothing must be vertices or none"));
      AssertThrow(path == "helix" || path == "cut",
                  ExcMessage("path must be helix or cut"));
      AssertThrow(rhs == "constant" || rhs == "gaussian",
                  ExcMessage("rhs must be constant or gaussian"));
    }
  };



  // the hybrid mesh of step-40_hybrid
  // The coarse mesh: [0,1]^3 split into n x n columns and 5 layers in z:
  //   0: hexahedra,
  //   1: each cube split into the 6 pyramids from its faces to its center,
  //   2: each cube split into 2 wedges with their axis along y (so that their
  //      quadrilaterals match the pyramid bases below and above),
  //   3: pyramids as in layer 1, but the top pyramid is split into 2
  //      tetrahedra along the diagonal of the tetrahedra above,
  //   4: each cube split into the 6 tetrahedra around its diagonal from
  //      (0,0,0) to (1,1,1).
  void
  make_hybrid_mesh(Triangulation<dim> &tria, const unsigned int n)
  {
    const unsigned int    n_layers = 5;
    std::vector<Point<dim>> vertices;
    for (unsigned int z = 0; z <= n_layers; ++z)
      for (unsigned int y = 0; y <= n; ++y)
        for (unsigned int x = 0; x <= n; ++x)
          vertices.emplace_back(1. * x / n, 1. * y / n, 1. * z / n_layers);
    const auto vertex = [n](const unsigned int x,
                            const unsigned int y,
                            const unsigned int z) {
      return x + (n + 1) * (y + (n + 1) * z);
    };

    const auto volume = [&](const unsigned int a,
                            const unsigned int b,
                            const unsigned int c,
                            const unsigned int d) {
      return cross_product_3d(vertices[b] - vertices[a],
                              vertices[c] - vertices[a]) *
             (vertices[d] - vertices[a]);
    };

    std::vector<CellData<dim>> cells;
    const auto add_tetrahedron = [&](std::array<unsigned int, 4> v) {
      if (volume(v[0], v[1], v[2], v[3]) < 0)
        std::swap(v[2], v[3]);
      CellData<dim> cell(4);
      for (unsigned int i = 0; i < 4; ++i)
        cell.vertices[i] = v[i];
      cells.push_back(cell);
    };
    const auto add_pyramid = [&](std::array<unsigned int, 4> base,
                                 const unsigned int          apex) {
      if (volume(base[0], base[1], base[2], apex) < 0)
        std::swap(base[1], base[2]);
      CellData<dim> cell(5);
      for (unsigned int i = 0; i < 4; ++i)
        cell.vertices[i] = base[i];
      cell.vertices[4] = apex;
      cells.push_back(cell);
    };
    const auto add_wedge = [&](std::array<unsigned int, 6> v) {
      if (volume(v[0], v[1], v[2], v[3]) < 0)
        {
          std::swap(v[1], v[2]);
          std::swap(v[4], v[5]);
        }
      CellData<dim> cell(6);
      for (unsigned int i = 0; i < 6; ++i)
        cell.vertices[i] = v[i];
      cells.push_back(cell);
    };

    // the faces of a cube with the vertices in the order of the faces of the
    // hexahedron
    static const unsigned int cube_faces[6][4] = {{0, 2, 4, 6},
                                                  {1, 3, 5, 7},
                                                  {0, 4, 1, 5},
                                                  {2, 6, 3, 7},
                                                  {0, 1, 2, 3},
                                                  {4, 5, 6, 7}};

    for (unsigned int z = 0; z < n_layers; ++z)
      for (unsigned int y = 0; y < n; ++y)
        for (unsigned int x = 0; x < n; ++x)
          {
            std::array<unsigned int, 8> cube;
            for (unsigned int v = 0; v < 8; ++v)
              cube[v] = vertex(x + (v & 1), y + ((v >> 1) & 1), z + (v >> 2));

            if (z == 0)
              {
                CellData<dim> hex(8);
                for (unsigned int v = 0; v < 8; ++v)
                  hex.vertices[v] = cube[v];
                cells.push_back(hex);
              }
            else if (z == 1 || z == 3)
              {
                const unsigned int center = vertices.size();
                vertices.emplace_back((x + 0.5) / n,
                                      (y + 0.5) / n,
                                      (z + 0.5) / n_layers);
                for (unsigned int f = 0; f < 6; ++f)
                  {
                    const std::array<unsigned int, 4> base = {
                      {cube[cube_faces[f][0]],
                       cube[cube_faces[f][1]],
                       cube[cube_faces[f][2]],
                       cube[cube_faces[f][3]]}};
                    if (z == 3 && f == 5)
                      {
                        // along the diagonal from cube vertex 4 to 7
                        add_tetrahedron({{base[0], base[1], base[3], center}});
                        add_tetrahedron({{base[0], base[3], base[2], center}});
                      }
                    else
                      add_pyramid(base, center);
                  }
              }
            else if (z == 2)
              {
                // the triangles lie in the planes y = const
                add_wedge({{cube[0], cube[1], cube[4], cube[2], cube[3], cube[6]}});
                add_wedge({{cube[1], cube[5], cube[4], cube[3], cube[7], cube[6]}});
              }
            else
              {
                std::array<unsigned int, 3> directions = {{0, 1, 2}};
                do
                  {
                    const unsigned int v1 = 1u << directions[0];
                    const unsigned int v2 = v1 | (1u << directions[1]);
                    add_tetrahedron({{cube[0], cube[v1], cube[v2], cube[7]}});
                  }
                while (
                  std::next_permutation(directions.begin(), directions.end()));
              }
          }

    tria.create_triangulation(vertices, cells, SubCellData());
  }



  // The hybrid cube of the PASC benchmark (hybrid4 of t8_time_amr_cycle): the
  // octants of side 0.5 of the unit cube with 6 tetrahedra in the octant at the
  // origin (the Kuhn subdivision along its diagonal), 2 prisms in each of its
  // face neighbors (split along the face diagonals of the tetrahedra), 6
  // pyramids with the apex at the center in the octant [0.5,1]^3, and hexahedra
  // in the other three octants.
  //
  // Without pyramids (hybrid16), the octant [0.5,1]^3 is one hexahedron, at the
  // same position in the order of the coarse cells.
  void
  make_hybrid4_mesh(Triangulation<dim> &tria, const bool pyramids = true)
  {
    std::vector<Point<dim>> vertices;
    for (unsigned int z = 0; z <= 2; ++z)
      for (unsigned int y = 0; y <= 2; ++y)
        for (unsigned int x = 0; x <= 2; ++x)
          vertices.emplace_back(0.5 * x, 0.5 * y, 0.5 * z);
    const unsigned int apex = vertices.size();
    if (pyramids)
      vertices.emplace_back(0.75, 0.75, 0.75);
    const auto volume = [&](const unsigned int a,
                            const unsigned int b,
                            const unsigned int c,
                            const unsigned int d) {
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
    // the faces of a cube in the vertex order of the faces of a hexahedron
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
            // the vertex of the octant at the local corner (i, j, k)
            const auto corner = [&](const unsigned int i,
                                    const unsigned int j,
                                    const unsigned int k) {
              return (ox + i) + 3 * ((oy + j) + 3 * (oz + k));
            };
            std::array<unsigned int, 8> cube;
            for (unsigned int v = 0; v < 8; ++v)
              cube[v] = corner(v & 1, (v >> 1) & 1, v >> 2);
            const unsigned int n_offsets = ox + oy + oz;
            if (n_offsets == 0)
              {
                std::array<unsigned int, 3> directions = {{0, 1, 2}};
                do
                  {
                    const unsigned int v1 = 1u << directions[0];
                    const unsigned int v2 = v1 | (1u << directions[1]);
                    std::vector<unsigned int> t = {cube[0], cube[v1], cube[v2], cube[7]};
                    if (volume(t[0], t[1], t[2], t[3]) < 0)
                      std::swap(t[2], t[3]);
                    add_cell(t);
                  }
                while (
                  std::next_permutation(directions.begin(), directions.end()));
              }
            else if (n_offsets == 1)
              {
                // the axis a of the offset, the other axes b and c; the face
                // at the tetrahedra is split along the diagonal from (b, c) =
                // (0, 0) to (1, 1), like the faces of the Kuhn tetrahedra
                const unsigned int a = ox ? 0 : (oy ? 1 : 2);
                const unsigned int b = (a + 1) % 3, c = (a + 2) % 3;
                const auto local = [&](const unsigned int ia,
                                       const unsigned int ib,
                                       const unsigned int ic) {
                  unsigned int idx[3];
                  idx[a] = ia;
                  idx[b] = ib;
                  idx[c] = ic;
                  return corner(idx[0], idx[1], idx[2]);
                };
                const unsigned int triangles[2][3][2] = {
                  {{0, 0}, {1, 0}, {1, 1}}, {{0, 0}, {1, 1}, {0, 1}}};
                for (const auto &t : triangles)
                  {
                    std::vector<unsigned int> w(6);
                    for (unsigned int i = 0; i < 3; ++i)
                      {
                        w[i]     = local(0, t[i][0], t[i][1]);
                        w[i + 3] = local(1, t[i][0], t[i][1]);
                      }
                    if (volume(w[0], w[1], w[2], w[3]) < 0)
                      {
                        std::swap(w[1], w[2]);
                        std::swap(w[4], w[5]);
                      }
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



  // The shell of the PASC benchmark (t8_time_amr_cycle): whether a cell of
  // the given level with center x touches the sphere of the given center and
  // radius, for coarse cells of side tree_size
  bool
  touches_shell(const Point<dim>  &x,
                const unsigned int level,
                const Point<dim>  &center,
                const double       radius,
                const double       tree_size)
  {
    const double h = tree_size * std::ldexp(1.0, -static_cast<int>(level));
    return std::fabs(x.distance(center) - radius) < 0.5 * std::sqrt(3.0) * h;
  }




  // the point around which the mesh is refined in cycle k
  Point<dim>
  refinement_center(const unsigned int k,
                    const unsigned int n_cycles,
                    const std::string &path)
  {
    if (path == "cut")
      {
        // a full circle in the plane y = 0.5, so that the last cycle connects
        // to the first one
        const double t = 2 * numbers::PI * k / std::max(1u, n_cycles);
        return Point<dim>(0.5 + 0.3 * std::cos(t),
                          0.5,
                          0.5 + 0.33 * std::sin(t));
      }
    const double t = numbers::PI * k / std::max(1u, n_cycles);
    return Point<dim>(0.5 + 0.3 * std::cos(t),
                      0.5 + 0.3 * std::sin(t),
                      0.3 + 0.4 * k / std::max(1u, n_cycles));
  }



  class Benchmark
  {
  public:
    Benchmark(const Parameters &parameters);

    void
    run();

  private:
    void
    setup_dofs();
    void
    assemble(const unsigned int cycle);
    unsigned int
    solve();
    void
    mark_cells(const unsigned int cycle);
    // the side of the coarse cells
    double
    tree_size() const
    {
      return parameters.mesh == "hex" ? 1.0 / parameters.n :
             parameters.mesh == "hybrid" ? 1.0 / parameters.n : 0.5;
    }
    // add the parts of the last execute_coarsening_and_refinement()
    void
    record_refinement_parts();
    void
    output_results(const unsigned int cycle) const;
    void
    dump_cells(const unsigned int cycle) const;

    // time a phase: the wall time of each process for each cycle
    template <typename F>
    void
    time(const std::string &phase, const F &f);

    void
    record_cycle(const unsigned int cycle, const unsigned int iterations);

    const Parameters   parameters;
    MPI_Comm           mpi_communicator;
    ConditionalOStream pcout;

    parallel::distributed::Triangulation<dim> triangulation;
    hp::FECollection<dim>                     fe;
    hp::MappingCollection<dim>                mapping;
    hp::QCollection<dim>                      quadrature;
    DoFHandler<dim>                           dof_handler;

    IndexSet                  locally_owned_dofs, locally_relevant_dofs;
    AffineConstraints<double> constraints;
    LA::MPI::SparseMatrix     system_matrix;
    LA::MPI::Vector           solution, system_rhs;

    std::vector<std::string>      phases;
    std::map<std::string, double> cycle_times;
    std::map<std::string, double> total_times;
    std::ofstream                 csv;
  };



  Benchmark::Benchmark(const Parameters &parameters)
    : parameters(parameters)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , triangulation(mpi_communicator,
                    parameters.smoothing == "vertices" ?
                      Triangulation<dim>::limit_level_difference_at_vertices :
                      Triangulation<dim>::none)
    , dof_handler(triangulation)
  {
    const unsigned int p = parameters.degree;
    if (parameters.mesh == "hex" || parameters.mesh == "brick")
      {
        fe.push_back(FE_Q<dim>(p));
        mapping.push_back(MappingQ<dim>(1));
        quadrature.push_back(QGauss<dim>(p + 1));
      }
    else
      {
        // equidistant FE_Q so that the DoFs on faces with the other elements
        // coincide
        fe.push_back(FE_Q<dim>(QIterated<1>(QTrapezoid<1>(), p)));
        fe.push_back(FE_SimplexP<dim>(p));
        fe.push_back(FE_WedgeP<dim>(p));
        fe.push_back(FE_PyramidP<dim>(p));
        mapping.push_back(MappingQ<dim>(1));
        mapping.push_back(MappingFE<dim>(FE_SimplexP<dim>(1)));
        mapping.push_back(MappingFE<dim>(FE_WedgeP<dim>(1)));
        mapping.push_back(MappingFE<dim>(FE_PyramidP<dim>(1)));
        quadrature.push_back(QGauss<dim>(p + 1));
        quadrature.push_back(QGaussSimplex<dim>(p + 1));
        quadrature.push_back(QGaussWedge<dim>(p + 1));
        quadrature.push_back(QGaussPyramid<dim>(p + 1));
      }

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        csv.open(parameters.csv);
        csv << "cycle,phase,min,avg,max" << std::endl;
      }
  }



  template <typename F>
  void
  Benchmark::time(const std::string &phase, const F &f)
  {
    MPI_Barrier(mpi_communicator);
    Timer timer;
    f();
    timer.stop();
    if (std::find(phases.begin(), phases.end(), phase) == phases.end())
      phases.push_back(phase);
    cycle_times[phase] += timer.wall_time();
  }



  void
  Benchmark::record_refinement_parts()
  {
    const auto &t = triangulation.get_refinement_times();
    for (const auto &[phase, value] :
         std::vector<std::pair<std::string, double>>{{"refine: adapt", t.adapt},
                                                     {"refine: balance", t.balance},
                                                     {"refine: partition", t.partition},
                                                     {"refine: copy", t.copy}})
      {
        if (std::find(phases.begin(), phases.end(), phase) == phases.end())
          phases.push_back(phase);
        cycle_times[phase] += value;
      }
  }



  void
  Benchmark::record_cycle(const unsigned int cycle,
                          const unsigned int iterations)
  {
    // families of active siblings on more than one process, which cannot be
    // coarsened (partitioning for coarsening should avoid them)
    unsigned int n_split_families = 0;
    for (const auto &cell : triangulation.cell_iterators())
      if (cell->has_children() && cell->child(0)->is_locally_owned())
        {
          bool all_active = true, split = false;
          for (unsigned int c = 0; c < cell->n_children(); ++c)
            {
              all_active &= cell->child(c)->is_active();
              split |= !cell->child(c)->is_locally_owned();
            }
          n_split_families += (all_active && split);
        }
    n_split_families = Utilities::MPI::sum(n_split_families, mpi_communicator);
    pcout << "cycle " << cycle << ": " << n_split_families
          << " split families, " << triangulation.n_global_active_cells()
          << " cells, " << triangulation.n_global_levels() << " levels, "
          << dof_handler.n_dofs() << " dofs, " << iterations
          << " CG iterations" << std::endl;
    for (const std::string &phase : phases)
      {
        const auto t =
          Utilities::MPI::min_max_avg(cycle_times[phase], mpi_communicator);
        total_times[phase] += t.max;
        if (csv.is_open())
          csv << cycle << ',' << phase << ',' << t.min << ',' << t.avg << ','
              << t.max << std::endl;
      }
    cycle_times.clear();
  }



  void
  Benchmark::setup_dofs()
  {
    time("distribute dofs", [&]() {
      if (fe.size() > 1)
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto r = cell->reference_cell();
              cell->set_active_fe_index(
                r == ReferenceCells::Hexahedron  ? 0 :
                r == ReferenceCells::Tetrahedron ? 1 :
                r == ReferenceCells::Wedge       ? 2 :
                                                   3);
            }
      if (fe.size() > 1)
        dof_handler.distribute_dofs(fe);
      else
        dof_handler.distribute_dofs(fe[0]);
      locally_owned_dofs    = dof_handler.locally_owned_dofs();
      locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
    });

    time("constraints", [&]() {
      constraints.clear();
      constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      VectorTools::interpolate_boundary_values(
        mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
      constraints.close();
    });

    time("sparsity", [&]() {
      DynamicSparsityPattern dsp(locally_relevant_dofs);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(dsp,
                                                 locally_owned_dofs,
                                                 mpi_communicator,
                                                 locally_relevant_dofs);
      system_matrix.reinit(locally_owned_dofs,
                           locally_owned_dofs,
                           dsp,
                           mpi_communicator);
      system_rhs.reinit(locally_owned_dofs, mpi_communicator);
    });
  }



  void
  Benchmark::assemble(const unsigned int cycle)
  {
    // the source of rhs=gaussian, at the refinement point
    const Point<dim> center =
      refinement_center(cycle,
                        parameters.period > 0 ? parameters.period :
                                                parameters.cycles,
                        parameters.path);
    const bool   gaussian = parameters.rhs == "gaussian";
    const double width    = 0.08;
    time("assembly", [&]() {
      hp::FEValues<dim>                    hp_fe_values(mapping,
                                     fe,
                                     quadrature,
                                     update_values | update_gradients |
                                       update_JxW_values |
                                       (gaussian ? update_quadrature_points :
                                                   update_default));
      FullMatrix<double>                   cell_matrix;
      Vector<double>                       cell_rhs;
      std::vector<types::global_dof_index> dof_indices;
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            hp_fe_values.reinit(cell);
            const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
            const unsigned int   n         = cell->get_fe().n_dofs_per_cell();
            cell_matrix.reinit(n, n);
            cell_rhs.reinit(n);
            for (const unsigned int q : fe_values.quadrature_point_indices())
              {
                const double f =
                  gaussian ?
                    std::exp(-fe_values.quadrature_point(q).distance_square(
                               center) /
                             (width * width)) /
                      (width * width) :
                    1.;
                for (unsigned int i = 0; i < n; ++i)
                  {
                    for (unsigned int j = 0; j < n; ++j)
                      cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                           fe_values.shape_grad(j, q) *
                                           fe_values.JxW(q);
                    cell_rhs(i) +=
                      f * fe_values.shape_value(i, q) * fe_values.JxW(q);
                  }
              }
            dof_indices.resize(n);
            cell->get_dof_indices(dof_indices);
            constraints.distribute_local_to_global(
              cell_matrix, cell_rhs, dof_indices, system_matrix, system_rhs);
          }
      system_matrix.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);
    });
  }



  unsigned int
  Benchmark::solve()
  {
    LA::MPI::Vector                distributed(locally_owned_dofs, mpi_communicator);
    LA::MPI::PreconditionAMG       preconditioner;
    time("amg setup", [&]() {
      LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
      data.symmetric_operator = true;
#endif
      preconditioner.initialize(system_matrix, data);
    });
    SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
    time("cg", [&]() {
      LA::SolverCG solver(solver_control);
      solver.solve(system_matrix, distributed, system_rhs, preconditioner);
      constraints.distribute(distributed);
    });
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    solution = distributed;
    return solver_control.last_step();
  }



  void
  Benchmark::mark_cells(const unsigned int cycle)
  {
    if (parameters.refinement == "shell")
      {
        // refine the cells touching the shell, coarsen the others; the
        // center moves in every coordinate by the size of a cell of the
        // maximal level per cycle
        const double velocity =
          tree_size() * std::ldexp(1.0, -static_cast<int>(parameters.max_level));
        const Point<dim> center(0.45 + cycle * velocity,
                                0.47 + cycle * velocity,
                                0.49 + cycle * velocity);
        for (const auto &cell : triangulation.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const unsigned int level = cell->level();
              if (touches_shell(cell->center(), level, center, 0.25, tree_size()))
                {
                  if (level < parameters.max_level)
                    cell->set_refine_flag();
                }
              else if (parameters.coarsen && level > parameters.global)
                cell->set_coarsen_flag();
            }
        return;
      }
    // refine around the moving point, coarsen behind it
    const Point<dim> center =
      refinement_center(cycle,
                        parameters.period > 0 ? parameters.period :
                                                parameters.cycles,
                        parameters.path);
    const double     radius = 0.15;
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const double d = cell->center().distance(center);
          if (d < radius + cell->diameter() &&
              cell->level() < static_cast<int>(parameters.max_level))
            cell->set_refine_flag();
          else if (parameters.coarsen && d > 2 * radius &&
                   cell->level() > static_cast<int>(parameters.global))
            cell->set_coarsen_flag();
        }
  }



  void
  Benchmark::output_results(const unsigned int cycle) const
  {
    if (parameters.output.empty())
      return;
    Vector<float> cell_type(triangulation.n_active_cells());
    Vector<float> subdomain(triangulation.n_active_cells());
    Vector<float> level(triangulation.n_active_cells());
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto r = cell->reference_cell();
          // the fe indices: 0 hexahedron, 1 tetrahedron, 2 wedge, 3 pyramid
          cell_type(cell->active_cell_index()) =
            r == ReferenceCells::Hexahedron  ? 0 :
            r == ReferenceCells::Tetrahedron ? 1 :
            r == ReferenceCells::Wedge       ? 2 :
                                               3;
          subdomain(cell->active_cell_index()) = cell->subdomain_id();
          level(cell->active_cell_index())     = cell->level();
        }
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.add_data_vector(cell_type, "cell_type");
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.add_data_vector(level, "level");
    data_out.build_patches(mapping);
    data_out.write_vtu_with_pvtu_record(
      parameters.output + "/", "amr", cycle, mpi_communicator, 4);
  }



  void
  Benchmark::dump_cells(const unsigned int cycle) const
  {
    if (parameters.dump.empty() || cycle % parameters.dump_every != 0)
      return;
    std::ofstream out(parameters.dump + "/c" + std::to_string(cycle) + "_r" +
                      std::to_string(Utilities::MPI::this_mpi_process(
                        mpi_communicator)) +
                      ".txt");
    out << std::setprecision(17);
    // the vertices in the order of MFEM: hexahedron and pyramid base
    // counterclockwise instead of lexicographic
    static const unsigned int hex_order[8] = {0, 1, 3, 2, 4, 5, 7, 6};
    static const unsigned int pyr_order[5] = {0, 1, 3, 2, 4};
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto         r     = cell->reference_cell();
          const unsigned int shape = r == ReferenceCells::Hexahedron  ? 0 :
                                     r == ReferenceCells::Tetrahedron ? 1 :
                                     r == ReferenceCells::Wedge       ? 2 :
                                                                        3;
          out << shape << ' ' << cell->level() << ' ' << cell->n_vertices();
          for (const unsigned int i : cell->vertex_indices())
            {
              const unsigned int v = shape == 0 ? hex_order[i] :
                                     shape == 3 ? pyr_order[i] :
                                                  i;
              out << ' ' << cell->vertex(v);
            }
          out << '\n';
        }
  }



  void
  Benchmark::run()
  {
    pcout << "amr_benchmark: "
#if defined(DEAL_II_WITH_T8CODE)
          << "t8code"
#else
          << "p4est"
#endif
          << ", " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " processes, mesh " << parameters.mesh << ", refinement "
          << parameters.refinement << ", n " << parameters.n
          << ", degree " << parameters.degree << ", smoothing "
          << parameters.smoothing << std::endl;

    time("coarse mesh", [&]() {
      if (parameters.mesh == "hex")
        GridGenerator::subdivided_hyper_cube(triangulation, parameters.n);
      else if (parameters.mesh == "brick")
        GridGenerator::subdivided_hyper_cube(triangulation, 2);
      else if (parameters.mesh == "hybrid4")
        make_hybrid4_mesh(triangulation);
      else if (parameters.mesh == "hybrid16")
        make_hybrid4_mesh(triangulation, false);
      else
        make_hybrid_mesh(triangulation, parameters.n);
    });
    time("refine", [&]() { triangulation.refine_global(parameters.global); });
    if (parameters.refinement == "shell")
      // the setup of the PASC benchmark: level by level at the shell
      time("refine", [&]() {
        const Point<dim> center(0.45, 0.47, 0.49);
        for (unsigned int l = parameters.global; l < parameters.max_level; ++l)
          {
            for (const auto &cell : triangulation.active_cell_iterators())
              if (cell->is_locally_owned() &&
                  static_cast<unsigned int>(cell->level()) < parameters.max_level &&
                  touches_shell(cell->center(), cell->level(), center, 0.25, tree_size()))
                cell->set_refine_flag();
            triangulation.execute_coarsening_and_refinement();
          }
      });
    setup_dofs();
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    assemble(0);
    const unsigned int iterations = parameters.solve ? solve() : 0;
    record_cycle(0, iterations);
    output_results(0);
    dump_cells(0);

    for (unsigned int cycle = 1; cycle <= parameters.cycles; ++cycle)
      {
        time("mark", [&]() { mark_cells(cycle); });
        SolutionTransfer<dim, LA::MPI::Vector> transfer(dof_handler);
        time("prepare transfer", [&]() {
          triangulation.prepare_coarsening_and_refinement();
          transfer.prepare_for_coarsening_and_refinement(solution);
        });
        time("refine", [&]() {
          triangulation.execute_coarsening_and_refinement();
        });
        record_refinement_parts();
        setup_dofs();
        time("transfer", [&]() {
          LA::MPI::Vector transferred(locally_owned_dofs, mpi_communicator);
          transfer.interpolate(transferred);
          constraints.distribute(transferred);
          solution.reinit(locally_owned_dofs,
                          locally_relevant_dofs,
                          mpi_communicator);
          solution = transferred;
        });
        assemble(cycle);
        const unsigned int iterations = parameters.solve ? solve() : 0;
        record_cycle(cycle, iterations);
        output_results(cycle);
        dump_cells(cycle);
      }

    // the summary: the sums over the cycles of the maxima over the processes
    double total = 0;
    for (const auto &[phase, t] : total_times)
      total += t;
    pcout << std::endl << "phase                 time [s]  share" << std::endl;
    for (const std::string &phase : phases)
      pcout << std::left << std::setw(20) << phase << std::right
            << std::setw(10) << std::fixed << std::setprecision(3)
            << total_times[phase] << std::setw(7) << std::setprecision(1)
            << 100. * total_times[phase] / total << '%' << std::endl;
    pcout << std::left << std::setw(20) << "total" << std::right
          << std::setw(10) << std::setprecision(3) << total << std::endl;
  }
} // namespace AMRBenchmark



int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      AMRBenchmark::Parameters         parameters;
      parameters.parse(argc, argv);
      AMRBenchmark::Benchmark benchmark(parameters);
      benchmark.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << "Exception: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
