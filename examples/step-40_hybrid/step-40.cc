/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * ------------------------------------------------------------------------
 *
 * A variant of step-40 on a hybrid mesh of hexahedra, pyramids, wedges and
 * tetrahedra, distributed with parallel::distributed::Triangulation (with
 * t8code as backend): the Poisson equation -Laplace u = f with a known
 * solution u (a Gaussian peak), continuous elements of the same degree on all
 * cells (FE_Q, FE_SimplexP, FE_WedgeP, FE_PyramidP), and adaptive refinement
 * by the computed H1 error. Every cycle prints the number of cells of each
 * type, the number of DoFs and the L2 and H1 errors.
 *
 * Usage: mpirun -np N ./step-40 [degree] [n_cycles] [global]
 *   degree    polynomial degree (1 to 4, default 2)
 *   n_cycles  number of refinement cycles (default 6)
 *   global    1 for global instead of adaptive refinement (default 0)
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
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
#include <deal.II/numerics/vector_tools.h>

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>

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


namespace Step40Hybrid
{
  using namespace dealii;

  constexpr int dim = 3;

  // The exact solution, a Gaussian peak at the center of the domain, which
  // lies in the layer of wedges, and the right hand side f = -Laplace u.
  const Point<dim> peak(0.5, 0.5, 0.5);
  constexpr double width = 50.;

  // With the environment variable HYBRID_POLYNOMIAL=k, the exact solution is
  // instead the polynomial (1 + a.x)^k, which the discretization reproduces
  // exactly for degree >= k (a consistency check).
  const unsigned int polynomial_degree =
    std::getenv("HYBRID_POLYNOMIAL") ? std::atoi(std::getenv("HYBRID_POLYNOMIAL")) : 0;
  const Tensor<1, dim> a({0.3, -0.7, 0.5});

  class ExactSolution : public Function<dim>
  {
  public:
    double
    value(const Point<dim> &p, const unsigned int = 0) const override
    {
      if (polynomial_degree > 0)
        return std::pow(1 + a * p, polynomial_degree);
      return std::exp(-width * (p - peak).norm_square());
    }

    Tensor<1, dim>
    gradient(const Point<dim> &p, const unsigned int = 0) const override
    {
      if (polynomial_degree > 0)
        return polynomial_degree * std::pow(1 + a * p, polynomial_degree - 1) * a;
      return -2 * width * (p - peak) * value(p);
    }
  };

  class RightHandSide : public Function<dim>
  {
  public:
    double
    value(const Point<dim> &p, const unsigned int = 0) const override
    {
      if (polynomial_degree > 1)
        return -1. * polynomial_degree * (polynomial_degree - 1) *
               std::pow(1 + a * p, polynomial_degree - 2) * (a * a);
      else if (polynomial_degree == 1)
        return 0;
      const double r2 = (p - peak).norm_square();
      return (6 * width - 4 * width * width * r2) * std::exp(-width * r2);
    }
  };



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



  class LaplaceProblem
  {
  public:
    LaplaceProblem(const unsigned int degree,
                   const unsigned int n_cycles,
                   const bool         global_refinement);

    void
    run();

  private:
    void
    setup_system();
    void
    assemble_system();
    void
    solve();
    void
    compute_errors(Vector<float> &cell_errors);
    void
    refine_grid(const Vector<float> &cell_errors);
    void
    output_results(const unsigned int cycle) const;

    unsigned int
    fe_index(const ReferenceCell &reference_cell) const;

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    const unsigned int degree;
    const unsigned int n_cycles;
    const bool         global_refinement;

    // the elements, mappings and quadratures for hexahedra, tetrahedra,
    // wedges and pyramids, in this order
    hp::FECollection<dim>      fe;
    hp::MappingCollection<dim> mapping;
    hp::QCollection<dim>       quadrature;
    DoFHandler<dim>            dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    LA::MPI::SparseMatrix system_matrix;
    LA::MPI::Vector       locally_relevant_solution;
    LA::MPI::Vector       system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };



  LaplaceProblem::LaplaceProblem(const unsigned int degree,
                                 const unsigned int n_cycles,
                                 const bool         global_refinement)
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    Triangulation<dim>::limit_level_difference_at_vertices)
    , degree(degree)
    , n_cycles(n_cycles)
    , global_refinement(global_refinement)
    // FE_Q with equidistant support points, like the other elements, so that
    // their DoFs on common faces coincide
    , fe(FE_Q<dim>(QIterated<1>(QTrapezoid<1>(), degree)),
         FE_SimplexP<dim>(degree),
         FE_WedgeP<dim>(degree),
         FE_PyramidP<dim>(degree))
    , mapping(MappingQ<dim>(1),
              MappingFE<dim>(FE_SimplexP<dim>(1)),
              MappingFE<dim>(FE_WedgeP<dim>(1)),
              MappingFE<dim>(FE_PyramidP<dim>(1)))
    , quadrature(QGauss<dim>(degree + 1),
                 QGaussSimplex<dim>(degree + 1),
                 QGaussWedge<dim>(degree + 1),
                 QGaussPyramid<dim>(degree + 1))
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
  {}



  unsigned int
  LaplaceProblem::fe_index(const ReferenceCell &reference_cell) const
  {
    if (reference_cell == ReferenceCells::Hexahedron)
      return 0;
    else if (reference_cell == ReferenceCells::Tetrahedron)
      return 1;
    else if (reference_cell == ReferenceCells::Wedge)
      return 2;
    else
      return 3;
  }



  void
  LaplaceProblem::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_active_fe_index(fe_index(cell->reference_cell()));
    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(
      mapping, dof_handler, 0, ExactSolution(), constraints);
    constraints.close();

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
  }



  void
  LaplaceProblem::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    hp::FEValues<dim> hp_fe_values(mapping,
                                   fe,
                                   quadrature,
                                   update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values);
    const RightHandSide right_hand_side;

    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          hp_fe_values.reinit(cell);
          const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
          const unsigned int   dofs_per_cell = cell->get_fe().n_dofs_per_cell();

          cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
          cell_rhs.reinit(dofs_per_cell);

          for (const unsigned int q : fe_values.quadrature_point_indices())
            {
              const double rhs_value =
                right_hand_side.value(fe_values.quadrature_point(q));
              for (const unsigned int i : fe_values.dof_indices())
                {
                  for (const unsigned int j : fe_values.dof_indices())
                    cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                         fe_values.shape_grad(j, q) *
                                         fe_values.JxW(q);
                  cell_rhs(i) +=
                    rhs_value * fe_values.shape_value(i, q) * fe_values.JxW(q);
                }
            }

          local_dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }



  void
  LaplaceProblem::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    SolverControl solver_control(dof_handler.n_dofs(),
                                 1e-12 * system_rhs.l2_norm());
    LA::SolverCG  solver(solver_control);

    LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#endif
    LA::MPI::PreconditionAMG preconditioner;
    preconditioner.initialize(system_matrix, data);

    solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs,
                 preconditioner);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints.distribute(completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;
  }



  void
  LaplaceProblem::compute_errors(Vector<float> &cell_errors)
  {
    TimerOutput::Scope t(computing_timer, "errors");

    // a higher quadrature than for the assembly (the rules for tetrahedra
    // exist for at most 5 points)
    const hp::QCollection<dim> error_quadrature(
      QGauss<dim>(degree + 3),
      QGaussSimplex<dim>(std::min(degree + 3, 5u)),
      QGaussWedge<dim>(degree + 3),
      QGaussPyramid<dim>(degree + 3));

    Vector<float> l2_errors(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      locally_relevant_solution,
                                      ExactSolution(),
                                      l2_errors,
                                      error_quadrature,
                                      VectorTools::L2_norm);
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      locally_relevant_solution,
                                      ExactSolution(),
                                      cell_errors,
                                      error_quadrature,
                                      VectorTools::H1_seminorm);
    const double l2_error = VectorTools::compute_global_error(
      triangulation, l2_errors, VectorTools::L2_norm);
    const double h1_error = VectorTools::compute_global_error(
      triangulation, cell_errors, VectorTools::H1_seminorm);

    // the number of locally owned cells of each type
    std::array<unsigned int, 4> n_cells = {{0, 0, 0, 0}};
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        ++n_cells[fe_index(cell->reference_cell())];
    for (unsigned int &n : n_cells)
      n = Utilities::MPI::sum(n, mpi_communicator);

    pcout << "   cells: " << triangulation.n_global_active_cells() << " ("
          << n_cells[0] << " hexahedra, " << n_cells[1] << " tetrahedra, "
          << n_cells[2] << " wedges, " << n_cells[3] << " pyramids)"
          << std::endl
          << "   dofs: " << dof_handler.n_dofs() << std::endl
          << "   L2 error: " << l2_error << ", H1 error: " << h1_error
          << std::endl;
  }



  void
  LaplaceProblem::refine_grid(const Vector<float> &cell_errors)
  {
    TimerOutput::Scope t(computing_timer, "refine");

    if (global_refinement)
      for (const auto &cell : triangulation.active_cell_iterators())
        {
          if (cell->is_locally_owned())
            cell->set_refine_flag();
        }
    else if (polynomial_degree > 0)
      {
        // the errors are round-off for the polynomial solution: refine
        // around the peak instead, independently of the number of processes
        for (const auto &cell : triangulation.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              // an irregular pattern from the cell centers (with
              // HYBRID_IRREGULAR) or the cells around the peak
              const Point<dim> c = cell->center();
              const bool       irregular =
                std::getenv("HYBRID_IRREGULAR") != nullptr;
              const unsigned int hash =
                static_cast<unsigned int>(std::abs(
                  std::sin(12.9898 * c[0] + 78.233 * c[1] + 37.719 * c[2]) *
                  43758.5453 * 1000)) %
                5;
              if ((irregular && hash == 0) ||
                  (!irregular &&
                   c.distance(peak) < 0.4 / (1 + cell->level())))
                cell->set_refine_flag();
            }
      }
    else
      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        triangulation, cell_errors, 0.2, 0.03);
    triangulation.execute_coarsening_and_refinement();
  }



  void
  LaplaceProblem::output_results(const unsigned int cycle) const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "u");

    Vector<float> subdomain(triangulation.n_active_cells());
    Vector<float> cell_type(triangulation.n_active_cells());
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          subdomain(cell->active_cell_index()) = triangulation.locally_owned_subdomain();
          cell_type(cell->active_cell_index()) = fe_index(cell->reference_cell());
        }
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.add_data_vector(cell_type, "cell_type");

    data_out.build_patches(mapping);
    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2);
  }



  void
  LaplaceProblem::run()
  {
    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#else
          << "Trilinos"
#endif
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s), degree " << degree << ", "
          << (global_refinement ? "global" : "adaptive") << " refinement"
          << std::endl;

    make_hybrid_mesh(triangulation, 2);
    Vector<float> cell_errors;

    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;
        if (cycle > 0)
          refine_grid(cell_errors);

        setup_system();
        assemble_system();
        solve();

        cell_errors.reinit(triangulation.n_active_cells());
        compute_errors(cell_errors);
        output_results(cycle);

        computing_timer.print_summary();
        computing_timer.reset();
        pcout << std::endl;
      }
  }
} // namespace Step40Hybrid



int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step40Hybrid;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      const unsigned int degree   = argc > 1 ? std::stoi(argv[1]) : 2;
      const unsigned int n_cycles = argc > 2 ? std::stoi(argv[2]) : 6;
      const bool         global   = argc > 3 ? std::stoi(argv[3]) != 0 : false;

      LaplaceProblem laplace_problem(degree, n_cycles, global);
      laplace_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
