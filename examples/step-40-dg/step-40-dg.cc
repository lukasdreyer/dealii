/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2010 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Authors: Wolfgang Bangerth, Texas A&M University, 2009, 2010
 *          Timo Heister, University of Goettingen, 2009, 2010
 */


// @sect3{Include files}
//
// Most of the include files we need for this program have already been
// discussed in previous programs. In particular, all of the following should
// already be familiar friends:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

// This program can use either PETSc or Trilinos for its parallel
// algebra needs. By default, if deal.II has been configured with
// PETSc, it will use PETSc. Otherwise, the following few lines will
// check that deal.II has been configured with Trilinos and take that.
//
// But there may be cases where you want to use Trilinos, even though
// deal.II has *also* been configured with PETSc, for example to
// compare the performance of these two libraries. To do this,
// add the following \#define to the source code:
// @code
// #define FORCE_USE_OF_TRILINOS
// @endcode
//
// Using this logic, the following lines will then import either the
// PETSc or Trilinos wrappers into the namespace `LA` (for linear
// algebra). In the former case, we are also defining the macro
// `USE_PETSC_LA` so that we can detect if we are using PETSc (see
// solve() for an example where this is necessary).
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


#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>


// #include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>


#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>
// Here the discontinuous finite elements are defined. They are used in the same
// way as all other finite elements, though -- as you have seen in previous
// tutorial programs -- there isn't much user interaction with finite element
// classes at all: they are passed to <code>DoFHandler</code> and
// <code>FEValues</code> objects, and that is about it.
#include <deal.II/fe/fe_dgq.h>
// This header is needed for FEInterfaceValues to compute integrals on
// interfaces:
#include <deal.II/fe/fe_interface_values.h>
// We are going to use a standard solver, called Generalized minimal residual
// method (GMRES). It is an iterative solver which is applicable to arbitrary
// invertible matrices. This, in combination with a block SSOR preconditioner
// (defined in precondition_block.h), that uses the special block matrix
// structure of system matrices arising from DG discretizations.
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/trilinos_precondition.h>
// We are going to use gradients as refinement indicator.
#include <deal.II/numerics/derivative_approximation.h>

// Finally, the new include file for using the mesh_loop from the MeshWorker
// framework
#include <deal.II/meshworker/mesh_loop.h>


#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/error_estimator.h>

// The following, however, will be new or be used in new roles. Let's walk
// through them. The first of these will provide the tools of the
// Utilities::System namespace that we will use to query things like the
// number of processors associated with the current MPI universe, or the
// number within this universe the processor this job runs on is:
#include <deal.II/base/utilities.h>
// The next one provides a class, ConditionOStream that allows us to write
// code that would output things to a stream (such as <code>std::cout</code>)
// on every processor but throws the text away on all but one of them. We
// could achieve the same by simply putting an <code>if</code> statement in
// front of each place where we may generate output, but this doesn't make the
// code any prettier. In addition, the condition whether this processor should
// or should not produce output to the screen is the same every time -- and
// consequently it should be simple enough to put it into the statements that
// generate output itself.
#include <deal.II/base/conditional_ostream.h>
// After these preliminaries, here is where it becomes more interesting. As
// mentioned in the @ref distributed topic, one of the fundamental truths of
// solving problems on large numbers of processors is that there is no way for
// any processor to store everything (e.g. information about all cells in the
// mesh, all degrees of freedom, or the values of all elements of the solution
// vector). Rather, every processor will <i>own</i> a few of each of these
// and, if necessary, may <i>know</i> about a few more, for example the ones
// that are located on cells adjacent to the ones this processor owns
// itself. We typically call the latter <i>ghost cells</i>, <i>ghost nodes</i>
// or <i>ghost elements of a vector</i>. The point of this discussion here is
// that we need to have a way to indicate which elements a particular
// processor owns or need to know of. This is the realm of the IndexSet class:
// if there are a total of $N$ cells, degrees of freedom, or vector elements,
// associated with (non-negative) integral indices $[0,N)$, then both the set
// of elements the current processor owns as well as the (possibly larger) set
// of indices it needs to know about are subsets of the set $[0,N)$. IndexSet
// is a class that stores subsets of this set in an efficient format:
#include <deal.II/base/index_set.h>
// The next header file is necessary for a single function,
// SparsityTools::distribute_sparsity_pattern. The role of this function will
// be explained below.
#include <deal.II/lac/sparsity_tools.h>
// The final two, new header files provide the class
// parallel::distributed::Triangulation that provides meshes distributed
// across a potentially very large number of processors, while the second
// provides the namespace parallel::distributed::GridRefinement that offers
// functions that can adaptively refine such distributed meshes:
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <fstream>
#include <iostream>

namespace Step40DG
{
  using namespace dealii;


  // @sect3{Equation data}
  //
  // First, we define a class describing the inhomogeneous boundary data. Since
  // only its values are used, we implement value_list(), but leave all other
  // functions of Function undefined.
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues() = default;
    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<double>           &values,
                            const unsigned int component = 0) const override;
  };

  // Given the flow direction, the inflow boundary of the unit square $[0,1]^2$
  // are the right and the lower boundaries. We prescribe discontinuous boundary
  // values 1 and 0 on the x-axis and value 0 on the right boundary. The values
  // of this function on the outflow boundaries will not be used within the DG
  // scheme.
  template <int dim>
  void BoundaryValues<dim>::value_list(const std::vector<Point<dim>> &points,
                                       std::vector<double>           &values,
                                       const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    AssertDimension(values.size(), points.size());

    for (unsigned int i = 0; i < values.size(); ++i)
      {
        if (points[i][0] < 1. / 3 ||
            (points[i][0] > 2. / 3 && points[i][1] < 1. / 3) ||
            (points[i][1] > 2. / 3))
          values[i] = 1.;
        else
          values[i] = 0.;
      }
  }


  // Finally, a function that computes and returns the wind field
  // $\beta=\beta(\mathbf x)$. As explained in the introduction, we will use a
  // rotational field around the origin in 2d. In 3d, we simply leave the
  // $z$-component unset (i.e., at zero), whereas the function can not be used
  // in 1d in its current implementation:
  template <int dim>
  Tensor<1, dim> beta(const Point<dim> &p)
  {
    Assert(dim >= 2, ExcNotImplemented());

    Tensor<1, dim> wind_field;
    wind_field[0] = -p[1];
    wind_field[1] = p[0];

    if (wind_field.norm() > 1e-10)
      wind_field /= wind_field.norm();

    return wind_field;
  }


  // @sect3{The ScratchData and CopyData classes}
  //
  // The following objects are the scratch and copy objects we use in the call
  // to MeshWorker::mesh_loop(). The new object is the FEInterfaceValues object,
  // that works similar to FEValues or FEFaceValues, except that it acts on
  // an interface between two cells and allows us to assemble the interface
  // terms in our weak form.

  template <int dim>
  struct ScratchData
  {
    ScratchData(const Mapping<dim>        &mapping,
                const FiniteElement<dim>  &fe,
                const Quadrature<dim>     &quadrature,
                const Quadrature<dim - 1> &quadrature_face,
                const UpdateFlags          update_flags = update_values |
                                                 update_gradients |
                                                 update_quadrature_points |
                                                 update_JxW_values,
                const UpdateFlags interface_update_flags =
                  update_values | update_gradients | update_quadrature_points |
                  update_JxW_values | update_normal_vectors)
      : fe_values(mapping, fe, quadrature, update_flags)
      , fe_interface_values(mapping,
                            fe,
                            quadrature_face,
                            interface_update_flags)
    {}


    ScratchData(const ScratchData<dim> &scratch_data)
      : fe_values(scratch_data.fe_values.get_mapping(),
                  scratch_data.fe_values.get_fe(),
                  scratch_data.fe_values.get_quadrature(),
                  scratch_data.fe_values.get_update_flags())
      , fe_interface_values(scratch_data.fe_interface_values.get_mapping(),
                            scratch_data.fe_interface_values.get_fe(),
                            scratch_data.fe_interface_values.get_quadrature(),
                            scratch_data.fe_interface_values.get_update_flags())
    {}

    FEValues<dim>          fe_values;
    FEInterfaceValues<dim> fe_interface_values;
  };



  struct CopyDataFace
  {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
  };



  struct CopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace>            face_data;

    template <class Iterator>
    void reinit(const Iterator &cell, unsigned int dofs_per_cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);

      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
    }
  };

  // @sect3{The <code>equilibrium Advection</code> class template}

  // Next let's declare the main class of this program. Its structure is
  // almost exactly that of the step-6 tutorial program. The only significant
  // differences are:
  // - The <code>mpi_communicator</code> variable that
  //   describes the set of processors we want this code to run on. In practice,
  //   this will be MPI_COMM_WORLD, i.e. all processors the batch scheduling
  //   system has assigned to this particular job.
  // - The presence of the <code>pcout</code> variable of type ConditionOStream.
  // - The obvious use of parallel::distributed::Triangulation instead of
  // Triangulation.
  // - The presence of two IndexSet objects that denote which sets of degrees of
  //   freedom (and associated elements of solution and right hand side vectors)
  //   we own on the current processor and which we need (as ghost elements) for
  //   the algorithms in this program to work.
  // - The fact that all matrices and vectors are now distributed. We use
  //   either the PETSc or Trilinos wrapper classes so that we can use one of
  //   the sophisticated preconditioners offered by Hypre (with PETSc) or ML
  //   (with Trilinos). Note that as part of this class, we store a solution
  //   vector that does not only contain the degrees of freedom the current
  //   processor owns, but also (as ghost elements) all those vector elements
  //   that correspond to "locally relevant" degrees of freedom (i.e. all
  //   those that live on locally owned cells or the layer of ghost cells that
  //   surround it).
  template <int dim>
  class AdvectionProblem
  {
  public:
    AdvectionProblem();

    void run();

  private:
    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle);

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;
    const MappingQ1<dim>                      mapping;

    const FE_DGQ<dim>         fe;
    DoFHandler<dim>           dof_handler;
    AffineConstraints<double> constraints;

    const QGauss<dim>     quadrature;
    const QGauss<dim - 1> quadrature_face;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    LA::MPI::SparseMatrix system_matrix;
    LA::MPI::Vector       locally_relevant_solution;
    LA::MPI::Vector       system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };


  // @sect3{The <code>AdvectionProblem</code> class implementation}

  // @sect4{Constructor}

  // Constructors and destructors are rather trivial. In addition to what we
  // do in step-6, we set the set of processors we want to work on to all
  // machines available (MPI_COMM_WORLD); ask the triangulation to ensure that
  // the mesh remains smooth and free to refined islands, for example; and
  // initialize the <code>pcout</code> variable to only allow processor zero
  // to output anything. The final piece is to initialize a timer that we
  // use to determine how much compute time the different parts of the program
  // take:
  template <int dim>
  AdvectionProblem<dim>::AdvectionProblem()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , mapping()
    , fe(3)
    , dof_handler(triangulation)
    , quadrature(fe.tensor_degree() + 1)
    , quadrature_face(fe.tensor_degree() + 1)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
  {}



  // @sect4{AdvectionProblem::setup_system}

  // The following function is, arguably, the most interesting one in the
  // entire program since it goes to the heart of what distinguishes %parallel
  // step-40 from sequential step-6.
  //
  // At the top we do what we always do: tell the DoFHandler object to
  // distribute degrees of freedom. Since the triangulation we use here is
  // distributed, the DoFHandler object is smart enough to recognize that on
  // each processor it can only distribute degrees of freedom on cells it
  // owns; this is followed by an exchange step in which processors tell each
  // other about degrees of freedom on ghost cell. The result is a DoFHandler
  // that knows about the degrees of freedom on locally owned cells and ghost
  // cells (i.e. cells adjacent to locally owned cells) but nothing about
  // cells that are further away, consistent with the basic philosophy of
  // distributed computing that no processor can know everything.
  template <int dim>
  void AdvectionProblem<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    std::cout<<"n_levels"<<triangulation.n_levels()<<std::endl;
    std::cout<<"n_active"<<triangulation.n_active_cells()<<std::endl;
    std::cout<<"n_global_active_cells"<<triangulation.n_global_active_cells()<<std::endl;


    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(fe);

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;


    // The next two lines extract some information we will need later on,
    // namely two index sets that provide information about which degrees of
    // freedom are owned by the current processor (this information will be
    // used to initialize solution and right hand side vectors, and the system
    // matrix, indicating which elements to store on the current processor and
    // which to expect to be stored somewhere else); and an index set that
    // indicates which degrees of freedom are locally relevant (i.e. live on
    // cells that the current processor owns or on the layer of ghost cells
    // around the locally owned cells; we need all of these degrees of
    // freedom, for example, to estimate the error on the local cells).
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    // Next, let us initialize the solution and right hand side vectors. As
    // mentioned above, the solution vector we seek does not only store
    // elements we own, but also ghost entries; on the other hand, the right
    // hand side vector only needs to have the entries the current processor
    // owns since all we will ever do is write into it, never read from it on
    // locally owned cells (of course the linear solvers will read from it,
    // but they do not care about the geometric location of degrees of
    // freedom).
    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    constraints.close();

    // The last part of this function deals with initializing the matrix with
    // accompanying sparsity pattern. As in previous tutorial programs, we use
    // the DynamicSparsityPattern as an intermediate with which we
    // then initialize the system matrix. To do so, we have to tell the sparsity
    // pattern its size, but as above, there is no way the resulting object will
    // be able to store even a single pointer for each global degree of
    // freedom; the best we can hope for is that it stores information about
    // each locally relevant degree of freedom, i.e., all those that we may
    // ever touch in the process of assembling the matrix (the
    // @ref distributed_paper "distributed computing paper" has a long
    // discussion why one really needs the locally relevant, and not the small
    // set of locally active degrees of freedom in this context).
    //
    // So we tell the sparsity pattern its size and what DoFs to store
    // anything for and then ask DoFTools::make_sparsity_pattern to fill it
    // (this function ignores all cells that are not locally owned, mimicking
    // what we will do below in the assembly process). After this, we call a
    // function that exchanges entries in these sparsity pattern between
    // processors so that in the end each processor really knows about all the
    // entries that will exist in that part of the finite element matrix that
    // it will own. The final step is to initialize the matrix with the
    // sparsity pattern.
    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
  }



  // @sect4{AdvectionProblem::assemble_system}

  // The function that then assembles the linear system is comparatively
  // boring, being almost exactly what we've seen before. The points to watch
  // out for are:
  // - Assembly must only loop over locally owned cells. There
  //   are multiple ways to test that; for example, we could compare a cell's
  //   subdomain_id against information from the triangulation as in
  //   <code>cell->subdomain_id() ==
  //   triangulation.locally_owned_subdomain()</code>, or skip all cells for
  //   which the condition <code>cell->is_ghost() ||
  //   cell->is_artificial()</code> is true. The simplest way, however, is to
  //   simply ask the cell whether it is owned by the local processor.
  // - Copying local contributions into the global matrix must include
  //   distributing constraints and boundary values not just from the local
  //   matrix and vector into the global ones, but in the process
  //   also -- possibly -- from one MPI process to other processes if the
  //   entries we want to write to are not stored on the current process.
  //   Interestingly, this requires essentially no additional work: The
  //   AffineConstraints class we already used in step-6 is perfectly
  //   capable to also do this in parallel, and the only difference in this
  //   regard is that at the very end of the function, we have to call a
  //   `compress()` function on the global matrix and right hand side vector
  //   objects (see the description of what this does just before these calls).
  // - The way we compute the right hand side (given the
  //   formula stated in the introduction) may not be the most elegant but will
  //   do for a program whose focus lies somewhere entirely different.
  template <int dim>
  void AdvectionProblem<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
    const BoundaryValues<dim> boundary_function;

    // This is the function that will be executed for each cell.
    const auto cell_worker = [&](const Iterator   &cell,
                                 ScratchData<dim> &scratch_data,
                                 CopyData         &copy_data) {
      const unsigned int n_dofs =
        scratch_data.fe_values.get_fe().n_dofs_per_cell();
      copy_data.reinit(cell, n_dofs);
      scratch_data.fe_values.reinit(cell);

      const auto &q_points = scratch_data.fe_values.get_quadrature_points();

      const FEValues<dim>       &fe_v = scratch_data.fe_values;
      const std::vector<double> &JxW  = fe_v.get_JxW_values();

      // We solve a homogeneous equation, thus no right hand side shows up in
      // the cell term.  What's left is integrating the matrix entries.
      for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
        {
          auto beta_q = beta(q_points[point]);
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              {
                copy_data.cell_matrix(i, j) +=
                  -beta_q                      // -\beta
                  * fe_v.shape_grad(i, point)  // \nabla \phi_i
                  * fe_v.shape_value(j, point) // \phi_j
                  * JxW[point];                // dx
              }
        }
    };

    // This is the function called for boundary faces and consists of a normal
    // integration using FEFaceValues. New is the logic to decide if the term
    // goes into the system matrix (outflow) or the right-hand side (inflow).
    const auto boundary_worker = [&](const Iterator     &cell,
                                     const unsigned int &face_no,
                                     ScratchData<dim>   &scratch_data,
                                     CopyData           &copy_data) {
      scratch_data.fe_interface_values.reinit(cell, face_no);
      const FEFaceValuesBase<dim> &fe_face =
        scratch_data.fe_interface_values.get_fe_face_values(0);

      const auto &q_points = fe_face.get_quadrature_points();

      const unsigned int n_facet_dofs = fe_face.get_fe().n_dofs_per_cell();
      const std::vector<double>         &JxW     = fe_face.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals = fe_face.get_normal_vectors();

      std::vector<double> g(q_points.size());
      boundary_function.value_list(q_points, g);

      for (unsigned int point = 0; point < q_points.size(); ++point)
        {
          const double beta_dot_n = beta(q_points[point]) * normals[point];

          if (beta_dot_n > 0)
            {
              for (unsigned int i = 0; i < n_facet_dofs; ++i)
                for (unsigned int j = 0; j < n_facet_dofs; ++j)
                  copy_data.cell_matrix(i, j) +=
                    fe_face.shape_value(i, point)   // \phi_i
                    * fe_face.shape_value(j, point) // \phi_j
                    * beta_dot_n                    // \beta . n
                    * JxW[point];                   // dx
            }
          else
            for (unsigned int i = 0; i < n_facet_dofs; ++i)
              copy_data.cell_rhs(i) += -fe_face.shape_value(i, point) // \phi_i
                                       * g[point]                     // g
                                       * beta_dot_n  // \beta . n
                                       * JxW[point]; // dx
        }
    };

    // This is the function called on interior faces. The arguments specify
    // cells, face and subface indices (for adaptive refinement). We just pass
    // them along to the reinit() function of FEInterfaceValues.
    const auto face_worker = [&](const Iterator     &cell,
                                 const unsigned int &f,
                                 const unsigned int &sf,
                                 const Iterator     &ncell,
                                 const unsigned int &nf,
                                 const unsigned int &nsf,
                                 ScratchData<dim>   &scratch_data,
                                 CopyData           &copy_data) {
      FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values;
      fe_iv.reinit(cell, f, sf, ncell, nf, nsf);
      const auto &q_points = fe_iv.get_quadrature_points();

      copy_data.face_data.emplace_back();
      CopyDataFace &copy_data_face = copy_data.face_data.back();

      const unsigned int n_dofs        = fe_iv.n_current_interface_dofs();
      copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();

      copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

      const std::vector<double>         &JxW     = fe_iv.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();

      for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
        {
          const double beta_dot_n = beta(q_points[qpoint]) * normals[qpoint];
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              copy_data_face.cell_matrix(i, j) +=
                fe_iv.jump_in_shape_values(i, qpoint) // [\phi_i]
                *
                fe_iv.shape_value((beta_dot_n > 0), j, qpoint) // phi_j^{upwind}
                * beta_dot_n                                   // (\beta . n)
                * JxW[qpoint];                                 // dx
        }
    };

    // The following lambda function will handle copying the data from the
    // cell and face assembly into the global matrix and right-hand side.
    //
    // While we would not need an AffineConstraints object, because there are
    // no hanging node constraints in DG discretizations, we use an empty
    // object here as this allows us to use its `copy_local_to_global`
    // functionality.


    const auto copier = [&](const CopyData &c) {
      constraints.distribute_local_to_global(c.cell_matrix,
                                             c.cell_rhs,
                                             c.local_dof_indices,
                                             system_matrix,
                                             system_rhs);

      for (const auto &cdf : c.face_data)
        {
          constraints.distribute_local_to_global(cdf.cell_matrix,
                                                 cdf.joint_dof_indices,
                                                 system_matrix);
        }
    };

    ScratchData<dim> scratch_data(mapping, fe, quadrature, quadrature_face);
    CopyData         copy_data;

    // Here, we finally handle the assembly. We pass in ScratchData and
    // CopyData objects, the lambda functions from above, an specify that we
    // want to assemble interior faces once.
    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
                            MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_ghost_faces_once |
                            MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);

    // In the operations above, specifically the call to
    // `distribute_local_to_global()` in the last line, every MPI
    // process was only working on its local data. If the operation
    // required adding something to a matrix or vector entry that is
    // not actually stored on the current process, then the matrix or
    // vector object keeps track of this for a later data exchange,
    // but for efficiency reasons, this part of the operation is only
    // queued up, rather than executed right away. But now that we got
    // here, it is time to send these queued-up additions to those
    // processes that actually own these matrix or vector entries. In
    // other words, we want to "finalize" the global data
    // structures. This is done by invoking the function `compress()`
    // on both the matrix and vector objects. See
    // @ref GlossCompress "Compressing distributed objects"
    // for more information on what `compress()` actually does.
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }



  // @sect4{AdvectionProblem::solve}

  // Even though solving linear systems on potentially tens of thousands of
  // processors is by far not a trivial job, the function that does this is --
  // at least at the outside -- relatively simple. Most of the parts you've
  // seen before. There are really only two things worth mentioning:
  // - Solvers and preconditioners are built on the deal.II wrappers of PETSc
  //   and Trilinos functionality. It is relatively well known that the
  //   primary bottleneck of massively %parallel linear solvers is not
  //   actually the communication between processors, but the fact that it is
  //   difficult to produce preconditioners that scale well to large numbers
  //   of processors. Over the second half of the first decade of the 21st
  //   century, it has become clear that algebraic multigrid (AMG) methods
  //   turn out to be extremely efficient in this context, and we will use one
  //   of them -- either the BoomerAMG implementation of the Hypre package
  //   that can be interfaced to through PETSc, or a preconditioner provided
  //   by ML, which is part of Trilinos -- for the current program. The rest
  //   of the solver itself is boilerplate and has been shown before. Since
  //   the linear system is symmetric and positive definite, we can use the CG
  //   method as the outer solver.
  // - Ultimately, we want a vector that stores not only the elements
  //   of the solution for degrees of freedom the current processor owns, but
  //   also all other locally relevant degrees of freedom. On the other hand,
  //   the solver itself needs a vector that is uniquely split between
  //   processors, without any overlap. We therefore create a vector at the
  //   beginning of this function that has these properties, use it to solve the
  //   linear system, and only assign it to the vector we want at the very
  //   end. This last step ensures that all ghost elements are also copied as
  //   necessary.
  template <int dim>
  void AdvectionProblem<dim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    SolverControl solver_control(dof_handler.n_dofs(),
                                 1e-6 * system_rhs.l2_norm());


    LA::SolverGMRES::AdditionalData additional_data;
    //    additional_data.max_basis_size = 100;
    LA::SolverGMRES solver(solver_control, additional_data);
    // LA::MPI::PreconditionBlockSSOR preconditioner;
    TrilinosWrappers::PreconditionBlockSSOR preconditioner;

    preconditioner.initialize(system_matrix, fe.n_dofs_per_cell());
    solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs,
                 preconditioner);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints.distribute(completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;
  }



  // @sect4{AdvectionProblem::refine_grid}

  // The function that estimates the error and refines the grid is again
  // almost exactly like the one in step-6. The only difference is that the
  // function that flags cells to be refined is now in namespace
  // parallel::distributed::GridRefinement -- a namespace that has functions
  // that can communicate between all involved processors and determine global
  // thresholds to use in deciding which cells to refine and which to coarsen.
  //
  // Note that we didn't have to do anything special about the
  // KellyErrorEstimator class: we just give it a vector with as many elements
  // as the local triangulation has cells (locally owned cells, ghost cells,
  // and artificial ones), but it only fills those entries that correspond to
  // cells that are locally owned.
  template <int dim>
  void AdvectionProblem<dim>::refine_grid()
  {
    // The <code>DerivativeApproximation</code> class computes the gradients to
    // float precision. This is sufficient as they are approximate and serve as
    // refinement indicators only.
    Vector<float> gradient_indicator(triangulation.n_active_cells());

    // Now the approximate gradients are computed
    DerivativeApproximation::approximate_gradient(mapping,
                                                  dof_handler,
                                                  locally_relevant_solution,
                                                  gradient_indicator);

    // and they are cell-wise scaled by the factor $h^{1+d/2}$
    unsigned int cell_no = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      gradient_indicator(cell_no++) *=
        std::pow(cell->diameter(), 1 + 1.0 * dim / 2);

    // Finally they serve as refinement indicator.
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      triangulation, gradient_indicator, 0.3, 0.1);

    triangulation.execute_coarsening_and_refinement();
  }



  // @sect4{AdvectionProblem::output_results}

  // Compared to the corresponding function in step-6, the one here is
  // a tad more complicated. There are two reasons: the first one is
  // that we do not just want to output the solution but also for each
  // cell which processor owns it (i.e. which "subdomain" it is
  // in). Secondly, as discussed at length in step-17 and step-18,
  // generating graphical data can be a bottleneck in
  // parallelizing. In those two programs, we simply generate one
  // output file per process. That worked because the
  // parallel::shared::Triangulation cannot be used with large numbers
  // of MPI processes anyway.  But this doesn't scale: Creating a
  // single file per processor will overwhelm the filesystem with a
  // large number of processors.
  //
  // We here follow a more sophisticated approach that uses
  // high-performance, parallel IO routines using MPI I/O to write to
  // a small, fixed number of visualization files (here 8). We also
  // generate a .pvtu record referencing these .vtu files, which can
  // be opened directly in visualizatin tools like ParaView and VisIt.
  //
  // To start, the top of the function looks like it usually does. In addition
  // to attaching the solution vector (the one that has entries for all locally
  // relevant, not only the locally owned, elements), we attach a data vector
  // that stores, for each cell, the subdomain the cell belongs to. This is
  // slightly tricky, because of course not every processor knows about every
  // cell. The vector we attach therefore has an entry for every cell that the
  // current processor has in its mesh (locally owned ones, ghost cells, and
  // artificial cells), but the DataOut class will ignore all entries that
  // correspond to cells that are not owned by the current processor. As a
  // consequence, it doesn't actually matter what values we write into these
  // vector entries: we simply fill the entire vector with the number of the
  // current MPI process (i.e. the subdomain_id of the current process); this
  // correctly sets the values we care for, i.e. the entries that correspond
  // to locally owned cells, while providing the wrong value for all other
  // elements -- but these are then ignored anyway.
  template <int dim>
  void AdvectionProblem<dim>::output_results(const unsigned int cycle)
  {
    TimerOutput::Scope t(computing_timer, "output");

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution,
                             "u",
                             DataOut<dim>::type_dof_data);

    data_out.build_patches(mapping);

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2, 8);

    {
      Vector<float> values(triangulation.n_active_cells());
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        locally_relevant_solution,
                                        Functions::ZeroFunction<dim>(),
                                        values,
                                        quadrature,
                                        VectorTools::Linfty_norm);
      const double l_infty =
        VectorTools::compute_global_error(triangulation,
                                          values,
                                          VectorTools::Linfty_norm);
      std::cout << "  L-infinity norm: " << l_infty << std::endl;
    }
  }



  // @sect4{AdvectionProblem::run}

  // The function that controls the overall behavior of the program is again
  // like the one in step-6. The minor difference are the use of
  // <code>pcout</code> instead of <code>std::cout</code> for output to the
  // console (see also step-17).
  //
  // A functional difference to step-6 is the use of a square domain and that
  // we start with a slightly finer mesh (5 global refinement cycles) -- there
  // just isn't much of a point showing a massively %parallel program starting
  // on 4 cells (although admittedly the point is only slightly stronger
  // starting on 1024).
  template <int dim>
  void AdvectionProblem<dim>::run()
  {
    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#else
          << "Trilinos"
#endif
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    const unsigned int n_cycles = 1;
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation);
//            triangulation.refine_global(3);
          }
        else
//          refine_grid();

        pcout << "  Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;

        setup_system();

        pcout << "  Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

        assemble_system();
        solve();

        output_results(cycle);
      }
  }
} // namespace Step40DG



// @sect4{main()}

// The final function, <code>main()</code>, again has the same structure as in
// all other programs, in particular step-6. Like the other programs that use
// MPI, we have to initialize and finalize MPI, which is done using the helper
// object Utilities::MPI::MPI_InitFinalize. The constructor of that class also
// initializes libraries that depend on MPI, such as p4est, PETSc, SLEPc, and
// Zoltan (though the last two are not used in this tutorial). The order here
// is important: we cannot use any of these libraries until they are
// initialized, so it does not make sense to do anything before creating an
// instance of Utilities::MPI::MPI_InitFinalize.
//
// After the solver finishes, the AdvectionProblem destructor will run followed
// by Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize(). This order is
// also important: Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize() calls
// <code>PetscFinalize</code> (and finalization functions for other
// libraries), which will delete any in-use PETSc objects. This must be done
// after we destruct the Laplace solver to avoid double deletion
// errors. Fortunately, due to the order of destructor call rules of C++, we
// do not need to worry about any of this: everything happens in the correct
// order (i.e., the reverse of the order of construction). The last function
// called by Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize() is
// <code>MPI_Finalize</code>: i.e., once this object is destructed the program
// should exit since MPI will no longer be available.
int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step40DG;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      AdvectionProblem<2> advection_problem_2d;
      advection_problem_2d.run();
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
