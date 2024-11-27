// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2010 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


#include <deal.II/base/logstream.h>
#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <t8_data/t8_element_array_iterator.hxx>
#include <t8_cmesh/t8_cmesh_types.h>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_schemes/t8_default/t8_default.hxx>

#include <deal.II/distributed/t8code_wrappers.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>


DEAL_II_NAMESPACE_OPEN


namespace internal
{
  namespace parallel
  {
    namespace distributed
    {
      namespace TriangulationImplementation
      {
        /**
         * Communicate refinement flags on ghost cells from the owner of the
         * cell.
         *
         * This is necessary to get consistent refinement, as mesh smoothing
         * would undo some of the requested coarsening/refinement.
         */
        template <int dim, int spacedim>
        void
        exchange_refinement_flags(
          dealii::parallel::distributed::Triangulation<dim, spacedim> &tria)
        {
          auto pack =
            [](const typename Triangulation<dim, spacedim>::active_cell_iterator
                 &cell) -> std::uint8_t {
            if (cell->refine_flag_set())
              return 1;
            if (cell->coarsen_flag_set())
              return 2;
            return 0;
          };

          auto unpack =
            [](const typename Triangulation<dim, spacedim>::active_cell_iterator
                                  &cell,
               const std::uint8_t &flag) -> void {
            cell->clear_coarsen_flag();
            cell->clear_refine_flag();
            if (flag == 1)
              cell->set_refine_flag();
            else if (flag == 2)
              cell->set_coarsen_flag();
          };

          GridTools::exchange_cell_data_to_ghosts<std::uint8_t>(tria,
                                                                pack,
                                                                unpack);
        }
      } // namespace TriangulationImplementation
    }   // namespace distributed
  }     // namespace parallel
} // namespace internal



#ifdef DEAL_II_WITH_T8CODE

namespace parallel
{
  namespace distributed
  {
  template <int dim, int spacedim>
  void
  delete_all_children_and_self(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell)
  {
    if (cell->has_children())
      for (unsigned int c = 0; c < cell->n_children(); ++c)
        delete_all_children_and_self<dim, spacedim>(cell->child(c));
    else
      cell->set_coarsen_flag();
  }



  template <int dim, int spacedim>
  void
  delete_all_children(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell)
  {
    if (cell->has_children())
      for (unsigned int c = 0; c < cell->n_children(); ++c)
        delete_all_children_and_self<dim, spacedim>(cell->child(c));
  }

  template <int dim, int spacedim>
  DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
  typename dealii::internal::t8code::types::tree
    *Triangulation<dim, spacedim>::init_tree(
      const int dealii_coarse_cell_index) const
  {
    const unsigned int tree_index =
      coarse_cell_to_t8code_tree_permutation[dealii_coarse_cell_index];
    typename dealii::internal::t8code::types::tree *tree =
      static_cast<typename dealii::internal::t8code::types::tree *>(
        sc_array_index(parallel_forest->trees, tree_index));

    return tree;
  }

  template <int dim, int spacedim>
  bool
  tree_exists_locally(
    const typename dealii::internal::t8code::types::forest parallel_forest,
    const typename dealii::internal::t8code::types::gloidx  coarse_grid_cell)
  {
    Assert(coarse_grid_cell < parallel_forest->cmesh->num_trees,
           ExcInternalError());
    return ((coarse_grid_cell >= parallel_forest->first_local_tree) &&
            (coarse_grid_cell <= parallel_forest->last_local_tree));
  }


  template <int dim, int spacedim>
  void
  match_tree_recursively(
    const typename dealii::internal::t8code::types::tree           tree,
    const typename Triangulation<dim, spacedim>::cell_iterator &dealii_cell,
    const typename dealii::internal::t8code::types::element*       t8code_cell,
    const typename dealii::internal::t8code::types::forest         forest,
    const types::subdomain_id                                   my_subdomain)
  {
    dealii::internal::t8code::types::eclass tree_class = T8_ECLASS_QUAD;
    dealii::internal::t8code::types::eclass_scheme *eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class); 
    auto compare_lambda = [eclass_scheme](auto x, auto y){
        return eclass_scheme->t8_element_compare(x,y) < 0;
    };

    if (std::binary_search(t8_element_array_begin(&tree.elements),
                           t8_element_array_end(&tree.elements),
                           t8code_cell,
                           compare_lambda))
      {
        // yes, cell found in local part of p4est
        delete_all_children<dim, spacedim>(dealii_cell);
        if (dealii_cell->is_active())
          dealii_cell->set_subdomain_id(my_subdomain);
      }
    else
      {
        // no, cell not found in local part of p4est. this means that the
        // local part is more refined than the current cell. if this cell has
        // no children of its own, we need to refine it, and if it does
        // already have children then loop over all children and see if they
        // are locally available as well
        if (dealii_cell->is_active())
          dealii_cell->set_refine_flag();
        else
          {
            typename dealii::internal::t8code::types::element*
              t8code_child[GeometryInfo<dim>::max_children_per_cell];

            // TODO: remove, replcae by t8_element_nwe(length);
            for (unsigned int c = 0;
                 c < GeometryInfo<dim>::max_children_per_cell;
                 ++c)
              dealii::internal::t8code::element_new(forest, T8_ECLASS_QUAD, t8code_child + c);

            dealii::internal::t8code::element_children(forest, T8_ECLASS_QUAD, t8code_cell, t8code_child);

            for (unsigned int c = 0;
                 c < GeometryInfo<dim>::max_children_per_cell;
                 ++c)
              if (dealii::internal::t8code::element_overlaps_tree(
                    forest, tree, t8code_child[c]) == false) 
                {
                  // no, this child is locally not available in the p4est.
                  // delete all its children but, because this may not be
                  // successful, make sure to mark all children recursively
                  // as not local.
                  delete_all_children<dim, spacedim>(dealii_cell->child(c));
                  dealii_cell->child(c)->recursively_set_subdomain_id(
                    numbers::artificial_subdomain_id);
                }
              else
                {
                  // at least some part of the tree rooted in this child is
                  // locally available
                  match_tree_recursively<dim, spacedim>(tree,
                                                        dealii_cell->child(c),
                                                        t8code_child[c],
                                                        forest,
                                                        my_subdomain);
                }
          }
      }
  }



      template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    void Triangulation<dim,
                       spacedim>::setup_coarse_cell_to_t8code_tree_permutation()
    {
      DynamicSparsityPattern cell_connectivity;
      dealii::GridTools::get_vertex_connectivity_of_cells(*this,
                                                          cell_connectivity);
      coarse_cell_to_t8code_tree_permutation.resize(this->n_cells(0));
      SparsityTools::reorder_hierarchical(
        cell_connectivity, coarse_cell_to_t8code_tree_permutation);

      t8code_tree_to_coarse_cell_permutation =
        Utilities::invert_permutation(coarse_cell_to_t8code_tree_permutation);
    }

    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    bool Triangulation<dim, spacedim>::prepare_coarsening_and_refinement()
    {
      // First exchange coarsen/refinement flags on ghost cells. After this
      // collective communication call all flags on ghost cells match the
      // flags set by the user on the owning rank.
      dealii::internal::parallel::distributed::TriangulationImplementation::
        exchange_refinement_flags(*this);

      // Now we can call the sequential version to apply mesh smoothing and
      // other modifications:
      const bool any_changes = this->dealii::Triangulation<dim, spacedim>::
                                 prepare_coarsening_and_refinement();
      return any_changes;
    }


    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    void Triangulation<dim, spacedim>::copy_new_triangulation_to_t8code()
    {
      cmesh = t8_cmesh_new_hypercube(T8_ECLASS_QUAD, this->mpi_communicator, 0, 0, 0);
      scheme_collection = t8_scheme_new_default_cxx();
      parallel_forest = t8_forest_new_uniform(cmesh, scheme_collection, 0, 0, this->mpi_communicator);
    }

    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    void Triangulation<dim, spacedim>::create_triangulation(
      const std::vector<Point<spacedim>> &vertices,
      const std::vector<CellData<dim>>   &cells,
      const SubCellData                  &subcelldata)
    {
      dealii::Triangulation<dim, spacedim>::create_triangulation(
            vertices, cells, subcelldata);
      setup_coarse_cell_to_t8code_tree_permutation();
      copy_new_triangulation_to_t8code();
      copy_local_forest_to_triangulation();
    }

    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    void Triangulation<dim, spacedim>::copy_local_forest_to_triangulation()
    {
      bool mesh_changed = false;

      // Remove all deal.II refinements. Note that we could skip this and
      // start from our current state, because the algorithm later coarsens as
      // necessary. This has the advantage of being faster when large parts
      // of the local partition changes (likely) and gives a deterministic
      // ordering of the cells (useful for snapshot/resume).
      // TODO: is there a more efficient way to do this?
      if (settings & mesh_reconstruction_after_repartitioning)
        while (this->n_levels() > 1)
          {
            // Instead of marking all active cells, we slice off the finest
            // level, one level at a time. This takes the same number of
            // iterations but solves an issue where not all cells on a
            // periodic boundary are indeed coarsened and we run into an
            // irrelevant Assert() in update_periodic_face_map().
            for (const auto &cell :
                 this->active_cell_iterators_on_level(this->n_levels() - 1))
              {
                cell->set_coarsen_flag();
              }
            try
              {
                dealii::Triangulation<dim, spacedim>::
                  execute_coarsening_and_refinement();
              }
            catch (
              const typename Triangulation<dim, spacedim>::DistortedCellList &)
              {
                // the underlying triangulation should not be checking for
                // distorted cells
                DEAL_II_ASSERT_UNREACHABLE();
              }
          }

#if 0
      // query t8code for the ghost cells
      if (parallel_ghost != nullptr)
        {
          dealii::internal::t8code::functions::ghost_destroy(
            parallel_ghost);
          parallel_ghost = nullptr;
        }
      parallel_ghost = dealii::internal::t8code::functions<dim>::ghost_new(
        parallel_forest,
        (dim == 2 ? typename dealii::internal::t8code::types::balance_type(
                      P4EST_CONNECT_CORNER) :
                    typename dealii::internal::t8code::types::balance_type(
                      P8EST_CONNECT_CORNER)));

      Assert(parallel_ghost, ExcInternalError());

#endif

      // set all cells to artificial. we will later set it to the correct
      // subdomain in match_tree_recursively
      for (const auto &cell : this->cell_iterators_on_level(0))
        cell->recursively_set_subdomain_id(numbers::artificial_subdomain_id);

      do
        {
          for (const auto &cell : this->cell_iterators_on_level(0))
            {
              // if this processor stores no part of the forest that comes out
              // of this coarse grid cell, then we need to delete all children
              // of this cell (the coarse grid cell remains)
              if (tree_exists_locally<dim, spacedim>(
                    parallel_forest,
                    coarse_cell_to_t8code_tree_permutation[cell->index()]) ==
                  false)
                {
                  delete_all_children<dim, spacedim>(cell);
                  if (cell->is_active())
                    cell->set_subdomain_id(numbers::artificial_subdomain_id);
                }

              else
                {
                  // this processor stores at least a part of the tree that
                  // comes out of this cell.

                  typename dealii::internal::t8code::types::element *
                    t8code_coarse_cell;
                  
                  typename dealii::internal::t8code::types::tree *tree =
                    init_tree(cell->index());

                  dealii::internal::t8code::element_new(parallel_forest, T8_ECLASS_QUAD, &t8code_coarse_cell);

                  dealii::internal::t8code::types::eclass tree_class = T8_ECLASS_QUAD;
                  //TODO: replace by cell accessor call

                  dealii::internal::t8code::init_root(
                    parallel_forest,
                    tree_class,
                    t8code_coarse_cell);

                  match_tree_recursively<dim, spacedim>(*tree,
                                                        cell,
                                                        t8code_coarse_cell,
                                                        parallel_forest,
                                                        this->my_subdomain);
                }
            }


          // Fix all the flags to make sure we have a consistent local
          // mesh. For some reason periodic boundaries involving artificial
          // cells are not obeying the 2:1 ratio that we require (and that is
          // enforced by t8code between active cells). So, here we will loop
          // refining across periodic boundaries until 2:1 is satisfied. Note
          // that we are using the base class (sequential) prepare and execute
          // calls here, not involving communication, because we are only
          // trying to recreate a local triangulation from the t8code data.
          {
            bool         mesh_changed = true;
            unsigned int loop_counter = 0;

            do
              {
                this->dealii::Triangulation<dim, spacedim>::
                  prepare_coarsening_and_refinement();

                this->update_periodic_face_map();

//                mesh_changed =
//                  enforce_mesh_balance_over_periodic_boundaries(*this);

                // We can't be sure that we won't run into a situation where we
                // can not reconcile mesh smoothing and balancing of periodic
                // faces. As we don't know what else to do, at least abort with
                // an error message.
                ++loop_counter;

                AssertThrow(
                  loop_counter < 32,
                  ExcMessage(
                    "Infinite loop in "
                    "parallel::distributed::Triangulation::copy_local_forest_to_triangulation() "
                    "for periodic boundaries detected. Aborting."));
              }
            while (mesh_changed);
          }

          // see if any flags are still set
          mesh_changed =
            std::any_of(this->begin_active(),
                        active_cell_iterator{this->end()},
                        [](const CellAccessor<dim, spacedim> &cell) {
                          return cell.refine_flag_set() ||
                                 cell.coarsen_flag_set();
                        });

          // actually do the refinement to change the local mesh by
          // calling the base class refinement function directly
          try
            {
              dealii::Triangulation<dim, spacedim>::
                execute_coarsening_and_refinement();
            }
          catch (
            const typename Triangulation<dim, spacedim>::DistortedCellList &)
            {
              // the underlying triangulation should not be checking for
              // distorted cells
              DEAL_II_ASSERT_UNREACHABLE();
            }
        }
      while (mesh_changed);

#  ifdef DEBUG
      // check if correct number of ghosts is created
      unsigned int num_ghosts = 0;

      for (const auto &cell : this->active_cell_iterators())
        {
          if (cell->subdomain_id() != this->my_subdomain &&
              cell->subdomain_id() != numbers::artificial_subdomain_id)
            ++num_ghosts;
        }

//      Assert(num_ghosts == parallel_ghost->num_ghosts_elements,
//             ExcInternalError());
#  endif


#  ifdef DEBUG
      // check that our local copy has exactly as many cells as the t8code
      // original (at least if we are on only one processor); for parallel
      // computations, we want to check that we have at least as many as t8code
      // stores locally (in the future we should check that we have exactly as
      // many non-artificial cells as parallel_forest->local_num_elements)
      {
        const unsigned int total_local_cells = this->n_active_cells();


        if (Utilities::MPI::n_mpi_processes(this->mpi_communicator) == 1)
          {
            Assert(static_cast<unsigned int>(
                     parallel_forest->local_num_elements) == total_local_cells,
                   ExcInternalError());
          }
        else
          {
            Assert(static_cast<unsigned int>(
                     parallel_forest->local_num_elements) <= total_local_cells,
                   ExcInternalError());
          }

        // count the number of owned, active cells and compare with t8code.
        unsigned int n_owned = 0;
        for (const auto &cell : this->active_cell_iterators())
          {
            if (cell->subdomain_id() == this->my_subdomain)
              ++n_owned;
          }

        Assert(static_cast<unsigned int>(
                 parallel_forest->local_num_elements) == n_owned,
               ExcInternalError());
      }
#  endif

      // finally, after syncing the parallel_forest with the triangulation,
      // also update the cell_relations, which will be used for
      // repartitioning, further refinement/coarsening, and unpacking
      // of stored or transferred data.
//      update_cell_relations();
    }
    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    Triangulation<dim, spacedim>::Triangulation(
      const MPI_Comm mpi_communicator,
      const typename dealii::Triangulation<dim, spacedim>::MeshSmoothing
                     smooth_grid,
      const Settings settings)
      : // Do not check for distorted cells.
        // For multigrid, we need limit_level_difference_at_vertices
        // to make sure the transfer operators only need to consider two levels.
      dealii::parallel::DistributedTriangulationBase<dim, spacedim>(
        mpi_communicator,
        (settings & construct_multigrid_hierarchy) ?
          static_cast<
            typename dealii::Triangulation<dim, spacedim>::MeshSmoothing>(
            smooth_grid |
            Triangulation<dim, spacedim>::limit_level_difference_at_vertices) :
          smooth_grid,
        false)
      , settings(settings)
      , triangulation_has_content(false)
      , cmesh(nullptr)
      , parallel_forest(nullptr)
      , parallel_ghost(nullptr)
    {}

    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    Triangulation<dim, spacedim>::~Triangulation(){
      //TODO: destroy forest, cmesh, scheme_collection
    }

    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    void Triangulation<dim, spacedim>::clear(){
    }
    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    void Triangulation<dim, spacedim>::execute_coarsening_and_refinement(){
    }
    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    bool Triangulation<dim, spacedim>::is_multilevel_hierarchy_constructed() const{
      return false;
    }
    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    void Triangulation<dim, spacedim>::create_triangulation(
        const TriangulationDescription::Description<dim, spacedim>
          &construction_data) {
            (void)construction_data;
          }


  } // namespace distributed
} // namespace parallel

#endif // DEAL_II:WITH_T8CODE

/*-------------- Explicit Instantiations -------------------------------*/
#include "tria.inst"


DEAL_II_NAMESPACE_CLOSE
