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

#include <deal.II/distributed/t8code_wrappers.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>

#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_cmesh/t8_cmesh_types.h>
#include <t8_data/t8_element_array_iterator.hxx>
#include <t8_forest/t8_forest_ghost.h>
#include <t8_schemes/t8_default/t8_default.hxx>
#include <t8_schemes/t8_default/t8_default_tri/t8_dtri.h>
#include <t8_vec.h>

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
      const std::vector<std::vector<unsigned int> > deal_to_t8code_children = 
      {
        {0,1,3,2,4,5,7,6},
        {0,3,2,1,4,7,6,5},
        {1,3,0,2,5,7,4,6}, 
        {3,2,0,1,7,6,4,5}, 
        {3,0,1,2,7,4,5,6}, 
        {2,0,3,1,6,4,7,5}
      };
      const std::vector<std::vector<unsigned int> > t8code_to_deal_children = 
      {
        {0,1,3,2,4,5,7,6}, 
        {0,3,2,1,4,7,6,5}, 
        {2,0,3,1,6,4,7,5}, 
        {2,3,1,0,6,7,5,4}, 
        {1,2,3,0,5,6,7,4},
        {1,3,0,2,5,7,4,6}
      };

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
    dealii::internal::t8code::types::eclass 
    t8_eclass_from_reference_cell(const ReferenceCell &cell){
      if(cell.is_hyper_cube()){
        switch(cell.get_dimension()){
          case 0:
            return T8_ECLASS_VERTEX;
          case 1:
            return T8_ECLASS_LINE;
          case 2:
            return T8_ECLASS_QUAD;
          case 3:
            return T8_ECLASS_HEX;
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
      }else if(cell.is_simplex()){
        switch(cell.get_dimension()){
          case 0:
            return T8_ECLASS_VERTEX;
          case 1:
            return T8_ECLASS_LINE;
          case 2:
            return T8_ECLASS_TRIANGLE;
          case 3:
            return T8_ECLASS_TET;
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
      }
            DEAL_II_NOT_IMPLEMENTED();
          }


    template <int dim, int spacedim>
    void
    delete_all_children_and_self(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell)
    {
      if (cell->has_children())
        for (unsigned int c = 0; c < cell->n_children(); ++c)
          delete_all_children_and_self<dim, spacedim>(cell->child(c));
      else{
        std::cout<<"set coarsen for cell "<< cell->id()<<",rank: "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
//        cell->set_coarsen_flag();
      }
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
          sc_array_index(parallel_forest->trees, t8_forest_get_local_id(parallel_forest, tree_index)));

      return tree;
    }

    template <int dim, int spacedim>
    bool
    tree_exists_locally(
      const typename dealii::internal::t8code::types::forest parallel_forest,
      const typename dealii::internal::t8code::types::gloidx coarse_grid_cell)
    {
      Assert(coarse_grid_cell < parallel_forest->cmesh->num_trees,
             ExcInternalError());
      return ((coarse_grid_cell >= parallel_forest->first_local_tree) &&
              (coarse_grid_cell <= parallel_forest->last_local_tree));
    }


    template <int dim, int spacedim>
    void
    match_tree_recursively(
      const typename dealii::internal::t8code::types::tree        tree,
      const typename Triangulation<dim, spacedim>::cell_iterator &dealii_cell,
      const typename dealii::internal::t8code::types::element    *t8code_cell,
      const typename dealii::internal::t8code::types::forest      forest,
      const types::subdomain_id                                   my_subdomain,
      const int dealii_type)
    {
      std::cout<<"match_tree_recursively:"<<std::endl;
      std::cout<<dealii_cell->id()<<" with type "<< dealii_type <<std::endl;


      dealii::internal::t8code::types::eclass tree_class = t8_eclass_from_reference_cell(dealii_cell->reference_cell());
      dealii::internal::t8code::types::eclass_scheme *eclass_scheme =
        t8_forest_get_eclass_scheme(forest, tree_class);
      eclass_scheme->t8_element_debug_print(t8code_cell);
      auto compare_lambda = [eclass_scheme](auto x, auto y) {
        return eclass_scheme->t8_element_compare(x, y) < 0;
      };


      if (std::binary_search(t8_element_array_begin(&tree.elements),
                             t8_element_array_end(&tree.elements),
                             t8code_cell,
                             compare_lambda))
        {
          // yes, cell found in local part of p4est
          delete_all_children<dim, spacedim>(dealii_cell);
          std::cout<<"found local cell: "<<dealii_cell->id()<<std::endl;
          if (dealii_cell->is_active()){
            std::cout<<"found deal leaf: "<<dealii_cell->id()<<std::endl;
            dealii_cell->set_subdomain_id(my_subdomain);
          }
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
              typename dealii::internal::t8code::types::element
                *t8code_child[GeometryInfo<dim>::max_children_per_cell];

              // TODO: remove, replcae by t8_element_nwe(length);
              for (unsigned int c = 0;
                   c < GeometryInfo<dim>::max_children_per_cell;
                   ++c)
                dealii::internal::t8code::element_new(forest,
                                                      tree_class,
                                                      t8code_child + c);

              dealii::internal::t8code::element_children(forest,
                                                         tree_class,
                                                         t8code_cell,
                                                         t8code_child);


              for (unsigned int c = 0;
                   c < GeometryInfo<dim>::max_children_per_cell;
                   ++c)
{
                        const int t8_child_id = (tree_class == T8_ECLASS_TRIANGLE)? 
                                  dealii::internal::parallel::distributed::deal_to_t8code_children[dealii_type][c] : c;

                if (dealii::internal::t8code::element_overlaps_tree(
                      forest, tree, t8code_child[t8_child_id]) == false)
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
                    const int child_dealii_type = ((c%4)!=3) ? dealii_type : (dealii_type + 1) % 6;
                    const int t8_child_id = (tree_class == T8_ECLASS_TRIANGLE)? 
                                  dealii::internal::parallel::distributed::deal_to_t8code_children[dealii_type][c] : c;

                    match_tree_recursively<dim, spacedim>(tree,
                                                          dealii_cell->child(c),
                                                          t8code_child[t8_child_id],
                                                          forest,
                                                          my_subdomain, child_dealii_type);
                  }
            }
              for (unsigned int c = 0;
                   c < GeometryInfo<dim>::max_children_per_cell;
                   ++c)
                dealii::internal::t8code::element_destroy(forest,
                                                          tree_class,
                                                          t8code_child + c);
            }
        }
    }

    template <int dim, int spacedim>
    void
    match_element(
      const dealii::Triangulation<dim, spacedim>              *tria,
      const typename dealii::internal::t8code::types::forest   forest,
      unsigned int                                             dealii_index,
      const typename dealii::internal::t8code::types::element *ghost_element,
      types::subdomain_id                                      ghost_owner,
      dealii::internal::t8code::types::eclass ghost_eclass
      )
    {
      const int l = dealii::internal::t8code::element_level(forest,
                                                            ghost_eclass,
                                                            ghost_element);
      
      dealii::internal::t8code::types::element* t8code_cell;
      
      dealii::internal::t8code::element_new(forest,
                                            ghost_eclass,
                                            &t8code_cell);

      dealii::internal::t8code::init_root(forest,
                                          ghost_eclass,
                                          t8code_cell);

      int dealii_type = 0;
      for (int i = 0; i < l; ++i)
        {
          typename Triangulation<dim, spacedim>::cell_iterator cell(
            tria, i, dealii_index);
          if (cell->is_active())
            {
              cell->clear_coarsen_flag();
              cell->set_refine_flag();
                    dealii::internal::t8code::element_destroy(forest,
                                            ghost_eclass,
                                            &t8code_cell);
              return;
            }

          const int child_id = dealii::internal::t8code::element_ancestor_id(
            forest, ghost_eclass, ghost_element, i + 1);

          dealii::internal::t8code::element_child(forest, ghost_eclass, t8code_cell, child_id, t8code_cell);
          std::cout<<"child_id: "<<child_id<<", type: "<<dealii_type<<std::endl;
          int deal_child_id = (ghost_eclass == T8_ECLASS_TRIANGLE) ? dealii::internal::parallel::distributed::t8code_to_deal_children[dealii_type][child_id] : child_id;
          dealii_index = cell->child_index(deal_child_id);

          typename Triangulation<dim, spacedim>::cell_iterator child(
            tria, i+1, dealii_index);
          if (cell->child_iterator_to_index(child) % 4 == 3){
            dealii_type = (dealii_type + 1) % 6;
          }
        }

      dealii::internal::t8code::element_destroy(forest,
                                            ghost_eclass,
                                            &t8code_cell);

      typename Triangulation<dim, spacedim>::cell_iterator cell(tria,
                                                                l,
                                                                dealii_index);

      std::cout<<"corresponding cell: " << cell->id() << std::endl;
      if (cell->has_children())
        delete_all_children<dim, spacedim>(cell);
      else
        {
          cell->clear_coarsen_flag();
          std::cout<<"set subdomain id from cell "<<cell->id()<<"to "<<ghost_owner<<std::endl;
          cell->set_subdomain_id(ghost_owner);
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
      cmesh = //t8_cmesh_new_periodic_hybrid(this->mpi_communicator);
        t8_cmesh_new_hypercube(T8_ECLASS_QUAD, this->mpi_communicator, 0, 0, 0);
      scheme_collection = t8_scheme_new_default_cxx();
      parallel_forest   = t8_forest_new_uniform(
        cmesh, scheme_collection, 3, 1, this->mpi_communicator);

#if 1
      typename dealii::internal::t8code::types::forest partitioned_forest;
      const auto adapt_fn = [](t8_forest_t         forest,
                               t8_forest_t         forest_from,
                               t8_locidx_t         which_tree,
                               t8_locidx_t         lelement_id,
                               t8_eclass_scheme_c *ts,
                               const int           is_family,
                               const int           num_elements,
                               t8_element_t       *elements[]) {
        (void)forest;
        (void)lelement_id;
        (void)ts;
        (void)is_family;
        (void)num_elements;

        double min_corner[3];
        double max_corner[3];
        /* Compute the element's centroid coordinates. */
        t8_forest_element_coordinate(forest_from,
                                   which_tree,
                                   elements[0],0,
                                   min_corner);

        double dist_min = t8_vec_norm(min_corner);
        t8_forest_element_coordinate(forest_from,
                                   which_tree,
                                   elements[0],3, //TODO:generalize
                                   max_corner);

        double dist_max = t8_vec_norm(max_corner);

        if (dist_max >= 0.5 && dist_min <= 0.5)
          {
            /* Refine this element. */
            return 1;
          }
        else
          {
            return 0;
          }
      };


      t8_forest_init (&partitioned_forest);
      t8_forest_set_adapt(partitioned_forest, parallel_forest, adapt_fn, 0);
      t8_forest_set_balance(partitioned_forest, parallel_forest, 0);
      t8_forest_set_partition (partitioned_forest, parallel_forest, 0);
      t8_forest_set_ghost (partitioned_forest, 1, T8_GHOST_FACES);
      t8_forest_commit (partitioned_forest);
      parallel_forest = partitioned_forest;

      t8_forest_init (&partitioned_forest);
      t8_forest_set_adapt(partitioned_forest, parallel_forest, adapt_fn, 0);
      t8_forest_set_balance(partitioned_forest, parallel_forest, 0);
      t8_forest_set_partition (partitioned_forest, parallel_forest, 0);
      t8_forest_set_ghost (partitioned_forest, 1, T8_GHOST_FACES);
      t8_forest_commit (partitioned_forest);
      parallel_forest = partitioned_forest;

      t8_forest_init (&partitioned_forest);
      t8_forest_set_adapt(partitioned_forest, parallel_forest, adapt_fn, 0);
      t8_forest_set_balance(partitioned_forest, parallel_forest, 0);
      t8_forest_set_partition (partitioned_forest, parallel_forest, 0);
      t8_forest_set_ghost (partitioned_forest, 1, T8_GHOST_FACES);
      t8_forest_commit (partitioned_forest);
      parallel_forest = partitioned_forest;


#endif
    }

    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    const std::vector<types::global_dof_index>
      &Triangulation<dim, spacedim>::get_t8code_tree_to_coarse_cell_permutation()
        const
    {
      return t8code_tree_to_coarse_cell_permutation;
    }

    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    void Triangulation<dim, spacedim>::create_triangulation(
      const std::vector<Point<spacedim>> &vertices,
      const std::vector<CellData<dim>>   &cells,
      const SubCellData                  &subcelldata)
    {
      dealii::Triangulation<dim, spacedim>::create_triangulation(vertices,
                                                                 cells,
                                                                 subcelldata);
      triangulation_has_content = true;
      setup_coarse_cell_to_t8code_tree_permutation();
      copy_new_triangulation_to_t8code();
      copy_local_forest_to_triangulation();
      this->update_number_cache();
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
                std::cout<<"set coarsen for cell "<< cell->id()<<",rank: "<<Utilities::MPI::this_mpi_process(this->get_communicator())<<std::endl;
//                cell->set_coarsen_flag();
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


      // set all cells to artificial. we will later set it to the correct
      // subdomain in match_tree_recursively
      for (const auto &cell : this->cell_iterators_on_level(0))
        cell->recursively_set_subdomain_id(numbers::artificial_subdomain_id);

      int loop_iter = 0;
      do
        {
          std::cout<<"loop iter: "<<loop_iter<<std::endl;
          loop_iter++;
          for (const auto &cell : this->cell_iterators_on_level(0))
            {
                dealii::internal::t8code::types::eclass tree_class =
                  t8_eclass_from_reference_cell(cell->reference_cell());

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

                  typename dealii::internal::t8code::types::element
                    *t8code_coarse_cell;

                  typename dealii::internal::t8code::types::tree *tree =
                    init_tree(cell->index());


                  dealii::internal::t8code::element_new(parallel_forest,
                                                        tree_class,
                                                        &t8code_coarse_cell);

                  dealii::internal::t8code::init_root(parallel_forest,
                                                      tree_class,
                                                      t8code_coarse_cell);

                  match_tree_recursively<dim, spacedim>(*tree,
                                                        cell,
                                                        t8code_coarse_cell,
                                                        parallel_forest,
                                                        this->my_subdomain,
                                                        0);
                  dealii::internal::t8code::element_destroy(
                    parallel_forest, tree_class, &t8code_coarse_cell);
                }
            }

          types::subdomain_id                              ghost_owner     = 0;
          typename dealii::internal::t8code::types::gloidx global_tree_idx = 0;
          typename dealii::internal::t8code::types::locidx num_ghost_trees =
            t8_forest_get_num_ghost_trees(parallel_forest);
          typename dealii::internal::t8code::types::locidx num_ghosts_in_tree;


          for (typename dealii::internal::t8code::types::locidx
                 local_ghost_tree_idx = 0;
               local_ghost_tree_idx < num_ghost_trees;
               local_ghost_tree_idx++)
            {
              t8_element_array_t *element_array =
                t8_forest_ghost_get_tree_elements(parallel_forest,
                                                  local_ghost_tree_idx);
              dealii::internal::t8code::types::eclass ghost_eclass =
                t8_forest_ghost_get_tree_class(parallel_forest, local_ghost_tree_idx);
              
              global_tree_idx =
                t8_forest_ghost_get_global_treeid(parallel_forest,
                                                  local_ghost_tree_idx);

              num_ghosts_in_tree =
                t8_forest_ghost_tree_num_elements(parallel_forest,
                                                  local_ghost_tree_idx);
              for (dealii::internal::t8code::types::locidx
                     local_ghost_element_idx = 0;
                   local_ghost_element_idx < num_ghosts_in_tree;
                   local_ghost_element_idx++)
                {
                  typename dealii::internal::t8code::types::element
                    *local_ghost_element =
                      t8_element_array_index_locidx_mutable(element_array,
                                                    local_ghost_element_idx);
                  ghost_owner =
                    t8_forest_element_find_owner(parallel_forest,
                                                 global_tree_idx,
                                                 local_ghost_element,
                                                 ghost_eclass);
                  unsigned int coarse_cell_index =
                    t8code_tree_to_coarse_cell_permutation[global_tree_idx];

                  match_element<dim, spacedim>(this,
                                               parallel_forest,
                                               coarse_cell_index,
                                               local_ghost_element,
                                               ghost_owner, ghost_eclass);
                }
            }



          // see if any flags are still set
          for (const auto &cell : this->active_cell_iterators()){
            if(cell->refine_flag_set() ){
                std::cout<<"refine flag set for cell "<< cell->id()<<", loop:"<<loop_iter<<",rank: "<<Utilities::MPI::this_mpi_process(this->get_communicator()) <<std::endl;
              }
              if(cell->coarsen_flag_set() ){
                std::cout<<"coarsen flag set for cell "<< cell->id()<<", loop:"<<loop_iter<<",rank: "<<Utilities::MPI::this_mpi_process(this->get_communicator())<<std::endl;
              }
          }

            // clear coarsen flag if not all children were marked
            for (const auto &cell : this->cell_iterators())
              {
                // nothing to do if we are already on the finest level
                if (cell->is_active())
                  continue;

                const unsigned int n_children       = cell->n_children();
                unsigned int       flagged_children = 0;
                for (unsigned int child = 0; child < n_children; ++child)
                  if (cell->child(child)->is_active() &&
                      cell->child(child)->coarsen_flag_set())
                    ++flagged_children;

                // if not all children were flagged for coarsening, remove
                // coarsen flags
                if (flagged_children < n_children)
                  for (unsigned int child = 0; child < n_children; ++child)
                    if (cell->child(child)->is_active())
                      cell->child(child)->clear_coarsen_flag();
              }
          std::cout<<"cleared coarsening flags"<<std::endl;
          for (const auto &cell : this->active_cell_iterators()){
            if(cell->refine_flag_set() ){
                std::cout<<"refine flag set for cell "<< cell->id()<<", loop:"<<loop_iter<<",rank: "<<Utilities::MPI::this_mpi_process(this->get_communicator()) <<std::endl;
              }
              if(cell->coarsen_flag_set() ){
                std::cout<<"coarsen flag set for cell "<< cell->id()<<", loop:"<<loop_iter<<",rank: "<<Utilities::MPI::this_mpi_process(this->get_communicator())<<std::endl;
              }
          }

          mesh_changed =
            std::any_of(this->begin_active(),
                        active_cell_iterator{this->end()},
                        [loop_iter,this](const CellAccessor<dim, spacedim> &cell) {
//                          bool flag_set = false;
//                          const auto parent = cell.parent();
//                          for (unsigned int i = 0; i < GeometryInfo<dim>::max_children_per_cell; i++){
//                            const auto sibling = parent.child(i);
//                            if(sibling.refine_flag_set() || sibling.coarsen_flag_set()) {
//                              flag_set = true;
//                            }
//                          }


                          return (cell.refine_flag_set() ||
                                 cell.coarsen_flag_set());
                        });
          std::cout << "mesh_changed" << mesh_changed <<std::endl;
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

            // for (const auto &cell: this->active_cell_iterators()){
            //   cell->set_material_id(cell->subdomain_id());
            // }

            

            std::ofstream out("grid-" + std::to_string(loop_iter) + "." + std::to_string(Utilities::MPI::this_mpi_process(this->get_communicator())) + ".vtk");
            GridOut       grid_out;
            grid_out.write_vtk(*this, out);
        }
      while (mesh_changed);

#  ifdef DEBUG
      // check if correct number of ghosts is created
      int num_ghosts = 0;

      for (const auto &cell : this->active_cell_iterators())
        {
          if (cell->subdomain_id() != this->my_subdomain &&
              cell->subdomain_id() != numbers::artificial_subdomain_id)
            ++num_ghosts;
        }

      Assert(num_ghosts == t8_forest_get_num_ghosts(parallel_forest),
             ExcInternalError());
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

        Assert(static_cast<unsigned int>(parallel_forest->local_num_elements) ==
                 n_owned,
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
    Triangulation<dim, spacedim>::~Triangulation()
    {
      // virtual functions called in constructors and destructors never use the
      // override in a derived class
      // for clarity be explicit on which function is called
      try
        {
          dealii::parallel::distributed::Triangulation<dim, spacedim>::clear();
        }
      catch (...)
        {}

      AssertNothrow(triangulation_has_content == false, ExcInternalError());
      AssertNothrow(cmesh == nullptr, ExcInternalError());
      AssertNothrow(parallel_forest == nullptr, ExcInternalError());
    }

    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    void Triangulation<dim, spacedim>::clear()
    {
      if (triangulation_has_content)
        {
          t8_forest_unref(&parallel_forest);
          cmesh                     = nullptr;
          triangulation_has_content = 0;
          parallel_ghost            = nullptr;
          coarse_cell_to_t8code_tree_permutation.resize(0);
          t8code_tree_to_coarse_cell_permutation.resize(0);
          scheme_collection = nullptr;
        }
      dealii::parallel::DistributedTriangulationBase<dim, spacedim>::clear();

      this->update_number_cache();
    }
    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    void Triangulation<dim, spacedim>::execute_coarsening_and_refinement()
    {

    }
    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    bool Triangulation<dim, spacedim>::is_multilevel_hierarchy_constructed()
      const
    {
      return false;
    }
    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    void Triangulation<dim, spacedim>::create_triangulation(
      const TriangulationDescription::Description<dim, spacedim>
        &construction_data)
    {
      (void)construction_data;
    }


  } // namespace distributed
} // namespace parallel

#endif // DEAL_II:WITH_T8CODE

/*-------------- Explicit Instantiations -------------------------------*/
#include "tria.inst"


DEAL_II_NAMESPACE_CLOSE
