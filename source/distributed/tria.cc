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

#include <t8_cmesh/t8_cmesh.hxx>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_cmesh/t8_cmesh_types.h>
#include <t8_data/t8_element_array_iterator.hxx>
#include <t8_forest/t8_forest_ghost.h>
#include <t8_forest/t8_forest_partition.h>
#include <t8_vtk/t8_vtk_writer.h>
#  include <t8_schemes/t8_scheme.hxx>
#include <t8_schemes/t8_default/t8_default.hxx>
#include <t8_geometry/t8_geometry_implementations/t8_geometry_linear.hxx>
#include <t8_geometry/t8_geometry_with_vertices.hxx>
#include <t8_schemes/t8_default/t8_default_tri/t8_dtri.h>
#include <t8_types/t8_vec.h>

#include <algorithm>
#include <fstream>
#include <iostream>
//#include <limits>
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

      const std::vector<std::vector<int> >
        dealii_to_t8_faces = 
      {
        {}, //vertex has no faces
        {0,1}, //line is oriented the same
        {0,1,2,3}, //quad is oriented the same
        {2,0,1}, //tri is rotated by one
        {0,1,2,3,4,5,6,7},//hex
        {},//tet
        {},//prism
        {}//pyramid
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
//        std::cout<<"set coarsen for cell "<< cell->id()<<",rank: "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
        cell->set_coarsen_flag();
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

      dealii::internal::t8code::types::eclass eclass = t8_eclass_from_reference_cell(dealii_cell->reference_cell());
      dealii::internal::t8code::types::scheme_collection *scheme =
        t8_forest_get_scheme(forest);
//      scheme->t8_element_debug_print(eclass,t8code_cell);
      auto compare_lambda = [scheme,eclass](auto x, auto y) {
        return scheme->element_compare(eclass, x, y) < 0;
      };


      if (std::binary_search(t8_element_array_begin(&tree.elements),
                             t8_element_array_end(&tree.elements),
                             t8code_cell,
                             compare_lambda))
        {
          // yes, cell found in local part of p4est
          delete_all_children<dim, spacedim>(dealii_cell);
          if (dealii_cell->is_active()){
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

              dealii::internal::t8code::element_new(forest,
                                                      eclass,
                                                      GeometryInfo<dim>::max_children_per_cell,
                                                      t8code_child);

              dealii::internal::t8code::element_children(forest,
                                                         eclass,
                                                         t8code_cell,
                                                         t8code_child);


              for (unsigned int c = 0;
                   c < GeometryInfo<dim>::max_children_per_cell;
                   ++c)
{
                        const int t8_child_id = (eclass == T8_ECLASS_TRIANGLE)? 
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
                    const int t8_child_id = (eclass == T8_ECLASS_TRIANGLE)? 
                                  dealii::internal::parallel::distributed::deal_to_t8code_children[dealii_type][c] : c;

                    match_tree_recursively<dim, spacedim>(tree,
                                                          dealii_cell->child(c),
                                                          t8code_child[t8_child_id],
                                                          forest,
                                                          my_subdomain, child_dealii_type);
                  }
            }
              dealii::internal::t8code::element_destroy(forest,
                                                          eclass, GeometryInfo<dim>::max_children_per_cell,
                                                          t8code_child);
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
                                            ghost_eclass,1,
                                            &t8code_cell);

      dealii::internal::t8code::init_root(forest,
                                          ghost_eclass,
                                          t8code_cell);

      // t8_debugf("try to match ghost element in dealii index %i\n", dealii_index);
      // const t8_scheme *scheme = t8_forest_get_scheme(forest);
      // scheme->element_debug_print(ghost_eclass, ghost_element);

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
                                            ghost_eclass,1,
                                            &t8code_cell);
              return;
            }

          const int child_id = dealii::internal::t8code::element_ancestor_id(
            forest, ghost_eclass, ghost_element, i + 1);
          // std::cout<<"found childid " <<child_id << " at level "<<i<<std::endl;

          dealii::internal::t8code::element_child(forest, ghost_eclass, t8code_cell, child_id, t8code_cell);
//          std::cout<<"child_id: "<<child_id<<", type: "<<dealii_type<<std::endl;
          int deal_child_id = (ghost_eclass == T8_ECLASS_TRIANGLE) ? dealii::internal::parallel::distributed::t8code_to_deal_children[dealii_type][child_id] : child_id;
          dealii_index = cell->child_index(deal_child_id);

          typename Triangulation<dim, spacedim>::cell_iterator child(
            tria, i+1, dealii_index);
          if (cell->child_iterator_to_index(child) % 4 == 3){
            dealii_type = (dealii_type + 1) % 6;
          }
        }

      dealii::internal::t8code::element_destroy(forest,
                                            ghost_eclass,1,
                                            &t8code_cell);

      typename Triangulation<dim, spacedim>::cell_iterator cell(tria,
                                                                l,
                                                                dealii_index);

//      std::cout<<"corresponding cell: " << cell->id() << std::endl;
      if (cell->has_children())
        delete_all_children<dim, spacedim>(cell);
      else
        {
          cell->clear_coarsen_flag();
          // std::cout<<"set subdomain id from cell "<<cell->id()<<"to "<<ghost_owner<<std::endl;
          cell->set_subdomain_id(ghost_owner);
        }
    }

    template <int dim, int spacedim>
    using cell_relation_t = typename std::pair<
      typename dealii::Triangulation<dim, spacedim>::cell_iterator,
      CellStatus>;

    /**
     * Adds a pair of a @p dealii_cell and its @p status
     * to the vector containing all relations @p cell_rel.
     * The pair will be inserted in the position corresponding to the one
     * of the p4est quadrant in the underlying p4est sc_array. The position
     * will be determined from @p idx, which is the position of the quadrant
     * in its corresponding @p tree. The p4est quadrant will be deduced from
     * the @p tree by @p idx.
     */
    template <int dim, int spacedim>
    inline void
    add_single_cell_relation(
      std::vector<cell_relation_t<dim, spacedim>>                &cell_rel,
      const typename dealii::internal::t8code::types::tree   *tree,
      const unsigned int                                          idx,
      const typename Triangulation<dim, spacedim>::cell_iterator &dealii_cell,
      const CellStatus                                            status)
    {
      const unsigned int local_quadrant_index = tree->elements_offset + idx;

      // check if we will be writing into valid memory
      Assert(local_quadrant_index < cell_rel.size(), ExcInternalError());

      // store relation
      cell_rel[local_quadrant_index] = std::make_pair(dealii_cell, status);
    }



    /**
     * This is the recursive part of the member function
     * update_cell_relations().
     *
     * Find the relation between the @p t8code_cell and the @p dealii_cell in the
     * corresponding @p tree. Depending on the CellStatus relation between the two,
     * a new entry will either be inserted in @p cell_rel or the recursion
     * will be continued.
     */
    template <int dim, int spacedim>
    void
    update_cell_relations_recursively(
      const typename dealii::internal::t8code::types::forest forest,
      std::vector<cell_relation_t<dim, spacedim>>                  &cell_rel,
      const typename dealii::internal::t8code::types::locidx     ltreeid,
      const typename Triangulation<dim, spacedim>::cell_iterator   &dealii_cell,
      const typename dealii::internal::t8code::types::element *t8code_cell, const int dealii_type) //TODO: adapt dealii type for children
    {
      //get element array
      // use new function from forest to get idx
      const typename dealii::internal::t8code::types::locidx idx = t8_forest_element_leaf_index_in_tree (forest, t8code_cell, ltreeid);
      if (idx == -1 &&
          (dealii::internal::t8code::element_overlaps_tree(forest, 
            *t8_forest_get_tree(forest, ltreeid),
            t8code_cell) == false))
        // this element and none of its children belong to us.
        return;

      dealii::internal::t8code::types::eclass eclass = t8_eclass_from_reference_cell(dealii_cell->reference_cell());


      // recurse further if both p4est and dealii still have children
      const bool t8code_has_children = (idx == -1);
      if (t8code_has_children && dealii_cell->has_children())
        {
          // recurse further
          typename dealii::internal::t8code::types::element* //TODO: max children dependent on shape
            t8code_child[GeometryInfo<dim>::max_children_per_cell];
                  dealii::internal::t8code::element_new(forest,
                                                        eclass,GeometryInfo<dim>::max_children_per_cell,
                                                        t8code_child);


          dealii::internal::t8code::element_children(forest, eclass,
            t8code_cell, t8code_child);

          for (unsigned int t8_childid = 0; t8_childid < GeometryInfo<dim>::max_children_per_cell;
              ++t8_childid)
            {
            int deal_child_id = (eclass == T8_ECLASS_TRIANGLE) ? dealii::internal::parallel::distributed::t8code_to_deal_children[dealii_type][t8_childid] : t8_childid;
            int deal_child_type = (deal_child_id % 4 == 3)? (dealii_type + 1) % 6 : dealii_type;

              update_cell_relations_recursively<dim, spacedim>(forest, 
                cell_rel, ltreeid, dealii_cell->child(deal_child_id), t8code_child[t8_childid], deal_child_type);
            }
                              dealii::internal::t8code::element_destroy(forest,
                                                        eclass,GeometryInfo<dim>::max_children_per_cell,
                                                        t8code_child);
        }
      else if (!t8code_has_children && !dealii_cell->has_children())
        {
          // this active cell didn't change
          // save pair into corresponding position
          add_single_cell_relation<dim, spacedim>(
            cell_rel, t8_forest_get_tree(forest, ltreeid), idx, dealii_cell, CellStatus::cell_will_persist);
        }
      else if (t8code_has_children) // based on the conditions above, we know that
                                  // dealii_cell has no children
        {
          // this cell got refined in p4est, but the dealii_cell has not yet been
          // refined

          // this quadrant is not active
          // generate its children, and store information in those
          typename dealii::internal::t8code::types::element* //TODO: max children dependent on shape
            t8code_child[GeometryInfo<dim>::max_children_per_cell];
                  dealii::internal::t8code::element_new(forest,
                                                        eclass,GeometryInfo<dim>::max_children_per_cell,
                                                        t8code_child);


          dealii::internal::t8code::element_children(forest, eclass,
            t8code_cell, t8code_child);

          // mark first child with CellStatus::cell_will_be_refined and the
          // remaining children with CellStatus::cell_invalid, but associate them
          // all with the parent cell unpack algorithm will be called only on
          // CellStatus::cell_will_be_refined flagged quadrant
          CellStatus cell_status;
          for (unsigned int deal_childid = 0; deal_childid < GeometryInfo<dim>::max_children_per_cell;
              ++deal_childid)
            {
            const unsigned int t8_childid = (eclass == T8_ECLASS_TRIANGLE)? 
                                  dealii::internal::parallel::distributed::deal_to_t8code_children[dealii_type][deal_childid] : deal_childid;
            t8_locidx_t child_idx = 
              t8_forest_element_leaf_index_in_tree (forest, t8code_child[t8_childid], ltreeid);

              cell_status = (deal_childid == 0) ? CellStatus::cell_will_be_refined :
                                      CellStatus::cell_invalid;

              add_single_cell_relation<dim, spacedim>(
                cell_rel, t8_forest_get_tree(forest,ltreeid), child_idx, dealii_cell, cell_status);
            }
                              dealii::internal::t8code::element_destroy(forest,
                                                        eclass,GeometryInfo<dim>::max_children_per_cell,
                                                        t8code_child);
        }
      else // based on the conditions above, we know that t8code_cell has no
          // children, and the dealii_cell does
        {
          // its children got coarsened into this cell in p4est,
          // but the dealii_cell still has its children
          add_single_cell_relation<dim, spacedim>(
            cell_rel,
            t8_forest_get_tree(forest, ltreeid),
            idx,
            dealii_cell,
            CellStatus::children_will_be_coarsened);
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
      //TODO!! coarse mesh permutation!!!
      t8_cmesh_init(&cmesh);
      t8_cmesh_register_geometry<t8_geometry_linear> (cmesh);

      for(const auto & cell:this->active_cell_iterators()){ 
        const auto t8_index = coarse_cell_to_t8code_tree_permutation[cell->index()];
        t8_eclass_t eclass = t8_eclass_from_reference_cell(cell->reference_cell());
        t8_cmesh_set_tree_class (cmesh, t8_index, eclass);
        std::vector<double> coords(3*cell->n_vertices());
        for(unsigned int ivertex=0; ivertex < cell->n_vertices();ivertex++){
          const auto &vertex = cell->vertex(ivertex);
//          std::cout<<vertex<<std::endl;

          //TODO: Loop ?
          coords[3*ivertex+0]=vertex[0];
          if constexpr (dim >=2) {
            coords[3*ivertex+1]=vertex[1];
          }
          if constexpr (dim >=3) {
            coords[3*ivertex+2]=0;
          }
        }
        t8_cmesh_set_tree_vertices (cmesh, t8_index, coords.data(), cell->n_vertices());

        for(unsigned int iface=0; iface< cell->n_faces();iface++){
          if(cell->neighbor_index(iface)==-1){
            continue;
          }
          const auto &t8_neighbor_index = coarse_cell_to_t8code_tree_permutation[cell->neighbor_index(iface)];
          if(t8_index<t8_neighbor_index){
              const auto ineighface = cell->neighbor_of_neighbor(iface);
              int t8_iface = dealii::internal::parallel::distributed::dealii_to_t8_faces[eclass][iface];
              t8_eclass_t neigh_eclass = t8_eclass_from_reference_cell(cell->neighbor(iface)->reference_cell());
              int t8_ineighface = dealii::internal::parallel::distributed::dealii_to_t8_faces[neigh_eclass][ineighface];

              unsigned int orientation = (cell->face_orientation(iface) != cell->neighbor(iface)->face_orientation(ineighface));//cell->combined_face_orientation(iface);
              if(eclass == T8_ECLASS_TRIANGLE && iface == 2){
//                std::cout<<"switched orientation because own cell is triangle and on face 2"<<std::endl;
                orientation = !orientation;
              }
              if(neigh_eclass == T8_ECLASS_TRIANGLE && ineighface == 2){
//                std::cout<<"switched orientation because neighbor cell is triangle and on face 2"<<std::endl;
                orientation = !orientation;
              }

//              std::cout<<"added face join from cell "<<t8_index<<" face "<<t8_iface<<" to cell "<<t8_neighbor_index<<" face " << t8_ineighface <<" with orientation "<<orientation <<std::endl;
              std::cout<<cell->face_orientation(iface)<<" "<<cell->neighbor(iface)->face_orientation(ineighface)<<std::endl;
//              std::cout<<"t8_cmesh_set_join(cmesh,"<< t8_index<<", "<<t8_neighbor_index<<", "<< t8_iface <<", "<< t8_ineighface<<", "<<orientation <<")"<<std::endl;
              t8_cmesh_set_join(cmesh, t8_index, t8_neighbor_index, t8_iface, t8_ineighface, orientation);
            }
        }

        std::vector<t8_gloidx_t> vertex_list(cell->n_vertices());
        for(unsigned int ivertex=0; ivertex< cell->n_vertices();ivertex++){
          vertex_list[ivertex] = cell->vertex_index(ivertex);
          std::cout << "local vertex "<<ivertex<<" connected to global vertex " << cell->vertex_index(ivertex);
        }
        t8_cmesh_set_global_vertices_of_tree(cmesh, t8_index, vertex_list.data(), cell->n_vertices());
      }
      t8_cmesh_commit(cmesh, this->mpi_communicator);

      scheme_collection = t8_scheme_new_default();
      parallel_forest   = t8_forest_new_uniform(
        cmesh, scheme_collection, 0, 1, this->mpi_communicator);
      t8_forest_set_user_data(parallel_forest, this);

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


      // set all cells to artificial. we will later set it to the correct
      // subdomain in match_tree_recursively
      for (const auto &cell : this->cell_iterators_on_level(0))
        cell->recursively_set_subdomain_id(numbers::artificial_subdomain_id);

      int loop_iter = 0;
      do
        {
//          std::cout<<"loop iter: "<<loop_iter<<std::endl;
          loop_iter++;
          for (const auto &cell : this->cell_iterators_on_level(0))
            {
                dealii::internal::t8code::types::eclass eclass =
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
                                                        eclass,1,
                                                        &t8code_coarse_cell);

                  dealii::internal::t8code::init_root(parallel_forest,
                                                      eclass,
                                                      t8code_coarse_cell);

                  match_tree_recursively<dim, spacedim>(*tree,
                                                        cell,
                                                        t8code_coarse_cell,
                                                        parallel_forest,
                                                        this->my_subdomain,
                                                        0);
                  dealii::internal::t8code::element_destroy(
                    parallel_forest, eclass, 1,&t8code_coarse_cell);
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


          this->dealii::Triangulation<dim, spacedim>::
                  prepare_coarsening_and_refinement();
          //TODO? enforce periodic balance

          std::cout << "mesh flags:" <<std::endl;

          mesh_changed =
            std::any_of(this->begin_active(),
                        active_cell_iterator{this->end()},
                        [loop_iter,this](const CellAccessor<dim, spacedim> &cell) {
                          return (cell.refine_flag_set() ||
                                 cell.coarsen_flag_set());
                        });
//          std::cout << "mesh_changed" << mesh_changed <<std::endl;
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

            for (const auto &cell: this->active_cell_iterators()){
               cell->set_material_id(cell->subdomain_id());
            }

            

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
            update_cell_relations();
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







  /**
   * A data structure that we use to store which cells (indicated by
   * internal::t8code::types::element objects) shall be refined and which
   * shall be coarsened.
   */
  template <int dim, int spacedim>
  class RefineAndCoarsenList
  {
  public:
    RefineAndCoarsenList(const Triangulation<dim, spacedim> &triangulation,
    dealii::internal::t8code::types::forest forest,
                         const std::vector<types::global_dof_index>
                           &t8code_tree_to_coarse_cell_permutation,
                         const types::subdomain_id my_subdomain);

    /**
     * A callback function that we pass to the t8code data structures when a
     * forest is to be refined. The t8code functions call it back with a tree
     * (the index of the tree that grows out of a given coarse cell) and a
     * refinement path from that coarse cell to a terminal/leaf cell. The
     * function returns whether the corresponding cell in the deal.II
     * triangulation has the refined flag set.
     */

    static int
    adapt_callback(
      typename dealii::internal::t8code::types::forest    forest,
      typename dealii::internal::t8code::types::forest    forest_from,
      typename dealii::internal::t8code::types::locidx    which_tree,
      typename dealii::internal::t8code::types::eclass    eclass,
      typename dealii::internal::t8code::types::locidx    lelement_id,
      const typename dealii::internal::t8code::types::scheme_collection    *scheme,
      const int is_family,
      const int num_elements,
      typename dealii::internal::t8code::types::element *elements[]);




  private:
    std::vector<std::vector<int> >adapt_list;

    void
    build_list(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const typename dealii::internal::t8code::types::forest        forest,
      const typename dealii::internal::t8code::types::locidx        which_tree,
      const typename dealii::internal::t8code::types::element       *t8code_cell,
      const types::subdomain_id                                   myid, const int dealii_type = 0);
  };



  template <int dim, int spacedim>
  RefineAndCoarsenList<dim, spacedim>::RefineAndCoarsenList(
    const Triangulation<dim, spacedim> &triangulation,
    dealii::internal::t8code::types::forest forest,
    const std::vector<types::global_dof_index>
                             &t8code_tree_to_coarse_cell_permutation,
    const types::subdomain_id my_subdomain)
  {

    t8_locidx_t n_coarse_elements = t8_forest_get_num_local_trees(forest);
    adapt_list.clear();
    adapt_list.resize(n_coarse_elements);
    for(t8_locidx_t itree=0; itree <n_coarse_elements; itree++){
      t8_locidx_t n_fine_elements_in_coarse_element = t8_forest_get_tree_num_elements(forest, itree);
      adapt_list[itree].clear();
      adapt_list[itree].reserve(n_fine_elements_in_coarse_element);
    }


    // now build the lists of cells that are flagged. note that t8code will
    // traverse its cells in the order in which trees appear in the
    // forest. this order is not the same as the order of coarse cells in the
    // deal.II Triangulation because we have translated everything by the
    // coarse_cell_to_t8code_tree_permutation permutation. in order to make
    // sure that the output array is already in the correct order, traverse
    // our coarse cells in the same order in which t8code will:
    const auto first_local = t8_forest_get_first_local_tree_id(forest);
    typename dealii::internal::t8code::types::element *t8code_cell;
    for (int c = 0; c < n_coarse_elements; ++c)
      {
        unsigned int global_coarse_cell_index =
          t8code_tree_to_coarse_cell_permutation[c + first_local];
        
        typename dealii::internal::t8code::types::eclass eclass = t8_forest_get_tree_class(forest, c);
        dealii::internal::t8code::element_new(forest, eclass, 1 , &t8code_cell);

        const typename Triangulation<dim, spacedim>::cell_iterator cell(
          &triangulation, 0, global_coarse_cell_index);

        dealii::internal::t8code::init_root(forest, eclass, t8code_cell);
        build_list(cell, forest, c, t8code_cell, my_subdomain, 0);
        dealii::internal::t8code::element_destroy(forest, eclass, 1 , &t8code_cell);
        // std::cout<<"root "<<c<<":"<<std::endl;
        // for (const auto &entry : adapt_list[c]){
        //   std::cout<<entry<<std::endl;
        // }
      }
  }



  template <int dim, int spacedim>
  void
  RefineAndCoarsenList<dim, spacedim>::build_list(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const typename dealii::internal::t8code::types::forest forest,
    const typename dealii::internal::t8code::types::locidx  which_tree,
    const typename dealii::internal::t8code::types::element *t8code_cell,
    const types::subdomain_id                       my_subdomain,
    const int dealii_type)
  {
//    std::cout<<"call build list recursively on tree: "<< which_tree<<", dealii_type"<<dealii_type<<", cellid: "<<cell->id()<<std::endl;

    if (cell->is_active())
      {
        if (cell->subdomain_id() == my_subdomain)
          {
            if (cell->refine_flag_set()){
              adapt_list[which_tree].push_back(1);
              std::cout<<"set refine flag for cell "<<cell->id()<<std::endl;
            }
            else if (cell->coarsen_flag_set())
              adapt_list[which_tree].push_back(-1);
            else 
              adapt_list[which_tree].push_back(0);

          }
      }
    else
      {
        typename dealii::internal::t8code::types::element*
          t8code_child[GeometryInfo<dim>::max_children_per_cell];
        typename dealii::internal::t8code::types::eclass eclass = t8_forest_get_eclass(forest, which_tree);

        dealii::internal::t8code::element_new(forest, eclass, GeometryInfo<dim>::max_children_per_cell, t8code_child);
        dealii::internal::t8code::element_children(forest, eclass, t8code_cell,
                                                            t8code_child);

        bool any_child_coarsen_flag_set=false;
        bool all_child_coarsen_flag_set=true;
        for (unsigned int c = 0; c < GeometryInfo<dim>::max_children_per_cell;
             ++c)
          {
            int deal_child_id = (eclass == T8_ECLASS_TRIANGLE) ? dealii::internal::parallel::distributed::t8code_to_deal_children[dealii_type][c] : c;
            int deal_child_type = (deal_child_id % 4 == 3)? (dealii_type + 1) % 6 : dealii_type;
            // std::cout<<"deal_child_id: "<<deal_child_id<<", t8_child_id: "<<c<<std::endl;
            build_list(cell->child(deal_child_id), forest, which_tree, t8code_child[c], my_subdomain, deal_child_type);
            if(cell->child(deal_child_id)->is_active() && cell->child(deal_child_id)->coarsen_flag_set()){
              any_child_coarsen_flag_set = true;
            }else{
              all_child_coarsen_flag_set = false;
            }
          }
        dealii::internal::t8code::element_destroy(forest, eclass, GeometryInfo<dim>::max_children_per_cell, t8code_child);
        if(any_child_coarsen_flag_set && !all_child_coarsen_flag_set){
          dealii::internal::t8code::types::locidx num_elements_in_tree = adapt_list[which_tree].size();
          unsigned int max_delete = std::min(GeometryInfo<dim>::max_children_per_cell, (unsigned int) num_elements_in_tree);
          for (unsigned int c = 0; c < max_delete;
             ++c)
          {
            if (adapt_list[which_tree][num_elements_in_tree-1-c] == -1){
              adapt_list[which_tree][num_elements_in_tree-1-c] = 0;
            }
          }
        }
      }
  }


  template <int dim, int spacedim>
  int
  RefineAndCoarsenList<dim, spacedim>::adapt_callback(
      typename dealii::internal::t8code::types::forest    forest,
      typename dealii::internal::t8code::types::forest    forest_from,
      typename dealii::internal::t8code::types::locidx    which_tree,
      typename dealii::internal::t8code::types::eclass    eclass,
      typename dealii::internal::t8code::types::locidx    lelement_id,
      const typename dealii::internal::t8code::types::scheme_collection    *scheme,
      const int is_family,
      const int num_elements,
      typename dealii::internal::t8code::types::element *elements[])
  {
    (void) forest;
    (void) eclass;
    (void) scheme;
    (void) is_family;
    (void) num_elements;
    (void) elements;
    RefineAndCoarsenList<dim, spacedim> *this_object =
      reinterpret_cast<RefineAndCoarsenList<dim, spacedim> *>(t8_forest_get_user_data(forest_from));
      // std::cout<<"found value" << this_object->adapt_list[which_tree][lelement_id] <<" for tree "<<which_tree<<" and elem "<<lelement_id<<std::endl; 
      return this_object->adapt_list[which_tree][lelement_id];
  }


    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    void Triangulation<dim, spacedim>::execute_coarsening_and_refinement()
    {

      this->prepare_coarsening_and_refinement();

      // signal that refinement is going to happen
      this->signals.pre_distributed_refinement();

      // now do the work we're supposed to do when we are in charge
      // make sure all flags are cleared on cells we don't own, since nothing
      // good can come of that if they are still around
      for (const auto &cell : this->active_cell_iterators())
        if (cell->is_ghost() || cell->is_artificial())
          {
            cell->clear_refine_flag();
            cell->clear_coarsen_flag();
          }


      // count how many cells will be refined and coarsened, and allocate that
      // much memory
      RefineAndCoarsenList<dim, spacedim> refine_and_coarsen_list(
        *this, parallel_forest, t8code_tree_to_coarse_cell_permutation, this->my_subdomain);

      std::cout<< "!!!!forest set balance before adapt: "<< parallel_forest->set_balance<<" !!!"<<std::endl;


      std::string fileprefix_o = "original";
      t8_forest_write_vtk_ext (parallel_forest, fileprefix_o.c_str(), 1, 1,
                         1, 1, 0,
                         0, 1, 0, nullptr);

      // copy refine and coarsen flags into t8code and execute the refinement
      // and coarsening. this uses the refine_and_coarsen_list just built,
      // which is communicated to the callback functions through
      // t8code's user_data object
      Assert(t8_forest_get_user_data(parallel_forest) == this, ExcInternalError());
      t8_forest_set_user_data(parallel_forest, &refine_and_coarsen_list);
      parallel_forest = dealii::internal::t8code::adapt(
        parallel_forest,
        &RefineAndCoarsenList<dim, spacedim>::adapt_callback);

      // reset the pointer
      t8_forest_set_user_data(parallel_forest, this);

      std::cout<< "!!!!forest set balance after adapt: "<< parallel_forest->set_balance<<" !!!"<<std::endl;
      std::string fileprefix_a = "adapt";
      t8_forest_write_vtk_ext (parallel_forest, fileprefix_a.c_str(), 1, 1,
                         1, 1, 0,
                         0, 1, 0, nullptr);

      // enforce 2:1 hanging node condition
      parallel_forest = dealii::internal::t8code::balance(parallel_forest);
      Assert(t8_forest_get_user_data(parallel_forest) == this, ExcInternalError());
      std::cout<< "!!!!forest set balance after balance: "<< parallel_forest->set_balance<<" !!!"<<std::endl;
      std::string fileprefix_b = "balanced";
      t8_forest_write_vtk_ext (parallel_forest, fileprefix_b.c_str(), 1, 1,
                         1, 1, 0,
                         0, 1, 0, nullptr);


      // since refinement and/or coarsening on the parallel forest
      // has happened, we need to update the quadrant cell relations

      update_cell_relations();

      // signals that parallel_forest has been refined and cell relations have
      // been updated
      this->signals.post_p4est_refinement();

      t8_forest_t old_forest; //remember old forest for data exchange
      if (this->cell_attached_data.n_attached_data_sets > 0)
        {
          old_forest = parallel_forest;
          t8_forest_ref(old_forest);
        }

      parallel_forest = dealii::internal::t8code::partition(parallel_forest, T8_GHOST_VERTICES);
      std::string fileprefix_p = "partitioned";
      Assert(t8_forest_get_user_data(parallel_forest) == this, ExcInternalError());
      std::cout<< "!!!!forest set balance after partition: "<< parallel_forest->set_balance<<" !!!"<<std::endl;
      t8_forest_write_vtk_ext (parallel_forest, fileprefix_p.c_str(), 1, 1,
                         1, 1, 0,
                         0, 1, 0, nullptr);


      // pack data before triangulation gets updated
      if (this->cell_attached_data.n_attached_data_sets > 0)
        {
          this->data_serializer.pack_data(
            this->local_cell_relations,
            this->cell_attached_data.pack_callbacks_fixed,
            this->cell_attached_data.pack_callbacks_variable,
            this->get_mpi_communicator());
        }

      // finally copy back from local part of tree to deal.II
      // triangulation. before doing so, make sure there are no refine or
      // coarsen flags pending
      for (const auto &cell : this->active_cell_iterators())
        {
          cell->clear_refine_flag();
          cell->clear_coarsen_flag();
        }

      try
        {
          copy_local_forest_to_triangulation();
        }
      catch (const typename Triangulation<dim>::DistortedCellList &)
        {
          // the underlying triangulation should not be checking for distorted
          // cells
          DEAL_II_ASSERT_UNREACHABLE();
        }

      // transfer data after triangulation got updated
      if (this->cell_attached_data.n_attached_data_sets > 0)
        {
          this->execute_transfer(parallel_forest,
                                 old_forest);
          t8_forest_unref(&old_forest);

          // also update the CellStatus information on the new mesh
          this->data_serializer.unpack_cell_status(this->local_cell_relations);
        }


      this->update_periodic_face_map();
      this->update_number_cache();

      // signal that refinement is finished
      this->signals.post_distributed_refinement();//
    }
    



    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    void Triangulation<dim, spacedim>::execute_transfer(
      const typename dealii::internal::t8code::types::forest
        parallel_forest,
      const typename dealii::internal::t8code::types::forest
        old_forest)
    {
      Assert(this->data_serializer.sizes_fixed_cumulative.size() > 0,
             ExcMessage("No data has been packed!"));

      // Resize memory according to the data that we will receive.
      this->data_serializer.dest_data_fixed.resize(
        parallel_forest->local_num_elements *
        this->data_serializer.sizes_fixed_cumulative.back());


      sc_array_t src_data_view, dest_data_view;
      sc_array_init_data(&src_data_view, this->data_serializer.src_data_fixed.data(), this->data_serializer.sizes_fixed_cumulative.back(), old_forest->local_num_elements);
      sc_array_init_data(&dest_data_view, this->data_serializer.dest_data_fixed.data(), this->data_serializer.sizes_fixed_cumulative.back(), parallel_forest->local_num_elements);

      t8_forest_partition_data(old_forest, parallel_forest, &src_data_view, &dest_data_view);

      // Release memory of previously packed data.
      this->data_serializer.src_data_fixed.clear();
      this->data_serializer.src_data_fixed.shrink_to_fit();
      if (this->data_serializer.variable_size_data_stored)
        {
          // Resize memory according to the data that we will receive.
          this->data_serializer.dest_sizes_variable.resize(
            parallel_forest->local_num_elements);
      sc_array_t src_size_view, dest_size_view;
      sc_array_init_data(&src_size_view, this->data_serializer.src_sizes_variable.data(), sizeof(int), old_forest->local_num_elements);
      sc_array_init_data(&dest_size_view, this->data_serializer.dest_sizes_variable.data(), sizeof(int), parallel_forest->local_num_elements);

      t8_forest_partition_data(old_forest, parallel_forest, &src_size_view, &dest_size_view);

          // Resize memory according to the data that we will receive.
          this->data_serializer.dest_data_variable.resize(
            std::accumulate(this->data_serializer.dest_sizes_variable.begin(),
                            this->data_serializer.dest_sizes_variable.end(),
                            std::vector<int>::size_type(0)));

      int max_data_size_loc =(this->data_serializer.src_sizes_variable.size() ? *std::max_element(this->data_serializer.src_sizes_variable.begin(),
                            this->data_serializer.src_sizes_variable.end()) : 0 ); //collective?

      int max_data_size;
      MPI_Allreduce(&max_data_size_loc, &max_data_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      std::cout <<"Max data size: "<<max_data_size<<", loc: "<<max_data_size_loc<<std::endl;

      sc_array_t src_data_padded, dest_data_padded;
      sc_array_init_count(&src_data_padded, max_data_size, old_forest->local_num_elements);
      sc_array_init_count(&dest_data_padded, max_data_size, parallel_forest->local_num_elements);


      std::cout<<"src representation:"<<std::endl;
      for(const auto &char_rep : this->data_serializer.src_data_variable){
        std::cout<<(int)char_rep<<" ";
      }
      std::cout<<std::endl;

//memcpy src_data_variable in padded
      size_t cumulative_index=0;
      for(int idata = 0; idata< old_forest->local_num_elements; idata++){
        std::cout<<"memcpy to, idata "<<idata<<", cumulative_index "<<cumulative_index<<std::endl;
        if(this->data_serializer.src_sizes_variable[idata]){

          memcpy(sc_array_index_int(&src_data_padded, idata) , &(this->data_serializer.src_data_variable[cumulative_index]), this->data_serializer.src_sizes_variable[idata]);
        }
        cumulative_index += this->data_serializer.src_sizes_variable[idata];
      }

//communicate padded data
      t8_forest_partition_data(old_forest, parallel_forest, &src_data_padded, &dest_data_padded);

//memcpy dest_data_variable from padded
      cumulative_index = 0;
      for(int idata = 0; idata< parallel_forest->local_num_elements; idata++){
        std::cout<<"memcpy from, idata "<<idata<<", cumulative_index "<<cumulative_index<<std::endl;
        if(this->data_serializer.dest_sizes_variable[idata]){

          memcpy( &(this->data_serializer.dest_data_variable[cumulative_index]), sc_array_index_int(&dest_data_padded, idata), this->data_serializer.dest_sizes_variable[idata]);
        }
        cumulative_index += this->data_serializer.dest_sizes_variable[idata];
      }
      std::cout<<"dest representation:"<<std::endl;
      for(const auto &char_rep : this->data_serializer.dest_data_variable){
        std::cout<<(int)char_rep<<" ";
      }
      std::cout<<std::endl;


//Release memory of padded data
          sc_array_reset(&src_data_padded);
          sc_array_reset(&dest_data_padded);
          // Release memory of previously packed data.
          this->data_serializer.src_sizes_variable.clear();
          this->data_serializer.src_sizes_variable.shrink_to_fit();
          this->data_serializer.src_data_variable.clear();
          this->data_serializer.src_data_variable.shrink_to_fit();
        }
    }


    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    void Triangulation<dim, spacedim>::update_cell_relations()
    {
      // std::cout<<"enter ucr"<<std::endl;

      // reorganize memory for local_cell_relations
      this->local_cell_relations.resize(parallel_forest->local_num_elements);
      this->local_cell_relations.shrink_to_fit();

      // recurse over p4est
      for (const auto &cell : this->cell_iterators_on_level(0))
        {
          // skip coarse cells that are not ours
          if (tree_exists_locally<dim, spacedim>(
                parallel_forest,
                coarse_cell_to_t8code_tree_permutation[cell->index()]) == false)
            continue;

          // initialize auxiliary top level p4est quadrant
          typename dealii::internal::t8code::types::element
            *root;
          dealii::internal::t8code::types::eclass eclass = t8_eclass_from_reference_cell(cell->reference_cell());
          dealii::internal::t8code::element_new(parallel_forest, eclass, 1, &root);
          dealii::internal::t8code::init_root(parallel_forest, eclass, root);

          t8_locidx_t ltreeid = t8_forest_get_local_id(parallel_forest, coarse_cell_to_t8code_tree_permutation[cell->index()]);

          update_cell_relations_recursively<dim, spacedim>(parallel_forest,
            this->local_cell_relations, ltreeid, cell, root, 0);
          dealii::internal::t8code::element_destroy(parallel_forest, eclass, 1, &root);
        }
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
