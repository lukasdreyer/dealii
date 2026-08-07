// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2016 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


#include "deal.II/base/exception_macros.h"
#include <deal.II/distributed/t8code_wrappers.h>
#include <deal.II/distributed/tria.h>
#include <t8_forest/t8_forest_adapt.h>
#include <t8_forest/t8_forest_general.h>
#include <t8_geometry/t8_geometry_implementations/t8_geometry_linear.hxx>

DEAL_II_NAMESPACE_OPEN

#ifdef DEAL_II_WITH_T8CODE
#include <deal.II/distributed/p4est_wrappers.h>
#  include <t8_schemes/t8_scheme.hxx>
#  include <t8_forest/t8_forest_ghost.h>
#  include <t8_forest/t8_forest_types.h>
#  include <t8_cmesh/t8_cmesh.h>
#  include <t8_cmesh/t8_cmesh.hxx>

namespace internal
{
  namespace t8code
  {

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

      template <int dim, int spacedim>
      typename types<dim>::connectivity dealii_to_connectivity(typename ::dealii::parallel::distributed::Triangulation<dim,spacedim> *tria){
      t8_cmesh_t cmesh;
      //TODO!! coarse mesh permutation!!!
      t8_cmesh_init(&cmesh);
      t8_cmesh_register_geometry<t8_geometry_linear> (cmesh);
      const auto &coarse_cell_permutation = tria->get_p4est_tree_to_coarse_cell_permutation();
      for(const auto & cell:tria->active_cell_iterators()){ 
        const auto t8_index = coarse_cell_permutation[cell->index()];
        t8_eclass_t eclass = t8_eclass_from_reference_cell(cell->reference_cell());
        t8_cmesh_set_tree_class (cmesh, t8_index, eclass);
        std::vector<double> coords(3*cell->n_vertices());
        for(unsigned int ivertex=0; ivertex < cell->n_vertices();ivertex++){
          const auto &vertex = cell->vertex(ivertex);
          for(unsigned int idim=0;idim<dim;idim++){
            coords[3*ivertex+idim]=vertex[idim];
          }
        }
        t8_cmesh_set_tree_vertices (cmesh, t8_index, coords.data(), cell->n_vertices());

        for(unsigned int iface=0; iface< cell->n_faces();iface++){
          if(cell->neighbor_index(iface)==-1){
            continue;
          }

          
          const auto &t8_neighbor_index = coarse_cell_permutation[cell->neighbor_index(iface)];
          if(t8_index<t8_neighbor_index){
              const auto ineighface = cell->neighbor_of_neighbor(iface);
              int t8_iface = dealii_to_t8_faces[eclass][iface];
              t8_eclass_t neigh_eclass = t8_eclass_from_reference_cell(cell->neighbor(iface)->reference_cell());
              int t8_ineighface = dealii_to_t8_faces[neigh_eclass][ineighface];

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
        std::vector<t8_gloidx_t> edge_list(cell->n_lines());
        for(unsigned int iedge=0; iedge< cell->n_lines();iedge++){
          edge_list[iedge] = cell->line_index(iedge);
          // std::cout << "local edge "<<iedge<<" connected to global edge " << cell->line_index(iedge);
        }
        t8_cmesh_set_global_edges_of_tree(cmesh, t8_index, edge_list.data(), cell->n_lines());
      }
      t8_cmesh_commit(cmesh, tria->mpi_communicator);
      return cmesh;
    }

    int adapt_from_vec(t8_forest_t , t8_forest_t forest_from, t8_locidx_t which_tree,
                                  const t8_eclass_t , t8_locidx_t lelement_id, const t8_scheme_c *,
                                  const int , const int , t8_element_t *[]){
                                    std::vector<bool> *adapt_vec = (std::vector<bool> *)t8_forest_get_user_data(forest_from);
                                    const int idata = t8_forest_get_tree_element_offset(forest_from, which_tree) + lelement_id;
                                    return (*adapt_vec)[idata];
                                  }


    template <int dim, int spacedim>
    typename types<dim>::forest *
    adapt(typename types<dim>::forest  *parallel_forest,
          Triangulation<dim, spacedim> *triangulation){

      // count how many cells will be refined and coarsened, and allocate that
      // much memory
      std::vector<bool> adapt_list;
      // copy refine and coarsen flags into p4est and execute the refinement
      // and coarsening. this uses the refine_and_coarsen_list just built,
      // which is communicated to the callback functions through
      // p4est's user_pointer object
      Assert(forest_get_user_pointer<dim>(parallel_forest) == triangulation,
             ExcInternalError());
      forest_set_user_pointer<dim>(parallel_forest, &adapt_list);
      
      t8_forest_t new_forest;

      t8_forest_init (&new_forest);
      t8_forest_set_adapt (new_forest, parallel_forest, adapt_from_vec, false);
      t8_forest_set_ghost (new_forest, true, T8_GHOST_VERTICES);
      t8_forest_commit (new_forest);

      // reset the pointer
      forest_set_user_pointer<dim>(parallel_forest, triangulation);

      return parallel_forest;
          }

    template <int dim>
    typename types<dim>::forest* balance_full(typename types<dim>::forest *forest){
      t8_forest_t new_forest;
      t8_forest_init (&new_forest);
      t8_forest_set_balance (new_forest, forest, 1);
      t8_forest_set_user_data(new_forest, t8_forest_get_user_data(forest));
      t8_forest_commit (new_forest);
      return new_forest;
    }

    template <int dim>
    typename types<dim>::forest* partition(typename types<dim>::forest *forest, typename types<dim>::weight ){
  t8_forest_t new_forest;
  t8_forest_init (&new_forest);
  t8_forest_set_partition (new_forest, forest, 1);
  t8_forest_set_ghost (new_forest, 1, T8_GHOST_VERTICES);
  t8_forest_set_user_data(new_forest, t8_forest_get_user_data(forest));
  t8_forest_commit (new_forest);
  return new_forest;
    }

        template <int dim>
    void
    element_children(const typename types<dim>::forest *forest,
                     typename types<dim>::eclass eclass,
                     const typename types<dim>::element *element,
                     typename types<dim>::element       *children)
    {
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t >(forest));
      int num_children = scheme->element_get_num_children(eclass, (const t8_element_t*)element);
        for(int ichild = 0; ichild<num_children; ichild++){
          scheme->element_get_child(eclass, (const t8_element_t*)element, ichild,  (t8_element_t*)(children+ichild));
        }
    }

    template <int dim>
    void
    init_coarse_element(const typename types<dim>::forest   *forest,
              typename types<dim>::eclass   eclass,
              typename types<dim>::element *element)
    {
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t >(forest));
      scheme->element_get_level(eclass, (const t8_element_t*)element);
      scheme->set_to_root(eclass, (t8_element_t*)element);
    }

    template <int dim>
    int
    element_level(const typename types<dim>::forest   *forest,
                  typename types<dim>::eclass   eclass,
                  const typename types<dim>::element *element)
    {
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t >(forest));
      return scheme->element_get_level(eclass, reinterpret_cast<const t8_element_t *>(element));
    }

    template <int dim>
    bool
    element_is_equal(const typename types<dim>::forest   *forest,
                  typename types<dim>::eclass   eclass,
                  const typename types<dim>::element *element1,
                  const typename types<dim>::element *element2)
    {
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t >(forest));
      return scheme->element_is_equal(eclass, (const t8_element_t*)element1, (const t8_element_t*)element2);
    }


    template <int dim>
    void
    element_child(const typename types<dim>::forest         *forest,
                     typename types<dim>::eclass         eclass,
                     const typename types<dim>::element *element,
                     int childid,
                     typename types<dim>::element      *child)
    {
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t >(forest));
      scheme->element_get_child(eclass, (const t8_element_t*)element, childid, ( t8_element_t*)child);
    }


    template <int dim>
    bool
    cell_exists_in_tree(const typename types<dim>::forest   *forest,
                          typename types<dim>::tree     tree,
                          const typename types<dim>::element *element)
{

  typename types<dim>::eclass eclass = tree->eclass;
  typename types<dim>::scheme_collection *scheme =  t8_forest_get_scheme(const_cast<t8_forest_t >(forest));
//      scheme->t8_element_debug_print(eclass,t8code_cell);
auto compare_lambda = [scheme,eclass](auto x, auto y) {
        return scheme->element_compare(eclass, (t8_element_t*)x, (t8_element_t*)y) < 0;
      };

      
      return std::binary_search(t8_element_array_begin(&tree->leaf_elements),
                             t8_element_array_end(&tree->leaf_elements),
                             element,
                             compare_lambda);
                            }

    template <int dim>
    bool
    element_overlaps_tree(const typename types<dim>::forest   *forest,
                          typename types<dim>::tree     tree,
                          const typename types<dim>::element *element)
    {
      typename types<dim>::eclass         eclass = tree->eclass;
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t >(forest));
      typename types<dim>::element element_last_desc;
      bool            element_overlaps = true;

      const unsigned int maxlevel = scheme->get_maxlevel(eclass);
      scheme->element_get_last_descendant(eclass, (const t8_element_t*)element,
                                                (t8_element_t*)&element_last_desc,
                                                maxlevel);
      if (scheme->element_compare(eclass, (t8_element_t*)&element_last_desc,
                                            tree->first_desc) < 0)
        element_overlaps = false;

      /* check if q is after the last tree quadrant */
      if (scheme->element_compare(eclass, tree->last_desc, (const t8_element_t*)element) < 0)
        element_overlaps = false;

      return element_overlaps;
    }

    template <int dim>
    int
    element_ancestor_id(const typename types<dim>::forest   *forest,
                        typename types<dim>::eclass   eclass,
                        const typename types<dim>::element *element,
                        int             level)
    {
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t >(forest));
      return scheme->element_get_ancestor_id(eclass, (const t8_element_t*)element, level);
    }

    template <int dim>
    typename types<dim>::locidx
    leaf_index_in_tree(const typename types<dim>::forest  *forest,
                       const typename types<dim>::locidx   ltreeid,
                       const typename types<dim>::element *leaf)
    {
      return t8_forest_element_leaf_index_in_tree(const_cast<t8_forest_t >(forest), (const t8_element_t*)leaf, ltreeid);
    }

    template <int dim>
    typename types<dim>::eclass
    get_ghost_eclass(const typename types<dim>::forest *forest,
                     const typename types<dim>::locidx ghost_treeid)
    {
      return t8_forest_ghost_get_tree_class(const_cast<t8_forest_t >(forest), ghost_treeid);
    }

        template <int dim>
    types<dim>::gloidx
    tree_get_offset(const typename types<dim>::tree tree){
      return tree->elements_offset;
    }

    template <int dim>
    typename types<dim>::tree
    forest_get_tree(const typename types<dim>::forest *forest,
                    const typename types<dim>::locidx  ltreeid)
    {
      return t8_forest_get_tree(const_cast<t8_forest_t >(forest), ltreeid);
    }

    template <int dim>
    typename types<dim>::locidx
    get_num_leafs(const typename types<dim>::forest *forest){
      return t8_forest_get_local_num_leaf_elements(const_cast<t8_forest_t >(forest));
    }


    template <int dim>
    typename types<dim>::ghost* ghost_new(typename types<dim>::forest      *forest)
                                                {
                                                  DEAL_II_NOT_IMPLEMENTED();
                                                  return forest->ghosts;
                                                }

    template <int dim>
    void ghost_destroy(typename types<dim>::ghost **){
      DEAL_II_NOT_IMPLEMENTED();
    }

           template <int dim> void
      forest_destroy(typename types<dim>::forest **){
        DEAL_II_NOT_IMPLEMENTED();
      }

      template <int dim>   void
      vtk_write_file(const typename types<dim>::forest *, const char *){
        DEAL_II_NOT_IMPLEMENTED();
      }


        template <int dim>
        ::dealii::types::subdomain_id
    comm_find_owner(const typename types<dim>::forest  *,
                    const typename types<dim>::locidx   ,
                    const typename types<dim>::element *,
                    const ::dealii::types::subdomain_id                           ){
                      DEAL_II_NOT_IMPLEMENTED();
                    }

  } // namespace t8code
} // namespace internal

#endif // DEAL_II_WITH_T8CODE

/*-------------- Explicit Instantiations -------------------------------*/
#include "distributed/t8code_wrappers.inst"


DEAL_II_NAMESPACE_CLOSE