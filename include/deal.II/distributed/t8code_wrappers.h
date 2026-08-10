// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2016 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

#ifndef dealii_t8code_wrappers_h
#define dealii_t8code_wrappers_h

#include <deal.II/base/config.h>

#include "deal.II/base/types.h"

#include "deal.II/grid/tria.h"
#include <t8_schemes/t8_standalone/t8_standalone_elements.hxx>

#ifdef DEAL_II_WITH_T8CODE
#  include <t8.h>
//#  include <t8_cmesh/t8_cmesh.h>
#  include <t8_cmesh/t8_cmesh_internal/t8_cmesh_types.h>
#  include <t8_element.h>
#  include <t8_forest/t8_forest.h>
#  include <t8_forest/t8_forest_general.h>
#  include <t8_forest/t8_forest_types.h>
#  include <t8_schemes/t8_scheme.hxx>
#include <t8_forest/t8_forest_ghost.h>
#include <t8_forest/t8_forest_partition.h>
#include <t8_data/t8_element_array_iterator.hxx>

DEAL_II_NAMESPACE_OPEN
namespace internal
{
  namespace t8code
  {
    using t8code_weight_dummy_t = int (*)(t8_forest_t   forest,
                                          long          local_tree, // TODO
                                          t8_element_t *element);
    template <int dim>
    struct types
    { // TODO: check which should be pointer, and which not
      using connectivity      = t8_cmesh_t;
      using forest            = struct t8_forest;
      using tree              = t8_tree_t;
      using element           = t8_standalone_element<dim == 3 ? (t8_eclass_t)4 : (t8_eclass_t)dim>; //TODO: make standalone dependent on dim!
      using element_coord     = t8_element_coord;
      using eclass            = t8_eclass_t;
      using scheme_collection = const t8_scheme;
      using locidx            = t8_locidx_t;
      using gloidx            = t8_gloidx_t;
      using topidx            = t8_gloidx_t;
      using ghost             = struct t8_forest_ghost;
      using ghost_type        = t8_ghost_type_t;
      using balance_type      = t8_ghost_type_t;
      using weight            = t8code_weight_dummy_t;
      using element_array     = t8_element_array_t *;
    };

    template <int dim>
    struct functions
    {

      static types<dim>::forest
      adapt_balance_partition(types<dim>::forest,
                              t8_forest_adapt_t adapt_callback,
                              t8_ghost_type_t   ghost_type);


      static void
      connectivity_destroy(types<dim>::connectivity *connectivity);

      static std::size_t
      forest_memory_used(const types<dim>::forest *forest);
      static std::size_t
      connectivity_memory_used(const types<dim>::connectivity *cmesh);


      static unsigned int
      checksum(const types<dim>::forest *forest);


      static void
      element_destroy(const types<dim>::forest *forest,
                      types<dim>::eclass        eclass,
                      types<dim>::locidx        length,
                      types<dim>::element      *element);

     static unsigned int
      get_max_level(const typename types<dim>::forest *parallel_forest)
      {
        return t8_forest_get_maxlevel(
          const_cast<types<dim>::forest *>(parallel_forest));
      };

      static types<dim>::forest *
      copy_forest(types<dim>::forest *input, int copy_data);




    }; // struct functions

      template <int dim, int spacedim>
      typename types<dim>::connectivity dealii_to_connectivity(typename ::dealii::parallel::distributed::Triangulation<dim,spacedim> *tria);

    template <int dim> types<dim>::forest *
      partition(typename types<dim>::forest *, typename types<dim>::weight weights);

      template <int dim>   void
      vtk_write_file(const typename types<dim>::forest *forest, const char *baseName);

      // TODO: forest const?
      template <int dim>  types<dim>::eclass
      get_eclass(const typename types<dim>::forest *parallel_forest,
                 typename types<dim>::locidx        local_tree)
      {
        return t8_forest_get_eclass(
          const_cast<types<dim>::forest *>(parallel_forest), local_tree);
      }

 
       template <int dim> void
      ghost_destroy(typename types<dim>::ghost **ghost);
       template <int dim> types<dim>::ghost *
      ghost_new(typename types<dim>::forest *forest);
       template <int dim> void
      forest_destroy(typename types<dim>::forest **forest);


       template <int dim> int
      element_level(const typename types<dim>::forest *forest,
                    typename types<dim>::eclass        eclass,
                    const typename types<dim>::element *element);


       template <int dim>  void
      element_child(const typename types<dim>::forest *forest,
                   typename  types<dim>::eclass        tree_class,
                    const typename types<dim>::element *element,
                    int                       childid,
                    typename types<dim>::element       *child);

       template <int dim>  bool
      element_overlaps_tree(const typename types<dim>::forest *forest,
                            const typename types<dim>::tree    tree,
                            const typename types<dim>::element *element);



       template <int dim>  bool
      element_is_equal(const typename types<dim>::forest *forest,
                       typename types<dim>::eclass        eclass,
                       const typename types<dim>::element       *element_1,
                       const typename types<dim>::element       *element_2);

                       template <int dim>  bool
      cell_exists_in_tree(const typename types<dim>::forest *forest,
                          const typename types<dim>::tree    tree,
                          const typename types<dim>::element *element);

                          template <int dim>  int
      element_ancestor_id(const typename types<dim>::forest *forest,
                          typename types<dim>::eclass        eclass,
                          const typename types<dim>::element *element,
                          int                       level);


      template <int dim>  void
      element_children(const typename types<dim>::forest *forest,
                       typename types<dim>::eclass        eclass,
                       const typename types<dim>::element *element,
                       typename types<dim>::element      *children);

      template <int dim> dealii::types::subdomain_id
      comm_find_owner(const typename types<dim>::forest *forest,
                      const typename types<dim>::locidx  ltreeid,
                      const typename types<dim>::element *amr_cell,
                      dealii::types::subdomain_id        subdomain);



    template <int dim, int spacedim>
    typename types<dim>::forest *
    adapt(typename types<dim>::forest  *parallel_forest,
           typename ::dealii::parallel::distributed::Triangulation<dim,spacedim> *triangulation);

    template <int dim>
    types<dim>::forest *
    balance_full(typename types<dim>::forest *forest);


    template <int dim>
    void
    forest_set_user_pointer(typename types<dim>::forest *forest,
                            void                              *user_pointer);

    template <int dim>
    void *
    forest_get_user_pointer(const typename types<dim>::forest *forest);

    template <int dim>
    types<dim>::scheme_collection *
    forest_get_scheme(const typename types<dim>::forest *forest);


/*    template <int dim>
    types<dim>::element
    get_ghost_elem_and_owner(
      const typename types<dim>::forest *parallel_forest,
      const typename types<dim>::topidx  global_tree_idx,
      const typename types<dim>::locidx  ghost_in_tree_idx,
      const typename types<dim>::eclass  ghost_eclass,
      dealii::types::subdomain_id       &subdomain);
*/
    template <int dim>
    types<dim>::eclass
    get_ghost_eclass(const typename types<dim>::forest *parallel_forest,
                     const typename types<dim>::locidx  local_ghost_tree_idx);

      template <int dim>
      types<dim>::eclass
      get_eclass_from_tree(const typename types<dim>::tree tree)
      {
        return tree->eclass;
      }

    template <int dim>
    types<dim>::tree
    forest_get_tree(const typename types<dim>::forest *forest,
                    const typename types<dim>::locidx  ltreeid);

    template <int dim>
    types<dim>::gloidx
    tree_get_offset(const typename types<dim>::tree tree);



    template <int dim>
    bool
    tree_exists_locally(const typename types<dim>::forest *parallel_forest,
                        const typename types<dim>::topidx  coarse_grid_cell)
    {
      return ((coarse_grid_cell >= parallel_forest->first_local_tree) &&
              (coarse_grid_cell <= parallel_forest->last_local_tree));
    }

    template <int dim>
    typename types<dim>::connectivity *
    get_connectivity(const typename types<dim>::forest *forest);



    template <int dim>
    typename types<dim>::locidx
    leaf_index_in_tree(const typename types<dim>::forest *forest,
                       const typename types<dim>::locidx  ltreeid,
                       const typename types<dim>::element *leaf);

    template <int dim>
    typename types<dim>::locidx
    get_num_leafs(const typename types<dim>::forest *forest);


    template <int dim>
    typename types<dim>::connectivity *
    copy_connectivity(const typename types<dim>::connectivity *connectivity);


    template <int dim>
    void
    init_coarse_element(const typename types<dim>::forest *forest,
                        typename types<dim>::eclass        eclass,
                        typename types<dim>::element       *element);


  } // namespace t8code
} // namespace internal
DEAL_II_NAMESPACE_CLOSE

#else

// Make sure the scripts that create the C++20 module input files have
// something to latch on if the preprocessor #ifdef above would
// otherwise lead to an empty content of the file.
DEAL_II_NAMESPACE_OPEN
DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_WITH_T8CODE

#endif // dealii_t8code_wrappers_h
