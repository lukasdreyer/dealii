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

#ifdef DEAL_II_WITH_T8CODE
#  include <t8.h>
#  include <t8_cmesh/t8_cmesh.h>
#  include <t8_element.h>
#  include <t8_forest/t8_forest.h>
#  include <t8_forest/t8_forest_general.h>
#  include <t8_forest/t8_forest_types.h>
#  include <t8_schemes/t8_scheme.hxx>


DEAL_II_NAMESPACE_OPEN
namespace internal
{
  namespace t8code
  {
    using t8code_weight_dummy_t = int (*)(t8_forest_t   forest,
                                          long          local_tree, // TODO
                                          t8_element_t *element);
    template <int>
    struct types
    { // TODO: check which should be pointer, and which not
      using connectivity      = t8_cmesh_t;
      using forest            = struct t8_forest;
      using tree              = t8_tree_t;
      using element           = t8_element_t *;
      using element_coord     = t8_element_coord;
      using eclass            = t8_eclass_t;
      using scheme_collection = const t8_scheme;
      using locidx            = t8_locidx_t;
      using gloidx            = t8_gloidx_t;
      using topidx            = t8_gloidx_t;
      using ghost             = t8_forest_ghost_t;
      using ghost_type        = t8_ghost_type_t;
      using balance_type      = t8_ghost_type_t;
      using weight            = t8code_weight_dummy_t;
      using element_array     = t8_element_array_t *;
    };

    template <int dim>
    struct functions
    {
      static types<dim>::forest *
      partition(types<dim>::forest *, types<dim>::weight weights);

      static types<dim>::forest
      adapt_balance_partition(types<dim>::forest,
                              t8_forest_adapt_t adapt_callback,
                              t8_ghost_type_t   ghost_type);


      static void
      ghost_destroy(types<dim>::ghost *ghost);
      static types<dim>::ghost *
      ghost_new(types<dim>::forest *forest);
      static void
      destroy(types<dim>::forest *forest);
      static void
      connectivity_destroy(types<dim>::connectivity *connectivity);

      static std::size_t
      forest_memory_used(const types<dim>::forest *forest);
      static std::size_t
      connectivity_memory_used(const types<dim>::connectivity *cmesh);

      static void
      vtk_write_file(const types<dim>::forest *forest, const char *baseName);

      static unsigned int
      checksum(const types<dim>::forest *forest);


      // TODO: forest const?
      static types<dim>::eclass
      get_eclass(const types<dim>::forest *parallel_forest,
                 types<dim>::locidx        local_tree)
      {
        return t8_forest_get_eclass(
          const_cast<types<dim>::forest *>(parallel_forest), local_tree);
      }
      static types<dim>::eclass
      get_eclass_from_tree(const types<dim>::tree tree)
      {
        return tree->eclass;
      }

      static unsigned int
      get_max_level(const types<dim>::forest *parallel_forest)
      {
        return t8_forest_get_maxlevel(
          const_cast<types<dim>::forest *>(parallel_forest));
      };


      static void
      element_new(const types<dim>::forest *forest,
                  types<dim>::eclass        eclass,
                  //                types<dim>::locidx       length,
                  types<dim>::element *element);

      static void
      element_init(const types<dim>::forest *forest,
                   types<dim>::eclass        eclass,
                   //                types<dim>::locidx       length,
                   types<dim>::element element);


      static int
      element_level(const types<dim>::forest *forest,
                    types<dim>::eclass        eclass,
                    const types<dim>::element element);
      static void
      element_destroy(const types<dim>::forest *forest,
                      types<dim>::eclass        eclass,
                      types<dim>::locidx        length,
                      types<dim>::element      *element);
      static void
      element_children(const types<dim>::forest *forest,
                       types<dim>::eclass        eclass,
                       const types<dim>::element element,
                       types<dim>::element      *children);

      static void
      element_child(const types<dim>::forest *forest,
                    types<dim>::eclass        tree_class,
                    const types<dim>::element element,
                    int                       childid,
                    types<dim>::element       child);

      static bool
      element_overlaps_tree(const types<dim>::forest *forest,
                            const types<dim>::tree    tree,
                            const types<dim>::element element);


        
    static bool
    element_is_equal(const typename types<dim>::forest *forest,
                     typename types<dim>::eclass        eclass,
                     typename types<dim>::element       element_1,
                     typename types<dim>::element       element_2);
      static bool
      cell_exists_in_tree(const types<dim>::tree    tree,
                          const types<dim>::element element);
      static int
      element_ancestor_id(const types<dim>::forest *forest,
                          types<dim>::eclass        eclass,
                          const types<dim>::element element,
                          int                       level);

      static types<dim>::forest *
      copy_forest(types<dim>::forest *input, int copy_data);


      static dealii::types::subdomain_id
      comm_find_owner(const typename types<dim>::forest *forest,
                      const typename types<dim>::locidx  ltreeid,
                      const typename types<dim>::element amr_cell,
                      dealii::types::subdomain_id        subdomain);



    }; // struct functions

    template <int dim, int spacedim>
    types<dim>::forest *
    adapt(typename types<dim>::forest  *forest,
          Triangulation<dim, spacedim> &triangulation);

    template <int dim>
    types<dim>::forest *
    balance_full(typename types<dim>::forest *forest);


    template <int dim>
    void
    forest_set_user_pointer(const typename types<dim>::forest *forest,
                            void                              *user_pointer);

    template <int dim>
    void *
    forest_get_user_pointer(const typename types<dim>::forest *forest);

    template <int dim>
    types<dim>::element
    get_ghost_elem_and_owner(
      const typename types<dim>::forest *parallel_forest,
      const typename types<dim>::topidx  global_tree_idx,
      const typename types<dim>::locidx  ghost_in_tree_idx,
      const typename types<dim>::eclass  ghost_eclass,
      dealii::types::subdomain_id       &subdomain);

    template <int dim>
    types<dim>::eclass
    get_ghost_eclass(const typename types<dim>::forest *parallel_forest,
                     const typename types<dim>::locidx  local_ghost_tree_idx);


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
                       const typename types<dim>::element leaf);

    template <int dim>
    typename types<dim>::locidx
    get_num_leafs(const typename types<dim>::forest *forest);


    template <int dim>
    typename types<dim>::connectivity *
    copy_connectivity(const typename types<dim>::connectivity *connectivity);


    template <int dim>
    void
    init_coarse_element(const typename types<dim>::forest *forest,
                        typename types<dim>::locidx        local_tree,
                        typename types<dim>::element       element);


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
