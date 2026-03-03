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
    template <int>
    struct types
    { // TODO: check which should be pointer, and which not
      using connectivity      = t8_cmesh_t;
      using forest            = t8_forest_t;
      using tree              = t8_tree_struct_t;
      using element           = t8_element_t *;
      using eclass            = t8_eclass_t;
      using scheme_collection = const t8_scheme;
      using locidx            = t8_locidx_t;
      using gloidx            = t8_gloidx_t;
      using ghost             = t8_forest_ghost_t;
      using ghost_type        = t8_ghost_type_t;
    };

    template <int dim>
    struct functions
    {
      types<dim>::forest
      adapt(types<dim>::forest, t8_forest_adapt_t adapt_callback);

      types<dim>::forest balance(types<dim>::forest);

      types<dim>::forest
      partition(types<dim>::forest, t8_ghost_type_t ghost_type);

      types<dim>::forest
      adapt_balance_partition(types<dim>::forest,
                              t8_forest_adapt_t adapt_callback,
                              t8_ghost_type_t   ghost_type);


      void
      init_root(const types<dim>::forest forest,
                types<dim>::eclass       eclass,
                types<dim>::element     *element);
      void
      element_new(const types<dim>::forest forest,
                  types<dim>::eclass       eclass,
                  types<dim>::locidx       length,
                  types<dim>::element    **element);
      int
      element_level(const types<dim>::forest   forest,
                    types<dim>::eclass         eclass,
                    const types<dim>::element *element);
      void
      element_destroy(const types<dim>::forest forest,
                      types<dim>::eclass       eclass,
                      types<dim>::locidx       length,
                      types<dim>::element    **element);
      void
      element_children(const types<dim>::forest   forest,
                       types<dim>::eclass         eclass,
                       const types<dim>::element *element,
                       types<dim>::element      **children);

      void
      element_child(const types<dim>::forest   forest,
                    types<dim>::eclass         tree_class,
                    const types<dim>::element *element,
                    int                        childid,
                    types<dim>::element       *child);

      bool
      element_overlaps_tree(const types<dim>::forest   forest,
                            const types<dim>::tree     tree,
                            const types<dim>::element *element);
      int
      element_ancestor_id(const types<dim>::forest   forest,
                          types<dim>::eclass         eclass,
                          const types<dim>::element *element,
                          int                        level);
    }; // struct functions

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
