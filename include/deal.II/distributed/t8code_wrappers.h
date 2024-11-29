// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2016 - 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef T8CODE_WRAPPERS_20COPY
#define T8CODE_WRAPPERS_20COPY

#ifndef dealii_t8code_wrappers_h
#  define dealii_t8code_wrappers_h

#  include <deal.II/base/config.h>

#  include <deal.II/base/geometry_info.h>

#  ifdef DEAL_II_WITH_T8CODE
#    include <t8.h>
#    include <t8_cmesh.h>
#    include <t8_element.h>
#    include <t8_element.hxx>
#    include <t8_forest/t8_forest.h>
#    include <t8_forest/t8_forest_general.h>
#    include <t8_forest/t8_forest_types.h>

#    include <limits>

DEAL_II_NAMESPACE_OPEN
namespace internal
{
  namespace t8code
  {
    struct types
    { // TODO: check which should be pointer, and which not
      using cmesh  = t8_cmesh_t;
      using forest = t8_forest_t;
      using tree   = t8_tree_struct_t;
      typedef t8_element_t element;
      using eclass            = t8_eclass_t;
      using eclass_scheme     = struct t8_eclass_scheme;
      using scheme_collection = t8_scheme_cxx_t;
      using locidx            = t8_locidx_t;
      using gloidx            = t8_gloidx_t;
      using ghost             = t8_forest_ghost_t;
      using ghost_type        = t8_ghost_type_t;
    };

    struct functions
    {
#    if 0
        static types::ghost *(&ghost_new)(types::forest      forest,
                                                types::ghost_type btype);

        static void (&ghost_destroy)(types::ghost *ghost);



        static void ()
#    endif
    };
    void
    init_root(const types::forest   forest,
              types::eclass   eclass,
              types::element *element);
    void
    element_new(const types::forest    forest,
                types::eclass    eclass,
                types::element **element);
    int
    element_level(const types::forest    forest,
                  types::eclass    eclass,
                  const types::element *element);
    void
    element_destroy(const types::forest    forest,
                    types::eclass    eclass,
                    types::element **element);
    void
    element_children(const types::forest         forest,
                     types::eclass         eclass,
                     const types::element *element,
                     types::element      **children);

    void
    element_child(const types::forest         forest,
                     types::eclass         tree_class,
                     const types::element *element,
                     int childid,
                     types::element      *child);

    bool
    element_overlaps_tree(const types::forest   forest,
                          const types::tree     tree,
                          const types::element *element);
    int
    element_ancestor_id(const types::forest   forest,
                        types::eclass         eclass,
                        const types::element *element,
                        int             level);
  } // namespace t8code
} // namespace internal
DEAL_II_NAMESPACE_CLOSE

#  endif // DEAL_II_WITH_T8CODE

#endif // dealii_t8code_wrappers_h


#endif /* T8CODE_WRAPPERS_20COPY */
