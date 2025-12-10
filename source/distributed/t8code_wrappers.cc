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


#include <deal.II/distributed/t8code_wrappers.h>
#include <deal.II/distributed/tria.h>

DEAL_II_NAMESPACE_OPEN

#ifdef DEAL_II_WITH_T8CODE
#  include <t8_schemes/t8_scheme.hxx>
#  include <t8_forest/t8_forest_ghost.h>
#  include <t8_forest/t8_forest_types.h>

namespace internal
{
  namespace t8code
  {

    types::forest adapt(types::forest forest, t8_forest_adapt_t adapt_callback){
  t8_forest_t new_forest;
  t8_forest_init (&new_forest);
  t8_forest_set_adapt (new_forest, forest, adapt_callback, 0);
  t8_forest_set_user_data(new_forest, t8_forest_get_user_data(forest));
  t8_forest_commit (new_forest);
  return new_forest;
    }
    types::forest balance(types::forest forest){
  t8_forest_t new_forest;
  t8_forest_init (&new_forest);
  t8_forest_set_balance (new_forest, forest, 1);
  t8_forest_set_user_data(new_forest, t8_forest_get_user_data(forest));
  t8_forest_commit (new_forest);
  return new_forest;
    }
    types::forest partition(types::forest forest, t8_ghost_type_t ghost_type){
  t8_forest_t new_forest;
  t8_forest_init (&new_forest);
  t8_forest_set_partition (new_forest, forest, 1);
  t8_forest_set_ghost (new_forest, 1, ghost_type);
  t8_forest_set_user_data(new_forest, t8_forest_get_user_data(forest));
  t8_forest_commit (new_forest);
  return new_forest;
    }


    void
    init_root(const types::forest   forest,
              types::eclass   eclass,
              types::element *element)
    {
      types::scheme_collection *scheme =
        t8_forest_get_scheme(forest);
      scheme->element_get_level(eclass, element);
      scheme->set_to_root(eclass, element);
    }

    void
    element_new(const types::forest    forest,
                types::eclass    eclass,
                types::locidx length,
                types::element **pelement)
    {
      types::scheme_collection *scheme =
        t8_forest_get_scheme(forest);
      scheme->element_new(eclass, length, pelement);
    }
    int
    element_level(const types::forest   forest,
                  types::eclass   eclass,
                  const types::element *element)
    {
      types::scheme_collection *scheme =
        t8_forest_get_scheme(forest);
      return scheme->element_get_level(eclass, element);
    }
    void
    element_destroy(const types::forest    forest,
                    types::eclass    eclass,
                types::locidx length,
                    types::element **pelement)
    {
      types::scheme_collection *scheme =
        t8_forest_get_scheme(forest);
      scheme->element_destroy(eclass, length, pelement);
    }


    void
    element_child(const types::forest         forest,
                     types::eclass         eclass,
                     const types::element *element,
                     int childid,
                     types::element      *child)
    {
      types::scheme_collection *scheme =
        t8_forest_get_scheme(forest);
      scheme->element_get_child(eclass, element, childid, child);
    }


    void
    element_children(const types::forest         forest,
                     types::eclass         eclass,
                     const types::element *element,
                     types::element      **children)
    {
      types::scheme_collection *scheme =
        t8_forest_get_scheme(forest);
      int num_children = scheme->element_get_num_children(eclass, element);
      scheme->element_get_children(eclass, element, num_children, children);
    }
    bool
    element_overlaps_tree(const types::forest   forest,
                          types::tree     tree,
                          const types::element *element)
    {
      types::eclass         eclass = tree.eclass;
      types::scheme_collection *scheme =
        t8_forest_get_scheme(forest);
      types::element *element_last_desc;
      bool            element_overlaps = true;

      element_new(forest, eclass, 1, &element_last_desc);
      const unsigned int maxlevel = scheme->get_maxlevel(eclass);
      scheme->element_get_last_descendant(eclass, element,
                                                element_last_desc,
                                                maxlevel);
      if (scheme->element_compare(eclass, element_last_desc,
                                            tree.first_desc) < 0)
        element_overlaps = false;

      element_destroy(forest, eclass, 1, &element_last_desc);

      /* check if q is after the last tree quadrant */
      if (scheme->element_compare(eclass, tree.last_desc, element) < 0)
        element_overlaps = false;

      return element_overlaps;
    }
    int
    element_ancestor_id(const types::forest   forest,
                        types::eclass   eclass,
                        const types::element *element,
                        int             level)
    {
      types::scheme_collection *scheme =
        t8_forest_get_scheme(forest);
      return scheme->element_get_ancestor_id(eclass, element, level);
    }


#  if 0
    types::ghost *(&functions::ghost_new)(types::forest      *forest,
                                                types::ghost_type btype) =
      t8_forest_ghost_init;

    void (&functions::ghost_destroy)(types:ghost *ghost) =
      t8_forest_ghost_destroy;
#  endif
  } // namespace t8code
} // namespace internal

#endif // DEAL_II_WITH_T8CODE

/*-------------- Explicit Instantiations -------------------------------*/
#include "t8code_wrappers.inst"


DEAL_II_NAMESPACE_CLOSE
