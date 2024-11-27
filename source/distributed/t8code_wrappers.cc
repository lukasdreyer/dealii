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
#include <t8_forest/t8_forest_ghost.h>
#include <t8_forest/t8_forest_types.h>
#include <t8_element.hxx>

namespace internal
{
  namespace t8code
  {
    void init_root (types::forest forest, types::eclass tree_class, types::element *element)
    {
          types::eclass_scheme *eclass_scheme = t8_forest_get_eclass_scheme (forest, tree_class);
          eclass_scheme->t8_element_level(element);
          eclass_scheme->t8_element_root(element);
    }

    void element_new (types::forest forest, types::eclass tree_class, types::element **pelement){
          types::eclass_scheme *eclass_scheme = t8_forest_get_eclass_scheme (forest, tree_class);
          eclass_scheme->t8_element_new(1, pelement);    
    }
    void element_children (types::forest forest, types::eclass tree_class, const types::element *element, types::element **children){
          types::eclass_scheme *eclass_scheme = t8_forest_get_eclass_scheme (forest, tree_class);
          int num_children = eclass_scheme->t8_element_num_children(element);
          eclass_scheme->t8_element_children(element, num_children, children);    

    }
    bool element_overlaps_tree (types::forest forest, types::tree tree, types::element *element){
      types::eclass tree_class = tree.eclass;
      types::eclass_scheme *eclass_scheme = t8_forest_get_eclass_scheme (forest, tree_class);
      types::element* element_last_desc;

      element_new(forest, T8_ECLASS_QUAD, &element_last_desc);
      const unsigned int maxlevel = eclass_scheme->t8_element_maxlevel();
      eclass_scheme->t8_element_last_descendant (element, element_last_desc, maxlevel);
      if (eclass_scheme->t8_element_compare (element_last_desc, tree.first_desc) < 0)
        return false;

      /* check if q is after the last tree quadrant */
      if (eclass_scheme->t8_element_compare (tree.last_desc, element) < 0)
        return false;

      return true;
    }


#if 0
    types::ghost *(&functions::ghost_new)(types::forest      *forest,
                                                types::ghost_type btype) =
      t8_forest_ghost_init;

    void (&functions::ghost_destroy)(types:ghost *ghost) =
      t8_forest_ghost_destroy;
#endif
  } // namespace t8code
} // namespace internal

#endif // DEAL_II_WITH_T8CODE

/*-------------- Explicit Instantiations -------------------------------*/
#include "t8code_wrappers.inst"


DEAL_II_NAMESPACE_CLOSE
