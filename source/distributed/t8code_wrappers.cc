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
#include "deal.II/base/types.h"

#include <deal.II/distributed/t8code_wrappers.h>
#include <deal.II/distributed/tria.h>

#include "deal.II/grid/reference_cell.h"
#include "deal.II/grid/tria.h"

#include <t8_eclass.h>
#include <t8_forest/t8_forest_adapt.h>
#include <t8_forest/t8_forest_general.h>
#include <t8_forest/t8_forest_io.h>
#include <t8_geometry/t8_geometry_implementations/t8_geometry_linear.hxx>

#include <cstdint>
#include <vector>


#ifdef DEAL_II_WITH_T8CODE
#  include <deal.II/distributed/p4est_wrappers.h>

#  include <t8_forest/t8_forest_ghost.h>
#  include <t8_forest/t8_forest_types.h>
#  include <t8_schemes/t8_scheme.hxx>
// #  include <t8_cmesh/t8_cmesh.h>
#  include <t8_cmesh/t8_cmesh.hxx>
#  include <t8_cmesh/t8_cmesh_internal/t8_cmesh_types.h>


DEAL_II_NAMESPACE_OPEN

namespace internal
{
  namespace t8code
  {
    std::pair<unsigned int, unsigned int>
    amr_to_dealii_child_index_and_type(const ReferenceCell &ref_cell,
                                       const unsigned int   dealii_type,
                                       const unsigned int   child)
    {
      if (ref_cell.is_hyper_cube())
        return std::pair<unsigned int, unsigned int>{child, dealii_type};
      else if (ref_cell == ReferenceCells::Triangle)
        {
          const ::dealii::ndarray<unsigned int, 6, 4> triangle_perms = {
            {{{0, 1, 3, 2}},
             {{0, 3, 2, 1}},
             {{2, 0, 3, 1}},
             {{2, 3, 1, 0}},
             {{1, 2, 3, 0}},
             {{1, 3, 0, 2}}}};

          const unsigned int child_index = triangle_perms[dealii_type][child];
          const unsigned int child_type =
            (child_index == 3) ? (dealii_type + 1) % 6 : dealii_type;

          return std::pair<unsigned int, unsigned int>{child_index, child_type};
        }
      else if (ref_cell == ReferenceCells::Tetrahedron)
        {
// State = 4 * compact_t8_type + allowed_permutation_index
// compact t8 type 0..5 maps to sparse t8 type {0,1,2,5,6,7}
const ::dealii::ndarray<unsigned int, 24, 8> tet_perms_t8_to_dealii = {{
  {{0, 1, 4, 5, 2, 7, 6, 3}},
  {{1, 0, 5, 4, 3, 6, 7, 2}},
  {{2, 3, 6, 7, 0, 5, 4, 1}},
  {{3, 2, 7, 6, 1, 4, 5, 0}},
  {{0, 6, 3, 5, 2, 7, 4, 1}},
  {{3, 5, 0, 6, 1, 4, 7, 2}},
  {{2, 4, 1, 7, 0, 5, 6, 3}},
  {{1, 7, 2, 4, 3, 6, 5, 0}},
  {{0, 3, 6, 5, 7, 4, 2, 1}},
  {{3, 0, 5, 6, 4, 7, 1, 2}},
  {{2, 1, 4, 7, 5, 6, 0, 3}},
  {{1, 2, 7, 4, 6, 5, 3, 0}},
  {{0, 1, 5, 4, 6, 7, 2, 3}},
  {{1, 0, 4, 5, 7, 6, 3, 2}},
  {{2, 3, 7, 6, 4, 5, 0, 1}},
  {{3, 2, 6, 7, 5, 4, 1, 0}},
  {{0, 5, 4, 1, 6, 2, 7, 3}},
  {{1, 4, 5, 0, 7, 3, 6, 2}},
  {{2, 7, 6, 3, 4, 0, 5, 1}},
  {{3, 6, 7, 2, 5, 1, 4, 0}},
  {{0, 5, 6, 3, 4, 7, 2, 1}},
  {{3, 6, 5, 0, 7, 4, 1, 2}},
  {{2, 7, 4, 1, 6, 5, 0, 3}},
  {{1, 4, 7, 2, 5, 6, 3, 0}}
}};
const ::dealii::ndarray<unsigned int, 24, 8>  tet_child_types= {{
  {{0, 0, 0, 0, 4, 12, 18, 9}},
  {{1, 1, 1, 1, 12, 4, 8, 17}},
  {{2, 2, 2, 2, 19, 11, 7, 14}},
  {{3, 3, 3, 3, 11, 19, 13, 6}},
  {{4, 4, 4, 4, 21, 10, 3, 15}},
  {{5, 5, 5, 5, 13, 2, 9, 23}},
  {{6, 6, 6, 6, 2, 13, 22, 8}},
  {{7, 7, 7, 7, 10, 21, 12, 0}},
  {{8, 8, 8, 8, 5, 22, 19, 3}},
  {{9, 9, 9, 9, 1, 18, 21, 7}},
  {{10, 10, 10, 10, 18, 1, 6, 20}},
  {{11, 11, 11, 11, 22, 5, 0, 16}},
  {{12, 12, 12, 12, 20, 16, 2, 5}},
  {{13, 13, 13, 13, 16, 20, 4, 1}},
  {{14, 14, 14, 14, 3, 7, 23, 18}},
  {{15, 15, 15, 15, 7, 3, 17, 22}},
  {{16, 16, 16, 16, 8, 0, 14, 21}},
  {{17, 17, 17, 17, 0, 8, 20, 13}},
  {{18, 18, 18, 18, 15, 23, 11, 2}},
  {{19, 19, 19, 19, 23, 15, 1, 10}},
  {{20, 20, 20, 20, 9, 6, 15, 19}},
  {{21, 21, 21, 21, 17, 14, 5, 11}},
  {{22, 22, 22, 22, 14, 17, 10, 4}},
  {{23, 23, 23, 23, 6, 9, 16, 12}}
}};



          const unsigned int dealii_child_index =
            tet_perms_t8_to_dealii[dealii_type][child];

          const unsigned int dealii_child_type =
            tet_child_types[dealii_type][dealii_child_index];

          return std::pair<unsigned int, unsigned int>{dealii_child_index,
                                                       dealii_child_type};
        }
      else
        {
          DEAL_II_NOT_IMPLEMENTED();
        }

      return std::pair<unsigned int, unsigned int>{
        numbers::invalid_unsigned_int, numbers::invalid_unsigned_int};
    }



    const std::vector<std::vector<int>> dealii_to_t8_faces = {
      {},                       // vertex has no faces
      {0, 1},                   // line is oriented the same
      {0, 1, 2, 3},             // quad is oriented the same
      {2, 0, 1},                // tri is rotated by one
      {0, 1, 2, 3, 4, 5, 6, 7}, // hex
      {},                       // tet
      {},                       // prism
      {}                        // pyramid
    };

    static t8_eclass_t
    t8_eclass_from_reference_cell(const ReferenceCell &ref_cell)
    {
      switch (ref_cell)
        {
          case ReferenceCells::Quadrilateral:
            return T8_ECLASS_QUAD;
          case ReferenceCells::Triangle:
            return T8_ECLASS_TRIANGLE;
          case ReferenceCells::Hexahedron:
            return T8_ECLASS_HEX;
          case ReferenceCells::Tetrahedron:
            return T8_ECLASS_TET;
          default:
            DEAL_II_NOT_IMPLEMENTED();
            return T8_ECLASS_INVALID;
        }
    }

    template <int dim, int spacedim>
    typename types<dim>::connectivity
    dealii_to_connectivity(
      typename ::dealii::parallel::distributed::Triangulation<dim, spacedim>
        *tria)
    {
      t8_cmesh_t cmesh;
      // TODO!! coarse mesh permutation!!!
      t8_cmesh_init(&cmesh);
      t8_cmesh_register_geometry<t8_geometry_linear>(cmesh);
      const auto &coarse_cell_permutation =
        tria->get_coarse_cell_to_p4est_tree_permutation();
      // const auto &coarse_cell_permutation =
      // tria->get_p4est_tree_to_coarse_cell_permutation();
      for (const auto &cell : tria->active_cell_iterators())
        {
          const auto  t8_index = coarse_cell_permutation[cell->index()];
          t8_eclass_t eclass =
            t8_eclass_from_reference_cell(cell->reference_cell());
          t8_cmesh_set_tree_class(cmesh, t8_index, eclass);
          std::vector<double> coords(3 * cell->n_vertices());
          for (unsigned int ivertex = 0; ivertex < cell->n_vertices();
               ivertex++)
            {
              const auto &vertex = cell->vertex(ivertex);
              for (unsigned int idim = 0; idim < dim; idim++)
                {
                  std::cout << "vertex[idim]" << vertex[idim] << std::endl;
                  coords[3 * ivertex + idim] = vertex[idim];
                }
            }

          t8_cmesh_disable_negative_volume_check(cmesh);
          t8_cmesh_set_tree_vertices(cmesh,
                                     t8_index,
                                     coords.data(),
                                     cell->n_vertices());

          for (unsigned int iface = 0; iface < cell->n_faces(); iface++)
            {
              if (cell->neighbor_index(iface) == -1)
                {
                  continue;
                }


              const auto &t8_neighbor_index =
                coarse_cell_permutation[cell->neighbor_index(iface)];
              if (t8_index < t8_neighbor_index)
                {
                  const auto  ineighface   = cell->neighbor_of_neighbor(iface);
                  int         t8_iface     = dealii_to_t8_faces[eclass][iface];
                  t8_eclass_t neigh_eclass = t8_eclass_from_reference_cell(
                    cell->neighbor(iface)->reference_cell());
                  int t8_ineighface =
                    dealii_to_t8_faces[neigh_eclass][ineighface];

                  bool orientation =
                    (cell->face_orientation(iface) !=
                     cell->neighbor(iface)->face_orientation(
                       ineighface)); // cell->combined_face_orientation(iface);
                  if (eclass == T8_ECLASS_TRIANGLE && iface == 2)
                    {
                      //                std::cout<<"switched orientation because
                      //                own cell is triangle and on face
                      //                2"<<std::endl;
                      orientation = !orientation;
                    }
                  if (neigh_eclass == T8_ECLASS_TRIANGLE && ineighface == 2)
                    {
                      //                std::cout<<"switched orientation because
                      //                neighbor cell is triangle and on face
                      //                2"<<std::endl;
                      orientation = !orientation;
                    }

                  //              std::cout<<"added face join from cell
                  //              "<<t8_index<<" face "<<t8_iface<<" to cell
                  //              "<<t8_neighbor_index<<" face " <<
                  //              t8_ineighface <<" with orientation
                  //              "<<orientation <<std::endl;
                  std::cout
                    << cell->face_orientation(iface) << " "
                    << cell->neighbor(iface)->face_orientation(ineighface)
                    << std::endl;
                  //              std::cout<<"t8_cmesh_set_join(cmesh,"<<
                  //              t8_index<<", "<<t8_neighbor_index<<", "<<
                  //              t8_iface <<", "<< t8_ineighface<<",
                  //              "<<orientation <<")"<<std::endl;
                  t8_cmesh_set_join(cmesh,
                                    t8_index,
                                    t8_neighbor_index,
                                    t8_iface,
                                    t8_ineighface,
                                    (int)orientation);
                }
            }

          std::vector<t8_gloidx_t> vertex_list(cell->n_vertices());
          for (unsigned int ivertex = 0; ivertex < cell->n_vertices();
               ivertex++)
            {
              vertex_list[ivertex] = cell->vertex_index(ivertex);
              std::cout << "local vertex " << ivertex
                        << " connected to global vertex "
                        << cell->vertex_index(ivertex);
            }
          t8_cmesh_set_global_vertices_of_tree(cmesh,
                                               t8_index,
                                               vertex_list.data(),
                                               cell->n_vertices());
          std::vector<t8_gloidx_t> edge_list(cell->n_lines());
          for (unsigned int iedge = 0; iedge < cell->n_lines(); iedge++)
            {
              edge_list[iedge] = cell->line_index(iedge);
              // std::cout << "local edge "<<iedge<<" connected to global edge "
              // << cell->line_index(iedge);
            }
          t8_cmesh_set_global_edges_of_tree(cmesh,
                                            t8_index,
                                            edge_list.data(),
                                            cell->n_lines());
        }
      t8_cmesh_commit(cmesh, tria->get_mpi_communicator());
      return cmesh;
    }

    template <int dim, int spacedim>
    static void
    fill_adapt_list_recursively(
      const typename Triangulation<dim, spacedim>::cell_iterator &dealii_cell,
      const unsigned int                                          dealii_type,
      std::vector<int>                                           &adapt_list)
    {
      if (!dealii_cell->has_children())
        {
          if (dealii_cell->is_locally_owned())
            {
              if (dealii_cell->refine_flag_set())
                {
                  adapt_list.push_back(1);
                }
              else if (dealii_cell->coarsen_flag_set())
                {
                  adapt_list.push_back(-1);
                }
              else
                {
                  adapt_list.push_back(0);
                }
            }
          return;
        }

      // loop over children in t8code order
      for (unsigned int t8code_child = 0;
           t8code_child < dealii_cell->n_children();
           ++t8code_child)
        {
          const std::pair<unsigned int, unsigned int>
            dealii_child_index_and_type =
              amr_to_dealii_child_index_and_type(dealii_cell->reference_cell(),
                                                 dealii_type,
                                                 t8code_child);
          const auto &dealii_child =
            dealii_cell->child(dealii_child_index_and_type.first);

          fill_adapt_list_recursively<dim, spacedim>(
            dealii_child, dealii_child_index_and_type.second, adapt_list);
        }
    }


    int
    adapt_from_vec(t8_forest_t,
                   t8_forest_t forest_from,
                   t8_locidx_t which_tree,
                   const t8_eclass_t,
                   t8_locidx_t lelement_id,
                   const t8_scheme_c *,
                   const int,
                   const int,
                   t8_element_t *[])
    {
      std::vector<int> *adapt_vec =
        (std::vector<int> *)t8_forest_get_user_data(forest_from);
      // std::cout<<"restored adapt_vec from adress "<<adapt_vec<<std::endl;
      const int idata =
        t8_forest_get_tree_element_offset(forest_from, which_tree) +
        lelement_id;
      return (*adapt_vec)[idata];
    }


    template <int dim, int spacedim>
    typename types<dim>::forest *
    adapt(typename types<dim>::forest *parallel_forest,
          typename ::dealii::parallel::distributed::Triangulation<dim, spacedim>
            *triangulation)
    {
      std::cout << "adapt forest with refcount " << parallel_forest->rc.refcount
                << std::endl;
      // count how many cells will be refined and coarsened, and allocate that
      // much memory
      std::vector<int> adapt_list;
      // copy refine and coarsen flags into p4est and execute the refinement
      // and coarsening. this uses the refine_and_coarsen_list just built,
      // which is communicated to the callback functions through
      // p4est's user_pointer object
      Assert(forest_get_user_pointer<dim>(parallel_forest) == triangulation,
             ExcInternalError());
      forest_set_user_pointer<dim>(parallel_forest, &adapt_list);
      std::cout << "set pointer to adapt list " << &adapt_list << std::endl;



      const auto &perm =
        triangulation->get_p4est_tree_to_coarse_cell_permutation();
      for (unsigned int i = 0; i < perm.size(); ++i)
        {
          const unsigned int cell_index = perm[i];

          typename dealii::Triangulation<dim, spacedim>::cell_iterator
            dealii_cell(triangulation, 0, cell_index);

          fill_adapt_list_recursively<dim, spacedim>(dealii_cell,
                                                     0,
                                                     adapt_list);
        }

      // std::cout<<"list size:"<<adapt_list.size()<<", tria
      // size:"<<triangulation->n_active_cells()<<std::endl; for (const auto
      // &entry: adapt_list){
      //   std::cout<<(int)entry<<std::endl;
      // }

      t8_forest_t new_forest;

      t8_forest_init(&new_forest);
      t8_forest_set_adapt(new_forest, parallel_forest, adapt_from_vec, false);
      t8_forest_set_ghost(new_forest, true, T8_GHOST_VERTICES);
      t8_forest_commit(new_forest);
      //      std::cout<<"old forest with refcount
      //      "<<parallel_forest->rc.refcount<<std::endl;
      std::cout << "new forest with refcount " << new_forest->rc.refcount
                << std::endl;


      // reset the pointer
      forest_set_user_pointer<dim>(new_forest, triangulation);

      return new_forest;
    }

    template <int dim>
    typename types<dim>::forest *
    balance_full(typename types<dim>::forest *forest)
    {
      t8_forest_t new_forest;
      t8_forest_init(&new_forest);
      t8_forest_set_balance(new_forest, forest, 1);
      t8_forest_set_user_data(new_forest, t8_forest_get_user_data(forest));
      t8_forest_commit(new_forest);
      return new_forest;
    }

    template <int dim>
    typename types<dim>::forest *
    partition(typename types<dim>::forest *forest, typename types<dim>::weight)
    {
      t8_forest_t new_forest;
      t8_forest_init(&new_forest);
      t8_forest_set_partition(new_forest, forest, 1);
      t8_forest_set_ghost(new_forest, 1, T8_GHOST_VERTICES);
      t8_forest_set_user_data(new_forest, t8_forest_get_user_data(forest));
      t8_forest_commit(new_forest);
      return new_forest;
    }

    template <int dim>
    void
    element_children(const typename types<dim>::forest  *forest,
                     typename types<dim>::eclass         eclass,
                     const typename types<dim>::element *element,
                     typename types<dim>::element       *children)
    {
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t>(forest));
      int num_children =
        scheme->element_get_num_children(eclass, (const t8_element_t *)element);
      for (int ichild = 0; ichild < num_children; ichild++)
        {
          scheme->element_get_child(eclass,
                                    (const t8_element_t *)element,
                                    ichild,
                                    (t8_element_t *)(children + ichild));
        }
    }

    template <int dim>
    void
    init_coarse_element(const typename types<dim>::forest *forest,
                        typename types<dim>::eclass        eclass,
                        typename types<dim>::element      *element)
    {
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t>(forest));
      //      scheme->element_get_level(eclass, (const t8_element_t*)element);
      scheme->set_to_root(eclass, (t8_element_t *)element);
    }

    template <int dim>
    int
    element_level(const typename types<dim>::forest  *forest,
                  typename types<dim>::eclass         eclass,
                  const typename types<dim>::element *element)
    {
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t>(forest));
      return scheme->element_get_level(
        eclass, reinterpret_cast<const t8_element_t *>(element));
    }

    template <int dim>
    bool
    element_is_equal(const typename types<dim>::forest  *forest,
                     typename types<dim>::eclass         eclass,
                     const typename types<dim>::element *element1,
                     const typename types<dim>::element *element2)
    {
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t>(forest));
      return scheme->element_is_equal(eclass,
                                      (const t8_element_t *)element1,
                                      (const t8_element_t *)element2);
    }


    template <int dim>
    void
    element_child(const typename types<dim>::forest  *forest,
                  typename types<dim>::eclass         eclass,
                  const typename types<dim>::element *element,
                  int                                 childid,
                  typename types<dim>::element       *child)
    {
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t>(forest));
      scheme->element_get_child(eclass,
                                (const t8_element_t *)element,
                                childid,
                                (t8_element_t *)child);
    }


    template <int dim>
    bool
    cell_exists_in_tree(const typename types<dim>::forest  *forest,
                        typename types<dim>::tree           tree,
                        const typename types<dim>::element *element)
    {
      typename types<dim>::eclass             eclass = tree->eclass;
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t>(forest));
      //      scheme->t8_element_debug_print(eclass,t8code_cell);
      auto compare_lambda = [scheme, eclass](auto x, auto y) {
        return scheme->element_compare(eclass,
                                       (t8_element_t *)x,
                                       (t8_element_t *)y) < 0;
      };


      return std::binary_search(t8_element_array_begin(&tree->leaf_elements),
                                t8_element_array_end(&tree->leaf_elements),
                                element,
                                compare_lambda);
    }

    template <int dim>
    bool
    element_overlaps_tree(const typename types<dim>::forest  *forest,
                          typename types<dim>::tree           tree,
                          const typename types<dim>::element *element)
    {
      typename types<dim>::eclass             eclass = tree->eclass;
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t>(forest));
      typename types<dim>::element element_last_desc;
      bool                         element_overlaps = true;

      const unsigned int maxlevel = scheme->get_maxlevel(eclass);
      scheme->element_get_last_descendant(eclass,
                                          (const t8_element_t *)element,
                                          (t8_element_t *)&element_last_desc,
                                          maxlevel);
      if (scheme->element_compare(eclass,
                                  (t8_element_t *)&element_last_desc,
                                  tree->first_desc) < 0)
        element_overlaps = false;

      /* check if q is after the last tree quadrant */
      if (scheme->element_compare(eclass,
                                  tree->last_desc,
                                  (const t8_element_t *)element) < 0)
        element_overlaps = false;

      return element_overlaps;
    }

    template <int dim>
    int
    element_ancestor_id(const typename types<dim>::forest  *forest,
                        typename types<dim>::eclass         eclass,
                        const typename types<dim>::element *element,
                        int                                 level)
    {
      typename types<dim>::scheme_collection *scheme =
        t8_forest_get_scheme(const_cast<t8_forest_t>(forest));
      return scheme->element_get_ancestor_id(eclass,
                                             (const t8_element_t *)element,
                                             level);
    }

    template <int dim>
    typename types<dim>::locidx
    leaf_index_in_tree(const typename types<dim>::forest  *forest,
                       const typename types<dim>::locidx   ltreeid,
                       const typename types<dim>::element *leaf)
    {
      return t8_forest_element_leaf_index_in_tree(
        const_cast<t8_forest_t>(forest), (const t8_element_t *)leaf, ltreeid);
    }

    template <int dim>
    typename types<dim>::eclass
    get_ghost_eclass(const typename types<dim>::forest *forest,
                     const typename types<dim>::locidx  ghost_treeid)
    {
      return t8_forest_ghost_get_tree_class(const_cast<t8_forest_t>(forest),
                                            ghost_treeid);
    }

    template <int dim>
    types<dim>::gloidx
    tree_get_offset(const typename types<dim>::tree tree)
    {
      return tree->elements_offset;
    }

    template <int dim>
    typename types<dim>::tree
    forest_get_tree(const typename types<dim>::forest *forest,
                    const typename types<dim>::locidx  ltreeid)
    {
      return t8_forest_get_tree(const_cast<t8_forest_t>(forest), ltreeid);
    }

    template <int dim>
    typename types<dim>::locidx
    get_num_leafs(const typename types<dim>::forest *forest)
    {
      return t8_forest_get_local_num_leaf_elements(
        const_cast<t8_forest_t>(forest));
    }


    template <int dim>
    typename types<dim>::ghost *
    ghost_new(typename types<dim>::forest *forest)
    {
      return forest->ghosts;
    }

    template <int dim>
    void
    ghost_destroy(typename types<dim>::ghost **)
    {
      // do nothing, ghost gets destroyed by forest?
    }

    template <int dim>
    void
    forest_destroy(typename types<dim>::forest **forest)
    {
      t8_forest_unref(forest);
    }

    template <int dim>
    void
    vtk_write_file(const typename types<dim>::forest *forest, const char *path)
    {
      t8_forest_write_vtk_ext(const_cast<types<dim>::forest *>(forest),
                              path,
                              1,
                              1,
                              1,
                              1,
                              1,
                              0,
                              0,
                              0,
                              NULL);
    }


    template <int dim>
    void
    forest_set_user_pointer(typename types<dim>::forest *forest,
                            void                        *user_pointer)
    {
      forest->user_data = user_pointer; // replace by function
    };

    template <int dim>
    void *
    forest_get_user_pointer(const typename types<dim>::forest *forest)
    {
      return forest->user_data; // replace by function
    };



    template <int dim>
    ::dealii::types::subdomain_id
    comm_find_owner(const typename types<dim>::forest *,
                    const typename types<dim>::locidx,
                    const typename types<dim>::element *,
                    const ::dealii::types::subdomain_id)
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

  } // namespace t8code
} // namespace internal

#endif // DEAL_II_WITH_T8CODE

/*-------------- Explicit Instantiations -------------------------------*/
#include "distributed/t8code_wrappers.inst"


DEAL_II_NAMESPACE_CLOSE
