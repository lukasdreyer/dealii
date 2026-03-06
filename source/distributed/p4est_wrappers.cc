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


#include <deal.II/distributed/p4est_wrappers.h>
#include <deal.II/distributed/tria.h>

#include <p4est_bits.h>

#include <type_traits>

#ifdef DEAL_II_WITH_P4EST
#  include <p4est.h>
#  include <p8est.h>
#  include <sc_containers.h>

// Below, we will use the P4EST_QUADRANT_INIT and P8EST_QUADRANT_INIT
// function-like macros. If we are building the library based on
// header files, we get these from the <p4est.h> and <p8est.h> header
// inclusions. But if we build a C++20 module, we only import
// declarations, not preprocessor macros. As a consequence, let us
// duplicate these macros here, hoping that at some point, the p4est
// library folks add regular functions that can do the job.
#  ifndef P4EST_QUADRANT_INIT
#    define P4EST_QUADRANT_INIT(q) \
      ((void)std::memset((q), -1, sizeof(p4est_quadrant_t)))
#  endif

#  ifndef P8EST_QUADRANT_INIT
#    define P8EST_QUADRANT_INIT(q) \
      ((void)std::memset((q), -1, sizeof(p8est_quadrant_t)))
#  endif

#endif


DEAL_II_NAMESPACE_OPEN

#ifdef DEAL_II_WITH_P4EST


namespace
{
  /**
   * A data structure that we use to store which cells (indicated by
   * dealii::internal::amr::types<dim>::element objects) shall be refined and
   * which shall be coarsened.
   */
  template <int dim, int spacedim>
  class RefineAndCoarsenList
  {
  public:
    RefineAndCoarsenList(
      const Triangulation<dim, spacedim>                       *triangulation,
      const typename dealii::internal::amr::types<dim>::forest *forest,
      const std::vector<dealii::types::global_dof_index>
                                       &p4est_tree_to_coarse_cell_permutation,
      const dealii::types::subdomain_id my_subdomain);

    /**
     * A callback function that we pass to the p4est data structures when a
     * forest is to be refined. The p4est functions call it back with a tree
     * (the index of the tree that grows out of a given coarse cell) and a
     * refinement path from that coarse cell to a terminal/leaf cell. The
     * function returns whether the corresponding cell in the deal.II
     * triangulation has the refined flag set.
     */
    static int
    refine_callback(
      typename dealii::internal::amr::types<dim>::forest *forest,
      typename dealii::internal::amr::types<dim>::topidx  coarse_cell_index,
      typename dealii::internal::amr::types<dim>::element element);

    /**
     * Same as the refine_callback function, but return whether all four of
     * the given children of a non-terminal cell are to be coarsened away.
     */
    static int
    coarsen_callback(
      typename dealii::internal::amr::types<dim>::forest *forest,
      typename dealii::internal::amr::types<dim>::topidx  coarse_cell_index,
      typename dealii::internal::amr::types<dim>::element children[]);

    bool
    pointers_are_at_end() const;

  private:
    std::vector<typename dealii::internal::amr::types<dim>::element>
      refine_list;
    typename std::vector<typename dealii::internal::amr::types<dim>::element>::
      const_iterator current_refine_pointer;

    std::vector<typename dealii::internal::amr::types<dim>::element>
      coarsen_list;
    typename std::vector<typename dealii::internal::amr::types<dim>::element>::
      const_iterator current_coarsen_pointer;

    void
    build_lists(
      const typename dealii::internal::amr::types<dim>::forest   *forest,
      const typename dealii::internal::amr::types<dim>::eclass    eclass,
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const typename dealii::internal::amr::types<dim>::element   amr_cell,
      const types::subdomain_id                                   myid);
  };



  template <int dim, int spacedim>
  bool
  RefineAndCoarsenList<dim, spacedim>::pointers_are_at_end() const
  {
    return ((current_refine_pointer == refine_list.end()) &&
            (current_coarsen_pointer == coarsen_list.end()));
  }



  template <int dim, int spacedim>
  RefineAndCoarsenList<dim, spacedim>::RefineAndCoarsenList(
    const Triangulation<dim, spacedim>                       *triangulation,
    const typename dealii::internal::amr::types<dim>::forest *forest,
    const std::vector<dealii::types::global_dof_index>
                                     &p4est_tree_to_coarse_cell_permutation,
    const dealii::types::subdomain_id my_subdomain)
  {
    // count how many flags are set and allocate that much memory
    unsigned int n_refine_flags = 0, n_coarsen_flags = 0;
    for (const auto &cell : triangulation->active_cell_iterators())
      {
        // skip cells that are not local
        if (cell->subdomain_id() != my_subdomain)
          continue;

        if (cell->refine_flag_set())
          ++n_refine_flags;
        else if (cell->coarsen_flag_set())
          ++n_coarsen_flags;
      }

    refine_list.reserve(n_refine_flags);
    coarsen_list.reserve(n_coarsen_flags);


    // now build the lists of cells that are flagged. note that p4est will
    // traverse its cells in the order in which trees appear in the
    // forest. this order is not the same as the order of coarse cells in the
    // deal.II Triangulation because we have translated everything by the
    // coarse_cell_to_p4est_tree_permutation permutation. in order to make
    // sure that the output array is already in the correct order, traverse
    // our coarse cells in the same order in which p4est will:
    for (unsigned int c = 0; c < triangulation->n_cells(0); ++c)
      {
        unsigned int coarse_cell_index =
          p4est_tree_to_coarse_cell_permutation[c];

        const typename Triangulation<dim, spacedim>::cell_iterator cell(
          triangulation, 0, coarse_cell_index);

        typename dealii::internal::amr::types<dim>::element amr_cell;
        typename dealii::internal::amr::types<dim>::eclass  eclass =
          dealii::internal::amr::get_eclass<dim>(forest, c);

        dealii::internal::amr::element_new<dim>(forest, eclass, &amr_cell, 1);
        dealii::internal::amr::init_coarse_element<dim>(forest,
                                                        eclass,
                                                        amr_cell);
        amr_cell->p.which_tree = c;
        build_lists(forest, eclass, cell, amr_cell, my_subdomain);
        dealii::internal::amr::element_destroy<dim>(forest,
                                                    eclass,
                                                    &amr_cell,
                                                    1);
      }


    Assert(refine_list.size() == n_refine_flags, ExcInternalError());
    Assert(coarsen_list.size() == n_coarsen_flags, ExcInternalError());

    // make sure that our ordering in fact worked
    for (unsigned int i = 1; i < refine_list.size(); ++i)
      Assert(refine_list[i]->p.which_tree >= refine_list[i - 1]->p.which_tree,
             ExcInternalError());
    for (unsigned int i = 1; i < coarsen_list.size(); ++i)
      Assert(coarsen_list[i]->p.which_tree >= coarsen_list[i - 1]->p.which_tree,
             ExcInternalError());

    current_refine_pointer  = refine_list.begin();
    current_coarsen_pointer = coarsen_list.begin();
  }



  template <int dim, int spacedim>
  void
  RefineAndCoarsenList<dim, spacedim>::build_lists(
    const typename dealii::internal::amr::types<dim>::forest   *forest,
    const typename dealii::internal::amr::types<dim>::eclass    eclass,
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const typename dealii::internal::amr::types<dim>::element   amr_cell,
    const types::subdomain_id                                   my_subdomain)
  {
    if (cell->is_active())
      {
        if (cell->subdomain_id() == my_subdomain)
          {
            if (cell->refine_flag_set())
              refine_list.push_back(amr_cell);
            else if (cell->coarsen_flag_set())
              coarsen_list.push_back(amr_cell);
          }
      }
    else
      {
        typename dealii::internal::amr::types<dim>::element
          p4est_child[GeometryInfo<dim>::max_children_per_cell];

        dealii::internal::amr::element_new<dim>(
          forest,
          eclass,
          p4est_child,
          GeometryInfo<dim>::max_children_per_cell);
        dealii::internal::amr::element_children<dim>(forest,
                                                     eclass,
                                                     amr_cell,
                                                     p4est_child);
        for (unsigned int c = 0; c < GeometryInfo<dim>::max_children_per_cell;
             ++c)
          {
            p4est_child[c]->p.which_tree = amr_cell->p.which_tree;
            build_lists(
              forest, eclass, cell->child(c), p4est_child[c], my_subdomain);
          }
      }
  }


  template <int dim, int spacedim>
  int
  RefineAndCoarsenList<dim, spacedim>::refine_callback(
    typename dealii::internal::amr::types<dim>::forest *forest,
    typename dealii::internal::amr::types<dim>::topidx  coarse_cell_index,
    typename dealii::internal::amr::types<dim>::element element)
  {
    RefineAndCoarsenList<dim, spacedim> *this_object =
      reinterpret_cast<RefineAndCoarsenList<dim, spacedim> *>(
        forest->user_pointer);

    // if there are no more cells in our list the current cell can't be
    // flagged for refinement
    if (this_object->current_refine_pointer == this_object->refine_list.end())
      return 0;

    Assert(coarse_cell_index <=
             (*(this_object->current_refine_pointer))->p.which_tree,
           ExcInternalError());

    // if p4est hasn't yet reached the tree of the next flagged cell the
    // current cell can't be flagged for refinement
    if (coarse_cell_index <
        (*(this_object->current_refine_pointer))->p.which_tree)
      return 0;

    // now we're in the right tree in the forest
    Assert(coarse_cell_index <=
             (*(this_object->current_refine_pointer))->p.which_tree,
           ExcInternalError());

    // make sure that the p4est loop over cells hasn't gotten ahead of our own
    // pointer
    Assert(dealii::internal::amr::functions<dim>::element_compare(
             element, *this_object->current_refine_pointer) <= 0,
           ExcInternalError());

    // now, if the p4est cell is one in the list, it is supposed to be refined
    if (dealii::internal::amr::element_is_equal<dim>(
          forest,
          0,
          element,
          *this_object->current_refine_pointer)) // TODO get eclass
      {
        ++this_object->current_refine_pointer;
        return 1;
      }

    // p4est cell is not in list
    return 0;
  }



  template <int dim, int spacedim>
  int
  RefineAndCoarsenList<dim, spacedim>::coarsen_callback(
    typename dealii::internal::amr::types<dim>::forest *forest,
    typename dealii::internal::amr::types<dim>::topidx  coarse_cell_index,
    typename dealii::internal::amr::types<dim>::element children[])
  {
    RefineAndCoarsenList<dim, spacedim> *this_object =
      reinterpret_cast<RefineAndCoarsenList<dim, spacedim> *>(
        forest->user_pointer);

    // if there are no more cells in our list the current cell can't be
    // flagged for coarsening
    if (this_object->current_coarsen_pointer == this_object->coarsen_list.end())
      return 0;

    Assert(coarse_cell_index <=
             (*(this_object->current_coarsen_pointer))->p.which_tree,
           ExcInternalError());

    // if p4est hasn't yet reached the tree of the next flagged cell the
    // current cell can't be flagged for coarsening
    if (coarse_cell_index <
        (*(this_object->current_coarsen_pointer))->p.which_tree)
      return 0;

    // now we're in the right tree in the forest
    Assert(coarse_cell_index <=
             (*(this_object->current_coarsen_pointer))->p.which_tree,
           ExcInternalError());

    // make sure that the p4est loop over cells hasn't gotten ahead of our own
    // pointer
    Assert(dealii::internal::amr::functions<dim>::element_compare(
             &children[0], &*this_object->current_coarsen_pointer) <= 0,
           ExcInternalError());

    // now, if the p4est cell is one in the list, it is supposed to be
    // coarsened
    if (dealii::internal::amr::element_is_equal<dim>(
          forest,
          0,
          children[0],
          *this_object->current_coarsen_pointer)) // TODO eclass
      {
        // move current pointer one up
        ++this_object->current_coarsen_pointer;

        // note that the next 3 cells in our list need to correspond to the
        // other siblings of the cell we have just found
        for (unsigned int c = 1; c < GeometryInfo<dim>::max_children_per_cell;
             ++c)
          {
            Assert(dealii::internal::amr::element_is_equal<dim>(
                     forest,
                     0,
                     children[c],
                     *this_object->current_coarsen_pointer), // TODO: eclass
                   ExcInternalError());
            ++this_object->current_coarsen_pointer;
          }

        return 1;
      }

    // p4est cell is not in list
    return 0;
  }
} // namespace


namespace internal
{
  namespace p4est
  {
    namespace
    {
      template <int dim, int spacedim>
      typename dealii::Triangulation<dim, spacedim>::cell_iterator
      cell_from_quad(
        const dealii::parallel::distributed::Triangulation<dim, spacedim>
          *triangulation,
        const typename dealii::internal::p4est::types<dim>::topidx  treeidx,
        const typename dealii::internal::p4est::types<dim>::element quad)
      {
        int                             i, l = quad->level;
        dealii::types::global_dof_index dealii_index =
          triangulation->get_p4est_tree_to_coarse_cell_permutation()[treeidx];

        for (i = 0; i < l; ++i)
          {
            typename dealii::Triangulation<dim, spacedim>::cell_iterator cell(
              triangulation, i, dealii_index);
            const int child_id =
              dealii::internal::p4est::element_ancestor_id<dim>(quad, i + 1);
            Assert(cell->has_children(),
                   ExcMessage("p4est quadrant does not correspond to a cell!"));
            dealii_index = cell->child_index(child_id);
          }

        typename dealii::Triangulation<dim, spacedim>::cell_iterator out_cell(
          triangulation, l, dealii_index);

        return out_cell;
      }

      /**
       * This is the callback data structure used to fill
       * vertices_with_ghost_neighbors via the p4est_iterate tool
       */
      template <int dim, int spacedim>
      struct FindGhosts
      {
        const typename dealii::parallel::distributed::Triangulation<dim,
                                                                    spacedim>
                   *triangulation;
        sc_array_t *subids;
        std::map<unsigned int, std::set<dealii::types::subdomain_id>>
          *vertices_with_ghost_neighbors;
      };


      /** At a corner (vertex), determine if any of the neighboring cells are
       * ghosts.  If there are, find out their subdomain ids, and if this is a
       * local vertex, then add these subdomain ids to the map
       * vertices_with_ghost_neighbors of that index
       */
      template <int dim, int spacedim>
      void
      find_ghosts_corner(
        typename dealii::internal::p4est::iter<dim>::corner_info *info,
        void                                                     *user_data)
      {
        int   i, j;
        int   nsides = info->sides.elem_count;
        auto *sides  = reinterpret_cast<
          typename dealii::internal::p4est::iter<dim>::corner_side *>(
          info->sides.array);
        FindGhosts<dim, spacedim> *fg =
          static_cast<FindGhosts<dim, spacedim> *>(user_data);
        sc_array_t *subids = fg->subids;
        const dealii::parallel::distributed::Triangulation<dim, spacedim>
                                    *triangulation = fg->triangulation;
        int                          nsubs;
        dealii::types::subdomain_id *subdomain_ids;
        std::map<unsigned int, std::set<dealii::types::subdomain_id>>
          *vertices_with_ghost_neighbors = fg->vertices_with_ghost_neighbors;

        subids->elem_count = 0;
        for (i = 0; i < nsides; ++i)
          {
            if (sides[i].is_ghost)
              {
                typename dealii::parallel::distributed::
                  Triangulation<dim, spacedim>::cell_iterator cell =
                    cell_from_quad(triangulation,
                                   sides[i].treeid,
                                   *(sides[i].quad));
                Assert(cell->is_ghost(),
                       ExcMessage("ghost quad did not find ghost cell"));
                dealii::types::subdomain_id *subid =
                  static_cast<dealii::types::subdomain_id *>(
                    sc_array_push(subids));
                *subid = cell->subdomain_id();
              }
          }

        if (!subids->elem_count)
          {
            return;
          }

        nsubs = static_cast<int>(subids->elem_count);
        subdomain_ids =
          reinterpret_cast<dealii::types::subdomain_id *>(subids->array);

        for (i = 0; i < nsides; ++i)
          {
            if (!sides[i].is_ghost)
              {
                typename dealii::parallel::distributed::
                  Triangulation<dim, spacedim>::cell_iterator cell =
                    cell_from_quad(triangulation,
                                   sides[i].treeid,
                                   *(sides[i].quad));

                Assert(!cell->is_ghost(),
                       ExcMessage("local quad found ghost cell"));

                for (j = 0; j < nsubs; ++j)
                  {
                    (*vertices_with_ghost_neighbors)[cell->vertex_index(
                                                       sides[i].corner)]
                      .insert(subdomain_ids[j]);
                  }
              }
          }

        subids->elem_count = 0;
      }

      /** Similar to find_ghosts_corner, but for the hanging vertex in the
       * middle of an edge
       */
      template <int dim, int spacedim>
      void
      find_ghosts_edge(
        typename dealii::internal::p4est::iter<dim>::edge_info *info,
        void                                                   *user_data)
      {
        int   i, j, k;
        int   nsides = info->sides.elem_count;
        auto *sides  = reinterpret_cast<
          typename dealii::internal::p4est::iter<dim>::edge_side *>(
          info->sides.array);
        auto       *fg = static_cast<FindGhosts<dim, spacedim> *>(user_data);
        sc_array_t *subids = fg->subids;
        const dealii::parallel::distributed::Triangulation<dim, spacedim>
                                    *triangulation = fg->triangulation;
        int                          nsubs;
        dealii::types::subdomain_id *subdomain_ids;
        std::map<unsigned int, std::set<dealii::types::subdomain_id>>
          *vertices_with_ghost_neighbors = fg->vertices_with_ghost_neighbors;

        subids->elem_count = 0;
        for (i = 0; i < nsides; ++i)
          {
            if (sides[i].is_hanging)
              {
                for (j = 0; j < 2; ++j)
                  {
                    if (sides[i].is.hanging.is_ghost[j])
                      {
                        typename dealii::parallel::distributed::
                          Triangulation<dim, spacedim>::cell_iterator cell =
                            cell_from_quad(triangulation,
                                           sides[i].treeid,
                                           *(sides[i].is.hanging.quad[j]));
                        dealii::types::subdomain_id *subid =
                          static_cast<dealii::types::subdomain_id *>(
                            sc_array_push(subids));
                        *subid = cell->subdomain_id();
                      }
                  }
              }
          }

        if (!subids->elem_count)
          {
            return;
          }

        nsubs = static_cast<int>(subids->elem_count);
        subdomain_ids =
          reinterpret_cast<dealii::types::subdomain_id *>(subids->array);

        for (i = 0; i < nsides; ++i)
          {
            if (sides[i].is_hanging)
              {
                for (j = 0; j < 2; ++j)
                  {
                    if (!sides[i].is.hanging.is_ghost[j])
                      {
                        typename dealii::parallel::distributed::
                          Triangulation<dim, spacedim>::cell_iterator cell =
                            cell_from_quad(triangulation,
                                           sides[i].treeid,
                                           *(sides[i].is.hanging.quad[j]));

                        for (k = 0; k < nsubs; ++k)
                          {
                            (*vertices_with_ghost_neighbors)
                              [cell->vertex_index(
                                 p8est_edge_corners[sides[i].edge][1 ^ j])]
                                .insert(subdomain_ids[k]);
                          }
                      }
                  }
              }
          }

        subids->elem_count = 0;
      }

      /** Similar to find_ghosts_corner, but for the hanging vertex in the
       * middle of a face
       */
      template <int dim, int spacedim>
      void
      find_ghosts_face(
        typename dealii::internal::p4est::iter<dim>::face_info *info,
        void                                                   *user_data)
      {
        int   i, j, k;
        int   nsides = info->sides.elem_count;
        auto *sides  = reinterpret_cast<
          typename dealii::internal::p4est::iter<dim>::face_side *>(
          info->sides.array);
        FindGhosts<dim, spacedim> *fg =
          static_cast<FindGhosts<dim, spacedim> *>(user_data);
        sc_array_t *subids = fg->subids;
        const dealii::parallel::distributed::Triangulation<dim, spacedim>
                                    *triangulation = fg->triangulation;
        int                          nsubs;
        dealii::types::subdomain_id *subdomain_ids;
        std::map<unsigned int, std::set<dealii::types::subdomain_id>>
           *vertices_with_ghost_neighbors = fg->vertices_with_ghost_neighbors;
        int limit                         = (dim == 2) ? 2 : 4;

        subids->elem_count = 0;
        for (i = 0; i < nsides; ++i)
          {
            if (sides[i].is_hanging)
              {
                for (j = 0; j < limit; ++j)
                  {
                    if (sides[i].is.hanging.is_ghost[j])
                      {
                        typename dealii::parallel::distributed::
                          Triangulation<dim, spacedim>::cell_iterator cell =
                            cell_from_quad(triangulation,
                                           sides[i].treeid,
                                           *(sides[i].is.hanging.quad[j]));
                        dealii::types::subdomain_id *subid =
                          static_cast<dealii::types::subdomain_id *>(
                            sc_array_push(subids));
                        *subid = cell->subdomain_id();
                      }
                  }
              }
          }

        if (!subids->elem_count)
          {
            return;
          }

        nsubs = static_cast<int>(subids->elem_count);
        subdomain_ids =
          reinterpret_cast<dealii::types::subdomain_id *>(subids->array);

        for (i = 0; i < nsides; ++i)
          {
            if (sides[i].is_hanging)
              {
                for (j = 0; j < limit; ++j)
                  {
                    if (!sides[i].is.hanging.is_ghost[j])
                      {
                        typename dealii::parallel::distributed::
                          Triangulation<dim, spacedim>::cell_iterator cell =
                            cell_from_quad(triangulation,
                                           sides[i].treeid,
                                           *(sides[i].is.hanging.quad[j]));

                        for (k = 0; k < nsubs; ++k)
                          {
                            if (dim == 2)
                              {
                                (*vertices_with_ghost_neighbors)
                                  [cell->vertex_index(
                                     p4est_face_corners[sides[i].face]
                                                       [(limit - 1) ^ j])]
                                    .insert(subdomain_ids[k]);
                              }
                            else
                              {
                                (*vertices_with_ghost_neighbors)
                                  [cell->vertex_index(
                                     p8est_face_corners[sides[i].face]
                                                       [(limit - 1) ^ j])]
                                    .insert(subdomain_ids[k]);
                              }
                          }
                      }
                  }
              }
          }

        subids->elem_count = 0;
      }
    } // namespace


    int (&functions<2>::element_compare)(const void *v1, const void *v2) =
      p4est_quadrant_compare;


    types<2>::connectivity *(&functions<2>::connectivity_new)(
      types<2>::topidx num_vertices,
      types<2>::topidx num_trees,
      types<2>::topidx num_corners,
      types<2>::topidx num_vtt) = p4est_connectivity_new;

    types<2>::connectivity *(&functions<2>::connectivity_new_copy)(
      types<2>::topidx        num_vertices,
      types<2>::topidx        num_trees,
      types<2>::topidx        num_corners,
      const double           *vertices,
      const types<2>::topidx *ttv,
      const types<2>::topidx *ttt,
      const int8_t           *ttf,
      const types<2>::topidx *ttc,
      const types<2>::topidx *coff,
      const types<2>::topidx *ctt,
      const int8_t           *ctc) = p4est_connectivity_new_copy;

    void (&functions<2>::connectivity_join_faces)(types<2>::connectivity *conn,
                                                  types<2>::topidx tree_left,
                                                  types<2>::topidx tree_right,
                                                  int              face_left,
                                                  int              face_right,
                                                  int orientation) =
      p4est_connectivity_join_faces;

    void (&functions<2>::connectivity_destroy)(
      p4est_connectivity_t *connectivity) = p4est_connectivity_destroy;

    types<2>::forest *(&functions<2>::new_forest)(
      MPI_Comm                mpicomm,
      types<2>::connectivity *connectivity,
      types<2>::locidx        min_quadrants,
      int                     min_level,
      int                     fill_uniform,
      std::size_t             data_size,
      p4est_init_t            init_fn,
      void                   *user_pointer) = p4est_new_ext;

    types<2>::forest *(&functions<2>::copy_forest)(types<2>::forest *input,
                                                   int copy_data) = p4est_copy;

    void (&functions<2>::refine)(types<2>::forest *p4est,
                                 int               refine_recursive,
                                 p4est_refine_t    refine_fn,
                                 p4est_init_t      init_fn) = p4est_refine;

    void (&functions<2>::coarsen)(types<2>::forest *p4est,
                                  int               coarsen_recursive,
                                  p4est_coarsen_t   coarsen_fn,
                                  p4est_init_t      init_fn) = p4est_coarsen;

    void (&functions<2>::save)(const char       *filename,
                               types<2>::forest *p4est,
                               int               save_data) = p4est_save;

    types<2>::forest *(&functions<2>::load_ext)(
      const char              *filename,
      MPI_Comm                 mpicomm,
      std::size_t              data_size,
      int                      load_data,
      int                      autopartition,
      int                      broadcasthead,
      void                    *user_pointer,
      types<2>::connectivity **p4est) = p4est_load_ext;

    int (&functions<2>::connectivity_save)(
      const char             *filename,
      types<2>::connectivity *connectivity) = p4est_connectivity_save;

    int (&functions<2>::connectivity_is_valid)(
      types<2>::connectivity *connectivity) = p4est_connectivity_is_valid;

    types<2>::connectivity *(&functions<2>::connectivity_load)(
      const char  *filename,
      std::size_t *length) = p4est_connectivity_load;

    unsigned int (&functions<2>::checksum)(types<2>::forest *p4est) =
      p4est_checksum;

    std::size_t (&functions<2>::forest_memory_used)(types<2>::forest *p4est) =
      p4est_memory_used;

    std::size_t (&functions<2>::connectivity_memory_used)(
      types<2>::connectivity *p4est) = p4est_connectivity_memory_used;


    void (&functions<2>::transfer_fixed)(const types<2>::gloidx *dest_gfq,
                                         const types<2>::gloidx *src_gfq,
                                         MPI_Comm                mpicomm,
                                         int                     tag,
                                         void                   *dest_data,
                                         const void             *src_data,
                                         std::size_t             data_size) =
      p4est_transfer_fixed;

    types<2>::transfer_context *(&functions<2>::transfer_fixed_begin)(
      const types<2>::gloidx *dest_gfq,
      const types<2>::gloidx *src_gfq,
      MPI_Comm                mpicomm,
      int                     tag,
      void                   *dest_data,
      const void             *src_data,
      std::size_t             data_size) = p4est_transfer_fixed_begin;

    void (&functions<2>::transfer_fixed_end)(types<2>::transfer_context *tc) =
      p4est_transfer_fixed_end;

    void (&functions<2>::transfer_custom)(const types<2>::gloidx *dest_gfq,
                                          const types<2>::gloidx *src_gfq,
                                          MPI_Comm                mpicomm,
                                          int                     tag,
                                          void                   *dest_data,
                                          const int              *dest_sizes,
                                          const void             *src_data,
                                          const int              *src_sizes) =
      p4est_transfer_custom;

    types<2>::transfer_context *(&functions<2>::transfer_custom_begin)(
      const types<2>::gloidx *dest_gfq,
      const types<2>::gloidx *src_gfq,
      MPI_Comm                mpicomm,
      int                     tag,
      void                   *dest_data,
      const int              *dest_sizes,
      const void             *src_data,
      const int              *src_sizes) = p4est_transfer_custom_begin;

    void (&functions<2>::transfer_custom_end)(types<2>::transfer_context *tc) =
      p4est_transfer_custom_end;

    void (&functions<2>::search_partition)(
      types<2>::forest                   *p4est,
      int                                 call_post,
      types<2>::search_partition_callback quadrant_fn,
      types<2>::search_partition_callback point_fn,
      sc_array_t                         *points) = p4est_search_partition;

    void (&functions<2>::element_coord_to_vertex)(
      types<2>::connectivity *connectivity,
      types<2>::topidx        treeid,
      types<2>::element_coord x,
      types<2>::element_coord y,
      double                  vxyz[3]) = p4est_qcoord_to_vertex;

    int (&functions<3>::element_compare)(const void *v1, const void *v2) =
      p8est_quadrant_compare;

    types<3>::connectivity *(&functions<3>::connectivity_new)(
      types<3>::topidx num_vertices,
      types<3>::topidx num_trees,
      types<3>::topidx num_edges,
      types<3>::topidx num_ett,
      types<3>::topidx num_corners,
      types<3>::topidx num_ctt) = p8est_connectivity_new;

    types<3>::connectivity *(&functions<3>::connectivity_new_copy)(
      types<3>::topidx        num_vertices,
      types<3>::topidx        num_trees,
      types<3>::topidx        num_edges,
      types<3>::topidx        num_corners,
      const double           *vertices,
      const types<3>::topidx *ttv,
      const types<3>::topidx *ttt,
      const int8_t           *ttf,
      const types<3>::topidx *tte,
      const types<3>::topidx *eoff,
      const types<3>::topidx *ett,
      const int8_t           *ete,
      const types<3>::topidx *ttc,
      const types<3>::topidx *coff,
      const types<3>::topidx *ctt,
      const int8_t           *ctc) = p8est_connectivity_new_copy;

    void (&functions<3>::connectivity_destroy)(
      p8est_connectivity_t *connectivity) = p8est_connectivity_destroy;

    void (&functions<3>::connectivity_join_faces)(types<3>::connectivity *conn,
                                                  types<3>::topidx tree_left,
                                                  types<3>::topidx tree_right,
                                                  int              face_left,
                                                  int              face_right,
                                                  int orientation) =
      p8est_connectivity_join_faces;

    types<3>::forest *(&functions<3>::new_forest)(
      MPI_Comm                mpicomm,
      types<3>::connectivity *connectivity,
      types<3>::locidx        min_quadrants,
      int                     min_level,
      int                     fill_uniform,
      std::size_t             data_size,
      p8est_init_t            init_fn,
      void                   *user_pointer) = p8est_new_ext;

    types<3>::forest *(&functions<3>::copy_forest)(types<3>::forest *input,
                                                   int copy_data) = p8est_copy;

    void (&functions<3>::refine)(types<3>::forest *p8est,
                                 int               refine_recursive,
                                 p8est_refine_t    refine_fn,
                                 p8est_init_t      init_fn) = p8est_refine;

    void (&functions<3>::coarsen)(types<3>::forest *p8est,
                                  int               coarsen_recursive,
                                  p8est_coarsen_t   coarsen_fn,
                                  p8est_init_t      init_fn) = p8est_coarsen;

    void (&functions<3>::save)(const char       *filename,
                               types<3>::forest *p4est,
                               int               save_data) = p8est_save;

    types<3>::forest *(&functions<3>::load_ext)(
      const char              *filename,
      MPI_Comm                 mpicomm,
      std::size_t              data_size,
      int                      load_data,
      int                      autopartition,
      int                      broadcasthead,
      void                    *user_pointer,
      types<3>::connectivity **p4est) = p8est_load_ext;

    int (&functions<3>::connectivity_save)(
      const char             *filename,
      types<3>::connectivity *connectivity) = p8est_connectivity_save;

    int (&functions<3>::connectivity_is_valid)(
      types<3>::connectivity *connectivity) = p8est_connectivity_is_valid;

    types<3>::connectivity *(&functions<3>::connectivity_load)(
      const char  *filename,
      std::size_t *length) = p8est_connectivity_load;

    unsigned int (&functions<3>::checksum)(types<3>::forest *p8est) =
      p8est_checksum;

    std::size_t (&functions<3>::forest_memory_used)(types<3>::forest *p4est) =
      p8est_memory_used;

    std::size_t (&functions<3>::connectivity_memory_used)(
      types<3>::connectivity *p4est) = p8est_connectivity_memory_used;



    void (&functions<3>::transfer_fixed)(const types<3>::gloidx *dest_gfq,
                                         const types<3>::gloidx *src_gfq,
                                         MPI_Comm                mpicomm,
                                         int                     tag,
                                         void                   *dest_data,
                                         const void             *src_data,
                                         std::size_t             data_size) =
      p8est_transfer_fixed;

    types<3>::transfer_context *(&functions<3>::transfer_fixed_begin)(
      const types<3>::gloidx *dest_gfq,
      const types<3>::gloidx *src_gfq,
      MPI_Comm                mpicomm,
      int                     tag,
      void                   *dest_data,
      const void             *src_data,
      std::size_t             data_size) = p8est_transfer_fixed_begin;

    void (&functions<3>::transfer_fixed_end)(types<3>::transfer_context *tc) =
      p8est_transfer_fixed_end;

    void (&functions<3>::transfer_custom)(const types<3>::gloidx *dest_gfq,
                                          const types<3>::gloidx *src_gfq,
                                          MPI_Comm                mpicomm,
                                          int                     tag,
                                          void                   *dest_data,
                                          const int              *dest_sizes,
                                          const void             *src_data,
                                          const int              *src_sizes) =
      p8est_transfer_custom;

    types<3>::transfer_context *(&functions<3>::transfer_custom_begin)(
      const types<3>::gloidx *dest_gfq,
      const types<3>::gloidx *src_gfq,
      MPI_Comm                mpicomm,
      int                     tag,
      void                   *dest_data,
      const int              *dest_sizes,
      const void             *src_data,
      const int              *src_sizes) = p8est_transfer_custom_begin;

    void (&functions<3>::transfer_custom_end)(types<3>::transfer_context *tc) =
      p8est_transfer_custom_end;

    void (&functions<3>::search_partition)(
      types<3>::forest                   *p4est,
      int                                 call_post,
      types<3>::search_partition_callback quadrant_fn,
      types<3>::search_partition_callback point_fn,
      sc_array_t                         *points) = p8est_search_partition;

    void (&functions<3>::element_coord_to_vertex)(
      types<3>::connectivity *connectivity,
      types<3>::topidx        treeid,
      types<3>::element_coord x,
      types<3>::element_coord y,
      types<3>::element_coord z,
      double                  vxyz[3]) = p8est_qcoord_to_vertex;

    template <int dim>
    typename types<dim>::element
    get_ghost_elem_and_owner(const typename types<dim>::forest *,
                             const typename types<dim>::topidx,
                             const typename types<dim>::locidx,
                             const typename types<dim>::eclass,
                             dealii::types::subdomain_id &)
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    template <int dim>
    typename types<dim>::gloidx
    tree_get_offset(const typename types<dim>::tree tree)
    {
      return tree->quadrants_offset;
    }

    template <int dim>
    typename types<dim>::eclass
    get_ghost_eclass(const typename types<dim>::forest *,
                     const typename types<dim>::locidx)
    {
      return 0;
    }

    template <int dim>
    void
    init_coarse_element(const typename types<dim>::forest *,
                        typename types<dim>::eclass,
                        typename types<dim>::element element)
    {
      if constexpr (dim == 2)
        {
          P4EST_QUADRANT_INIT(element);
          p4est_quadrant_set_morton(element,
                                    /*level=*/0,
                                    /*index=*/0);
        }
      if constexpr (dim == 3)
        {
          P8EST_QUADRANT_INIT(element);
          p8est_quadrant_set_morton(element,
                                    /*level=*/0,
                                    /*index=*/0);
        }
    }

    template <int dim>
    bool
    element_is_equal(const typename types<dim>::forest *,
                     typename types<dim>::eclass,
                     typename types<dim>::element element_1,
                     typename types<dim>::element element_2)
    {
      if constexpr (dim == 2)
        return static_cast<bool>(p4est_quadrant_is_equal(element_1, element_2));
      else if (dim == 3)
        return static_cast<bool>(p8est_quadrant_is_equal(element_1, element_2));

      DEAL_II_NOT_IMPLEMENTED();
      // return element_1 == element_2; 1D case
      return false;
    }

    template <int dim>
    bool
    tree_exists_locally(const typename types<dim>::forest *parallel_forest,
                        const typename types<dim>::topidx  coarse_grid_cell)
    {
      Assert(coarse_grid_cell < parallel_forest->connectivity->num_trees,
             ExcInternalError());
      return ((coarse_grid_cell >= parallel_forest->first_local_tree) &&
              (coarse_grid_cell <= parallel_forest->last_local_tree));
    }

    template <int dim>
    typename types<dim>::locidx
    leaf_index_in_tree(const typename types<dim>::forest *forest,
                       const typename types<dim>::locidx  ltreeid,
                       const typename types<dim>::element leaf)
    {
      const typename types<dim>::tree tree =
        forest_get_tree<dim>(forest, ltreeid);
      return sc_array_bsearch(const_cast<sc_array_t *>(&tree->quadrants),
                              leaf,
                              functions<dim>::element_compare);
    }


    template <int dim>
    typename types<dim>::locidx
    get_num_leafs(const typename types<dim>::forest *forest)
    {
      return forest->local_num_quadrants;
    }

    template <int dim>
    typename types<dim>::tree
    forest_get_tree(const typename types<dim>::forest *forest,
                    const typename types<dim>::locidx  ltreeid)
    {
      return static_cast<typename types<dim>::tree>(
        sc_array_index(forest->trees, ltreeid));
    }

    template <int dim, int spacedim>
    typename types<dim>::forest *
    adapt(typename types<dim>::forest  *parallel_forest,
          Triangulation<dim, spacedim> *triangulation,
          const std::vector<dealii::types::global_dof_index>
            &p4est_tree_to_coarse_cell_permutation,
          const dealii::types::subdomain_id subdomain_id)
    {
      // count how many cells will be refined and coarsened, and allocate that
      // much memory
      RefineAndCoarsenList<dim, spacedim> refine_and_coarsen_list(
        triangulation,
        parallel_forest,
        p4est_tree_to_coarse_cell_permutation,
        subdomain_id);

      // copy refine and coarsen flags into p4est and execute the refinement
      // and coarsening. this uses the refine_and_coarsen_list just built,
      // which is communicated to the callback functions through
      // p4est's user_pointer object
      Assert(forest_get_user_pointer<dim>(parallel_forest) == triangulation,
             ExcInternalError());
      forest_set_user_pointer<dim>(parallel_forest, &refine_and_coarsen_list);

      functions<dim>::refine(
        parallel_forest,
        /* refine_recursive */ false,
        &RefineAndCoarsenList<dim, spacedim>::refine_callback,
        /*init_callback=*/nullptr);
      functions<dim>::coarsen(
        parallel_forest,
        /* coarsen_recursive */ false,
        &RefineAndCoarsenList<dim, spacedim>::coarsen_callback,
        /*init_callback=*/nullptr);
      // make sure all cells in the lists have been consumed
      Assert(refine_and_coarsen_list.pointers_are_at_end(), ExcInternalError());

      // reset the pointer
      forest_set_user_pointer<dim>(parallel_forest, triangulation);

      return parallel_forest;
    }


    template <int dim>
    typename types<dim>::eclass
    get_eclass(const typename types<dim>::forest *, typename types<dim>::locidx)
    {
      return 0;
    }

    template <int dim>
    typename types<dim>::eclass
    get_eclass_from_tree(const typename types<dim>::tree)
    {
      return 0;
    }

    template <int dim>
    void
    element_children(const typename types<dim>::forest *,
                     typename types<dim>::eclass,
                     const typename types<dim>::element element,
                     typename types<dim>::element      *children)
    {
      for (unsigned int i = 0; i < GeometryInfo<dim>::max_children_per_cell;
           ++i)
        {
          if constexpr (dim == 2)
            p4est_quadrant_child(element, children[i], i);
          else if (dim == 3)
            p8est_quadrant_child(element, children[i], i);
          else
            DEAL_II_NOT_IMPLEMENTED();
        }
    }

    template <int dim>
    void
    element_new(const typename types<dim>::forest *,
                typename types<dim>::eclass,
                typename types<dim>::element *elements,
                const unsigned int            length)
    {
      using element_type = std::remove_pointer_t<typename types<dim>::element>;
      *elements          = new element_type[length];

      if constexpr (running_in_debug_mode())
        for (unsigned int i = 0; i < length; ++i)
          {
            if constexpr (dim == 2)
              P4EST_QUADRANT_INIT(elements[i]);
            else if (dim == 3)
              P8EST_QUADRANT_INIT(elements[i]);
            else
              DEAL_II_NOT_IMPLEMENTED();
          }
    }

    template <int dim>
    int
    element_level(const typename types<dim>::forest *,
                  typename types<dim>::eclass,
                  const typename types<dim>::element element)
    {
      return element->level;
    }

    template <int dim>
    bool
    cell_exists_in_tree(const typename types<dim>::tree    tree,
                        const typename types<dim>::element element)
    {
      return (
        sc_array_bsearch(const_cast<sc_array_t *>(&tree->quadrants),
                         element,
                         internal::p4est::functions<dim>::element_compare) !=
        -1);
    }

    template <int dim>
    bool
    element_overlaps_tree(const typename types<dim>::forest *,
                          const typename types<dim>::tree    tree,
                          const typename types<dim>::element element)
    {
      if constexpr (dim == 2)
        return p4est_quadrant_overlaps_tree(tree, element);
      else if (dim == 3)
        return p8est_quadrant_overlaps_tree(tree, element);

      DEAL_II_NOT_IMPLEMENTED();
      return false;
    }

    template <int dim>
    void
    element_destroy(const typename types<dim>::forest *,
                    typename types<dim>::eclass,
                    typename types<dim>::element *element,
                    const unsigned int)
    {
      delete[] element;
    }

    template <int dim>
    typename types<dim>::ghost *
    ghost_new(typename types<dim>::forest *forest)
    {
      if constexpr (dim == 2)
        return p4est_ghost_new(forest, P4EST_CONNECT_CORNER);
      else if (dim == 3)
        return p8est_ghost_new(forest, P8EST_CONNECT_CORNER);
      else
        DEAL_II_NOT_IMPLEMENTED();

      return nullptr;
    }

    template <int dim>
    void
    ghost_destroy(typename types<dim>::ghost **ghost)
    {
      if constexpr (dim == 2)
        return p4est_ghost_destroy(*ghost);
      else if (dim == 3)
        return p8est_ghost_destroy(*ghost);
      else
        DEAL_II_NOT_IMPLEMENTED();
    }

    template <int dim>
    int
    element_ancestor_id(const typename types<dim>::forest *,
                        typename types<dim>::eclass,
                        const typename types<dim>::element element,
                        int                                level)
    {
      if constexpr (dim == 2)
        return p4est_quadrant_ancestor_id(element, level);
      else if (dim == 3)
        return p8est_quadrant_ancestor_id(element, level);
      else
        DEAL_II_NOT_IMPLEMENTED();
      return -1;
    }



    template <int dim>
    int
    comm_find_owner(const typename types<dim>::forest *forest,
                    const typename types<dim>::locidx  which_tree,
                    const typename types<dim>::element element,
                    const int                          guess)
    {
      if constexpr (dim == 2)
        return p4est_comm_find_owner(const_cast<typename types<dim>::forest *>(
                                       forest),
                                     which_tree,
                                     element,
                                     guess);
      else if (dim == 3)
        return p8est_comm_find_owner(const_cast<typename types<dim>::forest *>(
                                       forest),
                                     which_tree,
                                     element,
                                     guess);
      else
        DEAL_II_NOT_IMPLEMENTED();
      return -1;
    }

    template <int dim>
    void
    forest_set_user_pointer(typename types<dim>::forest *forest,
                            void                        *user_pointer)
    {
      forest->user_pointer = user_pointer;
    };

    template <int dim>
    void *
    forest_get_user_pointer(const typename types<dim>::forest *forest)
    {
      return forest->user_pointer;
    };



    template <int dim>
    void
    forest_destroy(typename types<dim>::forest **forest)
    {
      if constexpr (dim == 2)
        return p4est_destroy(*forest);
      else if (dim == 3)
        return p8est_destroy(*forest);
      else
        DEAL_II_NOT_IMPLEMENTED();
    }


    template <int dim>
    typename types<dim>::forest *
    balance_full(typename types<dim>::forest *forest)
    {
      if constexpr (dim == 2)
        {
          p4est_balance(forest, P4EST_CONNECT_FULL, nullptr);
          return forest;
        }
      else if (dim == 3)
        {
          p8est_balance(forest, P8EST_CONNECT_FULL, nullptr);
          return forest;
        }
      else
        DEAL_II_NOT_IMPLEMENTED();

      return nullptr;
    }

    template <int dim>
    typename types<dim>::forest *
    partition(typename types<dim>::forest *forest,
              typename types<dim>::weight  weight_fn)
    {
      if constexpr (dim == 2)
        {
          p4est_partition_ext(forest, 1, weight_fn);
          return forest;
        }
      else if (dim == 3)
        {
          p8est_partition_ext(forest, 1, weight_fn);
          return forest;
        }
      else
        DEAL_II_NOT_IMPLEMENTED();
      return nullptr;
    }


    template <int dim>
    void
    vtk_write_file(typename types<dim>::forest *forest, const char *baseName)
    {
      if constexpr (dim == 2)
        {
          p4est_vtk_write_file(forest, nullptr, baseName);
        }
      else if (dim == 3)
        {
          p8est_vtk_write_file(forest, nullptr, baseName);
        }
      else
        DEAL_II_NOT_IMPLEMENTED();
    }



    // template specializations

    template <>
    typename types<2>::connectivity *
    copy_connectivity<2>(const typename types<2>::connectivity *connectivity)
    {
      return functions<2>::connectivity_new_copy(
        connectivity->num_vertices,
        connectivity->num_trees,
        connectivity->num_corners,
        connectivity->vertices,
        connectivity->tree_to_vertex,
        connectivity->tree_to_tree,
        connectivity->tree_to_face,
        connectivity->tree_to_corner,
        connectivity->ctt_offset,
        connectivity->corner_to_tree,
        connectivity->corner_to_corner);
    }

    template <>
    typename types<3>::connectivity *
    copy_connectivity<3>(const typename types<3>::connectivity *connectivity)
    {
      return functions<3>::connectivity_new_copy(
        connectivity->num_vertices,
        connectivity->num_trees,
        connectivity->num_edges,
        connectivity->num_corners,
        connectivity->vertices,
        connectivity->tree_to_vertex,
        connectivity->tree_to_tree,
        connectivity->tree_to_face,
        connectivity->tree_to_edge,
        connectivity->ett_offset,
        connectivity->edge_to_tree,
        connectivity->edge_to_edge,
        connectivity->tree_to_corner,
        connectivity->ctt_offset,
        connectivity->corner_to_tree,
        connectivity->corner_to_corner);
    }
  } // namespace p4est
} // namespace internal

#endif // DEAL_II_WITH_P4EST

/*-------------- Explicit Instantiations -------------------------------*/
#include "distributed/p4est_wrappers.inst"


DEAL_II_NAMESPACE_CLOSE
