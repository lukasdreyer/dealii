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

namespace internal
{
  namespace t8code
  {
    namespace
    {
      template <int dim, int spacedim>
      typename dealii::Triangulation<dim, spacedim>::cell_iterator
      cell_from_elem(
        const dealii::parallel::distributed::Triangulation<dim, spacedim>
          *triangulation,
        const typename dealii::internal::t8code::types<dim>::topidx    treeidx,
        const typename dealii::internal::t8code::types<dim>::element &elem)
      {
        int                             i, l = elem.level;
        dealii::types::global_dof_index dealii_index =
          triangulation->get_t8code_tree_to_coarse_cell_permutation()[treeidx];

        for (i = 0; i < l; ++i)
          {
            typename dealii::Triangulation<dim, spacedim>::cell_iterator cell(
              triangulation, i, dealii_index);
            const int child_id =
              dealii::internal::t8code::functions<dim>::element_ancestor_id(
                &elem, i + 1);
            Assert(cell->has_children(),
                   ExcMessage("t8code element does not correspond to a cell!"));
            dealii_index = cell->child_index(child_id);
          }

        typename dealii::Triangulation<dim, spacedim>::cell_iterator out_cell(
          triangulation, l, dealii_index);

        return out_cell;
      }

      /**
       * This is the callback data structure used to fill
       * vertices_with_ghost_neighbors via the t8code_iterate tool
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

#if 0
      /** At a corner (vertex), determine if any of the neighboring cells are
       * ghosts.  If there are, find out their subdomain ids, and if this is a
       * local vertex, then add these subdomain ids to the map
       * vertices_with_ghost_neighbors of that index
       */
      template <int dim, int spacedim>
      void
      find_ghosts_corner(
        typename dealii::internal::t8code::iter<dim>::corner_info *info,
        void                                                     *user_data)
      {
        int   i, j;
        int   nsides = info->sides.elem_count;
        auto *sides  = reinterpret_cast<
          typename dealii::internal::t8code::iter<dim>::corner_side *>(
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
                    cell_from_elem(triangulation,
                                   sides[i].treeid,
                                   *(sides[i].elem));
                Assert(cell->is_ghost(),
                       ExcMessage("ghost elem did not find ghost cell"));
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
                    cell_from_elem(triangulation,
                                   sides[i].treeid,
                                   *(sides[i].elem));

                Assert(!cell->is_ghost(),
                       ExcMessage("local elem found ghost cell"));

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
        typename dealii::internal::t8code::iter<dim>::edge_info *info,
        void                                                   *user_data)
      {
        int   i, j, k;
        int   nsides = info->sides.elem_count;
        auto *sides  = reinterpret_cast<
          typename dealii::internal::t8code::iter<dim>::edge_side *>(
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
                            cell_from_elem(triangulation,
                                           sides[i].treeid,
                                           *(sides[i].is.hanging.elem[j]));
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
                            cell_from_elem(triangulation,
                                           sides[i].treeid,
                                           *(sides[i].is.hanging.elem[j]));

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
#endif
      /** Similar to find_ghosts_corner, but for the hanging vertex in the
       * middle of a face
       */
      template <int dim, int spacedim>
      void
      find_ghosts_face(
        typename dealii::internal::t8code::iter<dim>::face_info *info,
        void                                                   *user_data)
      {
        int   i, j, k;
        int   nsides = info->sides.elem_count;
        auto *sides  = reinterpret_cast<
          typename dealii::internal::t8code::iter<dim>::face_side *>(
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
                            cell_from_elem(triangulation,
                                           sides[i].treeid,
                                           *(sides[i].is.hanging.elem[j]));
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
                            cell_from_elem(triangulation,
                                           sides[i].treeid,
                                           *(sides[i].is.hanging.elem[j]));

                        for (k = 0; k < nsubs; ++k)
                          {
                            if (dim == 2)
                              {
                                (*vertices_with_ghost_neighbors)
                                  [cell->vertex_index(
                                     t8code_face_corners[sides[i].face]
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
      t8_element_compare;

    void (&functions<2>::element_childrenv)(const types<2>::element *q,
                                             types<2>::element        c[]) =
      t8code_element_childrenv;

    int (&functions<2>::element_overlaps_tree)(types<2>::tree           *tree,
                                                const types<2>::element *q) =
      t8code_element_overlaps_tree;

    void (&functions<2>::element_set_morton)(types<2>::element *element,
                                              int                 level,
                                              std::uint64_t       id) =
      t8code_element_set_morton;

    int (&functions<2>::element_is_equal)(const types<2>::element *q1,
                                           const types<2>::element *q2) =
      t8_element_is_equal;

    int (&functions<2>::element_is_sibling)(const types<2>::element *q1,
                                             const types<2>::element *q2) =
      t8code_element_is_sibling;

    int (&functions<2>::element_is_ancestor)(const types<2>::element *q1,
                                              const types<2>::element *q2) =
      t8code_element_is_ancestor;

    int (&functions<2>::element_ancestor_id)(const types<2>::element *q,
                                              int                       level) =
      t8code_element_ancestor_id;

    int (&functions<2>::comm_find_owner)(types<2>::forest         *t8code,
                                         const types<2>::locidx    which_tree,
                                         const types<2>::element *q,
                                         const int                 guess) =
      t8code_comm_find_owner;

    types<2>::connectivity *(&functions<2>::connectivity_new)(
      types<2>::topidx num_vertices,
      types<2>::topidx num_trees,
      types<2>::topidx num_corners,
      types<2>::topidx num_vtt) = t8code_connectivity_new;

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
      const int8_t           *ctc) = t8code_connectivity_new_copy;

    void (&functions<2>::connectivity_join_faces)(types<2>::connectivity *conn,
                                                  types<2>::topidx tree_left,
                                                  types<2>::topidx tree_right,
                                                  int              face_left,
                                                  int              face_right,
                                                  int orientation) =
      t8code_connectivity_join_faces;

    void (&functions<2>::connectivity_destroy)(
      t8code_connectivity_t *connectivity) = t8code_connectivity_destroy;

    types<2>::forest *(&functions<2>::new_forest)(
      MPI_Comm                mpicomm,
      types<2>::connectivity *connectivity,
      types<2>::locidx        min_elements,
      int                     min_level,
      int                     fill_uniform,
      std::size_t             data_size,
      t8code_init_t            init_fn,
      void                   *user_pointer) = t8code_new_ext;

    types<2>::forest *(&functions<2>::copy_forest)(types<2>::forest *input,
                                                   int copy_data) = t8code_copy;

    void (&functions<2>::destroy)(types<2>::forest *t8code) = t8code_destroy;

    void (&functions<2>::refine)(types<2>::forest *t8code,
                                 int               refine_recursive,
                                 t8code_refine_t    refine_fn,
                                 t8code_init_t      init_fn) = t8code_refine;

    void (&functions<2>::coarsen)(types<2>::forest *t8code,
                                  int               coarsen_recursive,
                                  t8code_coarsen_t   coarsen_fn,
                                  t8code_init_t      init_fn) = t8code_coarsen;

    void (&functions<2>::balance)(types<2>::forest      *t8code,
                                  types<2>::balance_type btype,
                                  t8code_init_t init_fn) = t8code_balance;

    types<2>::gloidx (&functions<2>::partition)(types<2>::forest *t8code,
                                                int partition_for_coarsening,
                                                t8code_weight_t weight_fn) =
      t8code_partition_ext;

    void (&functions<2>::save)(const char       *filename,
                               types<2>::forest *t8code,
                               int               save_data) = t8code_save;

    types<2>::forest *(&functions<2>::load_ext)(
      const char              *filename,
      MPI_Comm                 mpicomm,
      std::size_t              data_size,
      int                      load_data,
      int                      autopartition,
      int                      broadcasthead,
      void                    *user_pointer,
      types<2>::connectivity **t8code) = t8code_load_ext;

    int (&functions<2>::connectivity_save)(
      const char             *filename,
      types<2>::connectivity *connectivity) = t8code_connectivity_save;

    int (&functions<2>::connectivity_is_valid)(
      types<2>::connectivity *connectivity) = t8code_connectivity_is_valid;

    types<2>::connectivity *(&functions<2>::connectivity_load)(
      const char  *filename,
      std::size_t *length) = t8code_connectivity_load;

    unsigned int (&functions<2>::checksum)(types<2>::forest *t8code) =
      t8code_checksum;

    void (&functions<2>::vtk_write_file)(types<2>::forest *t8code,
                                         t8code_geometry_t *,
                                         const char *baseName) =
      t8code_vtk_write_file;

    types<2>::ghost *(&functions<2>::ghost_new)(types<2>::forest      *t8code,
                                                types<2>::balance_type btype) =
      t8code_ghost_new;

    void (&functions<2>::ghost_destroy)(types<2>::ghost *ghost) =
      t8code_ghost_destroy;

    void (&functions<2>::reset_data)(types<2>::forest *t8code,
                                     std::size_t       data_size,
                                     t8code_init_t      init_fn,
                                     void *user_pointer) = t8code_reset_data;

    std::size_t (&functions<2>::forest_memory_used)(types<2>::forest *t8code) =
      t8code_memory_used;

    std::size_t (&functions<2>::connectivity_memory_used)(
      types<2>::connectivity *t8code) = t8code_connectivity_memory_used;

    constexpr unsigned int functions<2>::max_level;

    void (&functions<2>::transfer_fixed)(const types<2>::gloidx *dest_gfq,
                                         const types<2>::gloidx *src_gfq,
                                         MPI_Comm                mpicomm,
                                         int                     tag,
                                         void                   *dest_data,
                                         const void             *src_data,
                                         std::size_t             data_size) =
      t8code_transfer_fixed;

    types<2>::transfer_context *(&functions<2>::transfer_fixed_begin)(
      const types<2>::gloidx *dest_gfq,
      const types<2>::gloidx *src_gfq,
      MPI_Comm                mpicomm,
      int                     tag,
      void                   *dest_data,
      const void             *src_data,
      std::size_t             data_size) = t8code_transfer_fixed_begin;

    void (&functions<2>::transfer_fixed_end)(types<2>::transfer_context *tc) =
      t8code_transfer_fixed_end;

    void (&functions<2>::transfer_custom)(const types<2>::gloidx *dest_gfq,
                                          const types<2>::gloidx *src_gfq,
                                          MPI_Comm                mpicomm,
                                          int                     tag,
                                          void                   *dest_data,
                                          const int              *dest_sizes,
                                          const void             *src_data,
                                          const int              *src_sizes) =
      t8code_transfer_custom;

    types<2>::transfer_context *(&functions<2>::transfer_custom_begin)(
      const types<2>::gloidx *dest_gfq,
      const types<2>::gloidx *src_gfq,
      MPI_Comm                mpicomm,
      int                     tag,
      void                   *dest_data,
      const int              *dest_sizes,
      const void             *src_data,
      const int              *src_sizes) = t8code_transfer_custom_begin;

    void (&functions<2>::transfer_custom_end)(types<2>::transfer_context *tc) =
      t8code_transfer_custom_end;

#  ifdef T8CODE_SEARCH_LOCAL
    void (&functions<2>::search_partition)(
      types<2>::forest                   *t8code,
      int                                 call_post,
      types<2>::search_partition_callback element_fn,
      types<2>::search_partition_callback point_fn,
      sc_array_t                         *points) = t8code_search_partition;
#  endif

    void (&functions<2>::element_coord_to_vertex)(
      types<2>::connectivity  *connectivity,
      types<2>::topidx         treeid,
      types<2>::element_coord x,
      types<2>::element_coord y,
      double                   vxyz[3]) = t8code_qcoord_to_vertex;

    int (&functions<3>::element_compare)(const void *v1, const void *v2) =
      p8est_element_compare;

    void (&functions<3>::element_childrenv)(const types<3>::element *q,
                                             types<3>::element        c[]) =
      p8est_element_childrenv;

    int (&functions<3>::element_overlaps_tree)(types<3>::tree           *tree,
                                                const types<3>::element *q) =
      p8est_element_overlaps_tree;

    void (&functions<3>::element_set_morton)(types<3>::element *element,
                                              int                 level,
                                              std::uint64_t       id) =
      p8est_element_set_morton;

    int (&functions<3>::element_is_equal)(const types<3>::element *q1,
                                           const types<3>::element *q2) =
      p8est_element_is_equal;

    int (&functions<3>::element_is_sibling)(const types<3>::element *q1,
                                             const types<3>::element *q2) =
      p8est_element_is_sibling;

    int (&functions<3>::element_is_ancestor)(const types<3>::element *q1,
                                              const types<3>::element *q2) =
      p8est_element_is_ancestor;

    int (&functions<3>::element_ancestor_id)(const types<3>::element *q,
                                              int                       level) =
      p8est_element_ancestor_id;

    int (&functions<3>::comm_find_owner)(types<3>::forest         *t8code,
                                         const types<3>::locidx    which_tree,
                                         const types<3>::element *q,
                                         const int                 guess) =
      p8est_comm_find_owner;

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
      types<3>::locidx        min_elements,
      int                     min_level,
      int                     fill_uniform,
      std::size_t             data_size,
      p8est_init_t            init_fn,
      void                   *user_pointer) = p8est_new_ext;

    types<3>::forest *(&functions<3>::copy_forest)(types<3>::forest *input,
                                                   int copy_data) = p8est_copy;

    void (&functions<3>::destroy)(types<3>::forest *p8est) = p8est_destroy;

    void (&functions<3>::refine)(types<3>::forest *p8est,
                                 int               refine_recursive,
                                 p8est_refine_t    refine_fn,
                                 p8est_init_t      init_fn) = p8est_refine;

    void (&functions<3>::coarsen)(types<3>::forest *p8est,
                                  int               coarsen_recursive,
                                  p8est_coarsen_t   coarsen_fn,
                                  p8est_init_t      init_fn) = p8est_coarsen;

    void (&functions<3>::balance)(types<3>::forest      *p8est,
                                  types<3>::balance_type btype,
                                  p8est_init_t init_fn) = p8est_balance;

    types<3>::gloidx (&functions<3>::partition)(types<3>::forest *p8est,
                                                int partition_for_coarsening,
                                                p8est_weight_t weight_fn) =
      p8est_partition_ext;

    void (&functions<3>::save)(const char       *filename,
                               types<3>::forest *t8code,
                               int               save_data) = p8est_save;

    types<3>::forest *(&functions<3>::load_ext)(
      const char              *filename,
      MPI_Comm                 mpicomm,
      std::size_t              data_size,
      int                      load_data,
      int                      autopartition,
      int                      broadcasthead,
      void                    *user_pointer,
      types<3>::connectivity **t8code) = p8est_load_ext;

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

    void (&functions<3>::vtk_write_file)(types<3>::forest *p8est,
                                         p8est_geometry_t *,
                                         const char *baseName) =
      p8est_vtk_write_file;

    types<3>::ghost *(&functions<3>::ghost_new)(types<3>::forest      *t8code,
                                                types<3>::balance_type btype) =
      p8est_ghost_new;

    void (&functions<3>::ghost_destroy)(types<3>::ghost *ghost) =
      p8est_ghost_destroy;

    void (&functions<3>::reset_data)(types<3>::forest *t8code,
                                     std::size_t       data_size,
                                     p8est_init_t      init_fn,
                                     void *user_pointer) = p8est_reset_data;

    std::size_t (&functions<3>::forest_memory_used)(types<3>::forest *t8code) =
      p8est_memory_used;

    std::size_t (&functions<3>::connectivity_memory_used)(
      types<3>::connectivity *t8code) = p8est_connectivity_memory_used;

    constexpr unsigned int functions<3>::max_level;

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

#  ifdef T8CODE_SEARCH_LOCAL
    void (&functions<3>::search_partition)(
      types<3>::forest                   *t8code,
      int                                 call_post,
      types<3>::search_partition_callback element_fn,
      types<3>::search_partition_callback point_fn,
      sc_array_t                         *points) = p8est_search_partition;
#  endif

    void (&functions<3>::element_coord_to_vertex)(
      types<3>::connectivity  *connectivity,
      types<3>::topidx         treeid,
      types<3>::element_coord x,
      types<3>::element_coord y,
      types<3>::element_coord z,
      double                   vxyz[3]) = p8est_qcoord_to_vertex;

    template <int dim>
    void
    init_element_children(
      const typename types<dim>::element &t8code_cell,
      typename types<dim>::element (
        &t8code_children)[dealii::GeometryInfo<dim>::max_children_per_cell])
    {
      for (unsigned int c = 0;
           c < dealii::GeometryInfo<dim>::max_children_per_cell;
           ++c)
        switch (dim)
          {
            case 2:
              T8CODE_QUADRANT_INIT(&t8code_children[c]);
              break;
            case 3:
              P8EST_QUADRANT_INIT(&t8code_children[c]);
              break;
            default:
              DEAL_II_NOT_IMPLEMENTED();
          }


      functions<dim>::element_childrenv(&t8code_cell, t8code_children);
    }

    template <int dim>
    void
    init_coarse_element(typename types<dim>::element &elem)
    {
      switch (dim)
        {
          case 2:
            T8CODE_QUADRANT_INIT(&elem);
            break;
          case 3:
            P8EST_QUADRANT_INIT(&elem);
            break;
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
      functions<dim>::element_set_morton(&elem,
                                          /*level=*/0,
                                          /*index=*/0);
    }

    template <int dim>
    bool
    element_is_equal(const typename types<dim>::element &q1,
                      const typename types<dim>::element &q2)
    {
      return functions<dim>::element_is_equal(&q1, &q2);
    }



    template <int dim>
    bool
    element_is_ancestor(const typename types<dim>::element &q1,
                         const typename types<dim>::element &q2)
    {
      return functions<dim>::element_is_ancestor(&q1, &q2);
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



    template <>
    bool
    element_is_equal<1>(const typename types<1>::element &q1,
                         const typename types<1>::element &q2)
    {
      return q1 == q2;
    }



    template <>
    bool
    element_is_ancestor<1>(const types<1>::element &q1,
                            const types<1>::element &q2)
    {
      // determine level of elements
      const int level_1 = (q1 << types<1>::max_n_child_indices_bits) >>
                          types<1>::max_n_child_indices_bits;
      const int level_2 = (q2 << types<1>::max_n_child_indices_bits) >>
                          types<1>::max_n_child_indices_bits;

      // q1 can be an ancestor of q2 if q1's level is smaller
      if (level_1 >= level_2)
        return false;

      // extract path of elements up to level of possible ancestor q1
      const int truncated_id_1 = (q1 >> (types<1>::n_bits - 1 - level_1))
                                 << (types<1>::n_bits - 1 - level_1);
      const int truncated_id_2 = (q2 >> (types<1>::n_bits - 1 - level_1))
                                 << (types<1>::n_bits - 1 - level_1);

      // compare paths
      return truncated_id_1 == truncated_id_2;
    }



    template <>
    void
    init_element_children<1>(
      const typename types<1>::element &q,
      typename types<1>::element (
        &t8code_children)[dealii::GeometryInfo<1>::max_children_per_cell])
    {
      // determine the current level of element
      const int level_parent = (q << types<1>::max_n_child_indices_bits) >>
                               types<1>::max_n_child_indices_bits;
      const int level_child = level_parent + 1;

      // left child: only n_child_indices has to be incremented
      t8code_children[0] = (q + 1);

      // right child: increment and set a bit to 1 indicating that it is a right
      // child
      t8code_children[1] = (q + 1) | (1 << (types<1>::n_bits - 1 - level_child));
    }



    template <>
    void
    init_coarse_element<1>(typename types<1>::element &elem)
    {
      elem = 0;
    }

  } // namespace t8code
} // namespace internal

#endif // DEAL_II_WITH_T8CODE

/*-------------- Explicit Instantiations -------------------------------*/
#include "t8code_wrappers.inst"


DEAL_II_NAMESPACE_CLOSE
