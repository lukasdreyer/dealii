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
#define dealii_t8code_wrappers_h

#include <deal.II/base/config.h>

#include <deal.II/base/geometry_info.h>

#ifdef DEAL_II_WITH_T8CODE
#  include <t8.h>
#  include <t8_cmesh.h>
#  include <t8_forest/t8_forest.h>
#  include <t8_forest/t8_forest_general.h>
#  include <t8_forest/t8_forest_types.h>
#  include <limits>

DEAL_II_NAMESPACE_OPEN

// Forward declaration
#  ifndef DOXYGEN
namespace parallel
{
  namespace distributed
  {
    template <int dim, int spacedim>
    DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
    class Triangulation;
  }
} // namespace parallel
#  endif

namespace internal
{
  namespace t8code
  {
    /**
     * A structure whose explicit specializations contain alias to the
     * relevant t8code_* and p8est_* types. Using this structure, for example
     * by saying <tt>types<dim>::connectivity</tt> we can write code in a
     * dimension independent way, either referring to t8code_connectivity_t or
     * p8est_connectivity_t, depending on template argument.
     */
    template <int>
    struct types;

    // these struct mimics t8code for 1d
    template <>
    struct types<1>
    {
      // id of a element is an integeger
      using element = int;

      // maximum number of children
      static const int max_n_child_indices_bits = 27;

      // number of bits the data type of id has
      static const int n_bits = std::numeric_limits<element>::digits;
    };

    template <>
    struct types<2>
    {
      using connectivity     = t8_cmesh_t;
      using forest           = t8_forest_t;
      using tree             = t8_tree_struct_t;
      using element         = t8_element_t;
//      using element_coord   = t8_qcoord_t;
      using topidx           = t8_gloidx_t;
      using locidx           = t8_locidx_t;
      using gloidx           = t8_gloidx_t;
//      using balance_type     = t8_connect_type_t;
      using ghost            = t8_forest_ghost_t;
//      using transfer_context = t8_transfer_context_t;
#  ifdef T8CODE_SEARCH_LOCAL
      using search_partition_callback = t8code_search_partition_t;
#  endif
    };


    /**
     * A structure whose explicit specializations contain pointers to the
     * relevant t8code_* and p8est_* functions. Using this structure, for
     * example by saying functions<dim>::element_compare, we can write code
     * in a dimension independent way, either calling t8code_element_compare
     * or p8est_element_compare, depending on template argument.
     */
    template <int dim>
    struct functions;

    template <>
    struct functions<2>
    {
      static int (&element_compare)(const void *v1, const void *v2);

      static void (&element_childrenv)(const types<2>::element *q,
                                        types<2>::element        c[]);

      static int (&element_overlaps_tree)(types<2>::tree           *tree,
                                           const types<2>::element *q);

      static void (&element_set_morton)(types<2>::element *element,
                                         int                 level,
                                         std::uint64_t       id);

      static int (&element_is_equal)(const types<2>::element *q1,
                                      const types<2>::element *q2);

      static int (&element_is_sibling)(const types<2>::element *q1,
                                        const types<2>::element *q2);

      static int (&element_is_ancestor)(const types<2>::element *q1,
                                         const types<2>::element *q2);

      static int (&element_ancestor_id)(const types<2>::element *q,
                                         int                       level);

      static int (&comm_find_owner)(types<2>::forest         *t8code,
                                    const types<2>::locidx    which_tree,
                                    const types<2>::element *q,
                                    const int                 guess);

      static types<2>::connectivity *(&connectivity_new)(
        types<2>::topidx num_vertices,
        types<2>::topidx num_trees,
        types<2>::topidx num_corners,
        types<2>::topidx num_vtt);

      static types<2>::connectivity *(&connectivity_new_copy)(
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
        const int8_t           *ctc);

      static void (&connectivity_join_faces)(types<2>::connectivity *conn,
                                             types<2>::topidx        tree_left,
                                             types<2>::topidx        tree_right,
                                             int                     face_left,
                                             int                     face_right,
                                             int orientation);



      static void (&connectivity_destroy)(t8_cmesh_t *connectivity);

      static types<2>::forest *(&new_forest)(
        MPI_Comm                mpicomm,
        types<2>::connectivity *connectivity,
        types<2>::locidx        min_elements,
        int                     min_level,
        int                     fill_uniform,
        std::size_t             data_size,
//        t8code_init_t            init_fn,
        void                   *user_pointer);

      static types<2>::forest *(&copy_forest)(types<2>::forest *input,
                                              int               copy_data);

      static void (&destroy)(types<2>::forest *t8code);

      static void (&refine)(types<2>::forest *t8code,
                            int               refine_recursive,
                            t8_forest_adapt_t    refine_fn);

      static void (&coarsen)(types<2>::forest *t8code,
                             int               coarsen_recursive,
                             t8_forest_adapt_t  coarsen_fn);

      static void (&balance)(types<2>::forest      *t8code//,
//                             types<2>::balance_type btype,
//                             t8code_init_t           init_fn
                            );

      static types<2>::gloidx (&partition)(types<2>::forest *t8code,
                                           int partition_for_coarsening//,
                                           //t8code_weight_t weight_fn
                                           );

      static void (&save)(const char       *filename,
                          types<2>::forest *t8code,
                          int               save_data);

      static types<2>::forest *(&load_ext)(const char *filename,
                                           MPI_Comm    mpicomm,
                                           std::size_t data_size,
                                           int         load_data,
                                           int         autopartition,
                                           int         broadcasthead,
                                           void       *user_pointer,
                                           types<2>::connectivity **t8code);

      static int (&connectivity_save)(const char             *filename,
                                      types<2>::connectivity *connectivity);

      static int (&connectivity_is_valid)(types<2>::connectivity *connectivity);

      static types<2>::connectivity *(&connectivity_load)(const char  *filename,
                                                          std::size_t *length);

      static unsigned int (&checksum)(types<2>::forest *t8code);

      static void (&vtk_write_file)(types<2>::forest *t8code,
                                    t8_geometry_c *,
                                    const char *baseName);

      static types<2>::ghost *(&ghost_new)(types<2>::forest      *t8code//,
                                           //types<2>::balance_type btype
                                           );

      static void (&ghost_destroy)(types<2>::ghost *ghost);

      static void (&reset_data)(types<2>::forest *t8code,
                                std::size_t       data_size,
//                                t8code_init_t      init_fn,
                                void             *user_pointer);

      static std::size_t (&forest_memory_used)(types<2>::forest *t8code);

      static std::size_t (&connectivity_memory_used)(
        types<2>::connectivity *t8code);

      template <int spacedim>
      static void
      iterate(dealii::internal::t8code::types<2>::forest *parallel_forest,
              dealii::internal::t8code::types<2>::ghost  *parallel_ghost,
              void                                      *user_data);
#if 0
      static constexpr unsigned int max_level = T8_MAXLEVEL;

      static void (&transfer_fixed)(const types<2>::gloidx *dest_gfq,
                                    const types<2>::gloidx *src_gfq,
                                    MPI_Comm                mpicomm,
                                    int                     tag,
                                    void                   *dest_data,
                                    const void             *src_data,
                                    std::size_t             data_size);

      static types<2>::transfer_context *(&transfer_fixed_begin)(
        const types<2>::gloidx *dest_gfq,
        const types<2>::gloidx *src_gfq,
        MPI_Comm                mpicomm,
        int                     tag,
        void                   *dest_data,
        const void             *src_data,
        std::size_t             data_size);

      static void (&transfer_fixed_end)(types<2>::transfer_context *tc);

      static void (&transfer_custom)(const types<2>::gloidx *dest_gfq,
                                     const types<2>::gloidx *src_gfq,
                                     MPI_Comm                mpicomm,
                                     int                     tag,
                                     void                   *dest_data,
                                     const int              *dest_sizes,
                                     const void             *src_data,
                                     const int              *src_sizes);

      static types<2>::transfer_context *(&transfer_custom_begin)(
        const types<2>::gloidx *dest_gfq,
        const types<2>::gloidx *src_gfq,
        MPI_Comm                mpicomm,
        int                     tag,
        void                   *dest_data,
        const int              *dest_sizes,
        const void             *src_data,
        const int              *src_sizes);

      static void (&transfer_custom_end)(types<2>::transfer_context *tc);

#  ifdef T8CODE_SEARCH_LOCAL
      static void (&search_partition)(
        types<2>::forest                   *forest,
        int                                 call_post,
        types<2>::search_partition_callback element_fn,
        types<2>::search_partition_callback point_fn,
        sc_array_t                         *points);
#  endif

      static void (&element_coord_to_vertex)(
        types<2>::connectivity  *connectivity,
        types<2>::topidx         treeid,
        types<2>::element_coord x,
        types<2>::element_coord y,
        double                   vxyz[3]);
#endif
    };



#if 0
    /**
     * This struct templatizes the t8code iterate structs and function
     * prototypes, which are used to execute callback functions for faces,
     * edges, and corners that require local neighborhood information, i.e.
     * the neighboring cells
     */
    template <int dim>
    struct iter;

    template <>
    struct iter<2>
    {
      using corner_info = t8code_iter_corner_info_t;
      using corner_side = t8code_iter_corner_side_t;
      using corner_iter = t8code_iter_corner_t;
      using face_info   = t8code_iter_face_info_t;
      using face_side   = t8code_iter_face_side_t;
      using face_iter   = t8code_iter_face_t;
    };

    template <>
    struct iter<3>
    {
      using corner_info = p8est_iter_corner_info_t;
      using corner_side = p8est_iter_corner_side_t;
      using corner_iter = p8est_iter_corner_t;
      using edge_info   = p8est_iter_edge_info_t;
      using edge_side   = p8est_iter_edge_side_t;
      using edge_iter   = p8est_iter_edge_t;
      using face_info   = p8est_iter_face_info_t;
      using face_side   = p8est_iter_face_side_t;
      using face_iter   = p8est_iter_face_t;
    };
#endif


    /**
     * Initialize the GeometryInfo<dim>::max_children_per_cell children of the
     * cell t8code_cell.
     */
    template <int dim>
    void
    init_element_children(
      const typename types<dim>::element &t8code_cell,
      typename types<dim>::element (
        &t8code_children)[dealii::GeometryInfo<dim>::max_children_per_cell]);



    /**
     * Initialize element to represent a coarse cell.
     */
    template <int dim>
    void
    init_coarse_element(typename types<dim>::element &quad);



    /**
     * Return whether q1 and q2 are equal
     */
    template <int dim>
    bool
    element_is_equal(const typename types<dim>::element &q1,
                      const typename types<dim>::element &q2);



    /**
     * Return whether q1 is an ancestor of q2
     */
    template <int dim>
    bool
    element_is_ancestor(const typename types<dim>::element &q1,
                         const typename types<dim>::element &q2);



    /**
     * Return whether the children of a coarse cell are stored locally
     */
    template <int dim>
    bool
    tree_exists_locally(const typename types<dim>::forest *parallel_forest,
                        const typename types<dim>::topidx  coarse_grid_cell);


    /**
     * Deep copy a t8code connectivity object.
     */
    template <int dim>
    typename types<dim>::connectivity *
    copy_connectivity(const typename types<dim>::connectivity *connectivity);

#  ifndef DOXYGEN
    template <>
    typename types<2>::connectivity *
    copy_connectivity<2>(const typename types<2>::connectivity *connectivity);
#  endif
  } // namespace t8code
} // namespace internal

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_WITH_T8CODE

#endif // dealii_t8code_wrappers_h


#endif /* T8CODE_WRAPPERS_20COPY */
