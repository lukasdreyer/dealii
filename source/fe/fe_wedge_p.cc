// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2021 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <deal.II/base/polynomials_barycentric.h>
#include <deal.II/base/qprojector.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_wedge_p.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

DEAL_II_NAMESPACE_OPEN

namespace
{
  /**
   * Helper function to set up the dpo vector of FE_WedgeP for a given @p degree.
   */
  internal::GenericDoFsPerObject
  get_dpo_vector_fe_wedge_p(const unsigned int degree)
  {
    internal::GenericDoFsPerObject dpo;

    if (degree == 1)
      {
        dpo.dofs_per_object_exclusive  = {{1}, {0}, {0, 0, 0, 0, 0}, {0}};
        dpo.dofs_per_object_inclusive  = {{1}, {2}, {3, 3, 4, 4, 4}, {6}};
        dpo.object_index               = {{}, {6}, {6}, {6}};
        dpo.first_object_index_on_face = {{}, {3, 3, 4, 4, 4}, {3, 3, 4, 4, 4}};
      }
    else if (degree == 2)
      {
        dpo.dofs_per_object_exclusive  = {{1}, {1}, {0, 0, 1, 1, 1}, {0}};
        dpo.dofs_per_object_inclusive  = {{1}, {3}, {6, 6, 9, 9, 9}, {18}};
        dpo.object_index               = {{}, {6}, {15, 15, 15, 16, 17}, {18}};
        dpo.first_object_index_on_face = {{}, {3, 3, 4, 4, 4}, {6, 6, 8, 8, 8}};
      }
    else
      {
        DEAL_II_NOT_IMPLEMENTED();
      }

    return dpo;
  }

  /**
   * Helper function to set up the dpo vector of FE_WedgeDGP for a given @p degree.
   */
  internal::GenericDoFsPerObject
  get_dpo_vector_fe_wedge_dgp(const unsigned int degree)
  {
    unsigned int n_dofs = 0;

    if (degree == 1)
      n_dofs = 6;
    else if (degree == 2)
      n_dofs = 18;
    else
      DEAL_II_NOT_IMPLEMENTED();

    return internal::expand(3, {{0, 0, 0, n_dofs}}, ReferenceCells::Wedge);
  }
} // namespace



template <int dim, int spacedim>
FE_WedgePoly<dim, spacedim>::FE_WedgePoly(
  const unsigned int                                degree,
  const internal::GenericDoFsPerObject             &dpos,
  const bool                                        prolongation_is_additive,
  const typename FiniteElementData<dim>::Conformity conformity)
  : dealii::FE_Poly<dim, spacedim>(
      ScalarLagrangePolynomialWedge<dim>(degree),
      FiniteElementData<dim>(dpos,
                             ReferenceCells::Wedge,
                             1,
                             degree,
                             conformity),
      std::vector<bool>(
        FiniteElementData<dim>(dpos, ReferenceCells::Wedge, 1, degree)
          .dofs_per_cell,
        prolongation_is_additive),
      std::vector<ComponentMask>(
        FiniteElementData<dim>(dpos, ReferenceCells::Wedge, 1, degree)
          .dofs_per_cell,
        ComponentMask(std::vector<bool>(1, true))))
{
  AssertDimension(dim, 3);

  Assert(1 <= degree && degree <= 2, ExcNotImplemented());

  FE_SimplexP<2> fe_triangle(degree);
  FE_Q<1>        fe_line(degree);
  FE_Q<2>        fe_quad(degree);

  for (unsigned int i = 0; i < this->n_dofs_per_cell(); ++i)
    {
      const auto pair = this->degree == 1 ? internal::wedge_table_1[i] :
                                            internal::wedge_table_2[i];

      this->unit_support_points.emplace_back(
        fe_triangle.get_unit_support_points()[pair[0]][0],
        fe_triangle.get_unit_support_points()[pair[0]][1],
        fe_line.get_unit_support_points()[pair[1]][0]);
    }

  this->unit_face_support_points.resize(this->reference_cell().n_faces());

  for (const auto f : this->reference_cell().face_indices())
    if (this->reference_cell().face_reference_cell(f) ==
        ReferenceCells::Triangle)
      for (const auto &p : fe_triangle.get_unit_support_points())
        this->unit_face_support_points[f].emplace_back(p[0], p[1]);
    else if (this->reference_cell().face_reference_cell(f) ==
             ReferenceCells::Quadrilateral)
      for (const auto &p : fe_quad.get_unit_support_points())
        this->unit_face_support_points[f].emplace_back(p[0], p[1]);
    else
      DEAL_II_ASSERT_UNREACHABLE();
}



template <int dim, int spacedim>
void
FE_WedgePoly<dim, spacedim>::
  convert_generalized_support_point_values_to_dof_values(
    const std::vector<Vector<double>> &support_point_values,
    std::vector<double>               &nodal_values) const
{
  AssertDimension(support_point_values.size(),
                  this->get_unit_support_points().size());
  AssertDimension(support_point_values.size(), nodal_values.size());
  AssertDimension(this->dofs_per_cell, nodal_values.size());

  for (unsigned int i = 0; i < this->dofs_per_cell; ++i)
    {
      AssertDimension(support_point_values[i].size(), 1);

      nodal_values[i] = support_point_values[i](0);
    }
}



template <int dim, int spacedim>
const FullMatrix<double> &
FE_WedgePoly<dim, spacedim>::get_prolongation_matrix(
  const unsigned int         child,
  const RefinementCase<dim> &refinement_case) const
{
  AssertThrow(refinement_case == RefinementCase<dim>::isotropic_refinement,
              ExcNotImplemented());
  AssertDimension(dim, spacedim);

  // initialization upon first request
  if (this->prolongation[refinement_case - 1][child].n() == 0)
    {
      std::lock_guard<std::mutex> lock(prolongation_matrix_mutex);

      // if matrix got updated while waiting for the lock
      if (this->prolongation[refinement_case - 1][child].n() ==
          this->n_dofs_per_cell())
        return this->prolongation[refinement_case - 1][child];

      std::vector<std::vector<FullMatrix<double>>> isotropic_matrices(
        RefinementCase<dim>::isotropic_refinement);
      isotropic_matrices.back().resize(
        this->reference_cell().n_children(refinement_case),
        FullMatrix<double>(this->n_dofs_per_cell(), this->n_dofs_per_cell()));
      FETools::compute_embedding_matrices(*this, isotropic_matrices, true);

      const_cast<FE_WedgePoly<dim, spacedim> &>(*this)
        .prolongation[refinement_case - 1] =
        std::move(isotropic_matrices.back());
    }

  return this->prolongation[refinement_case - 1][child];
}



template <int dim, int spacedim>
const FullMatrix<double> &
FE_WedgePoly<dim, spacedim>::get_restriction_matrix(
  const unsigned int         child,
  const RefinementCase<dim> &refinement_case) const
{
  AssertThrow(refinement_case == RefinementCase<dim>::isotropic_refinement,
              ExcNotImplemented());
  AssertDimension(dim, spacedim);

  // initialization upon first request
  if (this->restriction[refinement_case - 1][child].n() == 0)
    {
      std::lock_guard<std::mutex> lock(restriction_matrix_mutex);

      // if matrix got updated while waiting for the lock
      if (this->restriction[refinement_case - 1][child].n() ==
          this->n_dofs_per_cell())
        return this->restriction[refinement_case - 1][child];

      auto &this_nonconst = const_cast<FE_WedgePoly<dim, spacedim> &>(*this);

      if (this->restriction_is_additive(0))
        {
          // discontinuous elements: L2 projection, the children add up
          std::vector<std::vector<FullMatrix<double>>> isotropic_matrices(
            RefinementCase<dim>::isotropic_refinement);
          isotropic_matrices.back().resize(
            this->reference_cell().n_children(refinement_case),
            FullMatrix<double>(this->n_dofs_per_cell(),
                               this->n_dofs_per_cell()));
          FETools::compute_projection_matrices(*this,
                                               isotropic_matrices,
                                               true);
          this_nonconst.restriction[refinement_case - 1] =
            std::move(isotropic_matrices.back());
        }
      else
        {
          // continuous elements: interpolate at the support points of the
          // parent that lie in a child, as FE_Q and FE_SimplexP do. The
          // children are affine images of the parent, x = v_0 + A x_child
          // with the columns of A the edges from vertex 0 to the vertices 1,
          // 2 and 3 of the child (the vertices on the coordinate axes of the
          // reference wedge), so we do not need a mapping (MappingFE would
          // need second derivatives of the wedge polynomials).
          Triangulation<dim, spacedim> tria;
          GridGenerator::reference_cell(tria, this->reference_cell());
          tria.refine_global(1);

          const std::vector<Point<dim>> &unit_support_points =
            this->get_unit_support_points();
          const double eps = 1e-12;

          std::vector<FullMatrix<double>> matrices(
            this->reference_cell().n_children(refinement_case));
          for (unsigned int c = 0; c < matrices.size(); ++c)
            {
              const auto     child = tria.begin(0)->child(c);
              Tensor<2, dim> edges;
              for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int e = 0; e < dim; ++e)
                  edges[e][d] = child->vertex(d + 1)[e] - child->vertex(0)[e];
              const Tensor<2, dim> inverse_edges = invert(edges);

              FullMatrix<double> restriction_matrix(this->n_dofs_per_cell(),
                                                    this->n_dofs_per_cell());
              for (unsigned int i = 0; i < unit_support_points.size(); ++i)
                {
                  Tensor<1, dim> x;
                  for (unsigned int d = 0; d < dim; ++d)
                    x[d] = unit_support_points[i][d] - child->vertex(0)[d];
                  const Point<dim> child_point(inverse_edges * x);

                  if (this->reference_cell().contains_point(child_point, eps))
                    for (unsigned int j = 0; j < this->n_dofs_per_cell(); ++j)
                      {
                        const double value = this->shape_value(j, child_point);
                        restriction_matrix(i, j) =
                          std::fabs(value) < eps ?
                            0. :
                            (std::fabs(value - 1.) < eps ? 1. : value);
                      }
                }
              matrices[c] = std::move(restriction_matrix);
            }
          this_nonconst.restriction[refinement_case - 1] = std::move(matrices);
        }
    }

  return this->restriction[refinement_case - 1][child];
}



namespace
{
  // The face of the reference cell of fe that has the same shape as face
  // face_no of reference_cell: face_no itself if possible, otherwise the
  // first such face. The face support points of the elements here do not
  // depend on the face but only on its shape.
  template <int dim, int spacedim>
  unsigned int
  equivalent_face(const FiniteElement<dim, spacedim> &fe,
                  const ReferenceCell                &reference_cell,
                  const unsigned int                  face_no)
  {
    const ReferenceCell face_reference_cell =
      reference_cell.face_reference_cell(face_no);
    if (face_no < fe.reference_cell().n_faces() &&
        fe.reference_cell().face_reference_cell(face_no) ==
          face_reference_cell)
      return face_no;
    for (const unsigned int f : fe.reference_cell().face_indices())
      if (fe.reference_cell().face_reference_cell(f) == face_reference_cell)
        return f;
    DEAL_II_ASSERT_UNREACHABLE();
    return numbers::invalid_unsigned_int;
  }



  // Evaluate the shape functions of fe on face face_no at the given points
  // on the cell of fe, which are the face support points of another element.
  template <int dim, int spacedim>
  void
  interpolate_at_face_points(const FiniteElement<dim, spacedim> &fe,
                             const unsigned int                  face_no,
                             const std::vector<Point<dim>>      &points,
                             FullMatrix<double> &interpolation_matrix)
  {
    AssertDimension(interpolation_matrix.m(), points.size());
    AssertDimension(interpolation_matrix.n(), fe.n_dofs_per_face(face_no));

    const double eps = 2e-13 * fe.degree * (dim - 1);

    for (unsigned int i = 0; i < points.size(); ++i)
      for (unsigned int j = 0; j < fe.n_dofs_per_face(face_no); ++j)
        {
          double matrix_entry =
            fe.shape_value(fe.face_to_cell_index(j, face_no), points[i]);

          // Correct the interpolated value. I.e. if it is close to 1 or
          // 0, make it exactly 1 or 0. Unfortunately, this is required to
          // avoid problems with higher order elements.
          if (std::fabs(matrix_entry - 1.0) < eps)
            matrix_entry = 1.0;
          if (std::fabs(matrix_entry) < eps)
            matrix_entry = 0.0;

          interpolation_matrix(i, j) = matrix_entry;
        }

    if constexpr (running_in_debug_mode())
      {
        for (unsigned int i = 0; i < points.size(); ++i)
          {
            double sum = 0.;
            for (unsigned int j = 0; j < fe.n_dofs_per_face(face_no); ++j)
              sum += interpolation_matrix(i, j);

            Assert(std::fabs(sum - 1) < eps, ExcInternalError());
          }
      }
  }
} // namespace



template <int dim, int spacedim>
void
FE_WedgePoly<dim, spacedim>::get_face_interpolation_matrix(
  const FiniteElement<dim, spacedim> &source_fe,
  FullMatrix<double>                 &interpolation_matrix,
  const unsigned int                  face_no) const
{
  if (dynamic_cast<const FE_Nothing<dim> *>(&source_fe) != nullptr)
    // nothing to do here, the FE_Nothing has no degrees of freedom anyway
    return;

  const unsigned int source_face_no =
    equivalent_face(source_fe, this->reference_cell(), face_no);
  AssertThrow(
    source_fe.has_face_support_points(source_face_no),
    (typename FiniteElement<dim, spacedim>::ExcInterpolationNotImplemented()));
  AssertDimension(interpolation_matrix.m(),
                  source_fe.n_dofs_per_face(source_face_no));

  const Quadrature<dim - 1> quad_face_support(
    source_fe.get_unit_face_support_points(source_face_no));
  interpolate_at_face_points(
    *this,
    face_no,
    QProjector<dim>::project_to_face(this->reference_cell(),
                                     quad_face_support,
                                     face_no,
                                     numbers::default_geometric_orientation)
      .get_points(),
    interpolation_matrix);
}



template <int dim, int spacedim>
void
FE_WedgePoly<dim, spacedim>::get_subface_interpolation_matrix(
  const FiniteElement<dim, spacedim> &source_fe,
  const unsigned int                  subface,
  FullMatrix<double>                 &interpolation_matrix,
  const unsigned int                  face_no) const
{
  if (dynamic_cast<const FE_Nothing<dim> *>(&source_fe) != nullptr)
    // nothing to do here, the FE_Nothing has no degrees of freedom anyway
    return;

  const unsigned int source_face_no =
    equivalent_face(source_fe, this->reference_cell(), face_no);
  AssertThrow(
    source_fe.has_face_support_points(source_face_no),
    (typename FiniteElement<dim, spacedim>::ExcInterpolationNotImplemented()));
  AssertDimension(interpolation_matrix.m(),
                  source_fe.n_dofs_per_face(source_face_no));

  const Quadrature<dim - 1> quad_face_support(
    source_fe.get_unit_face_support_points(source_face_no));
  interpolate_at_face_points(
    *this,
    face_no,
    QProjector<dim>::project_to_subface(
      this->reference_cell(),
      quad_face_support,
      face_no,
      subface,
      numbers::default_geometric_orientation,
      RefinementCase<dim - 1>::isotropic_refinement)
      .get_points(),
    interpolation_matrix);
}



template <int dim, int spacedim>
bool
FE_WedgePoly<dim, spacedim>::hp_constraints_are_implemented() const
{
  return true;
}



template <int dim, int spacedim>
unsigned int
FE_WedgePoly<dim, spacedim>::face_to_cell_index(
  const unsigned int                 face_dof_index,
  const unsigned int                 face,
  const types::geometric_orientation combined_orientation) const
{
  return FETools::face_to_cell_index(*this,
                                     face_dof_index,
                                     face,
                                     combined_orientation);
}



template <int dim, int spacedim>
FE_WedgeP<dim, spacedim>::FE_WedgeP(const unsigned int degree)
  : FE_WedgePoly<dim, spacedim>(degree,
                                get_dpo_vector_fe_wedge_p(degree),
                                false,
                                FiniteElementData<dim>::H1)
{}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_WedgeP<dim, spacedim>::clone() const
{
  return std::make_unique<FE_WedgeP<dim, spacedim>>(*this);
}



template <int dim, int spacedim>
std::string
FE_WedgeP<dim, spacedim>::get_name() const
{
  std::ostringstream namebuf;
  namebuf << "FE_WedgeP<" << Utilities::dim_string(dim, spacedim) << ">("
          << this->degree << ")";

  return namebuf.str();
}



template <int dim, int spacedim>
FiniteElementDomination::Domination
FE_WedgeP<dim, spacedim>::compare_for_domination(
  const FiniteElement<dim, spacedim> &fe_other,
  const unsigned int                  codim) const
{
  Assert(codim <= dim, ExcImpossibleInDim(dim));

  // vertex/line/face domination
  // (if fe_other is derived from FE_SimplexDGP)
  // ------------------------------------
  if (codim > 0)
    if (dynamic_cast<const FE_SimplexDGP<dim, spacedim> *>(&fe_other) !=
        nullptr)
      // there are no requirements between continuous and discontinuous
      // elements
      return FiniteElementDomination::no_requirements;


  // vertex/line/face domination
  // (if fe_other is not derived from FE_SimplexDGP)
  // & cell domination
  // ----------------------------------------
  if (const FE_WedgeP<dim, spacedim> *fe_wp_other =
        dynamic_cast<const FE_WedgeP<dim, spacedim> *>(&fe_other))
    {
      if (this->degree < fe_wp_other->degree)
        return FiniteElementDomination::this_element_dominates;
      else if (this->degree == fe_wp_other->degree)
        return FiniteElementDomination::either_element_can_dominate;
      else
        return FiniteElementDomination::other_element_dominates;
    }
  else if (const FE_SimplexP<dim, spacedim> *fe_p_other =
             dynamic_cast<const FE_SimplexP<dim, spacedim> *>(&fe_other))
    {
      if (this->degree < fe_p_other->degree)
        return FiniteElementDomination::this_element_dominates;
      else if (this->degree == fe_p_other->degree)
        return FiniteElementDomination::either_element_can_dominate;
      else
        return FiniteElementDomination::other_element_dominates;
    }
  else if (const FE_Q<dim, spacedim> *fe_q_other =
             dynamic_cast<const FE_Q<dim, spacedim> *>(&fe_other))
    {
      if (this->degree < fe_q_other->degree)
        return FiniteElementDomination::this_element_dominates;
      else if (this->degree == fe_q_other->degree)
        return FiniteElementDomination::either_element_can_dominate;
      else
        return FiniteElementDomination::other_element_dominates;
    }
  else if (const FE_Nothing<dim> *fe_nothing =
             dynamic_cast<const FE_Nothing<dim> *>(&fe_other))
    {
      if (fe_nothing->is_dominating())
        return FiniteElementDomination::other_element_dominates;
      else
        // the FE_Nothing has no degrees of freedom and it is typically used
        // in a context where we don't require any continuity along the
        // interface
        return FiniteElementDomination::no_requirements;
    }

  DEAL_II_NOT_IMPLEMENTED();
  return FiniteElementDomination::neither_element_dominates;
}



template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_WedgeP<dim, spacedim>::hp_vertex_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other) const
{
  (void)fe_other;

  Assert((dynamic_cast<const FE_SimplexP<dim, spacedim> *>(&fe_other)) ||
           (dynamic_cast<const FE_Q<dim, spacedim> *>(&fe_other)),
         ExcNotImplemented());

  return {{0, 0}};
}



template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_WedgeP<dim, spacedim>::hp_line_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other) const
{
  (void)fe_other;

  Assert((dynamic_cast<const FE_SimplexP<dim, spacedim> *>(&fe_other)) ||
           (dynamic_cast<const FE_Q<dim, spacedim> *>(&fe_other)),
         ExcNotImplemented());

  std::vector<std::pair<unsigned int, unsigned int>> result;

  result.reserve(this->degree - 1);
  for (unsigned int i = 0; i < this->degree - 1; ++i)
    result.emplace_back(i, i);

  return result;
}



template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_WedgeP<dim, spacedim>::hp_quad_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other,
  const unsigned int                  face_no) const
{
  (void)fe_other;

  AssertIndexRange(face_no, 5);

  if (face_no < 2)
    {
      Assert((dynamic_cast<const FE_SimplexP<dim, spacedim> *>(&fe_other)),
             ExcNotImplemented());
    }
  else
    {
      Assert((dynamic_cast<const FE_Q<dim, spacedim> *>(&fe_other)),
             ExcNotImplemented());
    }

  std::vector<std::pair<unsigned int, unsigned int>> result;

  result.reserve(this->n_dofs_per_quad(face_no));
  for (unsigned int i = 0; i < this->n_dofs_per_quad(face_no); ++i)
    result.emplace_back(i, i);

  return result;
}



template <int dim, int spacedim>
FE_WedgeDGP<dim, spacedim>::FE_WedgeDGP(const unsigned int degree)
  : FE_WedgePoly<dim, spacedim>(degree,
                                get_dpo_vector_fe_wedge_dgp(degree),
                                true,
                                FiniteElementData<dim>::L2)
{}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_WedgeDGP<dim, spacedim>::clone() const
{
  return std::make_unique<FE_WedgeDGP<dim, spacedim>>(*this);
}



template <int dim, int spacedim>
std::string
FE_WedgeDGP<dim, spacedim>::get_name() const
{
  std::ostringstream namebuf;
  namebuf << "FE_WedgeDGP<" << Utilities::dim_string(dim, spacedim) << ">("
          << this->degree << ")";

  return namebuf.str();
}

// explicit instantiations
#include "fe/fe_wedge_p.inst"

DEAL_II_NAMESPACE_CLOSE
