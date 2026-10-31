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
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face_dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_wedge_p.h>

DEAL_II_NAMESPACE_OPEN

namespace
{
  /**
   * Helper function to set up the dpo vector of FE_PyramidP for a given @p degree.
   */
  unsigned int
  compute_n_dofs(const unsigned int dim, const unsigned int degree)
  {
    AssertDimension(dim, 3);
    return (degree + 1) * (degree + 2) * (2 * degree + 3) / 6;
  }

  /**
   * The support points of FE_PyramidP of the given @p degree: the
   * equidistant lattice points, ordered by vertices, lines (from their first
   * to their second vertex), faces (in the order of the support points of
   * FE_Q with equidistant points on the quadrilateral and of FE_SimplexP on
   * the triangles) and the interior (by layers in z).
   */
  template <int dim>
  std::vector<Point<dim>>
  get_support_points(const unsigned int degree)
  {
    AssertDimension(dim, 3);
    Assert(degree > 0, ExcInternalError("Degree must be larger than 0."));

    const ReferenceCell     pyramid = ReferenceCells::Pyramid;
    std::vector<Point<dim>> support_points;

    // vertices
    for (const unsigned int v : pyramid.vertex_indices())
      support_points.push_back(pyramid.vertex<dim>(v));

    // lines
    for (const unsigned int l : pyramid.line_indices())
      {
        const Point<dim> p0 =
          pyramid.vertex<dim>(pyramid.line_to_cell_vertices(l, 0));
        const Point<dim> p1 =
          pyramid.vertex<dim>(pyramid.line_to_cell_vertices(l, 1));
        for (unsigned int i = 1; i < degree; ++i)
          support_points.push_back(p0 + (double(i) / degree) * (p1 - p0));
      }

    // faces: the support points inside the faces of the 2d elements, mapped
    // to the faces
    if (degree > 1)
      {
        const FE_Q<2>        fe_quad(QIterated<1>(QTrapezoid<1>(), degree));
        const FE_SimplexP<2> fe_triangle(degree);
        for (const unsigned int f : pyramid.face_indices())
          {
            const ReferenceCell face = pyramid.face_reference_cell(f);
            const FiniteElement<2> &fe_face =
              face == ReferenceCells::Quadrilateral ?
                static_cast<const FiniteElement<2> &>(fe_quad) :
                static_cast<const FiniteElement<2> &>(fe_triangle);
            std::vector<Point<dim>> face_vertices;
            for (const unsigned int v : face.vertex_indices())
              face_vertices.push_back(
                pyramid.vertex<dim>(pyramid.face_to_cell_vertices(
                  f, v, numbers::default_geometric_orientation)));
            const unsigned int first =
              face.n_vertices() * (1 + fe_face.n_dofs_per_line());
            for (unsigned int i = first; i < fe_face.n_dofs_per_cell(); ++i)
              support_points.push_back(
                internal::FEFaceDofTools::map_face_point(
                  face, fe_face.get_unit_support_points()[i], face_vertices));
          }
      }

    // interior: the lattice points of the layers z = k/degree
    for (unsigned int k = 1; k < degree; ++k)
      {
        const unsigned int m = degree - k;
        const double       z = double(k) / degree;
        for (unsigned int j = 1; j < m; ++j)
          for (unsigned int i = 1; i < m; ++i)
            support_points.emplace_back((1. - z) * (2. * i / m - 1.),
                                        (1. - z) * (2. * j / m - 1.),
                                        z);
      }

    AssertDimension(support_points.size(), compute_n_dofs(dim, degree));
    return support_points;
  }


  /**
   * Helper function to set up the dpo vector of FE_PyramidP and FE_PyramidDGP for a given @p degree.
   */
  template <int dim>
  internal::GenericDoFsPerObject
  get_dpo(const unsigned int                                degree,
          const typename FiniteElementData<dim>::Conformity conformity)
  {
    AssertDimension(dim, 3);
    internal::GenericDoFsPerObject dpo;

    if (conformity == FiniteElementData<dim>::L2)
      {
        dpo = internal::expand(3,
                               {{0, 0, 0, compute_n_dofs(dim, degree)}},
                               ReferenceCells::Pyramid);
      }
    else if (conformity == FiniteElementData<dim>::H1)
      {
        // the support points on the 8 lines excluding the vertices
        const unsigned int n_dofs_per_line = degree - 1;

        // support points on the bottom quad face and on the 4 triangular faces,
        // on the triangular faces the number of points is the sum from 1 to
        // (degree - 2) so 4*0.5*(degree - 2)*(degree - 1)
        const unsigned int n_dofs_per_quad = n_dofs_per_line * n_dofs_per_line;
        const unsigned int n_dofs_per_tri  = (degree - 2) * (degree - 1) / 2;
        const unsigned int total_dofs_faces =
          n_dofs_per_quad + 4 * n_dofs_per_tri;
        // total number of DoFs on a tri
        const unsigned int n_dofs_per_tri_inclusive =
          (degree + 1) * (degree + 2) / 2;

        const unsigned int n_dofs_total = compute_n_dofs(dim, degree);

        dpo.dofs_per_object_exclusive = {
          {1, 1, 1, 1, 1},
          {n_dofs_per_line,
           n_dofs_per_line,
           n_dofs_per_line,
           n_dofs_per_line,
           n_dofs_per_line,
           n_dofs_per_line,
           n_dofs_per_line,
           n_dofs_per_line},
          {n_dofs_per_quad,
           n_dofs_per_tri,
           n_dofs_per_tri,
           n_dofs_per_tri,
           n_dofs_per_tri},
          {n_dofs_total - 5 - 8 * n_dofs_per_line - total_dofs_faces}};

        dpo.dofs_per_object_inclusive = {{1, 1, 1, 1, 1},
                                         {
                                           degree + 1,
                                           degree + 1,
                                           degree + 1,
                                           degree + 1,
                                           degree + 1,
                                           degree + 1,
                                           degree + 1,
                                           degree + 1,
                                         },
                                         {(degree + 1) * (degree + 1),
                                          n_dofs_per_tri_inclusive,
                                          n_dofs_per_tri_inclusive,
                                          n_dofs_per_tri_inclusive,
                                          n_dofs_per_tri_inclusive},
                                         {n_dofs_total}};

        dpo.object_index = {
          {0, 1, 2, 3, 4},
          {5,
           5 + 1 * n_dofs_per_line,
           5 + 2 * n_dofs_per_line,
           5 + 3 * n_dofs_per_line,
           5 + 4 * n_dofs_per_line,
           5 + 5 * n_dofs_per_line,
           5 + 6 * n_dofs_per_line,
           5 + 7 * n_dofs_per_line},
          {5 + 8 * n_dofs_per_line,
           5 + 8 * n_dofs_per_line + n_dofs_per_quad,
           5 + 8 * n_dofs_per_line + n_dofs_per_quad + n_dofs_per_tri,
           5 + 8 * n_dofs_per_line + n_dofs_per_quad + 2 * n_dofs_per_tri,
           5 + 8 * n_dofs_per_line + n_dofs_per_quad + 3 * n_dofs_per_tri},
          {5 + 8 * n_dofs_per_line + total_dofs_faces}};

        dpo.first_object_index_on_face = {{0, 0, 0, 0, 0},
                                          {4, 3, 3, 3, 3},
                                          {4 + 4 * n_dofs_per_line,
                                           3 + 3 * n_dofs_per_line,
                                           3 + 3 * n_dofs_per_line,
                                           3 + 3 * n_dofs_per_line,
                                           3 + 3 * n_dofs_per_line}};
      }
    return dpo;
  }
} // namespace


template <int dim, int spacedim>
FE_PyramidPoly<dim, spacedim>::FE_PyramidPoly(
  const unsigned int                                degree,
  const internal::GenericDoFsPerObject              dpos,
  const std::vector<Point<dim>>                     support_points,
  const bool                                        prolongation_is_additive,
  const typename FiniteElementData<dim>::Conformity conformity)
  : dealii::FE_Poly<dim, spacedim>(
      ScalarLagrangePolynomialPyramid<dim>(degree,
                                           compute_n_dofs(dim, degree),
                                           support_points),
      FiniteElementData<dim>(dpos,
                             ReferenceCells::Pyramid,
                             1,
                             degree,
                             conformity),
      std::vector<bool>(
        FiniteElementData<dim>(dpos, ReferenceCells::Pyramid, 1, degree)
          .dofs_per_cell,
        prolongation_is_additive),
      std::vector<ComponentMask>(
        FiniteElementData<dim>(dpos, ReferenceCells::Pyramid, 1, degree)
          .dofs_per_cell,
        ComponentMask(std::vector<bool>(1, true))))
{
  AssertDimension(dim, 3);

  for (auto &support_point : support_points)
    this->unit_support_points.emplace_back(support_point);

  if (conformity == FiniteElementData<dim>::H1)
    {
      // face support points
      this->unit_face_support_points.resize(this->reference_cell().n_faces());

      for (const auto f : this->reference_cell().face_indices())
        {
          const auto face_reference_cell =
            this->reference_cell().face_reference_cell(f);

          if (face_reference_cell == ReferenceCells::Quadrilateral)
            {
              // switch to FE_Q when simplex supports electrostatic points
              // FE_Q<2> fe_face(degree);
              FE_Q<2> fe_face(QIterated<1>(QTrapezoid<1>(), degree));

              for (const auto &face_support_point :
                   fe_face.get_unit_support_points())
                {
                  Point<dim - 1> p;
                  for (unsigned int d = 0; d < dim - 1; ++d)
                    p[d] = face_support_point[d];
                  this->unit_face_support_points[f].emplace_back(p);
                }
            }
          else if (face_reference_cell == ReferenceCells::Triangle)
            {
              FE_SimplexP<2> fe_face(degree);
              for (const auto &face_support_point :
                   fe_face.get_unit_support_points())
                {
                  Point<dim - 1> p;
                  for (unsigned int d = 0; d < dim - 1; ++d)
                    p[d] = face_support_point[d];
                  this->unit_face_support_points[f].emplace_back(p);
                }
            }
        }
    }
}



template <int dim, int spacedim>
void
FE_PyramidPoly<dim, spacedim>::
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



namespace
{
  // As in fe_wedge_p.cc: the face of the reference cell of fe that has the same shape as face
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
FE_PyramidPoly<dim, spacedim>::get_face_interpolation_matrix(
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
FE_PyramidPoly<dim, spacedim>::get_subface_interpolation_matrix(
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
FE_PyramidPoly<dim, spacedim>::hp_constraints_are_implemented() const
{
  return true;
}



template <int dim, int spacedim>
unsigned int
FE_PyramidPoly<dim, spacedim>::face_to_cell_index(
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
FE_PyramidP<dim, spacedim>::FE_PyramidP(const unsigned int degree)
  : FE_PyramidPoly<dim, spacedim>(degree,
                                  get_dpo<dim>(degree,
                                               FiniteElementData<dim>::H1),
                                  get_support_points<dim>(degree),
                                  false,
                                  FiniteElementData<dim>::H1)
{
  internal::FEFaceDofTools::fill_line_dof_orientation_table(
    this->adjust_line_dof_index_for_line_orientation_table,
    this->n_dofs_per_line());

  // the DoFs inside the faces are permuted with the orientation of the face,
  // like the ones of FE_Q on the quadrilateral and of FE_SimplexP on the
  // triangles
  if (degree > 2)
    {
      const FE_Q<2>        fe_quad(QIterated<1>(QTrapezoid<1>(), degree));
      const FE_SimplexP<2> fe_triangle(degree);
      for (const unsigned int f : this->reference_cell().face_indices())
        {
          const ReferenceCell face = this->reference_cell().face_reference_cell(f);
          const FiniteElement<2> &fe_face =
            face == ReferenceCells::Quadrilateral ?
              static_cast<const FiniteElement<2> &>(fe_quad) :
              static_cast<const FiniteElement<2> &>(fe_triangle);
          const unsigned int first =
            face.n_vertices() * (1 + fe_face.n_dofs_per_line());
          const std::vector<Point<2>> interior_points(
            fe_face.get_unit_support_points().begin() + first,
            fe_face.get_unit_support_points().end());
          internal::FEFaceDofTools::fill_face_dof_orientation_table(
            this->adjust_quad_dof_index_for_face_orientation_table[f],
            face,
            interior_points);
        }
    }
}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_PyramidP<dim, spacedim>::clone() const
{
  return std::make_unique<FE_PyramidP<dim, spacedim>>(*this);
}



template <int dim, int spacedim>
std::string
FE_PyramidP<dim, spacedim>::get_name() const
{
  std::ostringstream namebuf;
  namebuf << "FE_PyramidP<" << Utilities::dim_string(dim, spacedim) << ">("
          << this->degree << ")";

  return namebuf.str();
}



template <int dim, int spacedim>
FiniteElementDomination::Domination
FE_PyramidP<dim, spacedim>::compare_for_domination(
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
  if (const FE_PyramidP<dim, spacedim> *fe_pp_other =
        dynamic_cast<const FE_PyramidP<dim, spacedim> *>(&fe_other))
    {
      if (this->degree < fe_pp_other->degree)
        return FiniteElementDomination::this_element_dominates;
      else if (this->degree == fe_pp_other->degree)
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
  else if (const FE_WedgeP<dim, spacedim> *fe_wp_other =
             dynamic_cast<const FE_WedgeP<dim, spacedim> *>(&fe_other))
    {
      if (this->degree < fe_wp_other->degree)
        return FiniteElementDomination::this_element_dominates;
      else if (this->degree == fe_wp_other->degree)
        return FiniteElementDomination::either_element_can_dominate;
      else
        return FiniteElementDomination::other_element_dominates;
    }
  else if (const FE_Nothing<dim, spacedim> *fe_nothing =
             dynamic_cast<const FE_Nothing<dim, spacedim> *>(&fe_other))
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
FE_PyramidP<dim, spacedim>::hp_vertex_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other) const
{
  (void)fe_other;

  Assert((dynamic_cast<const FE_SimplexP<dim, spacedim> *>(&fe_other)) ||
           (dynamic_cast<const FE_Q<dim, spacedim> *>(&fe_other)) ||
           (dynamic_cast<const FE_WedgeP<dim, spacedim> *>(&fe_other)),
         ExcNotImplemented());

  return {{0, 0}};
}



template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_PyramidP<dim, spacedim>::hp_line_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other) const
{
  // the DoFs at the same positions on the line (the elements next to this
  // one, FE_Q, FE_SimplexP, FE_WedgeP and FE_PyramidP, are Lagrange elements)
  return internal::FEFaceDofTools::line_dof_identities(*this, fe_other);
}



template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_PyramidP<dim, spacedim>::hp_quad_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other,
  const unsigned int                  face_no) const
{
  // the DoFs at the same positions inside the face
  return internal::FEFaceDofTools::face_dof_identities(*this,
                                                       fe_other,
                                                       face_no);
}



template <int dim, int spacedim>
FE_PyramidDGP<dim, spacedim>::FE_PyramidDGP(const unsigned int degree)
  : FE_PyramidPoly<dim, spacedim>(degree,
                                  get_dpo<dim>(degree,
                                               FiniteElementData<dim>::L2),
                                  get_support_points<dim>(degree),
                                  true,
                                  FiniteElementData<dim>::L2)
{}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_PyramidDGP<dim, spacedim>::clone() const
{
  return std::make_unique<FE_PyramidDGP<dim, spacedim>>(*this);
}



template <int dim, int spacedim>
std::string
FE_PyramidDGP<dim, spacedim>::get_name() const
{
  std::ostringstream namebuf;
  namebuf << "FE_PyramidDGP<" << Utilities::dim_string(dim, spacedim) << ">("
          << this->degree << ")";

  return namebuf.str();
}

// explicit instantiations
#include "fe/fe_pyramid_p.inst"

DEAL_II_NAMESPACE_CLOSE
