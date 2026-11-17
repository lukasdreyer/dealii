// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2020 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


#include <deal.II/base/polynomials_barycentric.h>
#include <deal.II/base/polynomials_wedge.h>

#include <deal.II/grid/reference_cell.h>

DEAL_II_NAMESPACE_OPEN

namespace
{
  unsigned int
  compute_n_polynomials_wedge(const unsigned int dim, const unsigned int degree)
  {
    AssertDimension(dim, 3);
    return (degree + 1) * (degree + 1) * (degree + 2) / 2;
  }
} // namespace



namespace internal
{
  std::vector<std::array<unsigned int, 2>>
  wedge_dof_pairs(const unsigned int degree)
  {
    Assert(degree > 0, ExcNotImplemented());
    const std::vector<Point<2>> triangle = simplex_lattice_points<2>(degree);
    const std::vector<Point<1>> line     = simplex_lattice_points<1>(degree);
    const unsigned int          n_interior_triangle =
      triangle.size() - 3 - 3 * (degree - 1);

    // the support points of the wedge, in the order of its DoFs
    const ReferenceCell   wedge = ReferenceCells::Wedge;
    std::vector<Point<3>> points;
    for (const unsigned int v : wedge.vertex_indices())
      points.push_back(wedge.vertex<3>(v));
    for (const unsigned int l : wedge.line_indices())
      {
        const Point<3> p0 = wedge.vertex<3>(wedge.line_to_cell_vertices(l, 0));
        const Point<3> p1 = wedge.vertex<3>(wedge.line_to_cell_vertices(l, 1));
        for (unsigned int i = 1; i < degree; ++i)
          points.push_back(p0 + (double(i) / degree) * (p1 - p0));
      }
    for (const unsigned int f : wedge.face_indices())
      {
        std::vector<Point<3>> vertices;
        for (const unsigned int v :
             wedge.face_reference_cell(f).vertex_indices())
          vertices.push_back(wedge.vertex<3>(wedge.face_to_cell_vertices(
            f, v, numbers::default_geometric_orientation)));
        if (wedge.face_reference_cell(f) == ReferenceCells::Triangle)
          // the points inside the triangle, by barycentric coordinates
          for (unsigned int i = triangle.size() - n_interior_triangle;
               i < triangle.size();
               ++i)
            points.push_back(
              (1. - triangle[i][0] - triangle[i][1]) * vertices[0] +
              triangle[i][0] * vertices[1] + triangle[i][1] * vertices[2]);
        else
          // the points inside the quadrilateral, lexicographically as for
          // FE_Q
          for (unsigned int j = 1; j < degree; ++j)
            for (unsigned int i = 1; i < degree; ++i)
              {
                const double x = double(i) / degree, y = double(j) / degree;
                points.push_back((1 - x) * (1 - y) * vertices[0] +
                                 x * (1 - y) * vertices[1] +
                                 (1 - x) * y * vertices[2] +
                                 x * y * vertices[3]);
              }
      }
    for (unsigned int k = 1; k < degree; ++k)
      for (unsigned int i = triangle.size() - n_interior_triangle;
           i < triangle.size();
           ++i)
        points.emplace_back(triangle[i][0], triangle[i][1], double(k) / degree);
    AssertDimension(points.size(), compute_n_polynomials_wedge(3, degree));

    // the triangle and line points of each point
    std::vector<std::array<unsigned int, 2>> pairs;
    for (const Point<3> &p : points)
      {
        std::array<unsigned int, 2> pair = {
          {numbers::invalid_unsigned_int, numbers::invalid_unsigned_int}};
        for (unsigned int i = 0; i < triangle.size(); ++i)
          if (std::abs(triangle[i][0] - p[0]) +
                std::abs(triangle[i][1] - p[1]) <
              1e-12)
            pair[0] = i;
        for (unsigned int i = 0; i < line.size(); ++i)
          if (std::abs(line[i][0] - p[2]) < 1e-12)
            pair[1] = i;
        Assert(pair[0] != numbers::invalid_unsigned_int &&
                 pair[1] != numbers::invalid_unsigned_int,
               ExcInternalError());
        pairs.push_back(pair);
      }
    return pairs;
  }
} // namespace internal



template <int dim>
ScalarLagrangePolynomialWedge<dim>::ScalarLagrangePolynomialWedge(
  const unsigned int degree)
  : ScalarPolynomialsBase<dim>(degree, compute_n_polynomials_wedge(dim, degree))
  , poly_tri(BarycentricPolynomials<2>::get_fe_p_basis(degree))
  , poly_line(BarycentricPolynomials<1>::get_fe_p_basis(degree))
  , dof_pairs(internal::wedge_dof_pairs(degree))
{}



template <int dim>
double
ScalarLagrangePolynomialWedge<dim>::compute_value(const unsigned int i,
                                                  const Point<dim>  &p) const
{
  const auto &pair = dof_pairs[i];

  const Point<2> p_tri(p[0], p[1]);
  const auto     v_tri = poly_tri.compute_value(pair[0], p_tri);

  const Point<1> p_line(p[2]);
  const auto     v_line = poly_line.compute_value(pair[1], p_line);

  return v_tri * v_line;
}



template <int dim>
Tensor<1, dim>
ScalarLagrangePolynomialWedge<dim>::compute_grad(const unsigned int i,
                                                 const Point<dim>  &p) const
{
  const auto &pair = dof_pairs[i];

  const Point<2> p_tri(p[0], p[1]);
  const auto     v_tri = poly_tri.compute_value(pair[0], p_tri);
  const auto     g_tri = poly_tri.compute_grad(pair[0], p_tri);

  const Point<1> p_line(p[2]);
  const auto     v_line = poly_line.compute_value(pair[1], p_line);
  const auto     g_line = poly_line.compute_grad(pair[1], p_line);

  Tensor<1, dim> grad;
  grad[0] = g_tri[0] * v_line;
  grad[1] = g_tri[1] * v_line;
  grad[2] = v_tri * g_line[0];

  return grad;
}



template <int dim>
Tensor<2, dim>
ScalarLagrangePolynomialWedge<dim>::compute_grad_grad(const unsigned int i,
                                                      const Point<dim> &p) const
{
  const auto &pair = dof_pairs[i];

  const Point<2> p_tri(p[0], p[1]);
  const auto     v_tri  = poly_tri.compute_value(pair[0], p_tri);
  const auto     g_tri  = poly_tri.compute_grad(pair[0], p_tri);
  const auto     gg_tri = poly_tri.compute_grad_grad(pair[0], p_tri);

  const Point<1> p_line(p[2]);
  const auto     v_line  = poly_line.compute_value(pair[1], p_line);
  const auto     g_line  = poly_line.compute_grad(pair[1], p_line);
  const auto     gg_line = poly_line.compute_grad_grad(pair[1], p_line);

  // the product rule for the triangle polynomial in x_0, x_1 times the line
  // polynomial in x_2 (wedges only exist in 3d; the guard avoids writing out
  // of bounds in the other instantiations)
  Tensor<2, dim> grad_grad;
  AssertDimension(dim, 3);
  if constexpr (dim == 3)
    {
      for (unsigned int a = 0; a < 2; ++a)
        {
          for (unsigned int b = 0; b < 2; ++b)
            grad_grad[a][b] = gg_tri[a][b] * v_line;
          grad_grad[a][2] = g_tri[a] * g_line[0];
          grad_grad[2][a] = g_tri[a] * g_line[0];
        }
      grad_grad[2][2] = v_tri * gg_line[0][0];
    }

  return grad_grad;
}



template <int dim>
void
ScalarLagrangePolynomialWedge<dim>::evaluate(
  const Point<dim>            &unit_point,
  std::vector<double>         &values,
  std::vector<Tensor<1, dim>> &grads,
  std::vector<Tensor<2, dim>> &grad_grads,
  std::vector<Tensor<3, dim>> &third_derivatives,
  std::vector<Tensor<4, dim>> &fourth_derivatives) const
{
  (void)third_derivatives;
  (void)fourth_derivatives;

  if (values.size() == this->n())
    for (unsigned int i = 0; i < this->n(); ++i)
      values[i] = compute_value(i, unit_point);

  if (grads.size() == this->n())
    for (unsigned int i = 0; i < this->n(); ++i)
      grads[i] = compute_grad(i, unit_point);

  if (grad_grads.size() == this->n())
    for (unsigned int i = 0; i < this->n(); ++i)
      grad_grads[i] = compute_grad_grad(i, unit_point);
}



template <int dim>
Tensor<1, dim>
ScalarLagrangePolynomialWedge<dim>::compute_1st_derivative(
  const unsigned int i,
  const Point<dim>  &p) const
{
  return compute_grad(i, p);
}



template <int dim>
Tensor<2, dim>
ScalarLagrangePolynomialWedge<dim>::compute_2nd_derivative(
  const unsigned int i,
  const Point<dim>  &p) const
{
  return compute_grad_grad(i, p);
}



template <int dim>
Tensor<3, dim>
ScalarLagrangePolynomialWedge<dim>::compute_3rd_derivative(
  const unsigned int i,
  const Point<dim>  &p) const
{
  (void)i;
  (void)p;

  DEAL_II_NOT_IMPLEMENTED();

  return {};
}



template <int dim>
Tensor<4, dim>
ScalarLagrangePolynomialWedge<dim>::compute_4th_derivative(
  const unsigned int i,
  const Point<dim>  &p) const
{
  (void)i;
  (void)p;

  DEAL_II_NOT_IMPLEMENTED();

  return {};
}



template <int dim>
std::string
ScalarLagrangePolynomialWedge<dim>::name() const
{
  return "ScalarLagrangePolynomialWedge";
}



template <int dim>
std::unique_ptr<ScalarPolynomialsBase<dim>>
ScalarLagrangePolynomialWedge<dim>::clone() const
{
  return std::make_unique<ScalarLagrangePolynomialWedge<dim>>(*this);
}



template class ScalarLagrangePolynomialWedge<1>;
template class ScalarLagrangePolynomialWedge<2>;
template class ScalarLagrangePolynomialWedge<3>;

DEAL_II_NAMESPACE_CLOSE
