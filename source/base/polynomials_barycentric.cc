// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2021 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

#include <deal.II/base/polynomials_barycentric.h>

#include <deal.II/grid/reference_cell.h>

DEAL_II_NAMESPACE_OPEN

namespace internal
{
  /**
   * Get the highest degree of the barycentric polynomial (in Cartesian
   * coordinates).
   */
  template <int dim>
  unsigned int
  get_degree(
    const std::vector<typename BarycentricPolynomials<dim>::PolyType> &polys)
  {
    // Since the first variable in a simplex polynomial is, e.g., in 2d,
    //
    // t0 = 1 - x - y
    //
    // (that is, it depends on the Cartesian variables), we have to compute
    // its degree separately. An example: t0*t1*t2 has degree 1 in the affine
    // polynomial basis but is degree 2 in the Cartesian polynomial basis.
    std::size_t max_degree = 0;
    for (const auto &poly : polys)
      {
        const TableIndices<dim + 1> degrees = poly.degrees();

        const auto  degree_0 = degrees[0];
        std::size_t degree_d = 0;
        for (unsigned int d = 1; d < dim + 1; ++d)
          degree_d = std::max(degree_d, degrees[d]);

        max_degree = std::max(max_degree, degree_d + degree_0);
      }

    return max_degree;
  }
} // namespace internal


namespace internal
{
  namespace
  {
    // the points of the degree-degree lattice inside the d-simplex with the
    // given vertices, see simplex_lattice_points()
    template <int d, int dim>
    std::vector<Point<dim>>
    interior_lattice_points(const unsigned int             degree,
                            const std::vector<Point<dim>> &vertices)
    {
      if (degree < d + 1)
        return {};
      const unsigned int inner_degree = degree - d - 1;

      Point<dim> sum;
      for (const Point<dim> &v : vertices)
        sum += v;
      std::vector<Point<dim>> inner_vertices;
      for (const Point<dim> &v : vertices)
        inner_vertices.push_back((sum + double(inner_degree) * v) / degree);

      if (inner_degree == 0)
        return {inner_vertices[0]};

      // map the lattice points of the reference d-simplex by their
      // barycentric coordinates
      std::vector<Point<dim>> points;
      for (const Point<d> &x : simplex_lattice_points<d>(inner_degree))
        {
          double     lambda_0 = 1.;
          Point<dim> y;
          for (unsigned int i = 0; i < d; ++i)
            {
              lambda_0 -= x[i];
              y += x[i] * inner_vertices[i + 1];
            }
          y += lambda_0 * inner_vertices[0];
          points.push_back(y);
        }
      return points;
    }
  } // namespace



  template <int dim>
  std::vector<Point<dim>>
  simplex_lattice_points(const unsigned int degree)
  {
    Assert(degree > 0, ExcInternalError());
    const auto              reference_cell = ReferenceCells::get_simplex<dim>();
    std::vector<Point<dim>> points;

    // vertices
    for (const unsigned int v : reference_cell.vertex_indices())
      points.push_back(reference_cell.template vertex<dim>(v));

    // lines
    if (dim > 0)
      for (const unsigned int l : reference_cell.line_indices())
        {
          const Point<dim> p0 =
            points[reference_cell.line_to_cell_vertices(l, 0)];
          const Point<dim> p1 =
            points[reference_cell.line_to_cell_vertices(l, 1)];
          for (unsigned int i = 1; i < degree; ++i)
            points.push_back(p0 + (double(i) / degree) * (p1 - p0));
        }

    // triangles: the cell in 2d, the faces in 3d
    if constexpr (dim == 2)
      {
        const std::vector<Point<dim>> vertices(points.begin(),
                                               points.begin() + 3);
        for (const Point<dim> &p : interior_lattice_points<2>(degree, vertices))
          points.push_back(p);
      }
    else if constexpr (dim == 3)
      {
        for (const unsigned int f : reference_cell.face_indices())
          {
            std::vector<Point<dim>> vertices;
            for (unsigned int v = 0; v < 3; ++v)
              vertices.push_back(points[reference_cell.face_to_cell_vertices(
                f, v, numbers::default_geometric_orientation)]);
            for (const Point<dim> &p :
                 interior_lattice_points<2>(degree, vertices))
              points.push_back(p);
          }
        const std::vector<Point<dim>> vertices(points.begin(),
                                               points.begin() + 4);
        for (const Point<dim> &p : interior_lattice_points<3>(degree, vertices))
          points.push_back(p);
      }

    return points;
  }

  template std::vector<Point<1>>
  simplex_lattice_points<1>(const unsigned int);
  template std::vector<Point<2>>
  simplex_lattice_points<2>(const unsigned int);
  template std::vector<Point<3>>
  simplex_lattice_points<3>(const unsigned int);
} // namespace internal


template <int dim>
BarycentricPolynomials<dim>
BarycentricPolynomials<dim>::get_fe_p_basis(const unsigned int degree)
{
  std::vector<PolyType> polys;

  const auto reference_cell = ReferenceCells::get_simplex<dim>();

  switch (degree)
    {
      case 0:
        polys.push_back(0 * BarycentricPolynomial<dim, double>::monomial(0) +
                        1);
        break;
      case 1:
        {
          for (const unsigned int v : reference_cell.vertex_indices())
            polys.push_back(BarycentricPolynomial<dim, double>::monomial(v));
          break;
        }
      case 2:
        {
          // vertices, then lines:
          for (const unsigned int v : reference_cell.vertex_indices())
            polys.push_back(
              BarycentricPolynomial<dim, double>::monomial(v) *
              (2 * BarycentricPolynomial<dim, double>::monomial(v) - 1));
          for (const unsigned int l : reference_cell.line_indices())
            {
              const auto v0 = reference_cell.line_to_cell_vertices(l, 0);
              const auto v1 = reference_cell.line_to_cell_vertices(l, 1);
              polys.push_back(4 *
                              BarycentricPolynomial<dim, double>::monomial(v0) *
                              BarycentricPolynomial<dim, double>::monomial(v1));
            }
          break;
        }
      case 3:
        {
          // vertices, then lines, then quads:
          for (const unsigned int v : reference_cell.vertex_indices())
            polys.push_back(
              0.5 * BarycentricPolynomial<dim, double>::monomial(v) *
              (3 * BarycentricPolynomial<dim, double>::monomial(v) - 1) *
              (3 * BarycentricPolynomial<dim, double>::monomial(v) - 2));
          for (unsigned int l : reference_cell.line_indices())
            {
              const auto v0 = reference_cell.line_to_cell_vertices(l, 0);
              const auto v1 = reference_cell.line_to_cell_vertices(l, 1);
              polys.push_back(
                4.5 * BarycentricPolynomial<dim, double>::monomial(v0) *
                (3 * BarycentricPolynomial<dim, double>::monomial(v0) - 1) *
                BarycentricPolynomial<dim, double>::monomial(v1));
              polys.push_back(
                4.5 * BarycentricPolynomial<dim, double>::monomial(v0) *
                (3 * BarycentricPolynomial<dim, double>::monomial(v1) - 1) *
                BarycentricPolynomial<dim, double>::monomial(v1));
            }

          if (dim == 2)
            {
              polys.push_back(27 *
                              BarycentricPolynomial<dim, double>::monomial(0) *
                              BarycentricPolynomial<dim, double>::monomial(1) *
                              BarycentricPolynomial<dim, double>::monomial(2));
            }
          else if (dim == 3)
            {
              polys.push_back(27 *
                              BarycentricPolynomial<dim, double>::monomial(0) *
                              BarycentricPolynomial<dim, double>::monomial(1) *
                              BarycentricPolynomial<dim, double>::monomial(2));
              polys.push_back(27 *
                              BarycentricPolynomial<dim, double>::monomial(0) *
                              BarycentricPolynomial<dim, double>::monomial(1) *
                              BarycentricPolynomial<dim, double>::monomial(3));
              polys.push_back(27 *
                              BarycentricPolynomial<dim, double>::monomial(0) *
                              BarycentricPolynomial<dim, double>::monomial(2) *
                              BarycentricPolynomial<dim, double>::monomial(3));
              polys.push_back(27 *
                              BarycentricPolynomial<dim, double>::monomial(1) *
                              BarycentricPolynomial<dim, double>::monomial(2) *
                              BarycentricPolynomial<dim, double>::monomial(3));
            }

          break;
        }
      default:
        {
          // the Lagrange basis of the lattice points: for the point with
          // barycentric coordinates a / degree, the product of
          // (degree lambda_i - m) / (m + 1) over m < a_i and all i
          for (const Point<dim> &x : internal::simplex_lattice_points<dim>(degree))
            {
              std::array<double, dim + 1> lambda;
              lambda[0] = 1.;
              for (unsigned int d = 0; d < dim; ++d)
                {
                  lambda[d + 1] = x[d];
                  lambda[0] -= x[d];
                }
              PolyType phi(TableIndices<dim + 1>(), 1.);
              for (unsigned int i = 0; i <= dim; ++i)
                {
                  const unsigned int a = std::lround(degree * lambda[i]);
                  for (unsigned int m = 0; m < a; ++m)
                    phi = phi * ((double(degree) * PolyType::monomial(i) -
                                  double(m)) /
                                 double(m + 1));
                }
              polys.push_back(phi);
            }
        }
    }

  return BarycentricPolynomials<dim>(polys);
}



template <int dim>
BarycentricPolynomials<dim>::BarycentricPolynomials(
  const std::vector<PolyType> &polynomials)
  : ScalarPolynomialsBase<dim>(internal::get_degree<dim>(polynomials),
                               polynomials.size())
  , polys(polynomials)
{
  poly_grads.resize(polynomials.size());
  poly_hessians.resize(polynomials.size());
  poly_third_derivatives.resize(polynomials.size());
  poly_fourth_derivatives.resize(polynomials.size());

  for (std::size_t i = 0; i < polynomials.size(); ++i)
    {
      // gradients
      for (unsigned int d = 0; d < dim; ++d)
        poly_grads[i][d] = polynomials[i].derivative(d);

      // hessians
      for (unsigned int d0 = 0; d0 < dim; ++d0)
        for (unsigned int d1 = 0; d1 < dim; ++d1)
          poly_hessians[i][d0][d1] = poly_grads[i][d0].derivative(d1);

      // third derivatives
      for (unsigned int d0 = 0; d0 < dim; ++d0)
        for (unsigned int d1 = 0; d1 < dim; ++d1)
          for (unsigned int d2 = 0; d2 < dim; ++d2)
            poly_third_derivatives[i][d0][d1][d2] =
              poly_hessians[i][d0][d1].derivative(d2);

      // fourth derivatives
      for (unsigned int d0 = 0; d0 < dim; ++d0)
        for (unsigned int d1 = 0; d1 < dim; ++d1)
          for (unsigned int d2 = 0; d2 < dim; ++d2)
            for (unsigned int d3 = 0; d3 < dim; ++d3)
              poly_fourth_derivatives[i][d0][d1][d2][d3] =
                poly_third_derivatives[i][d0][d1][d2].derivative(d3);
    }
}



template <int dim>
void
BarycentricPolynomials<dim>::evaluate(
  const Point<dim>            &unit_point,
  std::vector<double>         &values,
  std::vector<Tensor<1, dim>> &grads,
  std::vector<Tensor<2, dim>> &grad_grads,
  std::vector<Tensor<3, dim>> &third_derivatives,
  std::vector<Tensor<4, dim>> &fourth_derivatives) const
{
  Assert(values.size() == this->n() || values.empty(),
         ExcDimensionMismatch2(values.size(), this->n(), 0));
  Assert(grads.size() == this->n() || grads.empty(),
         ExcDimensionMismatch2(grads.size(), this->n(), 0));
  Assert(grad_grads.size() == this->n() || grad_grads.empty(),
         ExcDimensionMismatch2(grad_grads.size(), this->n(), 0));
  Assert(third_derivatives.size() == this->n() || third_derivatives.empty(),
         ExcDimensionMismatch2(third_derivatives.size(), this->n(), 0));
  Assert(fourth_derivatives.size() == this->n() || fourth_derivatives.empty(),
         ExcDimensionMismatch2(fourth_derivatives.size(), this->n(), 0));

  for (std::size_t i = 0; i < polys.size(); ++i)
    {
      if (values.size() == this->n())
        values[i] = polys[i].value(unit_point);

      // gradients
      if (grads.size() == this->n())
        for (unsigned int d = 0; d < dim; ++d)
          grads[i][d] = poly_grads[i][d].value(unit_point);

      // hessians
      if (grad_grads.size() == this->n())
        for (unsigned int d0 = 0; d0 < dim; ++d0)
          for (unsigned int d1 = 0; d1 < dim; ++d1)
            grad_grads[i][d0][d1] = poly_hessians[i][d0][d1].value(unit_point);

      // third derivatives
      if (third_derivatives.size() == this->n())
        for (unsigned int d0 = 0; d0 < dim; ++d0)
          for (unsigned int d1 = 0; d1 < dim; ++d1)
            for (unsigned int d2 = 0; d2 < dim; ++d2)
              third_derivatives[i][d0][d1][d2] =
                poly_third_derivatives[i][d0][d1][d2].value(unit_point);

      // fourth derivatives
      if (fourth_derivatives.size() == this->n())
        for (unsigned int d0 = 0; d0 < dim; ++d0)
          for (unsigned int d1 = 0; d1 < dim; ++d1)
            for (unsigned int d2 = 0; d2 < dim; ++d2)
              for (unsigned int d3 = 0; d3 < dim; ++d3)
                fourth_derivatives[i][d0][d1][d2][d3] =
                  poly_fourth_derivatives[i][d0][d1][d2][d3].value(unit_point);
    }
}



template <int dim>
double
BarycentricPolynomials<dim>::compute_value(const unsigned int i,
                                           const Point<dim>  &p) const
{
  AssertIndexRange(i, this->n());
  return polys[i].value(p);
}



template <int dim>
Tensor<1, dim>
BarycentricPolynomials<dim>::compute_1st_derivative(const unsigned int i,
                                                    const Point<dim>  &p) const
{
  Tensor<1, dim> result;
  for (unsigned int d = 0; d < dim; ++d)
    result[d] = poly_grads[i][d].value(p);
  return result;
}



template <int dim>
Tensor<2, dim>
BarycentricPolynomials<dim>::compute_2nd_derivative(const unsigned int i,
                                                    const Point<dim>  &p) const
{
  Tensor<2, dim> result;
  for (unsigned int d0 = 0; d0 < dim; ++d0)
    for (unsigned int d1 = 0; d1 < dim; ++d1)
      result[d0][d1] = poly_hessians[i][d0][d1].value(p);

  return result;
}



template <int dim>
Tensor<3, dim>
BarycentricPolynomials<dim>::compute_3rd_derivative(const unsigned int i,
                                                    const Point<dim>  &p) const
{
  Tensor<3, dim> result;
  for (unsigned int d0 = 0; d0 < dim; ++d0)
    for (unsigned int d1 = 0; d1 < dim; ++d1)
      for (unsigned int d2 = 0; d2 < dim; ++d2)
        result[d0][d1][d2] = poly_third_derivatives[i][d0][d1][d2].value(p);

  return result;
}



template <int dim>
Tensor<4, dim>
BarycentricPolynomials<dim>::compute_4th_derivative(const unsigned int i,
                                                    const Point<dim>  &p) const
{
  Tensor<4, dim> result;
  for (unsigned int d0 = 0; d0 < dim; ++d0)
    for (unsigned int d1 = 0; d1 < dim; ++d1)
      for (unsigned int d2 = 0; d2 < dim; ++d2)
        for (unsigned int d3 = 0; d3 < dim; ++d3)
          result[d0][d1][d2][d3] =
            poly_fourth_derivatives[i][d0][d1][d2][d3].value(p);

  return result;
}



template <int dim>
Tensor<1, dim>
BarycentricPolynomials<dim>::compute_grad(const unsigned int i,
                                          const Point<dim>  &p) const
{
  return compute_1st_derivative(i, p);
}



template <int dim>
Tensor<2, dim>
BarycentricPolynomials<dim>::compute_grad_grad(const unsigned int i,
                                               const Point<dim>  &p) const
{
  return compute_2nd_derivative(i, p);
}



template <int dim>
std::unique_ptr<ScalarPolynomialsBase<dim>>
BarycentricPolynomials<dim>::clone() const
{
  return std::make_unique<BarycentricPolynomials<dim>>(*this);
}



template <int dim>
std::string
BarycentricPolynomials<dim>::name() const
{
  return "BarycentricPolynomials<" + std::to_string(dim) + ">";
}



template <int dim>
std::size_t
BarycentricPolynomials<dim>::memory_consumption() const
{
  std::size_t poly_memory = 0;
  for (const auto &poly : polys)
    poly_memory += poly.memory_consumption();
  return ScalarPolynomialsBase<dim>::memory_consumption() + poly_memory +
         MemoryConsumption::memory_consumption(poly_grads) +
         MemoryConsumption::memory_consumption(poly_hessians) +
         MemoryConsumption::memory_consumption(poly_third_derivatives) +
         MemoryConsumption::memory_consumption(poly_fourth_derivatives);
}

template class BarycentricPolynomials<1>;
template class BarycentricPolynomials<2>;
template class BarycentricPolynomials<3>;

DEAL_II_NAMESPACE_CLOSE
