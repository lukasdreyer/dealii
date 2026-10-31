// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2020 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


#include <deal.II/base/point.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/polynomials_barycentric.h>
#include <deal.II/base/polynomials_pyramid.h>

#include <deal.II/grid/reference_cell.h>

#include <deal.II/lac/householder.h>
#include <deal.II/lac/vector.h>

DEAL_II_NAMESPACE_OPEN



namespace
{
  // Second derivative of the Jacobi polynomial, from
  // d/dx P_n^{a,b} = c P_{n-1}^{a+1,b+1} as in
  // Polynomials::jacobi_polynomial_derivative().
  double
  jacobi_polynomial_second_derivative(const unsigned int degree,
                                      const int          alpha,
                                      const int          beta,
                                      const double       x,
                                      const bool rescale_to_dealii_unit_interval)
  {
    if (degree < 2)
      return 0.;
    return (rescale_to_dealii_unit_interval ? 1. : 0.5) *
           (1 + alpha + beta + degree) *
           Polynomials::jacobi_polynomial_derivative<double>(
             degree - 1,
             alpha + 1,
             beta + 1,
             x,
             rescale_to_dealii_unit_interval);
  }



  // The Hessian of the orthogonal basis function A(xi) B(eta) C(z) with
  // xi = x/(1-z), eta = y/(1-z), A = P_i, B = P_j (Legendre on [-1,1]) and
  // C = (1-z)^m Q_k(z), m = max(i,j), Q_k = P_k^{2m+2,0} on [0,1].
  Tensor<2, 3>
  orthogonal_basis_hessian(const unsigned int i,
                           const unsigned int j,
                           const unsigned int k,
                           const Point<3>    &p)
  {
    const double       x = p[0], y = p[1], z = p[2];
    const unsigned int m = std::max(i, j);
    const int          a = 2 * m + 2;

    const double Q   = Polynomials::jacobi_polynomial_value<double>(k, a, 0, z, true);
    const double Q1  = Polynomials::jacobi_polynomial_derivative<double>(k, a, 0, z, true);
    const double Q2  = jacobi_polynomial_second_derivative(k, a, 0, z, true);

    Tensor<2, 3> H;

    // the cases in which 1/(1-z) cancels: Q_k(z), y Q_k(z), x Q_k(z)
    if (i == 0 && j == 0)
      {
        H[2][2] = Q2;
        return H;
      }
    else if (i == 0 && j == 1)
      {
        H[1][2] = H[2][1] = Q1;
        H[2][2]           = y * Q2;
        return H;
      }
    else if (i == 1 && j == 0)
      {
        H[0][2] = H[2][0] = Q1;
        H[2][2]           = x * Q2;
        return H;
      }
    else if (std::abs(z - 1.0) < 1e-14)
      // the rational functions are not smooth at the tip
      return H;

    const double r   = 1.0 / (1.0 - z);
    const double xi  = x * r;
    const double eta = y * r;

    const double A  = Polynomials::jacobi_polynomial_value<double>(i, 0, 0, xi, false);
    const double A1 = Polynomials::jacobi_polynomial_derivative<double>(i, 0, 0, xi, false);
    const double A2 = jacobi_polynomial_second_derivative(i, 0, 0, xi, false);
    const double B  = Polynomials::jacobi_polynomial_value<double>(j, 0, 0, eta, false);
    const double B1 = Polynomials::jacobi_polynomial_derivative<double>(j, 0, 0, eta, false);
    const double B2 = jacobi_polynomial_second_derivative(j, 0, 0, eta, false);

    const double s  = 1.0 - z;
    const double C  = std::pow(s, m) * Q;
    const double C1 = -1.0 * m * std::pow(s, m - 1) * Q + std::pow(s, m) * Q1;
    const double C2 = (m >= 2 ? m * (m - 1.0) * std::pow(s, m - 2) * Q : 0.) -
                      2.0 * m * std::pow(s, m - 1) * Q1 + std::pow(s, m) * Q2;

    const double r2 = r * r;
    H[0][0] = A2 * r2 * B * C;
    H[1][1] = A * B2 * r2 * C;
    H[0][1] = H[1][0] = A1 * B1 * r2 * C;
    H[0][2] = H[2][0] = A2 * xi * r2 * B * C + A1 * r2 * B * C +
                        A1 * B1 * eta * r2 * C + A1 * r * B * C1;
    H[1][2] = H[2][1] = A * B2 * eta * r2 * C + A * B1 * r2 * C +
                        A1 * xi * r2 * B1 * C + A * B1 * r * C1;
    H[2][2] = (A2 * xi * xi * r2 + 2 * A1 * xi * r2) * B * C +
              2 * A1 * xi * r * B1 * eta * r * C + 2 * A1 * xi * r * B * C1 +
              A * (B2 * eta * eta * r2 + 2 * B1 * eta * r2) * C +
              2 * A * B1 * eta * r * C1 + A * B * C2;
    return H;
  }
} // namespace



template <int dim>
ScalarLagrangePolynomialPyramid<dim>::ScalarLagrangePolynomialPyramid(
  const unsigned int degree)
  : ScalarLagrangePolynomialPyramid<dim>(
      1,
      5,
      {ReferenceCells::Pyramid.vertex<dim>(0),
       ReferenceCells::Pyramid.vertex<dim>(1),
       ReferenceCells::Pyramid.vertex<dim>(2),
       ReferenceCells::Pyramid.vertex<dim>(3),
       ReferenceCells::Pyramid.vertex<dim>(4)})
{
  AssertThrow(degree == 1,
              ExcNotImplemented(
                "This constructor only works for linear elements."));
}



template <int dim>
ScalarLagrangePolynomialPyramid<dim>::ScalarLagrangePolynomialPyramid(
  const unsigned int             degree,
  const unsigned int             n_dofs,
  const std::vector<Point<dim>> &support_points)
  : ScalarPolynomialsBase<dim>(degree, n_dofs)
{
  (void)n_dofs;

  AssertThrow(dim == 3,
              ExcNotImplemented("Pyramid elements only make sense in 3D."));

  AssertThrow(
    support_points.size() == n_dofs,
    ExcNotImplemented(
      "The number of DoF must be equal to the number of support points."));

  // fill VDM Matrix
  const unsigned int n = support_points.size();
  FullMatrix<double> VDM(n);

  for (unsigned i = 0; i < n; ++i)
    for (unsigned j = 0; j < n; ++j)
      VDM[i][j] =
        this->evaluate_orthogonal_basis_function(i, support_points[j]);

  // get the inverse matrix
  Householder<double> householder(VDM);
  Vector<double>      e(n);
  Vector<double>      x(n);

  // assume e_j are unit vectors
  // compute (VDM^-1)_ij by using (VDM^-1)_ij = e_i (VDM^-1) e_j
  // loop over all vectors e_j
  for (unsigned int j = 0; j < n; ++j)
    {
      e    = 0.;
      e[j] = 1.;

      x = 0.;
      // get x = (VDM^-1) e_j
      householder.least_squares(x, e);

      for (unsigned int i = 0; i < n; ++i)
        // (VDM^-1)_ij = e_i (VDM^-1) e_j
        VDM(i, j) = x[i];
    }

  vandermonde_matrix_inverse = VDM;

  // clean up small values
  for (unsigned int i = 0; i < n; ++i)
    for (unsigned int j = 0; j < n; ++j)
      if (std::abs(vandermonde_matrix_inverse[i][j]) < 1e-14)
        vandermonde_matrix_inverse[i][j] = 0.;
}



template <int dim>
double
ScalarLagrangePolynomialPyramid<
  dim>::evaluate_orthogonal_basis_function_by_degree(const unsigned int i,
                                                     const unsigned int j,
                                                     const unsigned int k,
                                                     const Point<dim>  &p) const
{
  AssertIndexRange(i, this->degree() + 1);
  AssertIndexRange(j, this->degree() + 1);
  AssertIndexRange(k, this->degree() + 1 - std::max(i, j));

  const double x = p[0];
  const double y = p[1];
  const double z = p[2];

  const unsigned int max_ij = std::max(i, j);

  // handle the special cases
  // assume (1-z)^0 = 1 even when z = 1
  if (max_ij == 0)
    return Polynomials::jacobi_polynomial_value<double>(k, 2, 0, z, true);

  // at the tip the basis looks like (x/(1-z))^i * (y/(1-z))^j * (1-z)^max(i,j)
  // when going to the tip of the cone the relation is roughly x = 1 - z, so
  // (x/(1-z)) = const and the (1-z)^max(i,j) term then pushes to 0
  if (std::fabs(z - 1.0) < 1e-14)
    return 0.0;

  const double ratio = 1.0 / (1.0 - z);

  const double phi =
    Polynomials::jacobi_polynomial_value<double>(i, 0, 0, x * ratio, false) *
    Polynomials::jacobi_polynomial_value<double>(j, 0, 0, y * ratio, false) *
    std::pow((1.0 - z), max_ij) *
    Polynomials::jacobi_polynomial_value<double>(k, 2 * max_ij + 2, 0, z, true);

  if (std::fabs(phi) < 1e-14)
    return 0.0;

  return phi;
}



template <int dim>
double
ScalarLagrangePolynomialPyramid<dim>::evaluate_orthogonal_basis_function(
  const unsigned int i,
  const Point<dim>  &p) const
{
  AssertIndexRange(i, this->n());

  // find corresponding entry to i
  // it holds 0 <= j, k <= degree,
  // 0 <= l <= degree - max(j, k)
  for (unsigned int j = 0, counter = 0; j <= this->degree(); ++j)
    for (unsigned int k = 0; k <= this->degree(); ++k)
      for (unsigned int l = 0; l <= this->degree() - std::max(j, k);
           ++l, ++counter)
        if (counter == i)
          return evaluate_orthogonal_basis_function_by_degree(j, k, l, p);

  DEAL_II_ASSERT_UNREACHABLE();
  return 0;
}



template <int dim>
Tensor<1, dim>
ScalarLagrangePolynomialPyramid<dim>::
  evaluate_orthogonal_basis_derivative_by_degree(const unsigned int i,
                                                 const unsigned int j,
                                                 const unsigned int k,
                                                 const Point<dim>  &p) const
{
  AssertIndexRange(i, this->degree() + 1);
  AssertIndexRange(j, this->degree() + 1);
  AssertIndexRange(k, this->degree() + 1 - std::max(i, j));

  Tensor<1, dim> grad;

  const double x = p[0];
  const double y = p[1];
  const double z = p[2];

  // handle the special cases where 1/(1-z) cancels and the tip
  if (i == 0 && j == 0)
    {
      // assume (1-z)^0 = 1 even when z = 1
      grad[0] = 0.;
      if constexpr (dim > 1)
        grad[1] = 0.;
      if constexpr (dim > 2)
        grad[2] =
          Polynomials::jacobi_polynomial_derivative<double>(k, 2, 0, z, true);

      return grad;
    }
  else if (i == 0 && j == 1)
    {
      grad[0] = 0.;
      if constexpr (dim > 1)
        grad[1] =
          Polynomials::jacobi_polynomial_value<double>(k, 4, 0, z, true);
      if constexpr (dim > 2)
        grad[2] =
          y *
          Polynomials::jacobi_polynomial_derivative<double>(k, 4, 0, z, true);

      return grad;
    }
  else if (i == 1 && j == 0)
    {
      grad[0] = Polynomials::jacobi_polynomial_value<double>(k, 4, 0, z, true);
      if constexpr (dim > 1)
        grad[1] = 0.0;
      if constexpr (dim > 2)
        grad[2] =
          x *
          Polynomials::jacobi_polynomial_derivative<double>(k, 4, 0, z, true);

      return grad;
    }
  else if (std::abs(p[2] - 1.0) < 1e-14)
    {
      grad = 0.;
      if (i == 1 && j == 1)
        // assume x/(1-z)-> 1 and y/(1-z)-> 1 at the tip
        for (unsigned int d = 0; d < dim; ++d)
          grad[d] =
            Polynomials::jacobi_polynomial_value<double>(k, 4, 0, z, true);

      return grad;
    }

  const unsigned int max_ij = std::max(i, j);
  const double       ratio  = 1.0 / (1.0 - z);

  grad[0] =
    Polynomials::jacobi_polynomial_derivative<double>(
      i, 0, 0, x * ratio, false) *
    ratio *
    Polynomials::jacobi_polynomial_value<double>(j, 0, 0, y * ratio, false) *
    std::pow((1.0 - z), max_ij) *
    Polynomials::jacobi_polynomial_value<double>(k, 2 * max_ij + 2, 0, z, true);
  if constexpr (dim > 1)
    grad[1] =
      Polynomials::jacobi_polynomial_value<double>(i, 0, 0, x * ratio, false) *
      Polynomials::jacobi_polynomial_derivative<double>(
        j, 0, 0, y * ratio, false) *
      ratio * std::pow((1.0 - z), max_ij) *
      Polynomials::jacobi_polynomial_value<double>(
        k, 2 * max_ij + 2, 0, z, true);
  if constexpr (dim > 2)
    grad[2] =
      Polynomials::jacobi_polynomial_derivative<double>(
        i, 0, 0, x * ratio, false) *
        x * ratio * ratio *
        Polynomials::jacobi_polynomial_value<double>(
          j, 0, 0, y * ratio, false) *
        std::pow((1.0 - z), max_ij) *
        Polynomials::jacobi_polynomial_value<double>(
          k, 2 * max_ij + 2, 0, z, true) +
      Polynomials::jacobi_polynomial_value<double>(i, 0, 0, x * ratio, false) *
        Polynomials::jacobi_polynomial_derivative<double>(
          j, 0, 0, y * ratio, false) *
        y * ratio * ratio * std::pow((1.0 - z), max_ij) *
        Polynomials::jacobi_polynomial_value<double>(
          k, 2 * max_ij + 2, 0, z, true) +
      Polynomials::jacobi_polynomial_value<double>(i, 0, 0, x * ratio, false) *
        Polynomials::jacobi_polynomial_value<double>(
          j, 0, 0, y * ratio, false) *
        (-1.0) * max_ij * std::pow((1.0 - z), max_ij - 1) *
        Polynomials::jacobi_polynomial_value<double>(
          k, 2 * max_ij + 2, 0, z, true) +
      Polynomials::jacobi_polynomial_value<double>(i, 0, 0, x * ratio, false) *
        Polynomials::jacobi_polynomial_value<double>(
          j, 0, 0, y * ratio, false) *
        std::pow((1.0 - z), max_ij) *
        Polynomials::jacobi_polynomial_derivative<double>(
          k, 2 * max_ij + 2, 0, 2.0 * z - 1.0, false) *
        2.0;

  for (unsigned int d = 0; d < dim; ++d)
    if (std::fabs(grad[d]) < 1e-14)
      grad[d] = 0.0;

  return grad;
}



template <int dim>
Tensor<1, dim>
ScalarLagrangePolynomialPyramid<dim>::evaluate_orthogonal_basis_derivative(
  const unsigned int i,
  const Point<dim>  &p) const
{
  AssertIndexRange(i, this->n());

  // find corresponding entrance to i
  for (unsigned int j = 0, counter = 0; j <= this->degree(); ++j)
    for (unsigned int k = 0; k <= this->degree(); ++k)
      for (unsigned int l = 0; l <= this->degree() - std::max(j, k);
           ++l, ++counter)
        if (counter == i)
          return evaluate_orthogonal_basis_derivative_by_degree(j, k, l, p);

  DEAL_II_ASSERT_UNREACHABLE();
  return Tensor<1, dim>();
}



template <int dim>
double
ScalarLagrangePolynomialPyramid<dim>::compute_value(const unsigned int i,
                                                    const Point<dim>  &p) const
{
  AssertDimension(dim, 3);
  AssertIndexRange(i, vandermonde_matrix_inverse.m());

  double result = 0.;
  for (unsigned int j = 0; j < vandermonde_matrix_inverse.n(); ++j)
    result += vandermonde_matrix_inverse[i][j] *
              this->evaluate_orthogonal_basis_function(j, p);

  if (std::fabs(result) < 1e-14)
    result = 0.0;

  return result;
}



template <int dim>
Tensor<1, dim>
ScalarLagrangePolynomialPyramid<dim>::compute_grad(const unsigned int i,
                                                   const Point<dim>  &p) const
{
  AssertDimension(dim, 3);
  AssertIndexRange(i, vandermonde_matrix_inverse.m());

  Tensor<1, dim> grad;

  grad = 0.;
  for (unsigned int j = 0; j < vandermonde_matrix_inverse.n(); ++j)
    grad += vandermonde_matrix_inverse[i][j] *
            this->evaluate_orthogonal_basis_derivative(j, p);

  for (unsigned int d = 0; d < dim; ++d)
    if (std::fabs(grad[d]) < 1e-14)
      grad[d] = 0.0;

  return grad;
}



template <int dim>
Tensor<2, dim>
ScalarLagrangePolynomialPyramid<dim>::compute_grad_grad(
  const unsigned int i,
  const Point<dim>  &p) const
{
  AssertDimension(dim, 3);
  AssertIndexRange(i, vandermonde_matrix_inverse.m());

  Tensor<2, dim> grad_grad;
  if constexpr (dim == 3)
    {
      // the orthogonal basis functions in the order of
      // evaluate_orthogonal_basis_function()
      unsigned int counter = 0;
      for (unsigned int j = 0; j <= this->degree(); ++j)
        for (unsigned int k = 0; k <= this->degree(); ++k)
          for (unsigned int l = 0; l <= this->degree() - std::max(j, k);
               ++l, ++counter)
            if (vandermonde_matrix_inverse[i][counter] != 0.)
              grad_grad += vandermonde_matrix_inverse[i][counter] *
                           orthogonal_basis_hessian(j, k, l, p);
    }
  else
    (void)p;

  return grad_grad;
}



template <int dim>
void
ScalarLagrangePolynomialPyramid<dim>::evaluate(
  const Point<dim>            &unit_point,
  std::vector<double>         &values,
  std::vector<Tensor<1, dim>> &grads,
  std::vector<Tensor<2, dim>> &grad_grads,
  std::vector<Tensor<3, dim>> &third_derivatives,
  std::vector<Tensor<4, dim>> &fourth_derivatives) const
{
  (void)grads;
  (void)grad_grads;
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
ScalarLagrangePolynomialPyramid<dim>::compute_1st_derivative(
  const unsigned int i,
  const Point<dim>  &p) const
{
  return compute_grad(i, p);
}



template <int dim>
Tensor<2, dim>
ScalarLagrangePolynomialPyramid<dim>::compute_2nd_derivative(
  const unsigned int i,
  const Point<dim>  &p) const
{
  return compute_grad_grad(i, p);
}



template <int dim>
Tensor<3, dim>
ScalarLagrangePolynomialPyramid<dim>::compute_3rd_derivative(
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
ScalarLagrangePolynomialPyramid<dim>::compute_4th_derivative(
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
ScalarLagrangePolynomialPyramid<dim>::name() const
{
  return "ScalarLagrangePolynomialPyramid";
}



template <int dim>
std::unique_ptr<ScalarPolynomialsBase<dim>>
ScalarLagrangePolynomialPyramid<dim>::clone() const
{
  return std::make_unique<ScalarLagrangePolynomialPyramid<dim>>(*this);
}



template class ScalarLagrangePolynomialPyramid<1>;
template class ScalarLagrangePolynomialPyramid<2>;
template class ScalarLagrangePolynomialPyramid<3>;

DEAL_II_NAMESPACE_CLOSE
