// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


#include <deal.II/fe/fe_nedelec_pyramid.h>

#include <deal.II/grid/reference_cell.h>

#include <array>
#include <memory>
#include <sstream>

DEAL_II_NAMESPACE_OPEN

namespace
{
  /**
   * Value, gradient and Hessian of the rational vertex function N_i of the
   * reference pyramid: for a base vertex (a, b, 0), with C = 1 - z,
   *   N    = (C + a x)(C + b y) / (4 C),
   *   grad = (a/4 + ab y/(4C), b/4 + ab x/(4C), -1/4 + ab xy/(4C^2)),
   *   d_xy = ab/(4C), d_xz = ab y/(4C^2), d_yz = ab x/(4C^2),
   *   d_zz = ab xy/(2C^3), d_xx = d_yy = 0,
   * and N_4 = z for the apex.
   */
  struct VertexFunction
  {
    double       value;
    Tensor<1, 3> grad;
    Tensor<2, 3> hessian;
  };

  VertexFunction
  vertex_function(const unsigned int i, const Point<3> &p)
  {
    VertexFunction n;
    if (i == 4)
      {
        n.value   = p[2];
        n.grad[2] = 1.;
        return n;
      }
    const Point<3>     v  = ReferenceCells::Pyramid.vertex<3>(i);
    const double       a  = v[0];
    const double       b  = v[1];
    const double       x  = p[0];
    const double       y  = p[1];
    // At the apex (C = 0, x = y = 0) the expressions are 0/0. The shape
    // functions of the element and their gradients are bounded there (the
    // vertex functions times the gradients and Hessians, as they appear in
    // theta and its gradient), so a tiny C gives the limit along the axis of
    // the pyramid; points with C > 1e-12 are not changed.
    const double       C  = std::max(1. - p[2], 1e-12);
    const double       ab = a * b;
    n.value               = (C + a * x) * (C + b * y) / (4. * C);
    n.grad[0]             = a / 4. + ab * y / (4. * C);
    n.grad[1]             = b / 4. + ab * x / (4. * C);
    n.grad[2]             = -0.25 + ab * x * y / (4. * C * C);
    n.hessian[0][1] = n.hessian[1][0] = ab / (4. * C);
    n.hessian[0][2] = n.hessian[2][0] = ab * y / (4. * C * C);
    n.hessian[1][2] = n.hessian[2][1] = ab * x / (4. * C * C);
    n.hessian[2][2]                   = ab * x * y / (2. * C * C * C);
    return n;
  }

  // theta_pq = N_p grad N_q - N_q grad N_p and its gradient
  // d_d theta_c = d_d N_p d_c N_q + N_p d_dc N_q - d_d N_q d_c N_p - N_q d_dc N_p
  // (indexed [component][direction])
  Tensor<1, 3>
  theta(const VertexFunction &np, const VertexFunction &nq)
  {
    return np.value * nq.grad - nq.value * np.grad;
  }

  Tensor<2, 3>
  grad_theta(const VertexFunction &np, const VertexFunction &nq)
  {
    Tensor<2, 3> g;
    for (unsigned int c = 0; c < 3; ++c)
      for (unsigned int d = 0; d < 3; ++d)
        g[c][d] = np.grad[d] * nq.grad[c] + np.value * nq.hessian[c][d] -
                  nq.grad[d] * np.grad[c] - nq.value * np.hessian[c][d];
    return g;
  }

  // the diagonals of the base, from vertex 0 to 3 and from vertex 1 to 2
  constexpr std::array<std::array<unsigned int, 2>, 2> diagonals = {
    {{{0, 3}}, {{1, 2}}}};

  // weights of theta along the two diagonals in the shape functions of the
  // base lines 0:(0,2), 1:(1,3), 2:(0,1), 3:(2,3): the line integrals along
  // the diagonals of the lowest-order function of the line on the
  // quadrilateral [-1,1]^2, e.g. for line 0 (x = -1, from y = -1 to y = 1)
  // the function (0, (1 - x)/4), whose integral along both diagonals is 1/2
  constexpr std::array<std::array<double, 2>, 4> diagonal_weights = {
    {{{0.5, 0.5}}, {{0.5, 0.5}}, {{0.5, -0.5}}, {{0.5, -0.5}}}};
} // namespace



template <int dim, int spacedim>
FE_NedelecPyramid<dim, spacedim>::FE_NedelecPyramid(const unsigned int order)
  : FE_NedelecLowestOrderBase<dim, spacedim>(ReferenceCells::Pyramid, order)
{}



template <int dim, int spacedim>
std::vector<std::vector<Tensor<1, dim>>>
FE_NedelecPyramid<dim, spacedim>::get_shape_values(
  const std::vector<Point<dim>> &p_list) const
{
  Assert(this->degree == 1, ExcMessage("Only order 0 is implemented."));
  const ReferenceCell ref_cell = ReferenceCells::Pyramid;

  std::vector<std::vector<Tensor<1, dim>>> values(
    ref_cell.n_lines(), std::vector<Tensor<1, dim>>(p_list.size()));
  std::array<VertexFunction, 5> n;
  for (unsigned int i = 0; i < p_list.size(); ++i)
    {
      for (unsigned int v = 0; v < 5; ++v)
        n[v] = vertex_function(v, p_list[i]);
      for (unsigned int l = 0; l < ref_cell.n_lines(); ++l)
        {
          values[l][i] = theta(n[ref_cell.line_to_cell_vertices(l, 0)],
                               n[ref_cell.line_to_cell_vertices(l, 1)]);
          if (l < 4)
            for (unsigned int k = 0; k < 2; ++k)
              values[l][i] +=
                diagonal_weights[l][k] *
                theta(n[diagonals[k][0]], n[diagonals[k][1]]);
        }
    }
  return values;
}



template <int dim, int spacedim>
std::vector<std::vector<Tensor<2, dim>>>
FE_NedelecPyramid<dim, spacedim>::get_shape_grads(
  const std::vector<Point<dim>> &p_list) const
{
  Assert(this->degree == 1, ExcMessage("Only order 0 is implemented."));
  const ReferenceCell ref_cell = ReferenceCells::Pyramid;

  std::vector<std::vector<Tensor<2, dim>>> grads(
    ref_cell.n_lines(), std::vector<Tensor<2, dim>>(p_list.size()));
  std::array<VertexFunction, 5> n;
  for (unsigned int i = 0; i < p_list.size(); ++i)
    {
      for (unsigned int v = 0; v < 5; ++v)
        n[v] = vertex_function(v, p_list[i]);
      for (unsigned int l = 0; l < ref_cell.n_lines(); ++l)
        {
          grads[l][i] = grad_theta(n[ref_cell.line_to_cell_vertices(l, 0)],
                                   n[ref_cell.line_to_cell_vertices(l, 1)]);
          if (l < 4)
            for (unsigned int k = 0; k < 2; ++k)
              grads[l][i] +=
                diagonal_weights[l][k] *
                grad_theta(n[diagonals[k][0]], n[diagonals[k][1]]);
        }
    }
  return grads;
}



template <int dim, int spacedim>
std::string
FE_NedelecPyramid<dim, spacedim>::get_name() const
{
  std::ostringstream namebuf;
  namebuf << "FE_NedelecPyramid<" << Utilities::dim_string(dim, spacedim)
          << ">(" << this->degree - 1 << ")";
  return namebuf.str();
}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, dim>>
FE_NedelecPyramid<dim, spacedim>::clone() const
{
  return std::make_unique<FE_NedelecPyramid<dim, spacedim>>(*this);
}



// explicit instantiations
template class FE_NedelecPyramid<3>;

DEAL_II_NAMESPACE_CLOSE
