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


#include <deal.II/fe/fe_nedelec_wedge.h>

#include <deal.II/grid/reference_cell.h>

#include <memory>
#include <sstream>

DEAL_II_NAMESPACE_OPEN

namespace
{
  // barycentric coordinates of the triangle (x,y) and their gradients
  double
  lambda(const unsigned int i, const Point<3> &p)
  {
    return (i == 0) ? 1. - p[0] - p[1] : p[i - 1];
  }

  Tensor<1, 3>
  grad_lambda(const unsigned int i)
  {
    Tensor<1, 3> g;
    if (i == 0)
      g[0] = g[1] = -1.;
    else
      g[i - 1] = 1.;
    return g;
  }

  // mu_0 = 1 - z, mu_1 = z
  double
  mu(const unsigned int k, const Point<3> &p)
  {
    return (k == 0) ? 1. - p[2] : p[2];
  }

  Tensor<1, 3>
  grad_mu(const unsigned int k)
  {
    Tensor<1, 3> g;
    g[2] = (k == 0) ? -1. : 1.;
    return g;
  }
} // namespace



template <int dim, int spacedim>
FE_NedelecWedge<dim, spacedim>::FE_NedelecWedge(const unsigned int order)
  : FE_NedelecLowestOrderBase<dim, spacedim>(ReferenceCells::Wedge, order)
{}



template <int dim, int spacedim>
std::vector<std::vector<Tensor<1, dim>>>
FE_NedelecWedge<dim, spacedim>::get_shape_values(
  const std::vector<Point<dim>> &p_list) const
{
  Assert(this->degree == 1, ExcMessage("Only order 0 is implemented."));
  const ReferenceCell ref_cell = ReferenceCells::Wedge;

  std::vector<std::vector<Tensor<1, dim>>> values(
    ref_cell.n_lines(), std::vector<Tensor<1, dim>>(p_list.size()));
  for (unsigned int l = 0; l < ref_cell.n_lines(); ++l)
    {
      // vertex v of the wedge is vertex v % 3 of the triangle at level v / 3
      const unsigned int p = ref_cell.line_to_cell_vertices(l, 0);
      const unsigned int q = ref_cell.line_to_cell_vertices(l, 1);
      for (unsigned int i = 0; i < p_list.size(); ++i)
        {
          const Point<dim> &x = p_list[i];
          if (p / 3 == q / 3)
            // horizontal line: Whitney function of the triangle times mu
            values[l][i] = (lambda(p % 3, x) * grad_lambda(q % 3) -
                            lambda(q % 3, x) * grad_lambda(p % 3)) *
                           mu(p / 3, x);
          else
            {
              // vertical line, from level p / 3 to level q / 3
              Assert(p % 3 == q % 3, ExcInternalError());
              values[l][i][2] =
                lambda(p % 3, x) * ((q / 3 > p / 3) ? 1. : -1.);
            }
        }
    }
  return values;
}



template <int dim, int spacedim>
std::vector<std::vector<Tensor<2, dim>>>
FE_NedelecWedge<dim, spacedim>::get_shape_grads(
  const std::vector<Point<dim>> &p_list) const
{
  Assert(this->degree == 1, ExcMessage("Only order 0 is implemented."));
  const ReferenceCell ref_cell = ReferenceCells::Wedge;

  // gradients are indexed [component][direction]
  std::vector<std::vector<Tensor<2, dim>>> grads(
    ref_cell.n_lines(), std::vector<Tensor<2, dim>>(p_list.size()));
  for (unsigned int l = 0; l < ref_cell.n_lines(); ++l)
    {
      const unsigned int p = ref_cell.line_to_cell_vertices(l, 0);
      const unsigned int q = ref_cell.line_to_cell_vertices(l, 1);
      for (unsigned int i = 0; i < p_list.size(); ++i)
        {
          const Point<dim> &x = p_list[i];
          if (p / 3 == q / 3)
            {
              const Tensor<1, dim> gp = grad_lambda(p % 3);
              const Tensor<1, dim> gq = grad_lambda(q % 3);
              const Tensor<1, dim> whitney =
                lambda(p % 3, x) * gq - lambda(q % 3, x) * gp;
              // d_j (W_i mu) = mu d_j W_i + W_i d_j mu with
              // d_j W_i = d_j lambda_p d_i lambda_q - d_j lambda_q d_i lambda_p
              for (unsigned int c = 0; c < dim; ++c)
                for (unsigned int d = 0; d < dim; ++d)
                  grads[l][i][c][d] =
                    mu(p / 3, x) * (gp[d] * gq[c] - gq[d] * gp[c]) +
                    whitney[c] * grad_mu(p / 3)[d];
            }
          else
            {
              const double         sign = (q / 3 > p / 3) ? 1. : -1.;
              const Tensor<1, dim> g    = grad_lambda(p % 3);
              for (unsigned int d = 0; d < dim; ++d)
                grads[l][i][2][d] = sign * g[d];
            }
        }
    }
  return grads;
}



template <int dim, int spacedim>
std::string
FE_NedelecWedge<dim, spacedim>::get_name() const
{
  std::ostringstream namebuf;
  namebuf << "FE_NedelecWedge<" << Utilities::dim_string(dim, spacedim)
          << ">(" << this->degree - 1 << ")";
  return namebuf.str();
}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, dim>>
FE_NedelecWedge<dim, spacedim>::clone() const
{
  return std::make_unique<FE_NedelecWedge<dim, spacedim>>(*this);
}



// explicit instantiations
template class FE_NedelecWedge<3>;

DEAL_II_NAMESPACE_CLOSE
