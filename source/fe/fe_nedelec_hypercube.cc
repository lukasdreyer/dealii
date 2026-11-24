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


#include <deal.II/fe/fe_nedelec_hypercube.h>

#include <deal.II/grid/reference_cell.h>

#include <memory>
#include <sstream>

DEAL_II_NAMESPACE_OPEN

namespace
{
  /**
   * The axis along which line l of the reference hypercube runs, the
   * direction (+1 or -1) of the line along it, and the coordinates of its
   * first vertex.
   */
  template <int dim>
  void
  line_geometry(const unsigned int l,
                unsigned int      &axis,
                double            &direction,
                Point<dim>        &start)
  {
    const ReferenceCell ref_cell = ReferenceCells::get_hypercube<dim>();
    start = ref_cell.template vertex<dim>(ref_cell.line_to_cell_vertices(l, 0));
    const Point<dim> end =
      ref_cell.template vertex<dim>(ref_cell.line_to_cell_vertices(l, 1));
    for (unsigned int d = 0; d < dim; ++d)
      if (start[d] != end[d])
        {
          axis      = d;
          direction = end[d] - start[d];
        }
  }

  // the factor of the other axis j in the weight: x_j or 1 - x_j
  inline double
  factor(const double start_j, const double x_j)
  {
    return (start_j == 1.) ? x_j : 1. - x_j;
  }

  inline double
  factor_derivative(const double start_j)
  {
    return (start_j == 1.) ? 1. : -1.;
  }
} // namespace



template <int dim, int spacedim>
FE_NedelecHypercube<dim, spacedim>::FE_NedelecHypercube(
  const unsigned int order)
  : FE_NedelecLowestOrderBase<dim, spacedim>(
      ReferenceCells::get_hypercube<dim>(),
      order)
{}



template <int dim, int spacedim>
std::vector<std::vector<Tensor<1, dim>>>
FE_NedelecHypercube<dim, spacedim>::get_shape_values(
  const std::vector<Point<dim>> &p_list) const
{
  Assert(this->degree == 1, ExcMessage("Only order 0 is implemented."));
  const unsigned int n_lines = this->reference_cell().n_lines();

  std::vector<std::vector<Tensor<1, dim>>> values(
    n_lines, std::vector<Tensor<1, dim>>(p_list.size()));
  for (unsigned int l = 0; l < n_lines; ++l)
    {
      unsigned int axis      = 0;
      double       direction = 0;
      Point<dim>   start;
      line_geometry(l, axis, direction, start);
      for (unsigned int i = 0; i < p_list.size(); ++i)
        {
          double w = 1.;
          for (unsigned int j = 0; j < dim; ++j)
            if (j != axis)
              w *= factor(start[j], p_list[i][j]);
          values[l][i][axis] = w * direction;
        }
    }
  return values;
}



template <int dim, int spacedim>
std::vector<std::vector<Tensor<2, dim>>>
FE_NedelecHypercube<dim, spacedim>::get_shape_grads(
  const std::vector<Point<dim>> &p_list) const
{
  Assert(this->degree == 1, ExcMessage("Only order 0 is implemented."));
  const unsigned int n_lines = this->reference_cell().n_lines();

  // gradients are indexed [component][direction]; only the component along
  // the axis of the line is nonzero, and it does not depend on that axis
  std::vector<std::vector<Tensor<2, dim>>> grads(
    n_lines, std::vector<Tensor<2, dim>>(p_list.size()));
  for (unsigned int l = 0; l < n_lines; ++l)
    {
      unsigned int axis      = 0;
      double       direction = 0;
      Point<dim>   start;
      line_geometry(l, axis, direction, start);
      for (unsigned int i = 0; i < p_list.size(); ++i)
        for (unsigned int d = 0; d < dim; ++d)
          if (d != axis)
            {
              double derivative = direction * factor_derivative(start[d]);
              for (unsigned int j = 0; j < dim; ++j)
                if (j != axis && j != d)
                  derivative *= factor(start[j], p_list[i][j]);
              grads[l][i][axis][d] = derivative;
            }
    }
  return grads;
}



template <int dim, int spacedim>
std::string
FE_NedelecHypercube<dim, spacedim>::get_name() const
{
  std::ostringstream namebuf;
  namebuf << "FE_NedelecHypercube<" << Utilities::dim_string(dim, spacedim)
          << ">(" << this->degree - 1 << ")";
  return namebuf.str();
}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, dim>>
FE_NedelecHypercube<dim, spacedim>::clone() const
{
  return std::make_unique<FE_NedelecHypercube<dim, spacedim>>(*this);
}



// explicit instantiations
template class FE_NedelecHypercube<2>;
template class FE_NedelecHypercube<3>;

DEAL_II_NAMESPACE_CLOSE
