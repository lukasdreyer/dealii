// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2018 - 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

#ifndef dealii_fe_nedelec_simplex_h
#define dealii_fe_nedelec_simplex_h

#include <deal.II/base/config.h>

#include <deal.II/fe/fe_nedelec_lowest_order.h>

DEAL_II_NAMESPACE_OPEN

/**
 * @addtogroup fe
 * @{
 */

/**
 * The lowest-order H(curl)-conforming Nédélec element on triangles and
 * tetrahedra (Whitney element): the shape function of the line from vertex p
 * to vertex q is lambda_p grad lambda_q - lambda_q grad lambda_p with the
 * barycentric coordinates lambda. See FE_NedelecLowestOrderBase for the
 * orientation of the DoFs.
 */
template <int dim, int spacedim = dim>
class FE_NedelecSimplex : public FE_NedelecLowestOrderBase<dim, spacedim>
{
public:
  /**
   * Constructor; only @p order = 0 (one DoF per line) is implemented.
   */
  FE_NedelecSimplex(const unsigned int order);

  virtual std::string
  get_name() const override;

  virtual std::unique_ptr<FiniteElement<dim, dim>>
  clone() const override;

protected:
  virtual std::vector<std::vector<Tensor<1, dim>>>
  get_shape_values(const std::vector<Point<dim>> &p_list) const override;

  virtual std::vector<std::vector<Tensor<2, dim>>>
  get_shape_grads(const std::vector<Point<dim>> &p_list) const override;
};

/** @} */

DEAL_II_NAMESPACE_CLOSE

#endif
