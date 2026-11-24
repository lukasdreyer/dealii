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

#ifndef dealii_fe_nedelec_hypercube_h
#define dealii_fe_nedelec_hypercube_h

#include <deal.II/base/config.h>

#include <deal.II/fe/fe_nedelec_lowest_order.h>

DEAL_II_NAMESPACE_OPEN

/**
 * @addtogroup fe
 * @{
 */

/**
 * The lowest-order H(curl)-conforming Nédélec element on quadrilaterals and
 * hexahedra, with one DoF per line. The shape function of a line from vertex
 * p to vertex q of the reference cell [0,1]^dim, which runs along the axis
 * k, is w(x) (q_k - p_k) e_k, where w is the product over the other axes j of
 * x_j (if p_j = 1) or 1 - x_j (if p_j = 0). This is the same space as
 * FE_Nedelec(0); the difference is the orientation of the DoFs (see
 * FE_NedelecLowestOrderBase), which is the same as for FE_NedelecSimplex,
 * FE_NedelecWedge and FE_NedelecPyramid, so that the elements can be mixed on
 * hybrid meshes and use the hanging node constraints of
 * DoFTools::make_hanging_node_constraints() and the functions in
 * dofs/edge_element_tools.h.
 */
template <int dim, int spacedim = dim>
class FE_NedelecHypercube : public FE_NedelecLowestOrderBase<dim, spacedim>
{
public:
  /**
   * Constructor; only @p order = 0 (one DoF per line) is implemented.
   */
  FE_NedelecHypercube(const unsigned int order);

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
