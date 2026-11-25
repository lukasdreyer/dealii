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

#ifndef dealii_fe_nedelec_wedge_h
#define dealii_fe_nedelec_wedge_h

#include <deal.II/base/config.h>

#include <deal.II/fe/fe_nedelec_lowest_order.h>

DEAL_II_NAMESPACE_OPEN

/**
 * @addtogroup fe
 * @{
 */

/**
 * The lowest-order H(curl)-conforming Nédélec element on wedges (prisms),
 * with one DoF per line (nine in total), following Zaglmayr's thesis
 * (Section 5.2.5). With the barycentric coordinates lambda of the triangle
 * and mu_0 = 1 - z, mu_1 = z, the shape function of a horizontal line from
 * vertex p to vertex q (both in the triangle at level k) is
 * (lambda_p grad lambda_q - lambda_q grad lambda_p) mu_k, and that of a
 * vertical line from p to q is lambda_p (z_q - z_p) e_z. The space is the
 * tensor product of the Whitney space of the triangle with P1 in z for the
 * horizontal components and of P1 on the triangle with P0 in z for the
 * vertical component; its traces agree with FE_NedelecSimplex on the
 * triangular faces and with the lowest-order element on quadrilaterals on
 * the quadrilateral faces.
 *
 * See FE_NedelecLowestOrderBase for the orientation of the DoFs; as for
 * FE_NedelecSimplex, the hanging node constraints of
 * DoFTools::make_hanging_node_constraints() and the functions in
 * dofs/edge_element_tools.h apply.
 */
template <int dim, int spacedim = dim>
class FE_NedelecWedge : public FE_NedelecLowestOrderBase<dim, spacedim>
{
public:
  static_assert(dim == 3, "FE_NedelecWedge is only implemented in 3d.");

  /**
   * Constructor; only @p order = 0 (one DoF per line) is implemented.
   */
  FE_NedelecWedge(const unsigned int order);

  virtual std::string
  get_name() const override;

  virtual std::unique_ptr<FiniteElement<dim, dim>>
  clone() const override;

protected:
  virtual std::vector<std::vector<Tensor<1, dim>>>
  get_shape_values(const std::vector<Point<dim>> &p_list) const override;

  virtual std::vector<std::vector<Tensor<2, dim>>>
  get_shape_grads(const std::vector<Point<dim>> &p_list) const override;

  virtual std::vector<std::vector<Tensor<3, dim>>>
  get_shape_hessians(const std::vector<Point<dim>> &p_list) const override;
};

/** @} */

DEAL_II_NAMESPACE_CLOSE

#endif
