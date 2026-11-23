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

#ifndef dealii_fe_nedelec_pyramid_h
#define dealii_fe_nedelec_pyramid_h

#include <deal.II/base/config.h>

#include <deal.II/fe/fe_nedelec_lowest_order.h>

DEAL_II_NAMESPACE_OPEN

/**
 * @addtogroup fe
 * @{
 */

/**
 * The lowest-order H(curl)-conforming Nédélec element on pyramids, with one
 * DoF per line (eight in total). The shape functions are rational: they are
 * the Whitney 1-forms on pyramids of Gradinaru and Hiptmair (ETNA 8, 1999),
 * which span the same space as the lowest-order elements of Nigam and
 * Phillips (IMA J. Numer. Anal. 32, 2012) and Bergot and Duruflé (J. Comput.
 * Phys. 232, 2013).
 *
 * On deal.II's reference pyramid (base [-1,1]^2 x {0}, apex (0,0,1)), let
 * N_i be the rational vertex functions of FE_PyramidP(1),
 * N_i = (1 - z + a_i x)(1 - z + b_i y) / (4 (1 - z)) for the base vertex
 * (a_i, b_i, 0) and N_4 = z, and theta_pq = N_p grad N_q - N_q grad N_p. The
 * shape function of a line from a base vertex to the apex is theta_pq. The
 * shape function of a base line is theta_pq plus multiples of theta_03 and
 * theta_12 along the diagonals of the base, with the weights +-1/2 given by
 * the line integrals along the diagonals of the lowest-order function on the
 * quadrilateral, so that the tangential trace on the base is that function.
 *
 * The tangential traces agree with FE_NedelecSimplex on the triangular faces
 * and with the lowest-order element on quadrilaterals on the base, the
 * space contains a + b x x and the gradients of the space of FE_PyramidP(1)
 * (span{1, x, y, z, xy/(1-z)}). The functions are bounded but not smooth at
 * the apex; QGaussPyramid does not evaluate them there.
 *
 * See FE_NedelecLowestOrderBase for the orientation of the DoFs; as for
 * FE_NedelecSimplex, the hanging node constraints of
 * DoFTools::make_hanging_node_constraints() and the functions in
 * dofs/edge_element_tools.h apply.
 */
template <int dim, int spacedim = dim>
class FE_NedelecPyramid : public FE_NedelecLowestOrderBase<dim, spacedim>
{
public:
  static_assert(dim == 3, "FE_NedelecPyramid is only implemented in 3d.");

  /**
   * Constructor; only @p order = 0 (one DoF per line) is implemented.
   */
  FE_NedelecPyramid(const unsigned int order);

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
