// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2018 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_fe_nedelec_simplex_linear_h
#define dealii_fe_nedelec_simplex_linear_h

#include <deal.II/base/config.h>

//#include <deal.II/base/derivative_form.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

DEAL_II_NAMESPACE_OPEN

/**
 * @addtogroup fe
 * @{
 */

/**
 * This class represents an implementation of the linear
 * H<sup>curl</sup>-conforming N&eacute;d&eacute;lec element for simplices
 */
template <int dim, int spacedim = dim>
class FE_NedelecSimplexLinear : public FiniteElement<dim, dim>
{
public:
  static_assert(dim == spacedim,
                "FE_NedelecSimplexLinear is only implemented for dim==spacedim!");

  /**
   * Constructor for the linear simplex Nedelec element
   */
  FE_NedelecSimplexLinear();

  virtual UpdateFlags
  requires_update_flags(const UpdateFlags update_flags) const override;

  virtual std::string
  get_name() const override;

  virtual std::unique_ptr<FiniteElement<dim, dim>>
  clone() const override;

};



/** @} */

DEAL_II_NAMESPACE_CLOSE

#endif
