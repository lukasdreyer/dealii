// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2018 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


#include <deal.II/fe/fe_nedelec_simplex_linear.h>
#include <deal.II/fe/fe_tools.h>

#include <memory>

DEAL_II_NAMESPACE_OPEN

// Constructor:
template <int dim, int spacedim>
FE_NedelecSimplexLinear<dim, spacedim>::FE_NedelecSimplexLinear()
  : FiniteElement<dim, spacedim>(
      FiniteElementData<dim>(std::vector<unsigned int>{0,1,0,0},
                             dim, 1,
                             FiniteElementData<dim>::Hcurl),
      std::vector<bool>(1, false), //restriction_is_additive (true in quad implementation)
      std::vector<ComponentMask>(1, ComponentMask(std::vector<bool>(dim, true))))
{
  Assert(dim >= 2, ExcImpossibleInDim(dim));

//  this->mapping_kind = mapping_nedelec;
  // Set up the table converting components to base components. Since we have
  // only one base element, everything remains zero except the component in the
  // base, which is the component itself.
  for (unsigned int comp = 0; comp < this->n_components(); ++comp)
  {
    this->component_to_base_table[comp].first.second = comp;
  }

}


template <int dim, int spacedim>
UpdateFlags
FE_NedelecSimplexLinear<dim, spacedim>::requires_update_flags(
  const UpdateFlags flags) const
{
  UpdateFlags out = update_default;

  if (flags & update_values)
    out |= update_values | update_covariant_transformation;

  if (flags & update_gradients)
    out |= update_gradients | update_values |
           update_jacobian_pushed_forward_grads |
           update_covariant_transformation;

  return out;
}



template <int dim, int spacedim>
std::string
FE_NedelecSimplexLinear<dim, spacedim>::get_name() const
{
  // note that the FETools::get_fe_by_name function depends on the particular
  // format of the string this function returns, so they have to be kept in sync
  std::ostringstream namebuf;
  namebuf << "FE_NedelecSimplexLinear<" << Utilities::dim_string(dim, spacedim) << ">()";

  return namebuf.str();
}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, dim>>
FE_NedelecSimplexLinear<dim, spacedim>::clone() const
{
  return std::make_unique<FE_NedelecSimplexLinear<dim, spacedim>>(*this);
}

// explicit instantiations
#include "fe/fe_nedelec_simplex_linear.inst"

DEAL_II_NAMESPACE_CLOSE
