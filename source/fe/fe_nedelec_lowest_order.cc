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


#include "deal.II/base/exception_macros.h"
#include "deal.II/base/exceptions.h"
#include "deal.II/base/types.h"
#include <deal.II/base/qprojector.h>

#include "deal.II/fe/fe_update_flags.h"
#include <deal.II/fe/fe_nedelec_lowest_order.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include "deal.II/grid/reference_cell.h"

#include <memory>

DEAL_II_NAMESPACE_OPEN

template <int dim, int spacedim>
FE_NedelecLowestOrderBase<dim, spacedim>::FE_NedelecLowestOrderBase(
  const ReferenceCell &reference_cell,
  const unsigned int   order)
  : FiniteElement<dim, dim>(
      FiniteElementData<dim>(get_dpo_vector(order),
                             reference_cell,
                             dim,
                             order + 1,
                             FiniteElementData<dim>::Hcurl),
      std::vector<bool>(reference_cell.n_lines(), true),
      std::vector<ComponentMask>(reference_cell.n_lines(),
                                 ComponentMask(std::vector<bool>(dim, true))))
{
  Assert(dim >= 2, ExcImpossibleInDim(dim));

  this->mapping_kind = mapping_nedelec;
  // Set up the table converting components to base components. Since we have
  // only one base element, everything remains zero except the component in the
  // base, which is the component itself.
  for (unsigned int comp = 0; comp < this->n_components(); ++comp)
    {
      this->component_to_base_table[comp].first.second = comp;
    }
}



template <int dim, int spacedim>
std::unique_ptr<typename dealii::FiniteElement<dim, spacedim>::InternalDataBase>
FE_NedelecLowestOrderBase<dim, spacedim>::get_data(
  const UpdateFlags update_flags,
  const Mapping<dim, spacedim> & /*mapping*/,
  const Quadrature<dim> &quadrature,
  dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                     spacedim>
    & /*output_data*/) const
{
  const auto &p_list = quadrature.get_points();
//  std::cout<<"enter get_data with quadrature with points:"<<std::endl;
//  for (const auto & p:p_list)
//    std::cout<<p<<std::endl;

  Assert(!(update_flags & (update_hessians | update_3rd_derivatives)),
         ExcMessage("FE_NedelecSimplex does not compute second or third "
                    "derivatives."));
  const UpdateFlags flags = requires_update_flags(update_flags);
  auto shape_values = (flags & update_values) ? get_shape_values(p_list): std::vector<std::vector<Tensor<1,dim>>>();
  auto shape_gradients = (flags & update_gradients) ? get_shape_grads(p_list): std::vector<std::vector<Tensor<2,dim>>>();

    std::unique_ptr<
    typename dealii::FiniteElement<dim, spacedim>::InternalDataBase>
    data_ptr = std::make_unique<InternalData>(std::move(shape_values), std::move(shape_gradients));
  data_ptr->update_each = requires_update_flags(update_flags);
  return data_ptr;
}

template <int dim, int spacedim>
std::vector<int>
FE_NedelecLowestOrderBase<dim, spacedim>::compute_edge_sign(
  const typename Triangulation<dim, dim>::cell_iterator &cell) const
{
  // The reference shape function of line l is directed from local vertex
  // line_to_cell_vertices(l, 0) to line_to_cell_vertices(l, 1). The global
  // DoF of a line refers to the direction of the line object itself (from
  // line->vertex(0) to line->vertex(1)). Both agree if the cell sees the line
  // in the default orientation. Lines inherit the direction of their parent
  // line during refinement, so the child DoFs of a hanging line have the same
  // direction as the parent DoF.
  std::vector<int> edge_sign(cell->n_lines());
  for (unsigned int l = 0; l < cell->n_lines(); ++l)
    edge_sign[l] =
      (cell->line_orientation(l) == numbers::default_geometric_orientation) ?
        1 :
        -1;
  return edge_sign;
}



template <int dim, int spacedim>
void
FE_NedelecLowestOrderBase<dim, spacedim>::fill_fe_values(
  const typename Triangulation<dim, dim>::cell_iterator &cell,
  const CellSimilarity::Similarity /*cell_similarity*/, //is not used as edge sign might differ
  const Quadrature<dim>                              &quadrature,
  const Mapping<dim, dim>                            &mapping,
  const typename Mapping<dim, dim>::InternalDataBase &mapping_internal,
  const internal::FEValuesImplementation::MappingRelatedData<dim, dim>
                                                           &mapping_data,
  const typename FiniteElement<dim, dim>::InternalDataBase &fe_internal,
  dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim, dim>
    &data) const
{
  // Convert to the correct internal data class for this FE class.
  Assert(dynamic_cast<const InternalData *>(&fe_internal) != nullptr,
         ExcInternalError());
  const InternalData &fe_data = static_cast<const InternalData &>(fe_internal);

  // Now update the edge-based DoFs, which depend on the cell.
  // This will fill in the missing items in the InternalData
  // (fe_internal/fe_data) which was not filled in by get_data.
  //  std::cout<<"fill fe_values for cell"<<cell->id()<<std::endl;
  //  std::cout<<"on quadrature points"<<std::endl;
  //  for(const auto  &point: quadrature.get_points()){
  //   std::cout<<point<<std::endl;
  //  }
   const auto edge_sign = compute_edge_sign(cell);
  //  std::cout<<"edge_sign"<<std::endl;
  //  for(const auto &value: edge_sign)
  //    std::cout<<value<<std::endl;

  const UpdateFlags  flags(fe_data.update_each);
  const unsigned int n_q_points = quadrature.size();

  if (flags & update_values)
    {
      // Now have all shape_values stored on the reference cell.
      // Must now transform to the physical cell.
      std::vector<Tensor<1, dim>> transformed_shape_values(n_q_points);
      for (unsigned int dof = 0; dof < this->n_dofs_per_cell(); ++dof)
        {
          const unsigned int first =
            data.shape_function_to_row_table[dof * this->n_components() +
                                             this->get_nonzero_components(dof)
                                               .first_selected_component()];

          // std::cout<<"shape values on reference for dof "<<dof<<std::endl;
          // for (const auto &value: fe_data.shape_values[dof])
          //   std::cout<<value<<std::endl;

          mapping.transform(make_array_view(fe_data.shape_values[dof]),
                            mapping_covariant,
                            mapping_internal,
                            make_array_view(transformed_shape_values));

          //  std::cout<<"transformed shape values"<<std::endl;
          //  for (const auto &value: transformed_shape_values)
          //    std::cout<<value<<std::endl;

          // std::cout<<"edge_sign: "<<edge_sign[dof]<<std::endl;

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int d = 0; d < dim; ++d)
                {
                  data.shape_values(first + d, q) =
                    transformed_shape_values[q][d] * edge_sign[dof];
//                  std::cout<<"filled value of dof "<<dof<<", q_point "<<q<<" and d "<<d<<"with "<< data.shape_values[first + d][q] <<std::endl;
                }
            }
        }
    }

  if (flags & update_gradients)
    {
      // Now have all shape_grads stored on the reference cell.
      // Must now transform to the physical cell.
      std::vector<Tensor<2, dim>> input(n_q_points);
      std::vector<Tensor<2, dim>> transformed_shape_grads(n_q_points);
      for (unsigned int dof = 0; dof < this->n_dofs_per_cell(); ++dof)
        {
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              input[q] = fe_data.shape_grads[dof][q];
            }
          mapping.transform(make_array_view(input),
                            mapping_covariant_gradient,
                            mapping_internal,
                            make_array_view(transformed_shape_grads));

          const unsigned int first =
            data.shape_function_to_row_table[dof * this->n_components() +
                                             this->get_nonzero_components(dof)
                                               .first_selected_component()];

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int d1 = 0; d1 < dim; ++d1)
                {
                  for (unsigned int d2 = 0; d2 < dim; ++d2)
                    {
                      transformed_shape_grads[q][d1] -=
                        data.shape_values(first + d2, q) *
                        mapping_data.jacobian_pushed_forward_grads[q][d2][d1];
                    }
                }
            }

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int d = 0; d < dim; ++d)
                {
                  data.shape_gradients[first + d][q] =
                    transformed_shape_grads[q][d] * edge_sign[dof];
//                  std::cout<<"filled grad of dof "<<dof<<", q_point "<<q<<" and d "<<d<<"with "<< data.shape_gradients[first + d][q] <<std::endl;
                }
            }
        }
    }
}



template <int dim, int spacedim>
void
FE_NedelecLowestOrderBase<dim, spacedim>::fill_fe_face_values(
  const typename Triangulation<dim, dim>::cell_iterator &cell,
  const unsigned int                                     face_no,
  const hp::QCollection<dim - 1>                        &quadrature,
  const Mapping<dim, dim>                               &mapping,
  const typename Mapping<dim, dim>::InternalDataBase    &mapping_internal,
  const internal::FEValuesImplementation::MappingRelatedData<dim, dim>
                                                           &mapping_data,
  const typename FiniteElement<dim, dim>::InternalDataBase &fe_internal,
  dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim, dim>
    &data) const
{
  // one quadrature for all faces, or one per face (e.g. triangles and
  // quadrilaterals on wedges)
  Assert(quadrature.size() == 1 || quadrature.size() == cell->n_faces(),
         ExcDimensionMismatch(quadrature.size(), cell->n_faces()));

  // Note for future improvement:
  // We don't have the full quadrature - should use QProjector to create the 2d
  // quadrature.
  //
  // For now I am effectively generating all of the shape function vals/grads,
  // etc. On all quad points on all faces and then only using them for one face.
  // This is obviously inefficient. I should cache the cell number and cache
  // all of the shape_values/gradients etc and then reuse them for each face.

  // convert data object to internal
  // data for this class. fails with
  // an exception if that is not
  // possible
  Assert(dynamic_cast<const InternalData *>(&fe_internal) != nullptr,
         ExcInternalError());
  const InternalData &fe_data = static_cast<const InternalData &>(fe_internal);

  // Now update the edge-based DoFs, which depend on the cell.
  // This will fill in the missing items in the InternalData
  // (fe_internal/fe_data) which was not filled in by get_data.
//  std::cout<<"fill fe_face_values for cell"<<cell->id()<<std::endl;
  const auto edge_sign = compute_edge_sign(cell);
  //std::cout<<"edge_sign"<<std::endl;
  //for(const auto &value: edge_sign)
  //  std::cout<<value<<std::endl;

  const UpdateFlags  flags(fe_data.update_each);
  const unsigned int n_q_points =
    quadrature[quadrature.size() == 1 ? 0 : face_no].size();
  const auto offset =
    QProjector<dim>::DataSetDescriptor::face(this->reference_cell(),
                                             face_no,
                                             cell->combined_face_orientation(
                                               face_no),
                                             quadrature);

  if (flags & update_values)
    {
      // Now have all shape_values stored on the reference cell.
      // Must now transform to the physical cell.
      std::vector<Tensor<1, dim>> transformed_shape_values(n_q_points);
      for (unsigned int dof = 0; dof < this->n_dofs_per_cell(); ++dof)
        {
          //std::cout<<"shape values on reference for face"<<face_no<<"and dof "<<dof<<std::endl;
          //for (unsigned int index=0; index< n_q_points; index++){
          //  std::cout<<fe_data.shape_values[dof][index+offset]<<std::endl;
          //}

          mapping.transform(make_array_view(fe_data.shape_values[dof],
                                            offset,
                                            n_q_points),
                            mapping_covariant,
                            mapping_internal,
                            make_array_view(transformed_shape_values));

          // std::cout<<"transformed shape values"<<std::endl;
          // for (const auto &value: transformed_shape_values)
          //   std::cout<<value<<std::endl;

          const unsigned int first =
            data.shape_function_to_row_table[dof * this->n_components() +
                                             this->get_nonzero_components(dof)
                                               .first_selected_component()];

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int d = 0; d < dim; ++d)
                {
                  data.shape_values(first + d, q) =
                    transformed_shape_values[q][d]*edge_sign[dof];
//                  std::cout<<"filled value of dof "<<dof<<", q_point "<<q<<" and d "<<d<<"with "<< data.shape_values[first + d][q] <<std::endl;
                }
            }
        }
    }
  if (flags & update_gradients)
    {
      // Now have all shape_grads stored on the reference cell.
      // Must now transform to the physical cell.
      std::vector<Tensor<2, dim>> input(n_q_points);
      std::vector<Tensor<2, dim>> transformed_shape_grads(n_q_points);
      for (unsigned int dof = 0; dof < this->n_dofs_per_cell(); ++dof)
        {
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              input[q] = fe_data.shape_grads[dof][offset + q];
            }
          mapping.transform(input,
                            mapping_covariant_gradient,
                            mapping_internal,
                            make_array_view(transformed_shape_grads));

          const unsigned int first =
            data.shape_function_to_row_table[dof * this->n_components() +
                                             this->get_nonzero_components(dof)
                                               .first_selected_component()];

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int d1 = 0; d1 < dim; ++d1)
                {
                  for (unsigned int d2 = 0; d2 < dim; ++d2)
                    {
                      transformed_shape_grads[q][d1] -=
                        data.shape_values(first + d2, q) *
                        mapping_data.jacobian_pushed_forward_grads[q][d2][d1];
                    }
                }
            }

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int d = 0; d < dim; ++d)
                {
                  data.shape_gradients[first + d][q] =
                    transformed_shape_grads[q][d]*edge_sign[dof];
//                  std::cout<<"filled gradient of dof "<<dof<<", q_point "<<q<<" and d "<<d<<"with "<< data.shape_gradients[first + d][q] <<std::endl;
                }
            }
        }
    }
}



  /**
   * Not implemented.
   */
  template <int dim, int spacedim>
  void
  FE_NedelecLowestOrderBase<dim, spacedim>::fill_fe_subface_values(
    const typename Triangulation<dim, dim>::cell_iterator &/*cell*/,
    const unsigned int                                     /*face_no*/,
    const unsigned int                                     /*sub_no*/,
    const Quadrature<dim - 1>                             &/*quadrature*/,
    const Mapping<dim, dim>                               &/*mapping*/,
    const typename Mapping<dim, dim>::InternalDataBase    &/*mapping_internal*/,
    const internal::FEValuesImplementation::MappingRelatedData<dim, dim>
                                                             &/*mapping_data*/,
    const typename FiniteElement<dim, dim>::InternalDataBase &/*fedata*/,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim, dim>
      &/*data*/) const

{
  DEAL_II_NOT_IMPLEMENTED();
}


template <int dim, int spacedim>
UpdateFlags
FE_NedelecLowestOrderBase<dim, spacedim>::requires_update_flags(
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
std::vector<unsigned int>
FE_NedelecLowestOrderBase<dim, spacedim>::get_dpo_vector(const unsigned int degree)
{
  // internal function to return a vector of "dofs per object"
  // where the objects inside the vector refer to:
  // 0 = vertex
  // 1 = edge
  // 2 = face (which is a cell in 2d)
  // 3 = cell
  std::vector<unsigned int> dpo;
  Assert(degree==0, ExcMessage(
           "Not yet implemented for degree>0!"));
  dpo.push_back(0);
  dpo.push_back(1);
  if (dim > 1)
    dpo.push_back(0);
  if (dim > 2)
    dpo.push_back(0);

  return dpo;
}


// explicit instantiations
#include "fe/fe_nedelec_lowest_order.inst"

DEAL_II_NAMESPACE_CLOSE
