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
#include <deal.II/fe/fe_nedelec_simplex.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include "deal.II/grid/reference_cell.h"

#include <memory>

DEAL_II_NAMESPACE_OPEN

// Constructor:
template <int dim, int spacedim>
FE_NedelecSimplex<dim, spacedim>::FE_NedelecSimplex(const unsigned int order)
  : FiniteElement<dim, dim>(
      FiniteElementData<dim>(get_dpo_vector(order),ReferenceCells::get_simplex<dim>(),
                             dim,
                             order + 1,
                             FiniteElementData<dim>::Hcurl),
      std::vector<bool>(compute_num_dofs(order), true),
      std::vector<ComponentMask>(compute_num_dofs(order),
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


  switch (dim)
    {
      case 1:
        {
          this->interface_constraints.reinit(0, 0);
          break;
        }
      case 2:
        {
          this->interface_constraints.reinit(2, 1);
          this->interface_constraints(0, 0)=0.5;
          this->interface_constraints(1, 0)=0.5;//Does this need to be -0.5?
          break;
        }
      case 3:
        {
          this->interface_constraints.reinit(12, 3);
          for (unsigned i=0; i<this->interface_constraints.m(); i++){
            for (unsigned j=0; j<this->interface_constraints.n(); j++){
              this->interface_constraints(i,j)=0; //TODO, which ones are 1/4?
            }
          }

          for (auto it = this->interface_constraints.begin(); it!= this->interface_constraints.end(); it++){
            *it =0;
          }
          this->interface_constraints(0,0)=0.5; //TODO, which ones are 1/4?
          this->interface_constraints(1,1)=0.5;
          this->interface_constraints(2,2)=0.5;
          this->interface_constraints(3,0)=0.5;
          this->interface_constraints(4,1)=0.5;
          this->interface_constraints(5,2)=0.5;
          this->interface_constraints(6,0)=0.5;
          this->interface_constraints(7,1)=0.5;
          this->interface_constraints(8,2)=0.5;
          this->interface_constraints(9,0)=0.5;
          this->interface_constraints(10,1)=0.5;
          this->interface_constraints(11,2)=0.5;


          break;
        }
      default:
        {
          break;
        }

    }
}


#if 0
template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_NedelecSimplex<dim, spacedim>::hp_vertex_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other) const
{
  return {};
}



template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_NedelecSimplex<dim, spacedim>::hp_line_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other) const
{
  AssertDimension(dim, 2);

  if (const FE_NedelecSimplex<dim, spacedim> *fe_p_other =
        dynamic_cast<const FE_NedelecSimplex<dim, spacedim> *>(&fe_other))
    {
      // dofs are located along lines, so two dofs are identical if they are
      // located at identical positions.
      // Therefore, read the points in unit_support_points for the
      // first coordinate direction. For FE_SimplexP, they are currently
      // hard-coded and we iterate over points on the first line which begin
      // after the 3 vertex points in the complete list of unit support points

      std::vector<std::pair<unsigned int, unsigned int>> identities;

      for (unsigned int i = 0; i < this->degree - 1; ++i)
        for (unsigned int j = 0; j < fe_p_other->degree - 1; ++j)
          if (std::fabs(this->unit_support_points[i + 3][0] -
                        fe_p_other->unit_support_points[i + 3][0]) < 1e-14)
            identities.emplace_back(i, j);
          else
            {
              // If nodes are not located in the same place, we have to
              // interpolate. This is then not handled through the
              // current function, but via interpolation matrices that
              // result in constraints, rather than identities. Since
              // that happens in a different function, there is nothing
              // for us to do here.
            }

      return identities;
    }
  else if (dynamic_cast<const FE_Nothing<dim> *>(&fe_other) != nullptr)
    {
      // The FE_Nothing has no degrees of freedom, so there are no
      // equivalencies to be recorded. (If the FE_Nothing is dominating,
      // then this will also leads to constraints, but we are not concerned
      // with this here.)
      return {};
    }
  else if (fe_other.n_unique_faces() == 1 && fe_other.n_dofs_per_face(0) == 0)
    {
      // if the other element has no elements on faces at all,
      // then it would be impossible to enforce any kind of
      // continuity even if we knew exactly what kind of element
      // we have -- simply because the other element declares
      // that it is discontinuous because it has no DoFs on
      // its faces. in that case, just state that we have no
      // constraints to declare
      return {};
    }
  else
    {
      DEAL_II_NOT_IMPLEMENTED();
      return {};
    }
}


#endif

template <int dim, int spacedim>
std::vector<std::vector<Tensor<1, dim>>>
FE_NedelecSimplex<dim, spacedim>::get_shape_values(
  const std::vector<Point<dim>> &p_list)
  const
{
  // Useful quantities:
  const unsigned int degree(this->degree - 1); // Note: FE holds input degree+1
  Assert(degree == 0,ExcMessage("Not yet implemented for higher order!"));

  const unsigned int n_q_points = p_list.size();

  std::vector<std::vector<Tensor<1, dim>>> shape_values(this->n_dofs_per_cell(),
                             std::vector<Tensor<1, dim>>(n_q_points));

    switch (dim) {
      case 2:
        for (unsigned int iquad = 0; iquad < n_q_points; iquad++){
          double x=p_list[iquad][0];
          double y=p_list[iquad][1];
//          std::cout<<"evaluate for x= "<<x<<" and y= "<<y<<std::endl;
          shape_values[0][iquad][0]=1-y;
          shape_values[0][iquad][1]=x;

          shape_values[1][iquad][0]=-y;
          shape_values[1][iquad][1]=x;
          
          shape_values[2][iquad][0]=-y;
          shape_values[2][iquad][1]=x-1;

//wrong?
//          shape_values[2][iquad][0]=y;
//          shape_values[2][iquad][1]=1-x;
        }
        break;
      case 3:
        for (unsigned int iquad = 0; iquad < n_q_points; iquad++){
          double x=p_list[iquad][0];
          double y=p_list[iquad][1];
          double z=p_list[iquad][2];
          // std::cout<<"evaluate shape functions at quadrature point"<<x<<" "<<y<<" "<<" "<<z<<std::endl;

          shape_values[0][iquad][0]=1-y-z;
          shape_values[0][iquad][1]=x;
          shape_values[0][iquad][2]=x;

          shape_values[1][iquad][0]=-y;
          shape_values[1][iquad][1]=x;
          shape_values[1][iquad][2]=0;

          shape_values[2][iquad][0]=-y;
          shape_values[2][iquad][1]=-1+x+z;
          shape_values[2][iquad][2]=-y;

          shape_values[3][iquad][0]=z;
          shape_values[3][iquad][1]=z;
          shape_values[3][iquad][2]=1-x-y;

          shape_values[4][iquad][0]=-z;
          shape_values[4][iquad][1]=0;
          shape_values[4][iquad][2]=x;

          shape_values[5][iquad][0]=0;
          shape_values[5][iquad][1]=-z;
          shape_values[5][iquad][2]=y;
          // for(unsigned idof=0; idof < 6; idof++){
          //   std::cout<<"shape_values["<<idof<<"]["<<iquad<<"]="<<shape_values[idof][iquad][0]<<" "<<shape_values[idof][iquad][1]<<" "<<shape_values[idof][iquad][2]<<" "<<std::endl;
          // }

        }      
        break;
      default:
        DEAL_II_NOT_IMPLEMENTED();
    
    }
    return shape_values;
}


template <int dim, int spacedim>
std::vector<std::vector<Tensor<2, dim>>>
FE_NedelecSimplex<dim, spacedim>::get_shape_grads(
  const std::vector<Point<dim>> &p_list)
  const
{
  const unsigned int degree(this->degree - 1); // Note: FE holds input degree+1
  Assert(degree == 0,ExcMessage("Not yet implemented for higher order!"));

  const unsigned int n_q_points = p_list.size();

  std::vector<std::vector<Tensor<2, dim>>> shape_grads(this->n_dofs_per_cell(),
                             std::vector<Tensor<2, dim>>(n_q_points));

    switch (dim) {
      case 2:
        for (unsigned int iquad = 0; iquad < n_q_points; iquad++){
          shape_grads[0][iquad][0][0]=0;
          shape_grads[0][iquad][0][1]=-1;

          shape_grads[0][iquad][1][0]=1;
          shape_grads[0][iquad][1][1]=0;


          shape_grads[1][iquad][0][0]=0;
          shape_grads[1][iquad][0][1]=-1;

          shape_grads[1][iquad][1][0]=1;
          shape_grads[1][iquad][1][1]=0;


          shape_grads[2][iquad][0][0]=0;
          shape_grads[2][iquad][0][1]=-1;

          shape_grads[2][iquad][1][0]=1;
          shape_grads[2][iquad][1][1]=0;
        }
        break;
      case 3:
        for (unsigned int iquad = 0; iquad < n_q_points; iquad++){
          //f_x = 1-y-z, f_y = x, f_z = x
          shape_grads[0][iquad][0][0]=0;
          shape_grads[0][iquad][0][1]=-1;
          shape_grads[0][iquad][0][2]=-1;
 
          shape_grads[0][iquad][1][0]=1;
          shape_grads[0][iquad][1][1]=0;
          shape_grads[0][iquad][1][2]=0;
 
          shape_grads[0][iquad][2][0]=1;
          shape_grads[0][iquad][2][1]=0;
          shape_grads[0][iquad][2][2]=0;

          
          //f_x = -y, f_y = x, f_z = 0
          shape_grads[1][iquad][0][0]=0;
          shape_grads[1][iquad][0][1]=-1;
          shape_grads[1][iquad][0][2]=0;
 
          shape_grads[1][iquad][1][0]=1;
          shape_grads[1][iquad][1][1]=0;
          shape_grads[1][iquad][1][2]=0;
 
          shape_grads[1][iquad][2][0]=0;
          shape_grads[1][iquad][2][1]=0;
          shape_grads[1][iquad][2][2]=0;

          
          //f_x = -y, f_y = -1 + x +z, f_z = -y
          shape_grads[2][iquad][0][0]=0;
          shape_grads[2][iquad][0][1]=-1;
          shape_grads[2][iquad][0][2]=0;
 
          shape_grads[2][iquad][1][0]=1;
          shape_grads[2][iquad][1][1]=0;
          shape_grads[2][iquad][1][2]=1;
 
          shape_grads[2][iquad][2][0]=0;
          shape_grads[2][iquad][2][1]=-1;
          shape_grads[2][iquad][2][2]=0;


          //f_x = z, f_y = z, f_z = 1 - x - y
          shape_grads[3][iquad][0][0]=0;
          shape_grads[3][iquad][0][1]=0;
          shape_grads[3][iquad][0][2]=1;
 
          shape_grads[3][iquad][1][0]=0;
          shape_grads[3][iquad][1][1]=0;
          shape_grads[3][iquad][1][2]=1;
 
          shape_grads[3][iquad][2][0]=-1;
          shape_grads[3][iquad][2][1]=-1;
          shape_grads[3][iquad][2][2]=0;


          //f_x = -z, f_y = 0, f_z = x
          shape_grads[4][iquad][0][0]=0;
          shape_grads[4][iquad][0][1]=0;
          shape_grads[4][iquad][0][2]=-1;
 
          shape_grads[4][iquad][1][0]=0;
          shape_grads[4][iquad][1][1]=0;
          shape_grads[4][iquad][1][2]=0;
 
          shape_grads[4][iquad][2][0]=1;
          shape_grads[4][iquad][2][1]=0;
          shape_grads[4][iquad][2][2]=0;


          //f_x = 0, f_y = -z, f_z = y
          shape_grads[5][iquad][0][0]=0;
          shape_grads[5][iquad][0][1]=0;
          shape_grads[5][iquad][0][2]=0;
 
          shape_grads[5][iquad][1][0]=0;
          shape_grads[5][iquad][1][1]=0;
          shape_grads[5][iquad][1][2]=-1;
 
          shape_grads[5][iquad][2][0]=0;
          shape_grads[5][iquad][2][1]=1;
          shape_grads[5][iquad][2][2]=0;

        }      
        break;
      default:
        DEAL_II_NOT_IMPLEMENTED();
    
    }
    return shape_grads;
  }




template <int dim, int spacedim>
std::unique_ptr<typename dealii::FiniteElement<dim, spacedim>::InternalDataBase>
FE_NedelecSimplex<dim, spacedim>::get_data(
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
FE_NedelecSimplex<dim, spacedim>::compute_edge_sign(
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
FE_NedelecSimplex<dim, spacedim>::fill_fe_values(
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
FE_NedelecSimplex<dim, spacedim>::fill_fe_face_values(
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
  AssertDimension(quadrature.size(), 1);

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
  const unsigned int n_q_points = quadrature[0].size();
  const auto         offset =
    QProjector<dim>::DataSetDescriptor::face(this->reference_cell(),
                                             face_no,
                                             cell->combined_face_orientation(
                                               face_no),
                                             n_q_points);

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
  FE_NedelecSimplex<dim, spacedim>::fill_fe_subface_values(
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
FE_NedelecSimplex<dim, spacedim>::requires_update_flags(
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
FE_NedelecSimplex<dim, spacedim>::get_name() const
{
  // note that the FETools::get_fe_by_name function depends on the particular
  // format of the string this function returns, so they have to be kept in sync
  std::ostringstream namebuf;
  namebuf << "FE_NedelecSimplex<" << Utilities::dim_string(dim, spacedim) << ">("
          << this->degree - 1 << ")";

  return namebuf.str();
}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, dim>>
FE_NedelecSimplex<dim, spacedim>::clone() const
{
  return std::make_unique<FE_NedelecSimplex<dim, spacedim>>(*this);
}



template <int dim, int spacedim>
std::vector<unsigned int>
FE_NedelecSimplex<dim, spacedim>::get_dpo_vector(const unsigned int degree)
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



template <int dim, int spacedim>
unsigned int
FE_NedelecSimplex<dim, spacedim>::compute_num_dofs(const unsigned int degree) const
{
  Assert(degree==0, ExcMessage(
           "Not yet implemented for degree>0!"));
  // Internal function to compute the number of DoFs
  // for a given dimension & polynomial order.
  switch (dim)
    {
      case 2:
        return 3;

      case 3:
        return 6;

      default:
        {
          DEAL_II_NOT_IMPLEMENTED();
          return 0;
        }
    }
}




// explicit instantiations
#include "fe/fe_nedelec_simplex.inst"

DEAL_II_NAMESPACE_CLOSE
