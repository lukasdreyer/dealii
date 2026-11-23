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
  : FE_NedelecLowestOrderBase<dim, spacedim>(ReferenceCells::get_simplex<dim>(),
                                             order)
{
  // The hanging node constraints are computed by
  // DoFTools::make_hanging_node_constraints() from the line objects, the
  // interface constraints below are not used.
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


// explicit instantiations
#include "fe/fe_nedelec_simplex.inst"

DEAL_II_NAMESPACE_CLOSE
