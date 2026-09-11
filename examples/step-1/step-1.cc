/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 1999 - 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 */

// @sect3{Include files}

// The most fundamental class in the library is the Triangulation class, which
// is declared here:
#include <deal.II/grid/tria.h>
// Here are some functions to generate standard grids:
#include <deal.II/grid/grid_generator.h>
// Output of grids in various graphics formats:
#include <deal.II/grid/grid_out.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/base/quadrature_lib.h>

// This is needed for C++ output:
#include <iostream>
#include <fstream>
// And this for the declarations of the `std::sqrt` and `std::fabs` functions:
#include <cmath>

// The final step in importing deal.II is this: All deal.II functions and
// classes are in a namespace <code>dealii</code>, to make sure they don't
// clash with symbols from other libraries you may want to use in conjunction
// with deal.II. One could use these functions and classes by prefixing every
// use of these names by <code>dealii::</code>, but that would quickly become
// cumbersome and annoying. Rather, we simply import the entire deal.II
// namespace for general use:
using namespace dealii;

template<int dim>
void print_cells(const Triangulation<dim> &tria)
{
  for(auto cell: tria.active_cell_iterators())
  {
    std::cout<<"cell"<<std::endl;
    for(auto vertex: cell->vertex_indices()){
      std::cout<<cell->vertex_index(vertex)<<" ";
    }
    std::cout<<std::endl;

    std::cout<<"lines:" <<std::endl;
    for(auto line: cell->line_indices()){
      std::cout<<cell->line(line)->index()<<": ";
      for(auto vertex: cell->line(line)->vertex_indices()){
        std::cout<<cell->line(line)->vertex_index(vertex)<<" ";
      }
      std::cout<<std::endl;

    }

    std::cout<<"faces:" <<std::endl;
    for(auto face: cell->face_indices()){
      std::cout<<cell->face(face)->index()<<": ";
      for(auto vertex: cell->face(face)->vertex_indices()){
        std::cout<<cell->face(face)->vertex_index(vertex)<<" ";
      }
      std::cout<<std::endl;
    }
  }
}

int main()
{

Triangulation<3> triangulation;

std::vector<Point<3>> vertices =
  {
    Point<3>(0, 0, 0),
    Point<3>(1, 0, 0),
    Point<3>(0, 1, 0),
    Point<3>(0, 0, 1),
    Point<3>(1, 0, 1)
  };

std::vector<CellData<3>> cells;


cells.emplace_back(ReferenceCells::Tetrahedron);

cells.emplace_back(ReferenceCells::Tetrahedron);
cells[0].vertices[0]=0;
cells[0].vertices[1]=1;
cells[0].vertices[2]=2;
cells[0].vertices[3]=3;
cells[1].vertices[0]=1;
cells[1].vertices[1]=2;
cells[1].vertices[2]=3;
cells[1].vertices[3]=4;


triangulation.create_triangulation(vertices,
                                    cells,
                                    SubCellData());
  //print grid
  triangulation.print_internal_structures();
  print_cells(triangulation);
  //refine global
  triangulation.refine_global(1);
  //print grid
  print_cells(triangulation);
  triangulation.print_internal_structures();
  return 0;
}
