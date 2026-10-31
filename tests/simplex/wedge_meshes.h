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


// Meshes of wedges, hexahedra and tetrahedra with faces in all orientations
// for the wedge_*.cc tests (and tests/distributed_grids/adaptive_meshes.h).

#ifndef dealii_tests_simplex_wedge_meshes_h
#define dealii_tests_simplex_wedge_meshes_h

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <vector>

#include "../tests.h"


// [0,n]^3 split into unit cubes. With hexahedra, the cubes with even x+y are
// hexahedra, all other cubes are split along a diagonal into two wedges. The
// vertices of every cell are permuted by a symmetry of its reference cell.
inline void
make_wedge_mesh(Triangulation<3> &tria,
                const unsigned int n,
                const bool         with_hexahedra)
{
  std::vector<Point<3>> vertices;
  for (unsigned int z = 0; z <= n; ++z)
    for (unsigned int y = 0; y <= n; ++y)
      for (unsigned int x = 0; x <= n; ++x)
        vertices.emplace_back(x, y, z);
  const auto vertex =
    [n](const unsigned int x, const unsigned int y, const unsigned int z) {
      return x + (n + 1) * (y + (n + 1) * z);
    };

  // the orientation preserving symmetries of the wedge
  const std::vector<std::array<unsigned int, 6>> wedge_symmetries = {
    {{0, 1, 2, 3, 4, 5}},
    {{1, 2, 0, 4, 5, 3}},
    {{2, 0, 1, 5, 3, 4}},
    {{3, 5, 4, 0, 2, 1}},
    {{5, 4, 3, 2, 1, 0}},
    {{4, 3, 5, 1, 0, 2}}};

  // the orientation preserving symmetries of the hexahedron: permute the
  // coordinate directions and reflect some of them
  std::vector<std::array<unsigned int, 8>> hex_symmetries;
  std::array<unsigned int, 3>              directions = {{0, 1, 2}};
  do
    {
      const unsigned int n_inversions = (directions[0] > directions[1]) +
                                        (directions[0] > directions[2]) +
                                        (directions[1] > directions[2]);
      for (unsigned int reflections = 0; reflections < 8; ++reflections)
        if ((n_inversions + std::bitset<3>(reflections).count()) % 2 == 0)
          {
            std::array<unsigned int, 8> symmetry;
            for (unsigned int v = 0; v < 8; ++v)
              {
                symmetry[v] = 0;
                for (unsigned int d = 0; d < 3; ++d)
                  symmetry[v] |= (((v >> d) & 1) ^ ((reflections >> d) & 1))
                                 << directions[d];
              }
            hex_symmetries.push_back(symmetry);
          }
    }
  while (std::next_permutation(directions.begin(), directions.end()));
  AssertDimension(hex_symmetries.size(), 24);

  std::vector<CellData<3>> cells;
  for (unsigned int z = 0; z < n; ++z)
    for (unsigned int y = 0; y < n; ++y)
      for (unsigned int x = 0; x < n; ++x)
        {
          std::array<unsigned int, 8> cube;
          for (unsigned int v = 0; v < 8; ++v)
            cube[v] = vertex(x + (v & 1), y + ((v >> 1) & 1), z + (v >> 2));

          if (with_hexahedra && (x + y) % 2 == 0)
            {
              const auto &symmetry =
                hex_symmetries[(7 * cells.size() + 3) % hex_symmetries.size()];
              CellData<3> hex(8);
              for (unsigned int v = 0; v < 8; ++v)
                hex.vertices[v] = cube[symmetry[v]];
              cells.push_back(hex);
            }
          else
            for (const std::array<unsigned int, 6> &wedge :
                 {std::array<unsigned int, 6>{
                    {cube[0], cube[1], cube[2], cube[4], cube[5], cube[6]}},
                  std::array<unsigned int, 6>{
                    {cube[1], cube[3], cube[2], cube[5], cube[7], cube[6]}}})
              {
                const auto &symmetry =
                  wedge_symmetries[cells.size() % wedge_symmetries.size()];
                CellData<3> cell(6);
                for (unsigned int v = 0; v < 6; ++v)
                  cell.vertices[v] = wedge[symmetry[v]];
                cells.push_back(cell);
              }
        }
  tria.create_triangulation(vertices, cells, SubCellData());
}



// [0,n]^2 x [0,2]: wedges for z < 1 and tetrahedra for z > 1. Each unit cube
// above z = 1 is split into the six tetrahedra around its diagonal from
// (0,0,0) to (1,1,1) (Kuhn's subdivision), the cubes below into two wedges
// along the same diagonal, so the triangles match at z = 1. The vertices of
// every cell are permuted by a symmetry of its reference cell.
inline void
make_wedge_tetrahedron_mesh(Triangulation<3> &tria, const unsigned int n)
{
  std::vector<Point<3>> vertices;
  for (unsigned int z = 0; z <= 2; ++z)
    for (unsigned int y = 0; y <= n; ++y)
      for (unsigned int x = 0; x <= n; ++x)
        vertices.emplace_back(x, y, z);
  const auto vertex =
    [n](const unsigned int x, const unsigned int y, const unsigned int z) {
      return x + (n + 1) * (y + (n + 1) * z);
    };

  const std::vector<std::array<unsigned int, 6>> wedge_symmetries = {
    {{0, 1, 2, 3, 4, 5}},
    {{1, 2, 0, 4, 5, 3}},
    {{2, 0, 1, 5, 3, 4}},
    {{3, 5, 4, 0, 2, 1}},
    {{5, 4, 3, 2, 1, 0}},
    {{4, 3, 5, 1, 0, 2}}};
  // the even permutations of the vertices of a tetrahedron
  const std::vector<std::array<unsigned int, 4>> tetrahedron_symmetries = {
    {{0, 1, 2, 3}},
    {{0, 2, 3, 1}},
    {{0, 3, 1, 2}},
    {{1, 0, 3, 2}},
    {{1, 2, 0, 3}},
    {{1, 3, 2, 0}},
    {{2, 0, 1, 3}},
    {{2, 1, 3, 0}},
    {{2, 3, 0, 1}},
    {{3, 0, 2, 1}},
    {{3, 1, 0, 2}},
    {{3, 2, 1, 0}}};

  std::vector<CellData<3>> cells;
  for (unsigned int z = 0; z < 2; ++z)
    for (unsigned int y = 0; y < n; ++y)
      for (unsigned int x = 0; x < n; ++x)
        {
          std::array<unsigned int, 8> cube;
          for (unsigned int v = 0; v < 8; ++v)
            cube[v] = vertex(x + (v & 1), y + ((v >> 1) & 1), z + (v >> 2));

          if (z == 0)
            for (const std::array<unsigned int, 6> &wedge :
                 {std::array<unsigned int, 6>{
                    {cube[0], cube[1], cube[3], cube[4], cube[5], cube[7]}},
                  std::array<unsigned int, 6>{
                    {cube[0], cube[3], cube[2], cube[4], cube[7], cube[6]}}})
              {
                const auto &symmetry =
                  wedge_symmetries[cells.size() % wedge_symmetries.size()];
                CellData<3> cell(6);
                for (unsigned int v = 0; v < 6; ++v)
                  cell.vertices[v] = wedge[symmetry[v]];
                cells.push_back(cell);
              }
          else
            {
              // the paths from vertex 0 to vertex 7 along the edges of the
              // cube, with the orientation of the tetrahedron
              std::array<unsigned int, 3> directions = {{0, 1, 2}};
              do
                {
                  const unsigned int v1 = 1u << directions[0];
                  const unsigned int v2 = v1 | (1u << directions[1]);
                  std::array<unsigned int, 4> tetrahedron = {
                    {cube[0], cube[v1], cube[v2], cube[7]}};
                  const unsigned int n_inversions =
                    (directions[0] > directions[1]) +
                    (directions[0] > directions[2]) +
                    (directions[1] > directions[2]);
                  if (n_inversions % 2 == 1)
                    std::swap(tetrahedron[1], tetrahedron[2]);

                  const auto &symmetry =
                    tetrahedron_symmetries[(5 * cells.size()) %
                                           tetrahedron_symmetries.size()];
                  CellData<3> cell(4);
                  for (unsigned int v = 0; v < 4; ++v)
                    cell.vertices[v] = tetrahedron[symmetry[v]];
                  cells.push_back(cell);
                }
              while (
                std::next_permutation(directions.begin(), directions.end()));
            }
        }
  tria.create_triangulation(vertices, cells, SubCellData());
}

#endif
