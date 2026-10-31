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


// Meshes of pyramids, hexahedra and tetrahedra with faces in all orientations
// for the pyramid_*.cc tests.

#ifndef dealii_tests_simplex_pyramid_meshes_h
#define dealii_tests_simplex_pyramid_meshes_h

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <vector>

#include "../tests.h"


namespace PyramidMeshes
{
  // the orientation preserving symmetries of the pyramid: rotations of the
  // base
  inline const std::vector<std::array<unsigned int, 5>> &
  pyramid_symmetries()
  {
    static const std::vector<std::array<unsigned int, 5>> symmetries = {
      {{0, 1, 2, 3, 4}},
      {{1, 3, 0, 2, 4}},
      {{3, 2, 1, 0, 4}},
      {{2, 0, 3, 1, 4}}};
    return symmetries;
  }

  // the even permutations of the vertices of a tetrahedron
  inline const std::vector<std::array<unsigned int, 4>> &
  tetrahedron_symmetries()
  {
    static const std::vector<std::array<unsigned int, 4>> symmetries = {
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
    return symmetries;
  }

  // the orientation preserving symmetries of the hexahedron: permute the
  // coordinate directions and reflect some of them
  inline std::vector<std::array<unsigned int, 8>>
  hexahedron_symmetries()
  {
    std::vector<std::array<unsigned int, 8>> symmetries;
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
              symmetries.push_back(symmetry);
            }
      }
    while (std::next_permutation(directions.begin(), directions.end()));
    return symmetries;
  }

  inline double
  signed_volume(const std::vector<Point<3>> &vertices,
                const unsigned int           a,
                const unsigned int           b,
                const unsigned int           c,
                const unsigned int           d)
  {
    const Tensor<1, 3> u = vertices[b] - vertices[a];
    const Tensor<1, 3> v = vertices[c] - vertices[a];
    const Tensor<1, 3> w = vertices[d] - vertices[a];
    return cross_product_3d(u, v) * w;
  }

  // a pyramid with the given base (in the order of the quadrilateral faces)
  // and apex, with positive volume, its vertices permuted by a symmetry
  inline CellData<3>
  make_pyramid(const std::vector<Point<3>> &vertices,
               std::array<unsigned int, 4>  base,
               const unsigned int           apex,
               const unsigned int           symmetry_index)
  {
    if (signed_volume(vertices, base[0], base[1], base[2], apex) < 0)
      std::swap(base[1], base[2]);
    const std::array<unsigned int, 5> pyramid = {
      {base[0], base[1], base[2], base[3], apex}};
    const auto &symmetry =
      pyramid_symmetries()[symmetry_index % pyramid_symmetries().size()];
    CellData<3> cell(5);
    for (unsigned int v = 0; v < 5; ++v)
      cell.vertices[v] = pyramid[symmetry[v]];
    return cell;
  }

  inline CellData<3>
  make_tetrahedron(const std::vector<Point<3>> &vertices,
                   std::array<unsigned int, 4>  tetrahedron,
                   const unsigned int           symmetry_index)
  {
    if (signed_volume(vertices,
                      tetrahedron[0],
                      tetrahedron[1],
                      tetrahedron[2],
                      tetrahedron[3]) < 0)
      std::swap(tetrahedron[2], tetrahedron[3]);
    const auto &symmetry =
      tetrahedron_symmetries()[symmetry_index % tetrahedron_symmetries().size()];
    CellData<3> cell(4);
    for (unsigned int v = 0; v < 4; ++v)
      cell.vertices[v] = tetrahedron[symmetry[v]];
    return cell;
  }
} // namespace PyramidMeshes



// [0,n]^3 split into unit cubes:
// - mesh 0: every cube is split into the 6 pyramids from its faces to its
//   center,
// - mesh 1: like mesh 0, but the cubes with even x+y+z are hexahedra,
// - mesh 2: [0,n]^2 x [0,2]: the cubes below z = 1 are split into pyramids
//   like in mesh 0, the cubes above into the 6 tetrahedra around their
//   diagonal from (0,0,0) to (1,1,1) (Kuhn's subdivision). The pyramids with
//   their base at z = 1 are split into 2 tetrahedra along the diagonal of the
//   tetrahedra above.
// The vertices of every cell are permuted by a symmetry of its reference
// cell.
inline void
make_pyramid_mesh(Triangulation<3> &tria,
                  const unsigned int n,
                  const unsigned int mesh)
{
  using namespace PyramidMeshes;

  const unsigned int nz = (mesh == 2 ? 2 : n);
  std::vector<Point<3>> vertices;
  for (unsigned int z = 0; z <= nz; ++z)
    for (unsigned int y = 0; y <= n; ++y)
      for (unsigned int x = 0; x <= n; ++x)
        vertices.emplace_back(x, y, z);
  const auto vertex =
    [n](const unsigned int x, const unsigned int y, const unsigned int z) {
      return x + (n + 1) * (y + (n + 1) * z);
    };

  const std::vector<std::array<unsigned int, 8>> hex_symmetries =
    hexahedron_symmetries();

  // the faces of the cube with the vertices in the order of the faces of the
  // hexahedron
  static const unsigned int cube_faces[6][4] = {{0, 2, 4, 6},
                                                {1, 3, 5, 7},
                                                {0, 4, 1, 5},
                                                {2, 6, 3, 7},
                                                {0, 1, 2, 3},
                                                {4, 5, 6, 7}};

  std::vector<CellData<3>> cells;
  for (unsigned int z = 0; z < nz; ++z)
    for (unsigned int y = 0; y < n; ++y)
      for (unsigned int x = 0; x < n; ++x)
        {
          std::array<unsigned int, 8> cube;
          for (unsigned int v = 0; v < 8; ++v)
            cube[v] = vertex(x + (v & 1), y + ((v >> 1) & 1), z + (v >> 2));

          if (mesh == 1 && (x + y + z) % 2 == 0)
            {
              const auto &symmetry =
                hex_symmetries[(7 * cells.size() + 3) % hex_symmetries.size()];
              CellData<3> hex(8);
              for (unsigned int v = 0; v < 8; ++v)
                hex.vertices[v] = cube[symmetry[v]];
              cells.push_back(hex);
            }
          else if (mesh == 2 && z == 1)
            {
              // the paths from vertex 0 to vertex 7 along the edges of the
              // cube
              std::array<unsigned int, 3> directions = {{0, 1, 2}};
              do
                {
                  const unsigned int v1 = 1u << directions[0];
                  const unsigned int v2 = v1 | (1u << directions[1]);
                  cells.push_back(make_tetrahedron(
                    vertices, {{cube[0], cube[v1], cube[v2], cube[7]}}, cells.size()));
                }
              while (std::next_permutation(directions.begin(), directions.end()));
            }
          else
            {
              const unsigned int center = vertices.size();
              vertices.emplace_back(x + 0.5, y + 0.5, z + 0.5);
              for (unsigned int f = 0; f < 6; ++f)
                {
                  const std::array<unsigned int, 4> base = {
                    {cube[cube_faces[f][0]],
                     cube[cube_faces[f][1]],
                     cube[cube_faces[f][2]],
                     cube[cube_faces[f][3]]}};
                  if (mesh == 2 && f == 5)
                    {
                      // split along the diagonal from cube vertex 4 to 7,
                      // which the tetrahedra above have on their face z = 0
                      cells.push_back(make_tetrahedron(
                        vertices, {{base[0], base[1], base[3], center}}, cells.size()));
                      cells.push_back(make_tetrahedron(
                        vertices, {{base[0], base[3], base[2], center}}, cells.size()));
                    }
                  else
                    cells.push_back(
                      make_pyramid(vertices, base, center, 3 * cells.size() + f));
                }
            }
        }
  tria.create_triangulation(vertices, cells, SubCellData());
}

#endif
