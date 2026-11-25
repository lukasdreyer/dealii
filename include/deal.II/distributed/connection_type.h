// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_distributed_connection_type_h
#define dealii_distributed_connection_type_h

#include <deal.II/base/config.h>

DEAL_II_NAMESPACE_OPEN

namespace parallel
{
  namespace distributed
  {
    /**
     * The cells that count as neighbors of a cell in a
     * parallel::distributed::Triangulation, for its ghost layer and for the
     * 2:1 balance of the mesh: cells sharing a face, an edge or a face, or any
     * vertex with it (in 2d, edges are vertices).
     */
    enum class ConnectionType
    {
      faces,
      edges,
      vertices
    };
  } // namespace distributed
} // namespace parallel

DEAL_II_NAMESPACE_CLOSE

#endif
