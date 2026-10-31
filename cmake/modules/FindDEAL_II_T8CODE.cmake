## ------------------------------------------------------------------------
##
## SPDX-License-Identifier: LGPL-2.1-or-later
## Copyright (C) 2012 - 2026 by the deal.II authors
##
## This file is part of the deal.II library.
##
## Part of the source code is dual licensed under Apache-2.0 WITH
## LLVM-exception OR LGPL-2.1-or-later. Detailed license information
## governing the source code and code contributions can be found in
## LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
##
## ------------------------------------------------------------------------

#
# Try to find the t8code library together with the sc and p4est libraries
# it was built with.
#
# T8CODE_DIR may point to the t8code installation prefix, to its cmake/
# directory containing T8CODEConfig.cmake, or to a t8code source directory
# with an install/ subdirectory.
#
# This module exports:
#   T8CODE_INCLUDE_DIR
#   T8CODE_WITH_MPI
#   T8CODE_VERSION
#

set(T8CODE_DIR "" CACHE PATH
  "An optional hint to a t8code installation/directory"
  )
set_if_empty(T8CODE_DIR "$ENV{T8CODE_DIR}")

set(_hints ${T8CODE_DIR} ${T8CODE_DIR}/install)

#
# t8code installs its own copies of sc and p4est; search them first so
# that T8CODEConfig.cmake picks them up as its dependencies:
#
find_package(SC CONFIG QUIET HINTS ${_hints})
find_package(P4EST CONFIG QUIET HINTS ${_hints})
find_package(T8CODE CONFIG QUIET HINTS ${_hints})

set(_targets)
if(T8CODE_FOUND)
  set(_targets T8CODE::T8)

  if(T8CODE_ENABLE_MPI)
    set(T8CODE_WITH_MPI TRUE)
  else()
    set(T8CODE_WITH_MPI FALSE)
  endif()

  #
  # T8CODE_DIR now is the directory of T8CODEConfig.cmake, i.e.,
  # <prefix>/cmake:
  #
  deal_ii_find_path(T8CODE_INCLUDE_DIR t8.h
    HINTS ${T8CODE_DIR}/.. ${T8CODE_DIR}
    PATH_SUFFIXES include
    )
endif()

process_feature(T8CODE
  TARGETS REQUIRED _targets
  LIBRARIES OPTIONAL LAPACK_LIBRARIES MPI_C_LIBRARIES
  INCLUDE_DIRS
    REQUIRED T8CODE_INCLUDE_DIR
  CLEAR
    T8CODE_INCLUDE_DIR
  )
