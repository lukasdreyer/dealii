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
# Configuration for the t8code library (with its own sc and p4est):
#

set(FEATURE_T8CODE_DEPENDS MPI)
set(FEATURE_T8CODE_AFTER P4EST)


macro(feature_t8code_find_external var)
  #
  # p4est and t8code both implement parallel::distributed::Triangulation,
  # so only one of them can be enabled. configure_50_p4est.cmake disables
  # p4est if t8code was requested explicitly; here t8code was only
  # autodetected after p4est had been found:
  #
  if(DEAL_II_WITH_P4EST)
    message(STATUS "t8code is not used because p4est is enabled: "
      "set DEAL_II_WITH_P4EST=OFF and DEAL_II_WITH_T8CODE=ON to use t8code."
      )
    set(T8CODE_ADDITIONAL_ERROR_STRING
      "t8code and p4est cannot be enabled at the same time.\n"
      )
    set(${var} FALSE)
  else()
    find_package(DEAL_II_T8CODE)

    if(T8CODE_FOUND)
      set(${var} TRUE)

      #
      # Check whether t8code supports mpi:
      #
      if(NOT T8CODE_WITH_MPI)
        message(STATUS "Insufficient t8code installation found: "
          "t8code has to be configured with MPI enabled."
          )
        set(T8CODE_ADDITIONAL_ERROR_STRING
          ${T8CODE_ADDITIONAL_ERROR_STRING}
          "Insufficient t8code installation found!\n"
          "t8code has to be configured with MPI enabled.\n"
          )
        set(${var} FALSE)
      endif()

      check_mpi_interface(T8CODE ${var})
    endif()
  endif()
endmacro()


configure_feature(T8CODE)
