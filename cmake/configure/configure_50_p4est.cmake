## -----------------------------------------------------------------------------
##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
## Copyright (C) 2012 - 2025 by the deal.II authors
##
## This file is part of the deal.II library.
##
## Detailed license information governing the source code and contributions
## can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
##
## -----------------------------------------------------------------------------

#
# Configuration for the p4est and sc libraries:
#

set(FEATURE_P4EST_DEPENDS MPI)


macro(feature_p4est_find_external var)
  find_package(DEAL_II_P4EST)

  if(P4EST_FOUND)
    set(${var} TRUE)

    #
    # We require at least version 2.2
    #
    set(_version_required 2.2)
    if(P4EST_VERSION VERSION_LESS ${_version_required})
      message(STATUS "Insufficient p4est installation found: "
        "At least version ${_version_required} is required."
        )
      set(P4EST_ADDITIONAL_ERROR_STRING
        "Insufficient p4est installation found!\n"
        "At least version ${_version_required} is required.\n"
        )
      set(${var} FALSE)
    endif()

    #
    # Check whether p4est supports mpi:
    #
    if(NOT P4EST_WITH_MPI)
      message(STATUS "Insufficient p4est installation found: "
        "p4est has to be configured with MPI enabled."
        )
      set(P4EST_ADDITIONAL_ERROR_STRING
        ${P4EST_ADDITIONAL_ERROR_STRING}
        "Insufficient p4est installation found!\n"
        "p4est has to be configured with MPI enabled.\n"
        )
      set(${var} FALSE)
    endif()

    #
    # Check whether p4est is built against zlib:
    #
    if(NOT P4EST_WITH_ZLIB)
      message(STATUS "Insufficient p4est installation found: "
        "p4est has to be configured with enabled zlib support."
        )
      set(P4EST_ADDITIONAL_ERROR_STRING
        ${P4EST_ADDITIONAL_ERROR_STRING}
        "Insufficient p4est installation found!\n"
        "p4est has to be configured with enabled zlib support.\n"
        )
      set(${var} FALSE)
    endif()

    check_mpi_interface(P4EST ${var})
  endif()
endmacro()

macro(feature_p4est_configure_external)
  set(DEAL_II_P4EST_WITH_VTK_BINARY ${P4EST_WITH_VTK_BINARY})
endmacro()


#
# p4est and t8code both implement parallel::distributed::Triangulation,
# so only one of them can be enabled. An explicitly requested t8code
# disables the autodetection of p4est:
#
if(DEAL_II_WITH_T8CODE)
  if(DEAL_II_WITH_P4EST)
    message(FATAL_ERROR "\n"
      "DEAL_II_WITH_P4EST and DEAL_II_WITH_T8CODE cannot both be set to "
      "\"ON\", since both implement parallel::distributed::Triangulation.\n\n"
      )
  elseif(NOT DEFINED DEAL_II_WITH_P4EST)
    message(STATUS "DEAL_II_WITH_T8CODE is set, disabling p4est.")
    set_cached_option(P4EST OFF)
  endif()
endif()


configure_feature(P4EST)
