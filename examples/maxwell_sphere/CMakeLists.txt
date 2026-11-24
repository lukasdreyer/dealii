##
#  CMake script for the dielectric sphere example (time-harmonic Maxwell)
##

set(TARGET "maxwell_sphere")
set(TARGET_SRC ${TARGET}.cc)

cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.8.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()

if(NOT DEAL_II_WITH_PETSC OR DEAL_II_PETSC_WITH_COMPLEX OR NOT DEAL_II_WITH_T8CODE)
  message(FATAL_ERROR "
Error! This program requires a deal.II library that was configured with
    DEAL_II_WITH_PETSC = ON (real scalars, with Hypre)
    DEAL_II_WITH_T8CODE = ON"
    )
endif()

deal_ii_initialize_cached_variables()
project(${TARGET})
deal_ii_invoke_autopilot()

# MUMPS (real arithmetic) for the subdomain problems of the Schwarz methods
set(MUMPS_DIR "$ENV{HOME}/bin/dealii/mumps/5.8.1.2" CACHE PATH "MUMPS installation")
set(SCALAPACK_DIR "$ENV{HOME}/bin/dealii/scalapack/2.2.2.2" CACHE PATH "ScaLAPACK installation")
find_library(DMUMPS_LIBRARY dmumps HINTS ${MUMPS_DIR}/lib REQUIRED)
find_library(MUMPS_COMMON_LIBRARY mumps_common HINTS ${MUMPS_DIR}/lib REQUIRED)
find_library(PORD_LIBRARY pord HINTS ${MUMPS_DIR}/lib REQUIRED)
target_include_directories(${TARGET} PRIVATE ${MUMPS_DIR}/include)
target_link_libraries(${TARGET} ${DMUMPS_LIBRARY} ${MUMPS_COMMON_LIBRARY} ${PORD_LIBRARY})
set_target_properties(${TARGET} PROPERTIES
  BUILD_RPATH "${MUMPS_DIR}/lib;${SCALAPACK_DIR}/lib;$ENV{HOME}/bin/dealii/libflame/5.2.0/lib;$ENV{HOME}/bin/dealii/blis/2.0/lib")
