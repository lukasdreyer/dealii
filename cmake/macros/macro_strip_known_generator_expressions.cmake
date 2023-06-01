## ---------------------------------------------------------------------
##
## Copyright (C) 2023 - 2023 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE.md at
## the top level directory of deal.II.
##
## ---------------------------------------------------------------------

#
# strip_known_generator_expressions(<variable>)
#
# Strip an enclosing generator expression from the variable. This macro is
# primarily used in copy_target_properties
#

macro(strip_known_generator_expressions _variable)
  set(generator_expressions
    "\\$<LINK_ONLY:([^>]*)>"
    "\\$<\\$<LINK_LANGUAGE:CXX>:([^>]*)>"
    "\\$<\\$<COMPILE_LANGUAGE:CXX>:([^>]*)>"
    )

  foreach(expression ${generator_expressions})
    string(REGEX REPLACE ${expression} "\\1" ${_variable} "${${_variable}}")
  endforeach()
endmacro()

