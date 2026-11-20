# Child tables of the t8code coupling

Scripts that derive the tables in `source/distributed/t8code_wrappers.cc` which match t8code's refinement
trees with deal.II's: for every t8code element type and vertex permutation (a *state*), the order of the
children and the states of the children. All scripts are plain Python 3 (`fractions`, `itertools`; the
search scripts also `numpy`) and print C++ tables or reports to the terminal.

## Generators of the tables in t8code_wrappers.cc

| script | tables |
|---|---|
| `generate_tet_mirrored.py` | `tetrahedron_child_order_t8code_to_dealii`, `tetrahedron_child_type_dealii_to_dealii`: left-handed tetrahedron trees (t8code's convention) and deal.II's refinement along the (6,8) diagonal with the rotated middle triangle child |
| `generate_prism.py` | `prism_perms_t8_to_dealii`, `prism_child_types` |
| `generate_pyramid.py` | `pyramid_perms_t8_to_dealii`, `pyramid_child_types`: pyramids and the tetrahedra inside pyramids |

The deal.II children are given by their vertex coordinates on the reference cell (as deal.II prints them after
refining it), so the scripts do not depend on the node numbering in `grid/tria.cc`. The output of all three
equals the tables in `t8code_wrappers.cc` (checked on 2026-11-20). `../dealii_to_t8code_childs_and_types.py`
is the first version for right-handed tetrahedron trees.

## Searches behind deal.II's refinement tables

These scripts found the child vertex orders of deal.II's wedge, tetrahedron and pyramid refinement such that
every face is in standard orientation from at least one of its two cells (the face orientation invariant the
coupling relies on). They are kept as the derivation of the tables in `grid/tria.cc`, not needed to rebuild.

| script | question |
|---|---|
| `pyramid_orient.py` | can t8code's pyramid scheme be refined in deal.II so that every face is standard from one side (states: t8code type and vertex permutation) |
| `pyramid_orient_sides.py` | the same, with a cell bound only on the faces where it is the standard side |
| `pyramid_search_tri_tables.py`, `pyramid_search_tri_tables_sides.py` | which vertex orders of the four triangle face children (one global table in deal.II) make the pyramid scheme feasible |
| `pyramid_realize.py` | fixed choices per state (child vertex orders, owners of the inner faces) with the rotated middle triangle child |
| `pyramid_dealii_search.py` | single deal.II child tables per cell kind (pyramid, tetrahedron in a pyramid) instead of per state |
| `pyramid_dealii_closure.py` | which of these tables close in the coupling (every t8code element reachable from a pyramid tree subdivides like the deal.II cell) |
| `pyramid_dealii_choose.py` | the chosen tables in parent-local vertex numbering, closest to the existing WIP pyramid table, with all checks |
| `tet_tree_sides.py` | the same question for tetrahedron trees with a given triangle face child table |
