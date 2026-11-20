"""Child order and child types between t8code's pyramid scheme and deal.II
pyramids and tetrahedra inside pyramids.

Like generate_prism.py: a state (t, p) is a type t of t8code's pyramid scheme
(0, 3 pyramids, 1, 2 tetrahedra) together with a vertex permutation p, where
vertex i of the deal.II cell lies on vertex p[i] of the t8code element. The
pyramid trees are created with the deal.II vertices in deal.II order, so the
root state is (0, identity).

The deal.II children are given by their vertex coordinates on the reference
pyramid and on the reference tetrahedron (for a tetrahedron inside a
pyramid), as printed by deal.II after refining them, so the script does not
depend on the node numbering in tria.cc.

Output: for every state reachable from the root, the deal.II child index of
each t8code child and the deal.II "type" of each deal.II child. The types of
deal.II pyramids are the state numbers, the ones of deal.II tetrahedra are
24 + state number, so that they differ from the 24 states of tetrahedral trees
(see amr_to_dealii_child_index_and_type in t8code_wrappers.cc). State 0 is the
root.
"""

from fractions import Fraction as Q
from itertools import permutations

# deal.II reference cells, coordinates times 2
DEALII_VERTICES = {
    "pyr": ((-2, -2, 0), (2, -2, 0), (-2, 2, 0), (2, 2, 0), (0, 0, 2)),
    "tet": ((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)),
}
DEALII_CHILDREN = {
    "pyr": (
        ("pyr", ((-2, -2, 0), (0, -2, 0), (-2, 0, 0), (0, 0, 0), (-1, -1, 1))),
        ("pyr", ((0, -2, 0), (2, -2, 0), (0, 0, 0), (2, 0, 0), (1, -1, 1))),
        ("pyr", ((-2, 0, 0), (0, 0, 0), (-2, 2, 0), (0, 2, 0), (-1, 1, 1))),
        ("pyr", ((0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0), (1, 1, 1))),
        ("tet", ((-1, -1, 1), (-1, 1, 1), (0, 0, 0), (-2, 0, 0))),
        ("tet", ((1, -1, 1), (1, 1, 1), (2, 0, 0), (0, 0, 0))),
        ("tet", ((-1, -1, 1), (1, -1, 1), (0, -2, 0), (0, 0, 0))),
        ("tet", ((-1, 1, 1), (1, 1, 1), (0, 0, 0), (0, 2, 0))),
        ("pyr", ((-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1), (0, 0, 0))),
        ("pyr", ((-1, -1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, 1), (0, 0, 2))),
    ),
    "tet": (
        ("tet", ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))),
        ("tet", ((1, 0, 0), (2, 0, 0), (1, 1, 0), (1, 0, 1))),
        ("tet", ((0, 1, 0), (1, 1, 0), (0, 2, 0), (0, 1, 1))),
        ("tet", ((0, 0, 1), (1, 0, 1), (0, 1, 1), (0, 0, 2))),
        ("pyr", ((1, 1, 0), (1, 0, 1), (0, 1, 0), (0, 0, 1), (1, 0, 0))),
        ("pyr", ((1, 1, 0), (0, 1, 0), (1, 0, 1), (0, 0, 1), (0, 1, 1))),
    ),
}

# t8code standalone pyramid scheme (t8_standalone_lut_pyra.hxx)
T8_VERTICES = {
    0: ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)),
    1: ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)),
    2: ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)),
    3: ((1, 1, 1), (0, 1, 1), (1, 0, 1), (0, 0, 1), (0, 0, 0)),
}
KIND = {0: "pyr", 3: "pyr", 1: "tet", 2: "tet"}
ILOC_TO_CHILDTYPE = {0: (0, 0, 2, 0, 1, 0, 1, 2, 3, 0), 1: (1, 1, 3, 0, 1, 1),
                     2: (2, 2, 3, 0, 2, 2), 3: (3, 0, 1, 2, 3, 2, 3, 1, 3, 3)}
ILOC_TO_CHILDCUBEID = {0: (0, 1, 1, 2, 2, 3, 3, 3, 3, 7), 1: (0, 1, 1, 5, 5, 7),
                       2: (0, 2, 2, 6, 6, 7), 3: (0, 4, 4, 4, 4, 5, 5, 6, 6, 7)}


def solve(m, b):
    """Solve the 3x3 system m x = b exactly."""
    a = [[Q(m[i][j]) for j in range(3)] + [Q(b[i])] for i in range(3)]
    for c in range(3):
        r = next(r for r in range(c, 3) if a[r][c] != 0)
        a[c], a[r] = a[r], a[c]
        for r in range(3):
            if r != c and a[r][c] != 0:
                f = a[r][c] / a[c][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [a[i][3] / a[i][i] for i in range(3)]


def affine(src, dst):
    """The affine map x -> A x + b with src[i] -> dst[i] for all i, from the
    vertices 0, 1, 2 and the last one, or None."""
    idx = (0, 1, 2, len(src) - 1)
    s = [[Q(src[i][d]) - Q(src[0][d]) for i in idx[1:]] for d in range(3)]
    t = [[Q(dst[i][d]) - Q(dst[0][d]) for i in idx[1:]] for d in range(3)]
    # A s = t, row by row: A[e] s = t[e]
    st = [[s[d][k] for d in range(3)] for k in range(3)]
    try:
        A = [solve(st, [t[e][k] for k in range(3)]) for e in range(3)]
    except StopIteration:
        return None
    f = lambda x: tuple(Q(dst[0][e]) + sum(A[e][d] * (Q(x[d]) - Q(src[0][d])) for d in range(3))
                        for e in range(3))
    if any(f(p) != tuple(map(Q, q)) for p, q in zip(src, dst)):
        return None
    det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
           - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
           + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
    return f, det


def t8_children(t):
    """The children of t8 type t in t8 coordinates of the parent, in t8 child
    order: (child type, vertices)."""
    out = []
    for ct, cid in zip(ILOC_TO_CHILDTYPE[t], ILOC_TO_CHILDCUBEID[t]):
        o = (cid & 1, (cid >> 1) & 1, (cid >> 2) & 1)
        out.append((ct, tuple(tuple(Q(o[d] + v[d], 2) for d in range(3)) for v in T8_VERTICES[ct])))
    return out


def main():
    root = (0, (0, 1, 2, 3, 4))
    states, todo, tables = [root], [root], {}
    while todo:
        t, p = todo.pop(0)
        kind = KIND[t]
        # map the deal.II children into t8 coordinates of this element
        to_t8, det = affine(DEALII_VERTICES[kind], [T8_VERTICES[t][p[i]] for i in range(len(p))])
        assert det > 0
        dealii_children = [(ck, tuple(to_t8(v) for v in verts)) for ck, verts in DEALII_CHILDREN[kind]]
        t8_to_dealii, child_states = [], [None] * len(dealii_children)
        for ct, verts in t8_children(t):
            hits = [c for c, (ck, dv) in enumerate(dealii_children) if set(dv) == set(verts)]
            assert len(hits) == 1, (t, p, ct)
            c = hits[0]
            assert dealii_children[c][0] == KIND[ct]
            q = tuple(verts.index(x) for x in dealii_children[c][1])
            # the child must be a positive affine image of the reference cell
            _, cdet = affine(DEALII_VERTICES[KIND[ct]], [T8_VERTICES[ct][q[i]] for i in range(len(q))])
            assert cdet > 0
            t8_to_dealii.append(c)
            child_states[c] = (ct, q)
            if (ct, q) not in states:
                states.append((ct, q))
                todo.append((ct, q))
        tables[(t, p)] = (t8_to_dealii, child_states)

    def dealii_type(s):
        i = states.index(s)
        return i if KIND[s[0]] == "pyr" else 24 + i

    print(f"// {len(states)} states (t8 type, deal.II vertex i on t8 vertex p[i]):")
    for i, s in enumerate(states):
        print(f"//   {dealii_type(s)}: {s}")
    print(f"const ::dealii::ndarray<unsigned int, {len(states)}, 10> pyramid_perms_t8_to_dealii = {{{{")
    for s in states:
        row = tables[s][0] + [0] * (10 - len(tables[s][0]))
        print("  {{" + ", ".join(map(str, row)) + "}},")
    print("}};")
    print(f"const ::dealii::ndarray<unsigned int, {len(states)}, 10> pyramid_child_types = {{{{")
    for s in states:
        row = [dealii_type(cs) for cs in tables[s][1]]
        row += [0] * (10 - len(row))
        print("  {{" + ", ".join(map(str, row)) + "}},")
    print("}};")


if __name__ == "__main__":
    main()
