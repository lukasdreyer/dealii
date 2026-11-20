"""Child order and child types between t8code prisms and deal.II wedges.

Like generate_tet.py: a state (t, p) is a t8code prism type t together with
a vertex permutation p, where vertex i of the deal.II cell lies on vertex p[i]
of the t8code element. The trees are created with the deal.II vertices in
deal.II order, so the root state is (0, identity).

The deal.II children are given by their vertex coordinates on the reference
wedge (as printed by deal.II after refining the reference cell), so the
script does not depend on the node numbering in tria.cc.
"""

from fractions import Fraction as Q
from itertools import permutations

# deal.II reference wedge and its children, coordinates times 2
DEALII_VERTICES = ((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 0, 2), (0, 2, 2))
DEALII_CHILDREN = (
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)),
    ((1, 0, 0), (2, 0, 0), (1, 1, 0), (1, 0, 1), (2, 0, 1), (1, 1, 1)),
    ((0, 1, 0), (1, 1, 0), (0, 2, 0), (0, 1, 1), (1, 1, 1), (0, 2, 1)),
    ((1, 1, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1), (0, 1, 1), (1, 0, 1)),
    ((0, 0, 1), (1, 0, 1), (0, 1, 1), (0, 0, 2), (1, 0, 2), (0, 1, 2)),
    ((1, 0, 1), (2, 0, 1), (1, 1, 1), (1, 0, 2), (2, 0, 2), (1, 1, 2)),
    ((0, 1, 1), (1, 1, 1), (0, 2, 1), (0, 1, 2), (1, 1, 2), (0, 2, 2)),
    ((1, 1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 2), (0, 1, 2), (1, 0, 2)),
)

# t8code standalone prism (t8_standalone_lut_prism.hxx)
T8_TYPES = (0, 1)

ILOC_TO_CHILDTYPE = (
    (0, 0, 1, 0, 0, 0, 1, 0),
    (1, 0, 1, 1, 1, 0, 1, 1),
)

ILOC_TO_CHILDCUBEID = (
    (0, 1, 1, 3, 4, 5, 5, 7),
    (0, 2, 2, 3, 4, 6, 6, 7),
)

TYPE_VERTICES = {
    0: ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)),
    1: ((0, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)),
}

N_CHILDREN = 8


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def det(u, x, y):
    return (u[0] * (x[1] * y[2] - x[2] * y[1])
            - u[1] * (x[0] * y[2] - x[2] * y[0])
            + u[2] * (x[0] * y[1] - x[1] * y[0]))


def affine_map(t, p):
    """The map from the deal.II reference wedge to the t8code element of type
    t with deal.II vertex i on t8code vertex p[i], or None if p is not a
    symmetry of the prism (then the map is not affine)."""
    v = [tuple(map(Q, TYPE_VERTICES[t][p[i]])) for i in range(6)]
    a = (sub(v[1], v[0]), sub(v[2], v[0]), sub(v[3], v[0]))

    def f(x):
        # x in deal.II reference coordinates times 2
        return tuple(v[0][d] + sum(Q(x[k], 2) * a[k][d] for k in range(3))
                     for d in range(3))

    if any(f(DEALII_VERTICES[i]) != v[i] for i in range(6)):
        return None
    if det(*a) <= 0:
        return None
    return f


# The 6 orientation preserving symmetries of every t8 type.
ALLOWED_PERMS = {
    t: tuple(p for p in permutations(range(6)) if affine_map(t, p) is not None)
    for t in T8_TYPES
}

STATES = tuple((t, p) for t in T8_TYPES for p in ALLOWED_PERMS[t])

STATE_ID = {state: i for i, state in enumerate(STATES)}

N_STATES = 12
assert len(STATES) == N_STATES
assert STATES[0] == (0, tuple(range(6)))


def origin(cid):
    return (cid & 1, (cid >> 1) & 1, (cid >> 2) & 1)


def t8_child(t, i):
    ct = ILOC_TO_CHILDTYPE[t][i]
    o = origin(ILOC_TO_CHILDCUBEID[t][i])
    return tuple(tuple(Q(o[d] + v[d], 2) for d in range(3))
                 for v in TYPE_VERTICES[ct])


def invert(p):
    p_inv = [0] * len(p)
    for i, j in enumerate(p):
        p_inv[j] = i
    return tuple(p_inv)


def transition(state):
    t, p = state
    f = affine_map(t, p)
    tc = tuple(t8_child(t, i) for i in range(N_CHILDREN))
    order = []
    children = []
    for child in DEALII_CHILDREN:
        dv = tuple(f(x) for x in child)  # vertices of the deal.II child
        matches = [i for i, tv in enumerate(tc) if frozenset(dv) == frozenset(tv)]
        assert len(matches) == 1
        i = matches[0]
        ct = ILOC_TO_CHILDTYPE[t][i]
        cp = tuple(tc[i].index(v) for v in dv)
        assert (ct, cp) in STATE_ID
        order.append(i)
        children.append(STATE_ID[(ct, cp)])
    return tuple(order), tuple(children)


def generate():
    orders = []
    child_states = []
    for state in STATES:
        order, children = transition(state)
        assert sorted(order) == list(range(N_CHILDREN))
        orders.append(invert(order))
        child_states.append(children)
    return tuple(orders), tuple(child_states)


def cpp2(name, a):
    return (f"const ::dealii::ndarray<unsigned int, {N_STATES}, {N_CHILDREN}> {name} = {{{{\n"
            + ",\n".join("  {{" + ", ".join(map(str, r)) + "}}" for r in a)
            + "\n}};")


def main():
    orders, types = generate()
    print("// State = 6 * t8_type + allowed_permutation_index")
    for i, (t, p) in enumerate(STATES):
        print(f"//   state {i}: t8 type {t}, deal.II vertex i on t8 vertex {p}")
    print(cpp2("prism_perms_t8_to_dealii", orders))
    print(cpp2("prism_child_types", types))


if __name__ == "__main__":
    main()
