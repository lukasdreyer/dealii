"""Can t8code's pyramid scheme be refined in deal.II such that every face is
seen in standard orientation by at least one of its two cells?

States: (t8 type, p) with deal.II vertex i on t8 vertex p[i] of an element of
the pyramid scheme (types 0, 3 pyramids, 1, 2 tetrahedra), such that the
affine map from the deal.II reference cell to the t8code element is
orientation preserving (deal.II cells have positive volume).
Inner faces can always be created in the orientation of one of their two
children. Faces on the faces of the parent: if the parent sees its face in
standard orientation, the children must see the face children (deal.II face
refinement) in standard orientation, then one side of every face is standard
by induction. A state is feasible if every child has an order whose state is
feasible and which satisfies these conditions; compute the largest feasible
set by fixpoint iteration."""
from fractions import Fraction as Q
from itertools import permutations
import numpy as np

# deal.II reference cells (vertex coordinates, faces in standard vertex order)
DEALII = {
    "pyr": ([(-1,-1,0),(1,-1,0),(-1,1,0),(1,1,0),(0,0,1)],
            [(0,1,2,3),(0,2,4),(3,1,4),(1,0,4),(2,3,4)]),
    "tet": ([(0,0,0),(1,0,0),(0,1,0),(0,0,1)],
            [(0,1,2),(1,0,3),(0,2,3),(2,1,3)]),
}
# t8code pyramid scheme (t8_standalone_lut_pyra.hxx)
T8_VERTICES = {
    0: [(0,0,0),(1,0,0),(0,1,0),(1,1,0),(1,1,1)],
    1: [(0,0,0),(1,0,0),(1,0,1),(1,1,1)],
    2: [(0,0,0),(0,1,0),(0,1,1),(1,1,1)],
    3: [(1,1,1),(0,1,1),(1,0,1),(0,0,1),(0,0,0)],
}
CHILDTYPE = {0: [0,0,2,0,1,0,1,2,3,0], 1: [1,1,3,0,1,1], 2: [2,2,3,0,2,2], 3: [3,0,1,2,3,2,3,1,3,3]}
CUBEID = {0: [0,1,1,2,2,3,3,3,3,7], 1: [0,1,1,5,5,7], 2: [0,2,2,6,6,7], 3: [0,4,4,4,4,5,5,6,6,7]}
KIND = {0: "pyr", 3: "pyr", 1: "tet", 2: "tet"}

def vec(a, b): return tuple(Q(y) - Q(x) for x, y in zip(a, b))

def affine(src, dst):
    """Affine map x -> A x + b with src[i] -> dst[i] (exact), or None."""
    s = [tuple(map(Q, p)) for p in src]; d = [tuple(map(Q, p)) for p in dst]
    # solve with the first 4 points (tet / pyramid base corner 0,1,2 and apex)
    idx = [0, 1, 2, len(s) - 1]
    S = np.array([[float(x) for x in vec(s[idx[0]], s[i])] for i in idx[1:]]).T
    D = np.array([[float(x) for x in vec(d[idx[0]], d[i])] for i in idx[1:]]).T
    if abs(np.linalg.det(S)) < 1e-12: return None
    A = D @ np.linalg.inv(S)
    f = lambda p: tuple(np.array([float(c) for c in d[idx[0]]]) + A @ (np.array([float(c) for c in p]) - np.array([float(c) for c in s[idx[0]]])))
    if any(np.linalg.norm(np.array(f(p)) - np.array([float(c) for c in q])) > 1e-9 for p, q in zip(s, d)):
        return None
    return A, f

def valid_perms(t):
    kind = KIND[t]; dv = DEALII[kind][0]; tv = T8_VERTICES[t]
    out = []
    for p in permutations(range(len(dv))):
        m = affine(dv, [tv[p[i]] for i in range(len(dv))])
        if m is not None and np.linalg.det(m[0]) > 0:
            out.append(p)
    return out

_vp = {}
def _cached_valid_perms(t, _orig=valid_perms):
    if t not in _vp:
        _vp[t] = _orig(t)
    return _vp[t]
valid_perms = _cached_valid_perms
STATES = [(t, p) for t in T8_VERTICES for p in valid_perms(t)]

def key(x): return tuple(round(c, 9) for c in x)
def mid(*ps): return tuple(sum(c) / len(ps) for c in zip(*ps))

# vertex orders of the triangle face children in deal.II's face refinement:
# indices into (v0, v1, v2, m01, m12, m20)
TRI_TABLE = [(0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)]

def face_children(fv):
    if len(fv) == 3:
        n = list(fv) + [mid(fv[0], fv[1]), mid(fv[1], fv[2]), mid(fv[2], fv[0])]
        return [tuple(n[i] for i in t) for t in TRI_TABLE]
    m01, m02, m13, m23, c = mid(fv[0], fv[1]), mid(fv[0], fv[2]), mid(fv[1], fv[3]), mid(fv[2], fv[3]), mid(*fv)
    return [(fv[0], m01, m02, c), (m01, fv[1], c, m13), (m02, c, fv[2], m23), (c, m13, m23, fv[3])]

def children_options(state):
    """For each child: the list of (child state) whose deal.II vertex order
    satisfies the face conditions, or None if the subdivision does not
    conform to the deal.II face refinement."""
    t, p = state
    kind = KIND[t]; dv, dfaces = DEALII[kind]
    A, f = affine(dv, [T8_VERTICES[t][p[i]] for i in range(len(dv))])
    Ainv = np.linalg.inv(A)
    base = np.array(f(dv[0]))
    to_dealii = lambda x: tuple(np.array([float(c) for c in dv[0]]) + Ainv @ (np.array(x) - base))
    # children in deal.II coordinates of the parent, as t8 vertex lists
    kids = []
    for ct, cid in zip(CHILDTYPE[t], CUBEID[t]):
        o = (cid & 1, (cid >> 1) & 1, (cid >> 2) & 1)
        pts = [to_dealii(tuple((o[d] + v[d]) / 2 for d in range(3))) for v in T8_VERTICES[ct]]
        kids.append((ct, pts))
    # face children of the parent faces in standard orientation
    fcs = [fc for fvs in dfaces for fc in face_children([tuple(map(float, dv[i])) for i in fvs])]
    options = []
    used = 0
    for ct, pts in kids:
        ckind = KIND[ct]; cfaces = DEALII[ckind][1]
        opts = []
        for q in valid_perms(ct):
            # deal.II vertex i of the child is t8 vertex q[i] of the child
            cverts = [pts[q[i]] for i in range(len(q))]
            ok = True
            for fv in cfaces:
                cf = tuple(key(cverts[i]) for i in fv)
                for fc in fcs:
                    if set(cf) == set(key(x) for x in fc):
                        if cf != tuple(key(x) for x in fc):
                            ok = False
            if ok:
                opts.append((ct, q))
        options.append(opts)
    # conformity: every face child is a face of exactly one child
    for fc in fcs:
        n = sum(1 for ct, pts in kids for fv in DEALII[KIND[ct]][1]
                if set(key(pts[i]) for i in fv) == set(key(x) for x in fc))
        if n != 1:
            return None
    return options

def fixpoint():
    feasible = set(STATES)
    info = {s: children_options(s) for s in STATES}
    changed = True
    while changed:
        changed = False
        for s in list(feasible):
            opts = info[s]
            if opts is None or any(not [o for o in child if o in feasible] for child in opts):
                feasible.discard(s); changed = True
    return feasible, info

if __name__ == "__main__":
    print("valid states per t8 type:", {t: len(valid_perms(t)) for t in T8_VERTICES})
    feasible, info = fixpoint()
    for t in T8_VERTICES:
        ss = [s for s in STATES if s[0] == t]
        print(f"type {t} ({KIND[t]}): {sum(s in feasible for s in ss)} of {len(ss)} states feasible")
    # why do states fail?
    for s in STATES:
        opts = info[s]
        if opts is None:
            print("non-conforming subdivision:", s)
    root = [s for s in feasible if s[0] == 0 and s[1] == (0,1,2,3,4)]
    print("root state (type 0, identity) feasible:", bool(root))
    for s in sorted(STATES)[:0]: pass
    # detail for the infeasible states: which children have no order
    for s in sorted(STATES):
        if s in feasible or info[s] is None: continue
        bad = [i for i, child in enumerate(info[s]) if not child]
        print("infeasible", s, "children without any admissible order:", bad)
