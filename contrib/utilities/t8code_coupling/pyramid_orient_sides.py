"""Like pyramid_orient.py, but a cell only has to satisfy the face condition
on the faces where it is the standard side. State: (t8 type, p, S) with S the
set of faces (deal.II numbering) the cell sees in standard orientation.
- Faces of the parent in S: the children see the face children in standard
  orientation (the face object has the parent's order), so the face children
  are in the children's S.
- Faces of the parent not in S: the other cell is the standard side; our
  children see those face children in non-standard orientation (both cells
  of a face use inward normals), so they are not in the children's S.
- Inner faces: exactly one of the two children is the standard side (the face
  is created in its order); chosen by the search.
Feasibility is monotone in S, the largest feasible set is a fixpoint."""
import itertools, sys, os
from functools import lru_cache
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyramid_orient as P

def geometry(state):
    t, p = state
    kind = P.KIND[t]; dv, dfaces = P.DEALII[kind]
    A, f = P.affine(dv, [P.T8_VERTICES[t][p[i]] for i in range(len(dv))])
    Ainv = np.linalg.inv(A); base = np.array(f(dv[0]))
    to_dealii = lambda x: tuple(np.array([float(c) for c in dv[0]]) + Ainv @ (np.array(x) - base))
    parent_faces = [[tuple(map(float, dv[i])) for i in fvs] for fvs in dfaces]
    pf_children = [[tuple(P.key(x) for x in c) for c in P.face_children(fv)] for fv in parent_faces]
    kids = []
    for ct, cid in zip(P.CHILDTYPE[t], P.CUBEID[t]):
        o = (cid & 1, (cid >> 1) & 1, (cid >> 2) & 1)
        pts = [to_dealii(tuple((o[d] + v[d]) / 2 for d in range(3))) for v in P.T8_VERTICES[ct]]
        kids.append((ct, pts))
    # for each child and order: the face tuples
    child_opts = []
    for ct, pts in kids:
        opts = []
        for q in P.valid_perms(ct):
            cverts = [pts[q[i]] for i in range(len(q))]
            opts.append((q, [tuple(P.key(cverts[i]) for i in fv) for fv in P.DEALII[P.KIND[ct]][1]]))
        child_opts.append((ct, opts))
    # inner faces: vertex sets shared by two children, not on a parent face
    on_parent = {frozenset(fc): (f, k) for f, fcs in enumerate(pf_children) for k, fc in enumerate(fcs)}
    face_sets = {}
    for c, (ct, opts) in enumerate(child_opts):
        for j, fset in enumerate(opts[0][1]):
            face_sets.setdefault(frozenset(fset), []).append(c)
    inner = [fs for fs, cs in face_sets.items() if fs not in on_parent]
    assert all(len(face_sets[fs]) == 2 for fs in inner)
    return pf_children, on_parent, child_opts, inner, face_sets

GEO = {s: geometry(s) for s in P.STATES}

def child_choices(state, S):
    """For each child: list of (q, forced standard faces j, {inner face set: j})."""
    pf_children, on_parent, child_opts, inner, face_sets = GEO[state]
    res = []
    for c, (ct, opts) in enumerate(child_opts):
        lst = []
        for q, faces in opts:
            forced, inner_j, ok = set(), {}, True
            for j, ft in enumerate(faces):
                fs = frozenset(ft)
                if fs in on_parent:
                    f, k = on_parent[fs]
                    if f in S:
                        if ft != pf_children[f][k]:
                            ok = False; break
                        forced.add(j)
                else:
                    inner_j[fs] = j
            if ok:
                lst.append((q, frozenset(forced), inner_j))
        res.append((ct, lst))
    return res, inner, face_sets

ALL_S = {s: [frozenset(c) for r in range(len(P.DEALII[P.KIND[s[0]]][1]) + 1)
             for c in itertools.combinations(range(len(P.DEALII[P.KIND[s[0]]][1])), r)] for s in P.STATES}

def feasible_with(state, S, feas):
    choices, inner, face_sets = child_choices(state, S)
    n = len(choices)
    assigned = [set() for _ in range(n)]
    def child_ok(c):
        ct, lst = choices[c]
        for q, forced, inner_j in lst:
            Sc = set(forced) | {inner_j[fs] for fs in assigned[c]}
            if (ct, q, frozenset(Sc)) in feas:
                return True
        return False
    if not all(child_ok(c) for c in range(n)):
        return False
    def bt(i):
        if i == len(inner):
            return True
        fs = inner[i]
        for c in face_sets[fs]:
            assigned[c].add(fs)
            if child_ok(c) and bt(i + 1):
                assigned[c].discard(fs); return True
            assigned[c].discard(fs)
        return False
    return bt(0)

def fixpoint():
    feas = {(t, p, S) for (t, p) in P.STATES for S in ALL_S[(t, p)]}
    changed = True
    while changed:
        changed = False
        for st in sorted(feas, key=lambda x: (-len(x[2]), x[0], x[1])):
            t, p, S = st
            if not feasible_with((t, p), S, feas):
                feas.discard(st); changed = True
    return feas

if __name__ == "__main__":
    feas = fixpoint()
    for t in P.T8_VERTICES:
        nf = len(P.DEALII[P.KIND[t]][1])
        tot = sum(1 for s in P.STATES if s[0] == t) * 2 ** nf
        full = sum(1 for (tt, p, S) in feas if tt == t and len(S) == nf)
        print(f"type {t} ({P.KIND[t]}): {sum(1 for x in feas if x[0]==t)} of {tot} (state, S) feasible; with all faces standard: {full}")
    root_full = (0, (0,1,2,3,4), frozenset(range(5))) in feas
    print("coarse pyramid seeing all faces in standard orientation (type 0, identity) feasible:", root_full)
    # which S are feasible for pyramid-family tets
    for t in (1, 2):
        Ss = sorted({tuple(sorted(S)) for (tt, p, S) in feas if tt == t}, key=lambda x: (len(x), x))
        print(f"type {t}: standard-face sets that occur feasible for some correspondence:", Ss)
