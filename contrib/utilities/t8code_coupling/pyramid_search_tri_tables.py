"""Search the vertex orders of the four triangle face children (one global
table in deal.II's face refinement) for which the pyramid scheme becomes
feasible. The geometry of the children is precomputed once."""
import itertools, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyramid_orient as P

def precompute(state):
    t, p = state
    kind = P.KIND[t]; dv, dfaces = P.DEALII[kind]
    A, f = P.affine(dv, [P.T8_VERTICES[t][p[i]] for i in range(len(dv))])
    Ainv = np.linalg.inv(A); base = np.array(f(dv[0]))
    to_dealii = lambda x: tuple(np.array([float(c) for c in dv[0]]) + Ainv @ (np.array(x) - base))
    parent_faces = [[tuple(map(float, dv[i])) for i in fvs] for fvs in dfaces]
    kids = []
    for ct, cid in zip(P.CHILDTYPE[t], P.CUBEID[t]):
        o = (cid & 1, (cid >> 1) & 1, (cid >> 2) & 1)
        pts = [to_dealii(tuple((o[d] + v[d]) / 2 for d in range(3))) for v in P.T8_VERTICES[ct]]
        opts = []
        for q in P.valid_perms(ct):
            cverts = [pts[q[i]] for i in range(len(q))]
            opts.append(((ct, q), [tuple(P.key(cverts[i]) for i in fv) for fv in P.DEALII[P.KIND[ct]][1]]))
        kids.append(opts)
    return parent_faces, kids

PRE = {s: precompute(s) for s in P.STATES}

def feasible_set():
    info = {}
    for s, (pfaces, kids) in PRE.items():
        fc = {}
        for fv in pfaces:
            for c in P.face_children(fv):
                k = tuple(P.key(x) for x in c)
                fc[frozenset(k)] = k
        opts = []
        for child in kids:
            ok = [cs for cs, faces in child
                  if all(fc.get(frozenset(f), f) == f for f in faces)]
            opts.append(ok)
        info[s] = opts
    feas = set(P.STATES); changed = True
    while changed:
        changed = False
        for s in list(feas):
            if any(not [o for o in child if o in feas] for child in info[s]):
                feas.discard(s); changed = True
    return feas

DEFAULT = list(P.TRI_TABLE)
assert not feasible_set()   # the current deal.II table: infeasible
results = []
for perms in itertools.product(list(itertools.permutations(range(3))), repeat=4):
    P.TRI_TABLE = [tuple(DEFAULT[k][perms[k][i]] for i in range(3)) for k in range(4)]
    feas = feasible_set()
    if feas:
        results.append((list(P.TRI_TABLE), len(feas), (0, (0,1,2,3,4)) in feas,
                        {t: sum(1 for s in feas if s[0] == t) for t in P.T8_VERTICES}))
print(f"{len(results)} of 1296 triangle child tables make some states feasible")
for r in results[:30]:
    print(r)
