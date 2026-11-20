"""pyramid_orient_sides.py for all 1296 vertex orders of the triangle face
children (only the face children depend on the table)."""
import itertools, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyramid_orient as P
import pyramid_orient_sides as M

DEFAULT = list(P.TRI_TABLE)
# table independent part of the geometry
BASE = {}
for s in P.STATES:
    t, p = s
    kind = P.KIND[t]; dv, dfaces = P.DEALII[kind]
    BASE[s] = [[tuple(map(float, dv[i])) for i in fvs] for fvs in dfaces]

def regeometry():
    for s, (pfc, onp, copts, inner, fsets) in list(M.GEO.items()):
        pf_children = [[tuple(P.key(x) for x in c) for c in P.face_children(fv)] for fv in BASE[s]]
        on_parent = {frozenset(fc): (f, k) for f, fcs in enumerate(pf_children) for k, fc in enumerate(fcs)}
        M.GEO[s] = (pf_children, on_parent, copts, inner, fsets)

if __name__ == "__main__":
    results = []
    for perms in itertools.product(list(itertools.permutations(range(3))), repeat=4):
        P.TRI_TABLE = [tuple(DEFAULT[k][perms[k][i]] for i in range(3)) for k in range(4)]
        regeometry()
        feas = M.fixpoint()
        if feas:
            root = (0, (0,1,2,3,4), frozenset(range(5))) in feas
            results.append((list(P.TRI_TABLE), len(feas), root))
    print(f"{len(results)} of 1296 triangle child tables give feasible states")
    for r in sorted(results, key=lambda r: (not r[2], -r[1]))[:20]:
        print(r)
