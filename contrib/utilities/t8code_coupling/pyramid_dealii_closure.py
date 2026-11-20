"""For the deal.II tables of dealii_search.py: which combinations close in the
coupling, i.e. every t8code element reachable from a pyramid tree (type 0,
deal.II vertex order) is subdivided into the same vertex sets as the deal.II
cell, and which t8 states (type, p) occur."""
import sys, os, itertools, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyramid_dealii_search as D
P = D.P; key = P.key

U = {"pyr": frozenset(range(5)), "tet": frozenset({0, 1})}
pyr_sols = D.solutions("pyr", D.pyr_children(), U["pyr"], U, first_owner_only=True)
tet_sols = D.solutions("tet", D.tetp_children((0, 5)), U["tet"], U, first_owner_only=True)
def distinct(sols):
    d = collections.OrderedDict()
    for orders, owners in sols:
        d.setdefault(tuple(tuple(key(x) for x in cv) for cv in orders), []).append(owners)
    return d
PO, TO = distinct(pyr_sols), distinct(tet_sols)
print("pyramid tables:", len(pyr_sols), "with", len(PO), "distinct child orders;",
      "tetP tables:", len(tet_sols), "with", len(TO), "distinct child orders")

REF = {"pyr": D.PYR, "tet": D.TET}
def to_cell(kind, cv):
    """affine map from the deal.II reference cell to the cell with vertices cv"""
    ref = REF[kind]; A, f = P.affine(ref, list(cv))
    return f

import functools
@functools.lru_cache(None)
def t8_children(t, p, kind):
    """t8 children of state (t, p) in deal.II reference coords of the parent"""
    dv = REF[kind]
    A, f = P.affine(dv, [P.T8_VERTICES[t][p[i]] for i in range(len(dv))])
    Ainv = np.linalg.inv(A); base = np.array(f(dv[0]))
    to_d = lambda x: tuple(np.array(dv[0]) + Ainv @ (np.array(x) - base))
    out = []
    for ct, cid in zip(P.CHILDTYPE[t], P.CUBEID[t]):
        o = (cid & 1, (cid >> 1) & 1, (cid >> 2) & 1)
        out.append((ct, tuple(key(to_d(tuple((o[d] + v[d]) / 2 for d in range(3)))) for v in P.T8_VERTICES[ct])))
    return tuple(out)

def closure(porder, torder):
    tables = {"pyr": porder, "tet": torder}
    root = (0, (0, 1, 2, 3, 4))
    seen = {root}; todo = [root]; maps = {}
    while todo:
        t, p = todo.pop()
        kind = P.KIND[t]
        t8k = t8_children(t, p, kind)
        dk = [tuple(c) for c in tables[kind]]
        m = []
        for ct, pts in t8k:
            hit = [i for i, cv in enumerate(dk) if set(cv) == set(pts)]
            if len(hit) != 1:
                return None
            cv = dk[hit[0]]
            q = tuple(pts.index(x) for x in cv)  # deal.II vertex i = t8 vertex q[i]
            s = (ct, q); m.append((hit[0], s))
            if s not in seen:
                seen.add(s); todo.append(s)
        maps[(t, p)] = m
    return maps

ok = []
for po in PO:
    for to in TO:
        r = closure(po, to)
        if r is not None:
            ok.append((po, to, r))
print("closing combinations:", len(ok))
if ok:
    po, to, r = ok[0]
    print("states:", sorted(r))
