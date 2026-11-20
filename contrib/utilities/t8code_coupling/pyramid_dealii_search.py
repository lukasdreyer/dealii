"""Single deal.II child tables for pyramids and pyramid-family tetrahedra
(rotated middle triangle child) such that every face is standard from at
least one side. Model as in pyramid_orient_sides.py, but the table
is global per deal.II cell kind, not per t8code state: a constraint on a
parent face f only applies if some reachable cell of that kind is the
standard side of f, so it suffices to track the union U(kind) of the
reachable standard face sets."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyramid_orient as P
P.TRI_TABLE = [(0, 3, 5), (3, 1, 4), (5, 4, 2), (4, 5, 3)]
key, mid = P.key, P.mid
F = lambda p: tuple(map(float, p))

PYR = [F(v) for v in P.DEALII["pyr"][0]]
TET = [F(v) for v in P.DEALII["tet"][0]]

def pyr_children():
    v = PYR; c = mid(*v[:4]); m = lambda a, b: mid(v[a], v[b])
    kids = []
    # base corner pyramids (sets; order chosen later)
    adj = {0: (1, 2), 1: (0, 3), 2: (0, 3), 3: (1, 2)}
    for i in range(4):
        kids.append(("pyr", [v[i], m(i, adj[i][0]), m(i, adj[i][1]), c, m(i, 4)]))
    lat = [m(i, 4) for i in range(4)]
    kids.append(("pyr", lat + [v[4]]))          # apex pyramid
    kids.append(("pyr", lat + [c]))             # inverted middle pyramid
    for a, b in [(0, 1), (1, 3), (3, 2), (2, 0)]:
        kids.append(("tet", [m(a, b), m(a, 4), m(b, 4), c]))
    return kids

TET_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]  # deal.II lines, midpoint 4+l
def tetp_children(axis):
    """axis: pair of line indices (opposite edges) whose midpoints are the apexes."""
    v = TET; m = {l: mid(v[a], v[b]) for l, (a, b) in enumerate(TET_EDGES)}
    kids = []
    for i in range(4):
        kids.append(("tet", [v[i]] + [m[l] for l, e in enumerate(TET_EDGES) if i in e]))
    sq = [m[l] for l in range(6) if l not in axis]
    for a in axis:
        kids.append(("pyr", sq + [m[a]]))
    return kids

def orders(kind, pts):
    """vertex orders (deal.II positive volume) as lists of points"""
    out = []
    ref = [F(x) for x in P.DEALII[kind][0]]
    import numpy as np
    for q in itertools.permutations(range(len(pts))):
        cv = [pts[i] for i in q]
        m = P.affine(ref, cv)
        if m is not None and np.linalg.det(m[0]) > 0:
            out.append(cv)
    return out

def faces_of(kind, cv):
    return [tuple(key(cv[i]) for i in fv) for fv in P.DEALII[kind][1]]

def parent_face_children(kind):
    ref = PYR if kind == "pyr" else TET
    res = {}
    for f, fv in enumerate(P.DEALII[kind][1]):
        for k, fc in enumerate(P.face_children([ref[i] for i in fv])):
            t = tuple(key(x) for x in fc)
            res[frozenset(t)] = (f, t)
    return res

def solutions(kind, kids, U_parent, U_allowed, limit=None, first_owner_only=False):
    """all (orders per child, owner per inner face) with constraints on faces
    f in U_parent and child standard sets within U_allowed[child kind]"""
    on_parent = parent_face_children(kind)
    opts = []
    for ck, pts in kids:
        lst = []
        for cv in orders(ck, pts):
            fs = faces_of(ck, cv); forced = set(); inner = {}; ok = True
            for j, ft in enumerate(fs):
                s = frozenset(ft)
                if s in on_parent:
                    f, t = on_parent[s]
                    if f in U_parent:
                        if ft != t: ok = False; break
                        forced.add(j)
                else:
                    inner[s] = j
            if ok and forced <= U_allowed[ck]:
                lst.append((cv, forced, inner))
        opts.append(lst)
    inner_sets = {}
    for c, lst in enumerate(opts):
        if not lst: return []
        for s in lst[0][2]: inner_sets.setdefault(s, []).append(c)
    inner_list = list(inner_sets)
    out = []; n_start = [0]
    def rec_child(c, chosen):
        if c == len(kids):
            # assign owners
            def rec_inner(i, S, owners):
                if limit and len(out) >= limit: return
                if first_owner_only and len(out) > n_start[0]: return
                if i == len(inner_list):
                    out.append(([x[0] for x in chosen], dict(owners))); return
                s = inner_list[i]
                for c2 in inner_sets[s]:
                    j = chosen[c2][2][s]
                    if j in U_allowed[kids[c2][0]]:
                        S[c2].add(j); owners[s] = c2
                        rec_inner(i + 1, S, owners)
                        S[c2].discard(j)
            n0 = len(out); n_start[0] = n0
            rec_inner(0, [set(x[1]) for x in chosen], {})
            if first_owner_only:
                del out[n0 + 1:]
            return
        for o in opts[c]:
            rec_child(c + 1, chosen + [o])
            if limit and len(out) >= limit: return
    rec_child(0, [])
    return out

if __name__ == "__main__":
    ALL = {"pyr": frozenset(range(5))}
    for r in range(5):
        for Ut in itertools.combinations(range(4), r):
            Ut = frozenset(Ut)
            U = {"pyr": ALL["pyr"], "tet": Ut}
            ps = solutions("pyr", pyr_children(), U["pyr"], U, limit=1)
            res = {}
            for ax in [(0, 5), (1, 3), (2, 4)]:
                res[ax] = bool(solutions("tet", tetp_children(ax), Ut, U, limit=1))
            print("U(tetP) =", sorted(Ut), "pyramid table:", bool(ps), "tetP tables per axis (lines):", res)
