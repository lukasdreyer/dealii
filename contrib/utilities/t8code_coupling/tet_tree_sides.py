"""Tetrahedron trees with a given triangle face child table: is there a
closed set of states (t8 tet type, p; left-handed trees, diagonal (6,8)
preserving) with fixed child orders and inner face owners such that every
face is standard from one side (S-model, coarse cells see all faces in
standard orientation)?"""
import sys, os, itertools
from fractions import Fraction as Q
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("gt", os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_tet_mirrored.py"))
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)

FACES = [(0,1,2),(1,0,3),(0,2,3),(2,1,3)]   # deal.II tetrahedron faces
def mid(a,b): return tuple((x+y)/2 for x,y in zip(a,b))
def face_children(fv, table):
    n = list(fv) + [mid(fv[0],fv[1]), mid(fv[1],fv[2]), mid(fv[2],fv[0])]
    return [tuple(n[i] for i in t) for t in table]

def analyse(table):
    states = list(G.STATES)
    geo = {}
    for (t, p) in states:
        nodes = G.deal_nodes(t, p)                 # parent in t8 coordinates, deal.II order
        pv = nodes[:4]
        pfc = {}
        for f, fv in enumerate(FACES):
            for k, c in enumerate(face_children([pv[i] for i in fv], table)):
                pfc[frozenset(c)] = (f, k, c)
        kids = []
        for i in range(8):
            ct = G.ILOC_TO_CHILDTYPE[t][i]
            tc = G.t8_child(t, i)
            opts = []
            for cp in G.ALLOWED_PERMS[ct]:          # deal.II vertex k of the child = t8 vertex cp[k]
                cverts = [tc[cp[k]] for k in range(4)]
                opts.append(((ct, cp), [tuple(cverts[j] for j in fv) for fv in FACES]))
            kids.append(opts)
        fsets = {}
        for c, opts in enumerate(kids):
            for ft in opts[0][1]:
                fsets.setdefault(frozenset(ft), []).append(c)
        inner = [fs for fs, cs in fsets.items() if fs not in pfc]
        geo[(t, p)] = (pfc, kids, inner, fsets)

    allS = [frozenset(c) for r in range(5) for c in itertools.combinations(range(4), r)]
    feas = {(s, S) for s in states for S in allS}
    def ok(s, S, feas):
        pfc, kids, inner, fsets = geo[s]
        opts = []
        for c, lst in enumerate(kids):
            good = []
            for cs, faces in lst:
                forced, inner_j, fine = set(), {}, True
                for j, ft in enumerate(faces):
                    fs = frozenset(ft)
                    if fs in pfc:
                        f, k, c_ = pfc[fs]
                        if f in S:
                            if ft != c_: fine = False; break
                            forced.add(j)
                    else:
                        inner_j[fs] = j
                if fine: good.append((cs, forced, inner_j))
            opts.append(good)
        assigned = [set() for _ in kids]
        def cq(c):
            return any((cs, frozenset(forced | {ij[fs] for fs in assigned[c]})) in feas for cs, forced, ij in opts[c])
        if not all(cq(c) for c in range(len(kids))): return False
        def bt(i):
            if i == len(inner): return True
            for c in fsets[inner[i]]:
                assigned[c].add(inner[i])
                if cq(c) and bt(i+1): return True
                assigned[c].discard(inner[i])
            return False
        return bt(0)
    changed = True
    while changed:
        changed = False
        for x in sorted(feas, key=lambda x: -len(x[1])):
            if not ok(x[0], x[1], feas):
                feas.discard(x); changed = True
    return feas

for name, table in [("current", [(0,3,5),(3,1,4),(5,4,2),(3,4,5)]), ("rotated middle child", [(0,3,5),(3,1,4),(5,4,2),(4,5,3)])]:
    feas = analyse(table)
    full = sum(1 for (s, S) in feas if len(S) == 4)
    root = (G.STATES[0], frozenset(range(4))) in feas
    print(f"{name}: {len(feas)} feasible (state, S); states feasible with all faces standard: {full} of 24; root state with all faces: {root}")
