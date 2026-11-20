"""Choose the deal.II child tables for pyramids and pyramid-family tets in
parent-local vertex numbering (vertices, line midpoints in line order, quad
face centers), closest to Dominik's WIP pyramid table, and verify them:
face conditions (U(pyr) = all faces, U(tet in pyramid family) = {0, 1}),
inner face owners, and the t8code closure."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyramid_dealii_search as D
import pyramid_dealii_closure as C
P = D.P; key = P.key; mid = P.mid

PYR_LINES = [(0, 2), (1, 3), (0, 1), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4)]
TET_LINES = D.TET_EDGES
def local_points(kind):
    if kind == "pyr":
        v = D.PYR; pts = v + [mid(v[a], v[b]) for a, b in PYR_LINES] + [mid(*v[:4])]
    else:
        v = D.TET; pts = v + [mid(v[a], v[b]) for a, b in TET_LINES]
    return [key(p) for p in pts]

DOMINIK_PYR = [[0, 7, 5, 13, 9], [7, 1, 13, 6, 10], [5, 13, 2, 8, 11], [13, 6, 8, 3, 12],
               [11, 13, 9, 5], [13, 12, 10, 6], [9, 13, 10, 7], [13, 11, 12, 8],
               [9, 11, 10, 12, 13], [9, 10, 11, 12, 4]]
# tets: corner tets 0-3 like the tet-family table, pyramids with apex 4, 9
TETP_PREF = [[0, 4, 6, 7], [4, 1, 5, 8], [6, 5, 2, 9], [7, 8, 9, 3],
             [5, 6, 8, 7, 4], [5, 6, 8, 7, 9]]

U = {"pyr": frozenset(range(5)), "tet": frozenset({0, 1})}

def choose(kind, kids_sets, pref):
    lp = local_points(kind)
    on_parent = D.parent_face_children(kind)
    U_parent = U[kind]
    table, kinds, child_faces = [], [], []
    for (ck, pts), pr in zip(kids_sets, pref):
        best = None
        for cv in D.orders(ck, pts):
            idx = [lp.index(key(x)) for x in cv]
            fs = D.faces_of(ck, cv); forced = set(); ok = True
            for j, ft in enumerate(fs):
                s = frozenset(ft)
                if s in on_parent:
                    f, t = on_parent[s]
                    if f in U_parent:
                        if ft != t: ok = False; break
                        forced.add(j)
            if not ok or not forced <= U[ck]:
                continue
            # closeness to the preferred order: number of equal positions,
            # then lexicographic
            score = (-sum(a == b for a, b in zip(idx, pr)), idx)
            if best is None or score < best[0]:
                best = (score, idx, fs)
        assert best, (kind, pr)
        table.append(best[1]); kinds.append(ck); child_faces.append(best[2])
    # inner faces and owners
    inner = {}
    for c, fs in enumerate(child_faces):
        for j, ft in enumerate(fs):
            if frozenset(ft) not in on_parent:
                inner.setdefault(frozenset(ft), []).append((c, j))
    owners = []
    for s, lst in inner.items():
        assert len(lst) == 2
        cand = [(c, j) for c, j in lst if j in U[kinds[c]]]
        assert cand, (kind, lst)
        owners.append(min(cand))
    owners.sort()
    # inner lines: edges of children that are not on parent faces
    face_pts = set()
    return table, kinds, owners, lp

def inner_lines(kind, table, kinds, lp):
    on_parent = D.parent_face_children(kind)
    parent_face_vsets = [set(fc) for fc in on_parent]   # face children vertex sets
    ref = D.PYR if kind == "pyr" else D.TET
    pfaces = [set(key(x) for x in [ref[i] for i in fv]) for fv in P.DEALII[kind][1]]
    # a line is on a parent face if both end points are on the same parent
    # face plane: test with the face children vertex sets of that face
    fc_by_face = {}
    for s, (f, t) in on_parent.items():
        fc_by_face.setdefault(f, set()).update(s)
    lines = set()
    for idx, ck in zip(table, kinds):
        cref = D.PYR if ck == "pyr" else D.TET
        nl = 8 if ck == "pyr" else 6
        L = PYR_LINES if ck == "pyr" else TET_LINES
        for a, b in L:
            pa, pb = lp[idx[a]], lp[idx[b]]
            if not any(pa in fc_by_face[f] and pb in fc_by_face[f] and
                       any(pa in s and pb in s for s, (ff, t) in on_parent.items() if ff == f)
                       for f in fc_by_face):
                lines.add(tuple(sorted((idx[a], idx[b]))))
    return sorted(lines)

for kind, kids, pref in [("pyr", D.pyr_children(), DOMINIK_PYR),
                         ("tet", D.tetp_children((0, 5)), TETP_PREF)]:
    # order the geometric children like the preference (same vertex sets)
    lp = local_points(kind)
    sets = {frozenset(lp.index(key(x)) for x in pts): (ck, pts) for ck, pts in kids}
    kids_sorted = [sets[frozenset(p)] for p in pref]
    table, kinds, owners, lp = choose(kind, kids_sorted, pref)
    print(kind, "children:")
    for c, (idx, ck) in enumerate(zip(table, kinds)):
        print(f"  {c}: {ck} {idx}" + ("" if idx == pref[c] else f"   (preferred {pref[c]})"))
    print("  inner faces (child, face):", owners)
    print("  inner lines:", inner_lines(kind, table, kinds, lp))
    globals()[kind + "_table"] = [[lp[i] for i in idx] for idx in table]

r = C.closure(tuple(tuple(x) for x in pyr_table), tuple(tuple(x) for x in tet_table))
print("t8code closure:", "ok," if r else "FAILS", len(r) if r else "", "states")
