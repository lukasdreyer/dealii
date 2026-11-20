"""With the rotated middle triangle child: find fixed choices per (t8 type, p)
(child vertex orders and owners of the inner faces), valid for the largest
set of standard faces S that occurs for that state, starting from a coarse
pyramid (type 0, identity) that sees all its faces in standard orientation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyramid_orient as P
import pyramid_orient_sides as M
from pyramid_search_tri_tables_sides import regeometry
P.TRI_TABLE = [(0, 3, 5), (3, 1, 4), (5, 4, 2), (4, 5, 3)]
regeometry()
feas = M.fixpoint()

def choose(state, S):
    """A choice at (state, S) whose children states are feasible: list of
    (child type, q, S_child) and the owner of each inner face."""
    choices, inner, face_sets = M.child_choices(state, S)
    n = len(choices); assigned = [set() for _ in range(n)]
    def child_q(c):
        ct, lst = choices[c]
        for q, forced, inner_j in lst:
            Sc = frozenset(set(forced) | {inner_j[fs] for fs in assigned[c]})
            if (ct, q, Sc) in feas:
                return (ct, q, Sc)
        return None
    def bt(i):
        if i == len(inner):
            return True
        for c in face_sets[inner[i]]:
            assigned[c].add(inner[i])
            if child_q(c) and bt(i + 1):
                return True
            assigned[c].discard(inner[i])
        return False
    if not all(child_q(c) for c in range(n)) or not bt(0):
        return None
    return [child_q(c) for c in range(n)]

smax = {(0, (0,1,2,3,4)): frozenset(range(5))}
chosen = {}
changed = True
while changed:
    changed = False
    for state, S in list(smax.items()):
        if chosen.get(state, (None,))[0] == S:
            continue
        r = choose(state, S)
        if r is None:
            print("NOT REALIZABLE at", state, sorted(S)); sys.exit(1)
        chosen[state] = (S, r)
        for ct, q, Sc in r:
            old = smax.get((ct, q), frozenset())
            new = old | Sc
            if new != old:
                smax[(ct, q)] = new; changed = True
        changed = True if chosen[state][0] != S else changed
print("realizable with fixed choices per state:", len(chosen), "states reached")
for t in P.T8_VERTICES:
    ss = [(s, sorted(chosen[s][0])) for s in chosen if s[0] == t]
    print(f" type {t} ({P.KIND[t]}): {len(ss)} states, largest standard face sets:", [x[1] for x in ss])
