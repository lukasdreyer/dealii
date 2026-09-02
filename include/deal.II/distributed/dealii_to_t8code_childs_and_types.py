from fractions import Fraction as Q
from itertools import permutations


EDGES = (
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 3),
    (1, 3),
    (2, 3),
)

# 6-8 deal.II refinement.
SCHEME = (
    (0, 4, 6, 7),
    (4, 1, 5, 8),
    (6, 5, 2, 9),
    (7, 8, 9, 3),

#    (5, 6, 4, 8),
#    (7, 8, 4, 6),
#    (9, 8, 7, 6),
#    (6, 9, 8, 5)

   
     (4, 8, 5, 6),
     (4, 6, 7, 8),
     (8, 9, 6, 7),
     (5, 8, 9, 6)
)


T8_TYPES = (0, 1, 2, 5, 6, 7)

ILOC_TO_CHILDTYPE = (
    (0, 0, 1, 5, 0, 2, 6, 0),
    (1, 0, 1, 2, 1, 5, 7, 1),
    (2, 2, 6, 7, 0, 1, 2, 2),
    (-1,) * 8,
    (-1,) * 8,
    (5, 5, 6, 7, 0, 1, 5, 5),
    (6, 0, 2, 6, 5, 6, 7, 6),
    (7, 1, 5, 7, 2, 6, 7, 7),
)

ILOC_TO_CHILDCUBEID = (
    (0, 1, 1, 1, 3, 3, 3, 7),
    (0, 2, 2, 2, 3, 3, 3, 7),
    (0, 1, 1, 1, 5, 5, 5, 7),
    (-1,) * 8,
    (-1,) * 8,
    (0, 2, 2, 2, 6, 6, 6, 7),
    (0, 4, 4, 4, 5, 5, 5, 7),
    (0, 4, 4, 4, 6, 6, 6, 7),
)

TYPE_VERTICES = {
    0: (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
    ),
    1: (
        (0, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, 1, 1),
    ),
    2: (
        (0, 0, 0),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 1),
    ),
    5: (
        (0, 0, 0),
        (0, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
    ),
    6: (
        (0, 0, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
    ),
    7: (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ),
}


def det6(v):
    a, b, c, d = v

    u = tuple(b[i] - a[i] for i in range(3))
    x = tuple(c[i] - a[i] for i in range(3))
    y = tuple(d[i] - a[i] for i in range(3))

    return (
          u[0] * (x[1] * y[2] - x[2] * y[1])
        - u[1] * (x[0] * y[2] - x[2] * y[0])
        + u[2] * (x[0] * y[1] - x[1] * y[0])
    )


def positive(t, p):
    return det6(tuple(TYPE_VERTICES[t][p[i]] for i in range(4))) > 0

def diagonal_preserving(p):
    assert(len(p)==4)
    return (p[0] in (0, 2) and p[2] in (0, 2)) or \
           (p[1] in (0, 2) and p[3] in (0, 2))

# The 12 positive permutations of every t8 type.
ALLOWED_PERMS = {
    t: tuple( p for p in permutations(range(4))
        if positive(t, p) and diagonal_preserving(p)
            ) for t in T8_TYPES
}

STATES = tuple(
    (t, p)
    for t in T8_TYPES
    for p in ALLOWED_PERMS[t]
)

STATE_ID = {
    state: i
    for i, state in enumerate(STATES)
}

N_STATES = 24
assert len(STATES) == N_STATES

def midpoint(a,b): return tuple((x+y)/2 for x,y in zip(a,b))
def origin(cid): return (cid&1,(cid>>1)&1,(cid>>2)&1)
def t8_child(t,i):
 ct=ILOC_TO_CHILDTYPE[t][i]; o=origin(ILOC_TO_CHILDCUBEID[t][i])
 return tuple(tuple(Q(o[d]+v[d],2) for d in range(3)) for v in TYPE_VERTICES[ct])
def deal_nodes(t,p):
 pv=[tuple(map(Q,TYPE_VERTICES[t][p[i]])) for i in range(4)]
 return tuple(pv+[midpoint(pv[a],pv[b]) for a,b in EDGES])

def invert(p):
 p_inv = [0] * len(p)
 for i, j in enumerate(p):
  p_inv[j] = i
 return tuple(p_inv)

def transition(state):
 t,p=state; nodes=deal_nodes(t,p); tc=tuple(t8_child(t,i) for i in range(8)); order=[]; children=[]
 for child in SCHEME:
  dv=tuple(nodes[j] for j in child) # vertices of dealii child
  matches=[i for i,tv in enumerate(tc) if frozenset(dv)==frozenset(tv)]
  assert(len(matches)==1)
  i=matches[0]; ct=ILOC_TO_CHILDTYPE[t][i]
  cp=tuple(tc[i].index(v) for v in dv)
  assert ((ct,cp) in STATE_ID)
  order.append(i); children.append(STATE_ID[(ct,cp)])
 return tuple(order),tuple(children)

def generate():
 orders=[]; child_states=[]
 for state in STATES:
  order, children =transition(state)
  assert sorted(order)==list(range(8))
  assert all(0<=x<N_STATES for x in children)
  orders.append(invert(order)); child_states.append(children)
 return tuple(orders),tuple(child_states)



def cpp2(name,a): return f"static constexpr std::array<std::array<unsigned int, 8>, {N_STATES}> {name} = {{{{\n"+",\n".join("  {{"+", ".join(map(str,r))+"}}" for r in a)+"\n}};"



def main():
 orders,types=generate()
 
 print()
 print("// State = 4 * compact_t8_type + allowed_permutation_index")
 print("// compact t8 type 0..5 maps to sparse t8 type {0,1,2,5,6,7}")
 print()
 print(cpp2("tetrahedron_child_order_t8code_to_dealii",orders)); print()
 print(cpp2("tetrahedron_child_type_dealii_to_dealii",types)); print()
 print()

if __name__=="__main__": main()
