"""Generate and validate the 72-state deal.II <-> t8code tet lookup.

State = (t8 type, positive-volume deal.II-to-t8 vertex permutation).
For each state the script chooses the unique compatible central-octahedron
cut, then emits
  * tetrahedron_refinement_scheme[state]
  * tetrahedron_child_order[state][deal_child]
  * tetrahedron_child_type[state][deal_child]
The returned child type is again a state in [0,72), so it is carried down.
"""
from fractions import Fraction as Q
from itertools import permutations

T8_TYPES=(0,1,2,5,6,7)
EDGES=((0,1),(1,2),(0,2),(0,3),(1,3),(2,3))
SCHEMES=(
 ((0,4,6,7),(4,1,5,8),(6,5,2,9),(7,8,9,3),
    (5, 6, 4, 8),(7, 8, 4, 6),(9, 8, 7, 6),(6, 9, 8, 5)), 
 ((0,4,6,7),(4,1,5,8),(6,5,2,9),(7,8,9,3),
   (5, 6, 4, 7),(7, 8, 4, 5),(9, 7, 6, 5),(8, 9, 5, 7)),
 ((0,4,6,7),(4,1,5,8),(6,5,2,9),(7,8,9,3),
   (5, 6, 4, 9),(7, 8, 4, 9),(9, 7, 6, 4),(8, 9, 5, 4))
)

ILOC_TO_CHILDTYPE=(
 (0,0,1,5,0,2,6,0),(1,0,1,2,1,5,7,1),(2,2,6,7,0,1,2,2),
 (-1,)*8,(-1,)*8,(5,5,6,7,0,1,5,5),(6,0,2,6,5,6,7,6),
 (7,1,5,7,2,6,7,7))
ILOC_TO_CHILDCUBEID=(
 (0,1,1,1,3,3,3,7),(0,2,2,2,3,3,3,7),(0,1,1,1,5,5,5,7),
 (-1,)*8,(-1,)*8,(0,2,2,2,6,6,6,7),(0,4,4,4,5,5,5,7),
 (0,4,4,4,6,6,6,7))
CUBEID_TO_PARENTTYPE=(
 (0,0,1,0,6,2,5,0),(1,0,1,1,7,2,5,1),(2,2,1,0,6,2,7,2),
 (-1,)*8,(-1,)*8,(5,0,5,1,7,6,5,5),(6,2,5,0,6,6,7,6),
 (7,2,5,1,7,6,7,7))
CUBEID_TO_ILOC=(
 (0,1,1,4,1,4,4,7),(0,2,2,4,1,5,5,7),(0,1,3,5,2,6,4,7),
 (-1,)*8,(-1,)*8,(0,3,1,5,2,4,6,7),(0,2,2,6,3,5,5,7),
 (0,3,3,6,3,6,6,7))

TYPE_VERTICES={
 0:((0,0,0),(1,0,0),(1,1,0),(1,1,1)),
 1:((0,0,0),(0,1,0),(1,1,0),(1,1,1)),
 2:((0,0,0),(1,0,0),(1,0,1),(1,1,1)),
 5:((0,0,0),(0,1,0),(0,1,1),(1,1,1)),
 6:((0,0,0),(0,0,1),(1,0,1),(1,1,1)),
 7:((0,0,0),(0,0,1),(0,1,1),(1,1,1)),
}

def det6(v):
 a,b,c,d=v; u=tuple(b[i]-a[i] for i in range(3)); x=tuple(c[i]-a[i] for i in range(3)); y=tuple(d[i]-a[i] for i in range(3))
 return u[0]*(x[1]*y[2]-x[2]*y[1])-u[1]*(x[0]*y[2]-x[2]*y[0])+u[2]*(x[0]*y[1]-x[1]*y[0])

def positive(t,p): return det6(tuple(TYPE_VERTICES[t][p[i]] for i in range(4)))>0
POSITIVE_PERMS={t:tuple(p for p in permutations(range(4)) if positive(t,p)) for t in T8_TYPES}
assert all(len(POSITIVE_PERMS[t])==12 for t in T8_TYPES)
STATES=tuple((t,p) for t in T8_TYPES for p in POSITIVE_PERMS[t])
STATE_ID={s:i for i,s in enumerate(STATES)}
assert len(STATES)==len(STATE_ID)==72

def midpoint(a,b): return tuple((x+y)/2 for x,y in zip(a,b))
def origin(cid): return (cid&1,(cid>>1)&1,(cid>>2)&1)
def t8_child(t,i):
 ct=ILOC_TO_CHILDTYPE[t][i]; o=origin(ILOC_TO_CHILDCUBEID[t][i])
 return tuple(tuple(Q(o[d]+v[d],2) for d in range(3)) for v in TYPE_VERTICES[ct])
def deal_nodes(t,p):
 pv=[tuple(map(Q,TYPE_VERTICES[t][p[i]])) for i in range(4)]
 return tuple(pv+[midpoint(pv[a],pv[b]) for a,b in EDGES])

def transition(state,scheme):
 t,p=state; nodes=deal_nodes(t,p); tc=tuple(t8_child(t,i) for i in range(8)); order=[]; children=[]
 for child in scheme:
  dv=tuple(nodes[j] for j in child)
  matches=[i for i,tv in enumerate(tc) if frozenset(dv)==frozenset(tv)]
  if len(matches)!=1: return None
  i=matches[0]; ct=ILOC_TO_CHILDTYPE[t][i]
  cp=tuple(tc[i].index(v) for v in dv)
  if (ct,cp) not in STATE_ID: return None
  order.append(i); children.append(STATE_ID[(ct,cp)])
 return tuple(order),tuple(children)

def validate_inputs():
 for t in T8_TYPES:
  for i in range(8):
   ct=ILOC_TO_CHILDTYPE[t][i]; cid=ILOC_TO_CHILDCUBEID[t][i]
   assert CUBEID_TO_PARENTTYPE[ct][cid]==t
   assert CUBEID_TO_ILOC[ct][cid]==i
 # Every supplied deal.II child ordering must be positive in a positive parent.
 pv=((Q(0),Q(0),Q(0)),(Q(1),Q(0),Q(0)),(Q(0),Q(1),Q(0)),(Q(0),Q(0),Q(1)))
 nodes=pv+tuple(midpoint(pv[a],pv[b]) for a,b in EDGES)
 for scheme in SCHEMES:
  assert all(det6(tuple(nodes[j] for j in child))>0 for child in scheme)

def generate():
 validate_inputs(); cuts=[]; orders=[]; child_states=[]
 for sid,state in enumerate(STATES):
  matches=[]
  for cut,scheme in enumerate(SCHEMES):
   result=transition(state,scheme)
   if result is not None: matches.append((cut,result))
  assert len(matches)==1,(sid,state,matches)
  cut,(order,children)=matches[0]
  assert sorted(order)==list(range(8))
  cuts.append(cut); orders.append(order); child_states.append(children)
 assert [cuts.count(i) for i in range(3)]==[24,24,24]
 assert all(0<=x<72 for row in child_states for x in row)
 # Exhaustive closure check for several levels from every state.
 reachable=set(range(72))
 for _ in range(6):
  reachable={child_states[s][c] for s in reachable for c in range(8)}
  assert reachable<=set(range(72))
 return tuple(cuts),tuple(orders),tuple(child_states)

def cpp1(name,a): return f"static constexpr std::array<unsigned int, 72> {name} = {{{{\n  "+", ".join(map(str,a))+"\n}};"
def cpp2(name,a): return f"static constexpr std::array<std::array<unsigned int, 8>, 72> {name} = {{{{\n"+",\n".join("  {{"+", ".join(map(str,r))+"}}" for r in a)+"\n}};"

def invert(orders, types):
  inverted_list_of_lists_order = list()
  inverted_list_of_lists_types = list()

  for j in range(len(orders)):
    order = orders[j]
    dealii_type = types[j]

    inverted_list_order = list()
    inverted_list_type = list()
    for _ in range(len(order)):
      inverted_list_order.append(0)
      inverted_list_type.append(0)

    for dealii_child in range(len(order)):
      t8_code_child = order[dealii_child]
      inverted_list_order[t8_code_child] = dealii_child

      current_dealii_type = dealii_type[dealii_child]
      inverted_list_type[t8_code_child] = current_dealii_type

    inverted_list_of_lists_order.append(inverted_list_order)
    inverted_list_of_lists_types.append(inverted_list_type)
  return inverted_list_of_lists_order, inverted_list_of_lists_types

def main():
 cuts,orders_invered,types_invered=generate()
 orders, types = invert(orders_invered, types_invered) 
 
 print()
 print("// State = 12 * compact_t8_type + positive_permutation_index")
 print("// compact t8 type 0..5 maps to sparse t8 type {0,1,2,5,6,7}")
 print("// cut ids: 0 = diagonal 6-8, 1 = diagonal 5-7, 2 = diagonal 4-9")
 print()
 print(cpp1("tetrahedron_refinement_scheme",cuts)); print()
 print(cpp2("tetrahedron_child_order_dealii_to_t8code",orders_invered)); print()
 print(cpp2("tetrahedron_child_type_dealii_to_t8code",types_invered)); print()
 print(cpp2("tetrahedron_child_order_t8code_to_dealii",orders)); print()
 print(cpp2("tetrahedron_child_type_t8code_to_dealii",types)); print()
 print("// Validation passed: 72 states; 24 per cut; exact geometric matching; closed child states.")

if __name__=="__main__": main()