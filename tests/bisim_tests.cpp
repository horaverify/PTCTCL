// Validation of the timed-bisimulation engine against the 7 hand-verified examples.
#include "hora/bisim.hpp"
#include <cstdio>
using namespace hora;
enum { a = 0, b = 1, c = 2, tau = 9 };
static Constraint ge(int clk, int k) { return c_ge(clk, k); }  // x_clk >= k
static Constraint leq(int clk, int k) { return c_le(clk, k); }  // x_clk <= k
using Inv = std::vector<std::vector<Constraint>>;

struct Case { const char* name; BAut A, B; std::set<int> obs;
              bool sBi, sAB, sBA, wBi, wAB, wBA; };

static int pass = 0, fail = 0;
static void chk(const char* name, const char* what, bool got, bool exp) {
  if (got == exp) pass++;
  else { fail++; printf("  FAIL %-6s %-8s got=%d exp=%d\n", name, what, got, exp); }
}

int main() {
  std::vector<Case> cases;
  // #1 identical: l0 -a-> l1
  { BAut A; A.nclocks=1; A.nlocs=2; A.edges={{0,1,{},{},a}}; BAut B=A;
    cases.push_back({"1",A,B,{a}, true,true,true, true,true,true}); }
  // #2 timing gap: a,x>=2  vs  a,x>=3
  { BAut A; A.nclocks=1; A.nlocs=2; A.edges={{0,1,{ge(1,2)},{},a}};
    BAut B; B.nclocks=1; B.nlocs=2; B.edges={{0,1,{ge(1,3)},{},a}};
    cases.push_back({"2",A,B,{a}, false,false,true, false,false,true}); }
  // #3 extra internal step: a  vs  t;a
  { BAut A; A.nclocks=1; A.nlocs=2; A.edges={{0,1,{},{},a}};
    BAut B; B.nclocks=1; B.nlocs=3; B.edges={{0,1,{},{},tau},{1,2,{},{},a}};
    cases.push_back({"3",A,B,{a}, false,false,false, true,true,true}); }
  // #4 tau before guarded delay: a,x>=2  vs  t,x>=1;a,x>=2
  { BAut A; A.nclocks=1; A.nlocs=2; A.edges={{0,1,{ge(1,2)},{},a}};
    BAut B; B.nclocks=1; B.nlocs=3; B.edges={{0,1,{ge(1,1)},{},tau},{1,2,{ge(1,2)},{},a}};
    cases.push_back({"4",A,B,{a}, false,false,false, true,true,true}); }
  // #5 reset asymmetry: a[reset x]; b,x>=1   vs   a; b,x>=1
  { BAut A; A.nclocks=1; A.nlocs=3; A.edges={{0,1,{},{1},a},{1,2,{ge(1,1)},{},b}};
    BAut B; B.nclocks=1; B.nlocs=3; B.edges={{0,1,{},{},a},{1,2,{ge(1,1)},{},b}};
    cases.push_back({"5",A,B,{a,b}, false,true,false, false,true,false}); }
  // #6 weak-sim asymmetry: a  vs  a;t;c
  { BAut A; A.nclocks=1; A.nlocs=2; A.edges={{0,1,{},{},a}};
    BAut B; B.nclocks=1; B.nlocs=4; B.edges={{0,1,{},{},a},{1,2,{},{},tau},{2,3,{},{},c}};
    cases.push_back({"6",A,B,{a,c}, false,true,false, false,true,false}); }
  // #7 a.b+a.c  vs  a.(b+c)
  { BAut A; A.nclocks=1; A.nlocs=5; A.edges={{0,1,{},{},a},{1,2,{},{},b},{0,3,{},{},a},{3,4,{},{},c}};
    BAut B; B.nclocks=1; B.nlocs=4; B.edges={{0,1,{},{},a},{1,2,{},{},b},{1,3,{},{},c}};
    cases.push_back({"7",A,B,{a,b,c}, false,true,false, false,true,false}); }

  // ---- location invariants -----------------------------------------------------------
  // #8 responder's invariant stops time: B may not sit at l0 past x=1, A may.
  { BAut A; A.nclocks=1; A.nlocs=2; A.edges={{0,1,{},{},a}};
    BAut B=A; B.invariants=Inv{{leq(1,1)},{}};
    cases.push_back({"8",A,B,{a}, false,false,true, false,false,true}); }
  // #9 challenger's own invariant bounds the delays it can challenge with, so the guard
  //    x<=1 on B's edge is enough: A<=B holds, and only because `avail` is A's invariant.
  { BAut A; A.nclocks=1; A.nlocs=2; A.edges={{0,1,{},{},a}}; A.invariants=Inv{{leq(1,1)},{}};
    BAut B; B.nclocks=1; B.nlocs=2; B.edges={{0,1,{leq(1,1)},{},a}};
    cases.push_back({"9",A,B,{a}, false,true,false, false,true,false}); }
  // #10 the target invariant kills the edge: guard x>=2 lands in l1 with x<=1, so A's `a`
  //     is not a transition at all and A is just a clock — same as the edgeless B.
  { BAut A; A.nclocks=1; A.nlocs=2; A.edges={{0,1,{ge(1,2)},{},a}}; A.invariants=Inv{{},{leq(1,1)}};
    BAut B; B.nclocks=1; B.nlocs=2;
    cases.push_back({"10",A,B,{a}, true,true,true, true,true,true}); }
  // #11 a responder τ-chain may not pass THROUGH a location whose invariant the valuation
  //     violates: B reaches its `a` only via l1 (x<=1), so past x=1 B is stuck at l0.
  { BAut A; A.nclocks=1; A.nlocs=2; A.edges={{0,1,{},{},a}};
    BAut B; B.nclocks=1; B.nlocs=4;
    B.edges={{0,1,{},{},tau},{1,2,{},{},tau},{2,3,{},{},a}};
    B.invariants=Inv{{},{leq(1,1)},{},{}};
    cases.push_back({"11",A,B,{a}, false,false,false, false,false,true}); }

  for (auto& t : cases) {
    chk(t.name, "sBi",  decide_bisim(t.A,t.B,BisimKind::StrongBisim),          t.sBi);
    chk(t.name, "sA<=B",decide_bisim(t.A,t.B,BisimKind::StrongSim),            t.sAB);
    chk(t.name, "sB<=A",decide_bisim(t.B,t.A,BisimKind::StrongSim),            t.sBA);
    chk(t.name, "wBi",  decide_bisim(t.A,t.B,BisimKind::WeakBisim,t.obs),      t.wBi);
    chk(t.name, "wA<=B",decide_bisim(t.A,t.B,BisimKind::WeakSim,t.obs),        t.wAB);
    chk(t.name, "wB<=A",decide_bisim(t.B,t.A,BisimKind::WeakSim,t.obs),        t.wBA);
  }
  // Independent sanity: reflexivity + bisim⟹sim on a non-trivial cyclic automaton.
  { BAut C; C.nclocks=1; C.nlocs=3;
    C.edges={{0,1,{ge(1,2)},{1},a},{1,2,{},{},b},{2,0,{ge(1,1)},{1},a},{1,0,{},{},tau}};
    chk("refl","A~A",  decide_bisim(C,C,BisimKind::StrongBisim),         true);
    chk("refl","wA~A", decide_bisim(C,C,BisimKind::WeakBisim,{a,b}),     true);
    chk("refl","A<=A", decide_bisim(C,C,BisimKind::StrongSim),           true);
    chk("refl","wA<=A",decide_bisim(C,C,BisimKind::WeakSim,{a,b}),       true);
  }
  // bisim ⟹ sim both ways (example 1 is bisimilar) and NOT for a non-bisimilar pair.
  { BAut A; A.nclocks=1; A.nlocs=2; A.edges={{0,1,{ge(1,2)},{},a}};
    BAut B; B.nclocks=1; B.nlocs=2; B.edges={{0,1,{ge(1,2)},{},a}};
    bool bi=decide_bisim(A,B,BisimKind::StrongBisim);
    bool ab=decide_bisim(A,B,BisimKind::StrongSim), ba=decide_bisim(B,A,BisimKind::StrongSim);
    chk("impl","bi=>ab", !bi||ab, true); chk("impl","bi=>ba", !bi||ba, true);
    chk("impl","selfbi", bi, true);  // same automaton twice must be bisimilar
  }
  // Reflexivity with invariants — the identity IS a bisimulation, so all four kinds must
  // hold on a cyclic automaton whose every location constrains time.
  { BAut C; C.nclocks=1; C.nlocs=3;
    C.edges={{0,1,{ge(1,2)},{1},a},{1,2,{},{},b},{2,0,{ge(1,1)},{1},a},{1,0,{},{},tau}};
    C.invariants=Inv{{},{leq(1,3)},{leq(1,5)}};
    chk("invrefl","A~A",  decide_bisim(C,C,BisimKind::StrongBisim),     true);
    chk("invrefl","wA~A", decide_bisim(C,C,BisimKind::WeakBisim,{a,b}), true);
    chk("invrefl","A<=A", decide_bisim(C,C,BisimKind::StrongSim),       true);
    chk("invrefl","wA<=A",decide_bisim(C,C,BisimKind::WeakSim,{a,b}),   true); }
  // An invariant the origin violates leaves the automaton with no initial state; the engine
  // reports `false` rather than treating the pair as vacuously related (see the spec).
  { BAut A; A.nclocks=1; A.nlocs=1; A.invariants=Inv{{ge(1,1)}};
    chk("noinit","A~A", decide_bisim(A,A,BisimKind::StrongBisim), false); }
  // The exposed relation must agree with the boolean it is derived from.
  { BAut A; A.nclocks=1; A.nlocs=2; A.edges={{0,1,{},{},a}};
    BAut B=A; B.invariants=Inv{{leq(1,1)},{}};
    bool r1=false, r2=false;
    auto RA = bisim_relation(A,B,BisimKind::StrongSim,{},&r1);
    auto RB = bisim_relation(B,A,BisimKind::StrongSim,{},&r2);
    chk("rel","size", RA.size()==static_cast<std::size_t>(A.nlocs)*B.nlocs, true);
    chk("rel","A<=B", r1, decide_bisim(A,B,BisimKind::StrongSim));
    chk("rel","B<=A", r2, decide_bisim(B,A,BisimKind::StrongSim));
    chk("rel","nonempty", !RB[B.initial*A.nlocs+A.initial].empty(), true); }
  printf("bisim validation: %d passed, %d failed\n", pass, fail);
  return fail ? 1 : 0;
}
