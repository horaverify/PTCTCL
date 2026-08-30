// dbm_tests.cpp — unit tests for the DBM zone algebra.
#include <cstdio>

#include "hora/zone.hpp"
#include "hora/reach.hpp"
#include "hora/ta.hpp"
#include "hora/detail/net_semantics.hpp"

namespace {
int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
using namespace hora;

// 1-clock zone [lo,hi]  (hi<0 => unbounded above).
Dbm zone1(int lo, int hi) {
  Dbm z(1);
  z.delay();                       // x >= 0
  if (hi >= 0) z.constrain(1, 0, le(hi));
  z.constrain(0, 1, le(-lo));      // x >= lo
  z.close();
  return z;
}

void test_basic() {
  Dbm z = zone1(0, 5);
  CHECK(!z.empty());
  CHECK(z.get(1, 0) == le(5));      // x <= 5
  CHECK(z.get(0, 1) == le(0));      // x >= 0

  Dbm e(1);                         // empty: x>=6 and x<=5
  e.delay();
  e.constrain(1, 0, le(5));
  bool ok = e.constrain(0, 1, le(-6));
  CHECK(!ok);
  CHECK(e.empty());
}

void test_reset_delay() {
  Dbm z = zone1(2, 5);              // x in [2,5]
  z.reset(1);                       // x := 0
  CHECK(z.get(1, 0) == le(0) && z.get(0, 1) == le(0));  // x == 0
  z.delay();                        // x >= 0
  CHECK(z.get(1, 0) == kInf);       // unbounded above
  CHECK(z.get(0, 1) == le(0));      // x >= 0
}

void test_inclusion() {
  Dbm a = zone1(0, 5), b = zone1(3, 5);
  CHECK(a.includes(b));    // [0,5] ⊇ [3,5]
  CHECK(!b.includes(a));   // [3,5] ⊉ [0,5]
  CHECK(a.includes(a));
}

void test_extrapolate() {
  Dbm z = zone1(0, 5);
  std::vector<std::int32_t> maxc = {0, 4};  // max constant for clock 1 is 4
  z.extrapolate(maxc);
  CHECK(z.get(1, 0) == kInf);       // 5 > 4  => upper bound removed
  CHECK(z.get(0, 1) == le(0));      // lower bound 0 kept
}

void test_two_clocks_diagonal() {
  // clocks x=1, y=2 ; delay ; x<=5 ; y>=2 ; x-y<=1
  Dbm z(2);
  z.delay();
  CHECK(z.constrain(1, 0, le(5)));   // x <= 5
  CHECK(z.constrain(0, 2, le(-2)));  // y >= 2
  CHECK(z.constrain(1, 2, le(1)));   // x - y <= 1
  CHECK(!z.empty());
  // x - y <= 1 and y >= 2 imply x <= ... but x also <= 5; canonical form tightens.
  // Contradiction check: add y - x <= -5 (i.e. x - y >= 5) -> empty with x-y<=1.
  Dbm w = z;
  bool ok = w.constrain(2, 1, le(-5));  // y - x <= -5  => x - y >= 5, clashes with <=1
  CHECK(!ok && w.empty());
}

void test_reachability() {
  // A --[x>=5]--> B ; A invariant x<=10 ; C unreachable. clock x = index 1.
  TimedAutomaton ta;
  ta.nclocks = 1;
  ta.clock_names = {"0", "x"};
  Location A; A.name = "A"; A.invariant = {c_le(1, 10)};
  Location B; B.name = "B";
  Location C; C.name = "C";
  ta.locations = {A, B, C};
  Edge e; e.src = 0; e.tgt = 1; e.guard = {c_ge(1, 5)};
  ta.edges = {e};
  ta.initial = 0;
  ta.compute_max_constants();

  CHECK(reachable(ta, 1));                 // B reachable
  CHECK(!reachable(ta, 2));                // C unreachable
  CHECK(!reachable(ta, 1, {c_le(1, 3)}));  // B with x<=3 impossible (entered x>=5)
  CHECK(reachable(ta, 1, {c_le(1, 8)}));   // B with x<=8 : x in [5,8]
  CHECK(reachable(ta, 1, {c_ge(1, 5)}));   // B with x>=5

  // Toggling loop with resets: Off <-> On, On invariant x<=3, resets on both edges.
  TimedAutomaton t2;
  t2.nclocks = 1;
  t2.clock_names = {"0", "x"};
  Location off; off.name = "Off";
  Location on; on.name = "On"; on.invariant = {c_le(1, 3)};
  t2.locations = {off, on};
  Edge e1; e1.src = 0; e1.tgt = 1; e1.guard = {c_ge(1, 2)}; e1.resets = {1};
  Edge e2; e2.src = 1; e2.tgt = 0; e2.guard = {c_ge(1, 1)}; e2.resets = {1};
  t2.edges = {e1, e2};
  t2.initial = 0;
  t2.compute_max_constants();

  ReachStats st;
  CHECK(reachable(t2, 1, {}, &st));        // On reachable (terminates via zones)
  CHECK(reachable(t2, 1, {c_le(1, 3)}));   // On with x<=3
  CHECK(!reachable(t2, 1, {c_gt(1, 3)}));  // On invariant x<=3 forbids x>3
  CHECK(st.explored > 0);

  // sequential and OpenMP engines must agree on every query
  for (int L = 0; L < 3; ++L)
    for (int b = 0; b <= 12; ++b) {
      CHECK(reachable(ta, L, {c_le(1, b)}, nullptr, Backend::Seq) ==
            reachable(ta, L, {c_le(1, b)}, nullptr, Backend::Omp));
      CHECK(reachable(t2, L < 2 ? L : 0, {c_le(1, b)}, nullptr, Backend::Seq) ==
            reachable(t2, L < 2 ? L : 0, {c_le(1, b)}, nullptr, Backend::Omp));
    }
}

}  // namespace

// ---- delay_avoiding: time may pass, but never through the convex region `forbid` ----
//
// The operator behind a liveness predicate that must be AVOIDED rather than maintained, i.e.
// the complement of a conjunction, which is a disjunction and so is not an invariant.
void test_delay_avoiding() {
  using detail::delay_avoiding;
  auto members = [](const Federation& f) {  // total measure as [lo,hi] pairs, for 1 clock
    std::vector<std::pair<int,int>> v;
    for (const auto& z : f) if (!z.empty()) v.push_back({-raw_bound(z.get(0,1)), raw_bound(z.get(1,0))});
    return v;
  };
  auto covers = [](const Federation& f, const Dbm& probe) { return fed_covers(f, probe); };

  Dbm forbid = zone1(5, 10);              // avoid 5 <= x <= 10
  Dbm fup = forbid; fup.delay();          // forbid^ = x >= 5
  Federation out;

  // From x = 0: may delay up to but NOT into [5,10]; x=4 reachable, x=5 and x=12 are not.
  delay_avoiding(zone1(0, 0), forbid, fup, out);
  CHECK(!out.empty());
  CHECK(covers(out, zone1(4, 4)));
  CHECK(!covers(out, zone1(5, 5)));
  CHECK(!covers(out, zone1(12, 12)));

  // From x = 12 -- already past the obstacle -- time is unconstrained.
  delay_avoiding(zone1(12, 12), forbid, fup, out);
  CHECK(covers(out, zone1(12, 12)));
  CHECK(covers(out, zone1(99, 99)));
  CHECK(!covers(out, zone1(7, 7)));

  // A zone straddling the obstacle keeps both sides and joins neither: the result is genuinely
  // non-convex, which is the whole reason this returns a federation.
  delay_avoiding(zone1(0, 20), forbid, fup, out);
  CHECK(covers(out, zone1(4, 4)));
  CHECK(covers(out, zone1(11, 11)));
  CHECK(covers(out, zone1(99, 99)));
  CHECK(!covers(out, zone1(5, 5)));
  CHECK(!covers(out, zone1(10, 10)));
  CHECK(out.size() >= 2);                 // no single zone denotes that set

  // Degenerate: nothing forbidden is an ordinary delay.
  Dbm none(1);                            // empty
  delay_avoiding(zone1(3, 3), none, none, out);
  CHECK(out.size() == 1);
  CHECK(covers(out, zone1(99, 99)));
  (void)members;
}

// ---- pre_avoiding: the exact inverse of delay_avoiding ----
void test_pre_avoiding() {
  using detail::pre_avoiding;
  using detail::delay_avoiding;
  auto covers = [](const Federation& f, const Dbm& probe) { return fed_covers(f, probe); };

  Dbm forbid = zone1(5, 10);                 // avoid 5 <= x <= 10
  Dbm fup = forbid; fup.delay();
  Federation out;

  // Landing at x = 12 is possible only from strictly after the obstacle.
  pre_avoiding(zone1(12, 12), forbid, fup, out);
  CHECK(covers(out, zone1(11, 11)));
  CHECK(covers(out, zone1(12, 12)));
  CHECK(!covers(out, zone1(10, 10)));        // inside forbid
  CHECK(!covers(out, zone1(3, 3)));          // would have to cross forbid

  // Landing at x = 3 is possible only from before it.
  pre_avoiding(zone1(3, 3), forbid, fup, out);
  CHECK(covers(out, zone1(0, 0)));
  CHECK(covers(out, zone1(3, 3)));
  CHECK(!covers(out, zone1(4, 4)));          // 4 > 3: time does not run backwards
  CHECK(!covers(out, zone1(12, 12)));

  // A target inside the forbidden region has no predecessor at all.
  pre_avoiding(zone1(7, 7), forbid, fup, out);
  CHECK(out.empty() || !covers(out, zone1(7, 7)));

  // Round trip: every point delay_avoiding reaches from 0 has 0 as a predecessor.
  Federation fwd;
  delay_avoiding(zone1(0, 0), forbid, fup, fwd);
  for (const auto& z : fwd) {
    if (z.empty()) continue;
    Federation back;
    pre_avoiding(z, forbid, fup, back);
    CHECK(covers(back, zone1(0, 0)));
  }

  // Degenerate: nothing forbidden is an ordinary time-predecessor.
  Dbm none(1);
  pre_avoiding(zone1(4, 4), none, none, out);
  CHECK(out.size() == 1);
  CHECK(covers(out, zone1(0, 0)));
}

int main() {
  test_delay_avoiding();
  test_pre_avoiding();
  test_basic();
  test_reset_delay();
  test_inclusion();
  test_extrapolate();
  test_two_clocks_diagonal();
  test_reachability();
  std::printf("dbm+reach tests: %d checks, %d failures\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
