// net_semantics.hpp — the transition semantics of a network of timed automata.
//
// What a transition IS, and what a successor zone IS: guard evaluation, the combined location
// invariant, the frozen (committed/urgent) test, the active-clock reduction, extrapolation, and
// the broadcast receiver/abstain enumeration. Everything above this layer -- the forward
// reachability search in netreach.cpp and the backward greatest-fixpoint liveness engine in
// netliveness.cpp -- is built out of exactly these pieces, and the two MUST agree on them: the
// broadcast-abstain bug in tests/models/liveness-regressions/bcast-receiver.xml was one copy of
// this logic drifting from another.
//
// Everything here is `inline` and lives in the header ON PURPOSE. The build has no LTO, so a
// definition in a .cpp would stop inlining at the translation-unit boundary -- and these are the
// innermost bodies of both engines, called once per generated successor. Each translation unit
// gets its own inlinable copy instead.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ptctlmc/fed.hpp"
#include "ptctlmc/network.hpp"
#include "ptctlmc/netreach.hpp"
#include "ptctlmc/detail/sym_state.hpp"

namespace ptctlmc {
namespace detail {

inline bool apply_all(Dbm& z, const std::vector<Constraint>& cs) {
  for (const auto& c : cs)
    if (!z.constrain(c.i, c.j, c.bound)) return false;
  return true;
}

inline bool int_ok(const std::vector<IntAtom>& g, const IntVec& ints) {
  for (const auto& a : g) {
    const long x = ints[a.var];
    bool r = false;
    switch (a.op) {
      case IntOp::Eq: r = x == a.val; break;
      case IntOp::Ne: r = x != a.val; break;
      case IntOp::Lt: r = x < a.val; break;
      case IntOp::Le: r = x <= a.val; break;
      case IntOp::Gt: r = x > a.val; break;
      case IntOp::Ge: r = x >= a.val; break;
    }
    if (!r) return false;
  }
  return true;
}

// Fills `inv` (cleared first) rather than returning a fresh vector: this runs
// 1-3x per successor in the hottest loop, so the allocation is pooled away by
// reusing a caller-owned scratch buffer.
inline void all_invariants(const Network& net, const LocVec& locs, std::vector<Constraint>& inv) {
  inv.clear();
  for (int p = 0; p < net.nprocs(); ++p)
    for (const auto& c : net.procs[p].locations[locs[p]].invariant) inv.push_back(c);
}

inline void frozen_flags(const Network& net, const LocVec& locs, bool* committed, bool* urgent) {
  *committed = *urgent = false;
  for (int p = 0; p < net.nprocs(); ++p) {
    const Location& l = net.procs[p].locations[locs[p]];
    if (l.committed) *committed = true;
    if (l.urgent) *urgent = true;
  }
}

// Reset clocks no process will read before resetting AND that the query does not
// read (qmask) — collapses zones that differ only in dead clocks, soundly.
inline void reset_inactive(const Network& net, const LocVec& locs, Dbm& z,
                    const std::vector<char>& qmask) {
  if (net.active.empty()) return;
  // BITMASK path: the active set of a location vector is the OR of the per-location masks, so
  // this is nprocs OR-s plus one word of arithmetic instead of an nprocs x nclocks byte scan
  // AND a heap allocation -- both of which ran once per GENERATED successor (23M times on
  // phil13). Only the clocks actually reset are then visited, via a popcount loop.
  if (!net.active_mask.empty()) {
    std::uint64_t a = 0;
    for (int p = 0; p < net.nprocs(); ++p) a |= net.active_mask[p][locs[p]];
    for (int c = 1; c <= net.nclocks; ++c) if (qmask[c]) a |= std::uint64_t(1) << c;
    std::uint64_t dead = ~a & (((std::uint64_t(1) << net.nclocks) - 1) << 1);
    while (dead) {
      const int c = __builtin_ctzll(dead);
      dead &= dead - 1;
      z.reset(c);
    }
    return;
  }
  std::vector<char> a(net.nclocks + 1, 0);
  for (int p = 0; p < net.nprocs(); ++p) {
    const auto& ap = net.active[p][locs[p]];
    for (int c = 1; c <= net.nclocks; ++c)
      if (ap[c]) a[c] = 1;
  }
  for (int c = 1; c <= net.nclocks; ++c)
    if (!a[c] && !qmask[c]) z.reset(c);
}

enum ExtrapMode { ExNone = 0, ExLuPlus = 1, ExNoDiag = 2 };

inline std::string disc_key_bytes(const LocVec& locs, const IntVec& ints) {
  std::string k;
  k.reserve(locs.size() * 4 + ints.size() * 8);
  for (int l : locs) k.append(reinterpret_cast<const char*>(&l), sizeof(int));
  for (long v : ints) { std::int64_t x = v; k.append(reinterpret_cast<const char*>(&x), sizeof(std::int64_t)); }
  return k;
}

// BROADCAST RECEIVER OPTIONS: participate via one enabled a?, or ABSTAIN.
//
// A broadcast `a!` fires whether or not any receiver is enabled, and each `a?` takes part exactly
// when its OWN guard holds. Collecting receivers by their INT guard alone and then conjoining
// their CLOCK guards -- what these loops used to do -- silently forces every int-enabled receiver
// to participate: the run in which a receiver's clock guard fails and it stays put is never
// generated, so reachable states are LOST. That is an `E<>` false negative and, worse, an `A[]`
// false positive. See tests/models/liveness-regressions/bcast-receiver.xml.
//
// The repair is an explicit ABSTAIN option per receiver process, guarded by the complement of the
// disjunction of its edges' clock guards. Each clock guard is a conjunction, so its complement is
// a disjunction of negated atoms, and the complement of the whole disjunction is the cross
// product of those -- a DNF whose disjuncts are each convex, hence each a separate option. Sizes
// are the product of the guard lengths, i.e. tiny in practice, and ZERO whenever any receiver
// edge has an empty clock guard: the complement of `true` is false, that receiver can never
// abstain, and the option list collapses to exactly what the old code produced. That is why csma,
// whose five `cd?` edges carry no guard at all, is unaffected.
struct RecvOpt { int edge; std::vector<Constraint> extra; };   // edge < 0 == abstain

inline void build_broadcast_parts(const Network& net, const SymState& s, const NEdge& se, int sender,
                                  std::vector<std::pair<int, std::vector<RecvOpt>>>& parts) {
  parts.clear();
  for (int j = 0; j < net.nprocs(); ++j) {
    if (j == sender) continue;
    std::vector<RecvOpt> opts;
    std::vector<const std::vector<Constraint>*> guards;
    for (int ej : net.procs[j].edges_of(s.locs[j])) {
      const NEdge& re = net.procs[j].edges[ej];
      if (re.sync_chan == se.sync_chan && re.sync_dir == -1 &&
          int_ok(re.int_guard, s.ints) && gen_ok(net, re, s.ints.data(), s.ints.size())) {
        opts.push_back({ej, {}});
        guards.push_back(&re.clock_guard);
      }
    }
    if (opts.empty()) continue;      // no int-enabled a?: j simply does not take part
    std::vector<std::vector<Constraint>> dnf{{}};
    for (const auto* g : guards) {
      if (g->empty()) { dnf.clear(); break; }        // complement of `true` is false
      std::vector<std::vector<Constraint>> next;
      next.reserve(dnf.size() * g->size());
      for (const auto& acc : dnf)
        for (const auto& c : *g) { auto cp = acc; cp.push_back(negate(c)); next.push_back(std::move(cp)); }
      dnf.swap(next);
    }
    for (auto& d : dnf) opts.push_back({-1, std::move(d)});
    parts.push_back({j, std::move(opts)});
  }
}

// Closes a successor zone once the transition's own guards and resets have been applied:
// conjoin the TARGET location invariant, let time pass (unless the target is frozen), drop the
// clocks nothing will read again, and extrapolate. `inv` is caller-owned scratch and is left
// holding the target invariant. Returns false when that invariant is unsatisfiable.
//
// The emptiness test is deliberately NOT folded in: the transition memo has to record an
// extrapolated-but-empty zone before it is discarded, so its call site needs the zone back even
// on the dead path. Every successor generator in this file ends with these same six steps in
// this same order -- local, binary-sync and broadcast edges, in both the accumulating and the
// slot-reusing variant -- and they must stay in step, so they share one body.
// always_inline, not merely inline: this is the tail of every successor generator in the file,
// and at nine call sites the inliner's cost model starts declining the big ones -- which cost
// ~2.5% on the liveness successor generation when these three sites first shared this body.
inline __attribute__((always_inline)) bool
close_succ_zone(const Network& net, const LocVec& tgt_locs, Dbm& z,
                            std::vector<Constraint>& inv, const std::vector<char>& qmask,
                            const std::vector<std::int32_t>& L,
                            const std::vector<std::int32_t>& U, int emode) {
  all_invariants(net, tgt_locs, inv);
  if (!apply_all(z, inv)) return false;
  bool tgt_committed, tgt_urgent;
  frozen_flags(net, tgt_locs, &tgt_committed, &tgt_urgent);
  if (!tgt_committed && !tgt_urgent) { z.delay(); apply_all(z, inv); }
  reset_inactive(net, tgt_locs, z, qmask);
  if (emode == ExLuPlus) z.extrapolate_lu_plus(L, U);
  else if (emode == ExNoDiag) z.extrapolate_lu_plus_nodiag(L, U);
  return true;
}

// Exact per-valuation deadlock test. A whole zone with no successor is the easy case; the hard
// one is a PARTIAL deadlock, where the zone has successors but a sub-region of it can never
// reach any of them. Defined in netreach.cpp -- it is ~170 lines dominated by federation
// subtraction and runs once per symbolic state, so an out-of-line call across the translation
// unit boundary is far below its own noise floor.
//
// `stuck`, when non-null, receives the EXACT set of valuations of s.zone from which no transition
// is ever enabled; `escapes` receives the enabled source regions themselves. Both are quantities
// the routine already computes internally and would otherwise discard -- see Otf::stop_set and
// Otf::build_atom_map, which need them.
bool state_has_deadlock(const Network& net, const SymState& s,
                        Federation* stuck = nullptr, Federation* escapes = nullptr);

}  // namespace detail
}  // namespace ptctlmc
