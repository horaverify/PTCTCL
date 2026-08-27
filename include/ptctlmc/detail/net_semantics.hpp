/// @file net_semantics.hpp
/// @brief The transition semantics of a network of timed automata.
///
/// Defines what a transition *is* and what a successor zone *is*: guard evaluation, the combined
/// location invariant, the frozen (committed or urgent) test, the active-clock reduction,
/// extrapolation, and the broadcast receiver/abstain enumeration.
///
/// Every layer above this one is built out of exactly these pieces — the forward reachability
/// search in @ref netreach.cpp and the backward greatest-fixpoint liveness engine in
/// @ref netliveness.cpp.
///
/// @warning The two engines must agree on these definitions exactly. A second copy of this logic
///          that drifts from this one produces a semantic difference between reachability and
///          liveness rather than a crash, which is why the semantics live here and not in either
///          engine.
/// @note Everything here is defined @c inline in the header deliberately. The build uses no LTO,
///       so an out-of-line definition would stop inlining at the translation-unit boundary, and
///       these are the innermost bodies of both engines — called once per generated successor.
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

/// @brief Conjoin every constraint of @p cs into @p z.
/// @param[in,out] z  the zone, tightened in place
/// @param cs         the constraints to apply
/// @return @c false as soon as a constraint makes @p z empty
/// @post On @c false, @p z is empty and the remaining constraints are not applied; callers treat
///       the zone as dead rather than reading it.
inline bool apply_all(Dbm& z, const std::vector<Constraint>& cs) {
  for (const auto& c : cs)
    if (!z.constrain(c.i, c.j, c.bound)) return false;
  return true;
}

/// @brief Evaluate a conjunction of integer atoms against a valuation.
/// @param g     the atoms, implicitly conjoined
/// @param ints  the integer variables of the state
/// @return @c true iff every atom of @p g holds; vacuously @c true for an empty @p g
/// @pre Every atom's variable index is in range for @p ints.
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

/// @brief Collect the combined invariant of a location vector.
///
/// The invariant of a network state is the conjunction of the invariants of the locations its
/// processes occupy.
///
/// @param net  the network under analysis
/// @param locs the location vector
/// @param[out] inv  caller-owned scratch; cleared, then filled with the combined invariant
///
/// @note Fills a caller-owned buffer rather than returning a fresh vector because this runs
///       several times per generated successor in the hottest loop; reusing the buffer keeps the
///       allocation out of that loop.
inline void all_invariants(const Network& net, const LocVec& locs, std::vector<Constraint>& inv) {
  inv.clear();
  for (int p = 0; p < net.nprocs(); ++p)
    for (const auto& c : net.procs[p].locations[locs[p]].invariant) inv.push_back(c);
}

/// @brief Determine whether a location vector is frozen, i.e. whether time may pass in it.
///
/// @param net  the network under analysis
/// @param locs the location vector
/// @param[out] committed  set iff some process sits in a committed location
/// @param[out] urgent     set iff some process sits in an urgent location
///
/// @note Either flag forbids delay. They are reported separately because committed additionally
///       restricts *which* edges may fire — only edges leaving a committed location — whereas
///       urgent only stops time.
inline void frozen_flags(const Network& net, const LocVec& locs, bool* committed, bool* urgent) {
  *committed = *urgent = false;
  for (int p = 0; p < net.nprocs(); ++p) {
    const Location& l = net.procs[p].locations[locs[p]];
    if (l.committed) *committed = true;
    if (l.urgent) *urgent = true;
  }
}

/// @brief Reset the clocks whose value can no longer be observed.
///
/// A clock that no process will read before resetting it, and that the query does not read
/// either, cannot influence any future behaviour. Zeroing it merges zones differing only in that
/// clock, which is sound precisely because nothing downstream can distinguish them.
///
/// @param net   the network under analysis
/// @param locs  the location vector, selecting the active-clock sets
/// @param[in,out] z  the zone; dead clocks are reset in place
/// @param qmask clocks the query reads, which are never treated as dead
///
/// @pre `qmask` has an entry per clock index.
/// @post Only dead clocks are modified; the projection onto live clocks is unchanged.
/// @note This is the continuous analogue of the discrete reduction ptctlmc::set_slice applies to
///       integer cells.
inline void reset_inactive(const Network& net, const LocVec& locs, Dbm& z,
                    const std::vector<char>& qmask) {
  if (net.active.empty()) return;
  // Bitmask path: the active set of a location vector is the OR of the per-location masks, so
  // this costs nprocs OR operations plus a word of arithmetic, rather than an nprocs x nclocks
  // byte scan and a heap allocation once per generated successor. Only the clocks actually reset
  // are then visited. The mask is built only when nclocks <= 63 (see Network::compute_active),
  // so the shifts below cannot overflow; the byte scan afterwards is the fallback for wider
  // networks.
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

/// @brief Which extrapolation ::close_succ_zone applies to a successor zone.
enum ExtrapMode {
  ExNone   = 0,  ///< none; finiteness comes from a_LU subsumption over exact zones
  ExLuPlus = 1,  ///< Extra_LU+; required when zones are compared exactly
  ExNoDiag = 2   ///< diagonal-preserving variant, for networks a_LU cannot abstract
};

/// @brief Serialise a discrete state to a byte string.
///
/// @param locs the location vector
/// @param ints the integer variables
/// @return a byte string equal for two discrete states exactly when they are equal
///
/// @note Unlike ::DiscKeyCodec this imposes no width limit, so it serves the paths that must
///       handle networks too wide for the packed 63-bit key. It allocates per call, so the hot
///       forward search uses the packed key instead.
inline std::string disc_key_bytes(const LocVec& locs, const IntVec& ints) {
  std::string k;
  k.reserve(locs.size() * 4 + ints.size() * 8);
  for (int l : locs) k.append(reinterpret_cast<const char*>(&l), sizeof(int));
  for (long v : ints) { std::int64_t x = v; k.append(reinterpret_cast<const char*>(&x), sizeof(std::int64_t)); }
  return k;
}

/// @brief One receiver process's choice in a broadcast: take a particular `a?` edge, or abstain.
///
/// A broadcast `a!` fires whether or not any receiver is enabled, and each `a?` participates
/// exactly when its own guard holds. A receiver whose clock guard fails must therefore be able to
/// stay put, which is what an abstain option represents.
///
/// @warning Collecting receivers by their integer guard alone and then conjoining their clock
///          guards forces every integer-enabled receiver to participate. The run in which a
///          receiver's clock guard fails and it stays put is then never generated, losing
///          reachable states — an `E<>` false negative and an `A[]` false positive.
struct RecvOpt {
  int edge;                       ///< the `a?` edge taken, or negative to abstain
  std::vector<Constraint> extra;  ///< for an abstain option, the guard under which abstaining is legal
};

/// @brief Enumerate each receiver process's options for a broadcast send.
///
/// @param net    the network under analysis
/// @param s      the source state
/// @param se     the sending `a!` edge
/// @param sender index of the sending process, which is never its own receiver
/// @param[out] parts  cleared, then filled with one (process, options) entry per process having
///                    at least one integer-enabled `a?`; the caller takes the cross product
///
/// @post Every process listed in @p parts has a non-empty option list.
///
/// @note An abstain option is guarded by the complement of the disjunction of that process's
///       enabled edges' clock guards. Each clock guard is a conjunction, so its complement is a
///       disjunction of negated atoms, and the complement of the whole disjunction is their cross
///       product — a DNF whose disjuncts are individually convex, hence one option each.
/// @note The option count is the product of the guard lengths, and is zero whenever any enabled
///       receiver edge has an empty clock guard: the complement of @c true is false, so that
///       receiver can never abstain. A model whose receivers carry no clock guards is therefore
///       unaffected by the abstain machinery.
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
    if (opts.empty()) continue;      // no integer-enabled a?: j simply does not take part
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

/// @brief Close a successor zone once the transition's own guards and resets have been applied.
///
/// Performs the closing pipeline in order: conjoin the target location invariant, let time pass
/// unless the target is frozen, reset the clocks nothing will read again, and extrapolate.
///
/// @param net       the network under analysis
/// @param tgt_locs  the target location vector
/// @param[in,out] z the successor zone, closed in place
/// @param[out] inv  caller-owned scratch; left holding the target invariant
/// @param qmask     clocks the query reads, preserved by the active-clock reduction
/// @param L         per-clock lower bounds for the extrapolation
/// @param U         per-clock upper bounds for the extrapolation
/// @param emode     which extrapolation to apply; see ::ExtrapMode
/// @return @c false when the target invariant is unsatisfiable
///
/// @post On @c true, @p z may still be **empty**; the emptiness test is deliberately left to the
///       caller, because the transition memo must record an extrapolated-but-empty zone before it
///       is discarded and so needs the zone back even on the dead path.
///
/// @warning Every successor generator in both engines ends with these steps in this order. They
///          must stay in step, which is why they share one body rather than repeating it.
/// @note Marked @c always_inline rather than merely @c inline: this is the tail of every
///       successor generator, and across that many call sites the inliner's cost model begins
///       declining the larger ones.
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

/// @brief Exact per-valuation deadlock test; see the definition in @ref netreach.cpp.
///
/// @param net  the network under analysis
/// @param s    the symbolic state to test
/// @param[out] stuck    if non-null, receives the exact set of valuations of `s.zone` from which
///                      no transition is ever enabled
/// @param[out] escapes  if non-null, receives the enabled source regions themselves
/// @return @c true iff some valuation of @p s is deadlocked
///
/// @note Both outputs are quantities the routine computes internally and would otherwise discard;
///       the liveness engine's stop set and atom map consume them.
/// @note Declared here but defined out of line, unlike the rest of this header: it is long,
///       dominated by federation subtraction, and runs once per symbolic state rather than once
///       per successor, so the call overhead is negligible against its own cost.
/// @warning At a frozen state this skips both the delay and the invariant. See the full contract
///          on the definition in @ref netreach.cpp before passing a zone that is not already
///          restricted to `Inv(q)`.
bool state_has_deadlock(const Network& net, const SymState& s,
                        Federation* stuck = nullptr, Federation* escapes = nullptr);

}  // namespace detail
}  // namespace ptctlmc
