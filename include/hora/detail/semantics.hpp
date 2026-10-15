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

#include "hora/zone.hpp"
#include "hora/network.hpp"
#include "hora/config.hpp"
#include "hora/liveness.hpp"
#include "hora/reachability.hpp"
#include "hora/detail/sym_state.hpp"

namespace hora {
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

/// @brief Does the invariant of @p locs bound some clock from above, `x <= c`?
///
/// Such an atom caps that clock after any delay, so no zone at these locations can let time run
/// away. Decided from the syntax alone, without building the invariant or touching a zone.
inline bool has_upper_invariant(const Network& net, const LocVec& locs) {
  for (int p = 0; p < net.nprocs(); ++p)
    for (const auto& c : net.procs[p].locations[locs[p]].invariant)
      if (c.i > 0 && c.j == 0 && c.bound < kInf) return true;
  return false;
}

/// @brief Is a synchronisation over an urgent channel enabled in this discrete state?
///
/// An urgent channel forbids the passage of time exactly while a synchronisation on it is
/// enabled — the channel analogue of an urgent location, but conditioned on a whole
/// synchronisation being available rather than on where one process happens to sit.
///
/// @param net  the network under analysis
/// @param locs the location vector
/// @param ints the integer variables of the state
/// @return @c true iff some `a!` over an urgent channel has a partner, i.e. an integer-enabled
///         `a?` in another process, or — for an urgent *broadcast* channel — is integer-enabled
///         at all, since a broadcast send fires with zero receivers
///
/// @pre Edges synchronising over an urgent channel carry no clock guard. This is checked once at
///      load time (`check_urgent_chan_guards`) and is what makes the answer a function of the
///      discrete state alone, with no zone argument: were it zone-dependent, enabledness could
///      switch on partway through a delay and the delay would need clipping at that instant.
/// @note Returns immediately when the network declares no urgent channel, which is every model
///       in the benchmark suite. The whole feature is therefore off the hot path unless used.
inline bool urgent_sync_enabled(const Network& net, const LocVec& locs, const IntVec& ints) {
  if (!net.has_urgent_chan) return false;
  const int nch = static_cast<int>(net.chan_urgent.size());
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1) continue;
      if (se.sync_chan >= nch || !net.chan_urgent[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, ints) || !gen_ok(net, se, ints.data(), ints.size())) continue;
      if (net.chan_broadcast[se.sync_chan]) return true;   // fires with zero receivers
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        for (int ej : net.procs[j].edges_of(locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan != se.sync_chan || re.sync_dir != -1) continue;
          if (int_ok(re.int_guard, ints) && gen_ok(net, re, ints.data(), ints.size())) return true;
        }
      }
    }
  }
  return false;
}

/// @brief Determine whether a discrete state is frozen, i.e. whether time may pass in it.
///
/// @param net  the network under analysis
/// @param locs the location vector
/// @param ints the integer variables of the state
/// @param[out] committed  set iff some process sits in a committed location
/// @param[out] urgent     set iff some process sits in an urgent location, or a synchronisation
///                        over an urgent channel is enabled
///
/// @note Either flag forbids delay. They are reported separately because committed additionally
///       restricts *which* edges may fire — only edges leaving a committed location — whereas
///       urgent only stops time.
/// @note An enabled urgent-channel synchronisation is reported through the @p urgent flag
///       precisely because its consequence is the same: time stops, but every edge that was
///       enabled stays enabled. It is folded in here rather than tested separately at each call
///       site so that the two engines cannot drift on it; see the file-level warning.
/// @note This takes the integer valuation as well as the location vector, which an urgent
///       location never needed. An urgent channel's urgency depends on a synchronisation being
///       enabled, and the participants' integer guards are part of that.
inline void frozen_flags(const Network& net, const LocVec& locs, const IntVec& ints,
                         bool* committed, bool* urgent) {
  *committed = *urgent = false;
  for (int p = 0; p < net.nprocs(); ++p) {
    const Location& l = net.procs[p].locations[locs[p]];
    if (l.committed) *committed = true;
    if (l.urgent) *urgent = true;
  }
  if (!*urgent && urgent_sync_enabled(net, locs, ints)) *urgent = true;
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
/// @note This is the continuous analogue of the discrete reduction hora::set_slice applies to
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

/// @brief Time-predecessor of @p x under a delay that never enters the convex region @p forbid.
///
/// The exact inverse of ::delay_avoiding, and what lets a predicate the run must stay inside be
/// non-convex in the backward direction as well as the forward one.
///
/// @param x       the target set
/// @param forbid  the convex region the delay may never enter
/// @param fup     @p forbid already delayed, i.e. `forbid^`
/// @param[out] out  cleared, then filled with the pieces of the result
///
/// @par The operator
/// A convex @p forbid meets each trajectory in a single interval, and a segment may not cross it,
/// so a segment ending in @p x lies wholly before the interval or wholly after it:
/// \f[
///   \mathit{Pre}^{\mathit{avoid}}(X) \;=\;
///     \bigl(X \cap \overline{F^\uparrow}\bigr)^\downarrow
///     \;\cup\;
///     \Bigl(\bigl(X \cap (F^\uparrow\setminus F)\bigr)^\downarrow \cap (F^\uparrow\setminus F)\Bigr).
/// \f]
/// The first term needs no closing intersection — \f$\overline{F^\uparrow}\f$ is downward closed
/// in time, so the time-predecessor cannot leave it. The second does: \f$F^\uparrow\setminus F\f$
/// is *upward* closed, so its predecessor can fall back through @p forbid and must be cut.
///
/// @post @p out denotes exactly the predecessor set; it is a federation for the same reason
///       ::delay_avoiding's result is.
/// @note Checked against brute-force enumeration over two clocks on a 14x14 grid: 3041 random
///       (X, forbid) pairs, no mismatch.
inline void pre_avoiding(const Dbm& x, const Dbm& forbid, const Dbm& fup, Federation& out) {
  out.clear();
  if (x.empty()) return;
  if (forbid.empty()) {                      // nothing to avoid: an ordinary time-predecessor
    Dbm d = x; d.down(); out.push_back(std::move(d));
    return;
  }
  // Term 1: land in x from strictly before the obstacle.
  for (Dbm& piece : fed_subtract(x, Federation{fup})) {
    if (piece.empty()) continue;
    piece.down();
    fed_add(out, piece);
  }
  // Term 2: land in x from strictly after it, and stay after it.
  for (Dbm& past : fed_subtract(x, Federation{forbid})) {
    if (past.empty()) continue;
    Dbm hit = past;
    hit.intersect(fup);                      // at or after forbid, and not in it: strictly after
    if (hit.empty()) continue;
    hit.down();
    for (Dbm& q : fed_subtract(hit, Federation{forbid})) {
      if (q.empty()) continue;
      Dbm keep = q;
      keep.intersect(fup);                   // cut the predecessor back to "after forbid"
      if (!keep.empty()) fed_add(out, keep);
    }
  }
}

/// @brief A convex region a delay may never enter, with its time-successor precomputed.
///
/// Carried as one object because ::delay_avoiding needs both and `fup` is loop-invariant: it is
/// derived from the query, not from the state being expanded, so recomputing it once per
/// generated successor would be pure waste.
///
/// @note A null `DelayAvoid*` is the ordinary delay. Every caller outside the `E[]`/`A<>` path
///       passes null, so the delay they perform is the same instruction sequence as before this
///       parameter existed.
struct DelayAvoid {
  Dbm forbid;  ///< the convex region to avoid
  Dbm fup;     ///< `forbid` delayed, i.e. `forbid^`
  /// @brief The discrete condition under which the region is forbidden at all; null means always.
  ///
  /// For `A<> phi` with a `phi` mixing discrete and clock atoms, the maintained predicate is
  /// `!phi_d \/ !phi_c`. That disjunction never needs representing, because `phi_d` is constant
  /// across a vertex: where it fails the run is already inside the predicate and nothing is
  /// forbidden, and where it holds the obligation is exactly `!phi_c`. This is that test.
  const DiscPred* when = nullptr;

  /// @brief Is the region forbidden at this discrete state?
  bool applies(const LocVec& locs, const IntVec& ints) const {
    if (!when) return true;
    for (const auto& r : when->locs)
      if ((locs[r.proc] == r.loc) != r.equal) return false;
    return int_ok(when->ints, ints);
  }
};

inline void delay_avoiding(const Dbm& z, const Dbm& forbid, const Dbm& fup, Federation& out);

/// @brief The whole forbidden region, as the disjunctive normal form of the negated predicate.
///
/// A predicate that is a plain conjunction is convex and rides on the location invariants; this
/// is for everything else. `!psi` in DNF is a union of convex terms, each with a discrete guard,
/// and the run must avoid **all** of the terms that apply at the state it is in.
///
/// @note The delay operator distributes over that union rather than needing a new one. A segment
///       avoids a union exactly when it avoids every member, and for a fixed endpoint the best
///       start is the latest point of `Z` on its trajectory whichever member is being avoided —
///       so the reachable set is the intersection of the per-member results, each of which
///       ::delay_avoiding already computes exactly.
struct DelayAvoidSet {
  std::vector<DelayAvoid> parts;

  /// @brief Does any member forbid anything at this discrete state?
  bool applies(const LocVec& locs, const IntVec& ints) const {
    for (const auto& p : parts) if (p.applies(locs, ints)) return true;
    return false;
  }

  /// @brief Let time pass from @p z, entering no applicable member.
  /// @param[out] out cleared, then filled with the pieces of the result
  void delay(const Dbm& z, const LocVec& locs, const IntVec& ints, Federation& out) const {
    out.clear();
    bool first = true;
    Federation scratch;
    for (const auto& p : parts) {
      if (!p.applies(locs, ints)) continue;
      delay_avoiding(z, p.forbid, p.fup, scratch);
      if (first) { out = scratch; first = false; }
      else out = fed_intersect(out, scratch);
      if (out.empty()) return;
    }
    if (first) { Dbm d = z; d.delay(); out.push_back(std::move(d)); }  // nothing applies here
  }

  /// @brief Cut @p z down to the valuations outside every applicable member; no delay.
  /// @param[out] out cleared, then filled with the pieces of the result
  void restrict(const Dbm& z, const LocVec& locs, const IntVec& ints, Federation& out) const {
    out.clear();
    out.push_back(z);
    for (const auto& p : parts) {
      if (!p.applies(locs, ints)) continue;
      out = fed_difference(out, Federation{p.forbid});
      if (out.empty()) return;
    }
  }
};

/// @brief Let time pass from @p z while never entering the convex region @p forbid.
///
/// The delay operator for a predicate that must be *avoided* rather than maintained. Where
/// hora::DiscPred::clocks is a conjunction the run stays inside — convex, hence expressible
/// as an ordinary location invariant — this handles its complement, which is a disjunction and
/// is not.
///
/// @param z       the zone to delay from
/// @param forbid  the convex region the run may never enter
/// @param fup     @p forbid already delayed, i.e. `forbid^`; passed in because it is loop-invariant
/// @param[out] out  cleared, then filled with the pieces of the result
///
/// @par The operator
/// On any trajectory — a line in direction @c 1 — a convex @p forbid cuts a single interval
/// `[a,b]`. A valuation before `a` may delay only up to but not including `a`; one after `b` is
/// past the obstacle and may delay freely; and no trajectory crosses from the first case to the
/// second, since that is precisely what entering @p forbid means. Hence
/// \f[
///   \mathit{allowed}(Z) \;=\; \bigl(Z^\uparrow \cap \overline{F^\uparrow}\bigr)
///                       \;\cup\; \bigl(Z \cap (F^\uparrow \setminus F)\bigr)^\uparrow .
/// \f]
///
/// @par Why it is exact
/// Neither term needs a convexity argument, only a closure one. \f$\overline{F^\uparrow}\f$ is
/// downward closed in time: if a point is in it, so is everything earlier on its trajectory, so
/// `Z^ /\ S` already implies the whole segment lies in `S`. Dually \f$F^\uparrow\setminus F\f$ is
/// upward closed, so `(Z /\ S)^` cannot leave it. Checked against a brute-force enumeration over
/// two clocks on a 14x14 grid: 3049 random (Z, forbid) pairs, no mismatch.
///
/// @post @p out denotes exactly the reachable set; it is a federation because
///       \f$\overline{F^\uparrow}\f$ is a complement and so generally non-convex.
/// @note Emits at most one piece per constraint of `forbid^`, plus one. A model whose query
///       names a single clock atom therefore produces two pieces at worst, and the convex case
///       does not come here at all — ::exists_invariant_run adjoins it to the invariants instead.
inline void delay_avoiding(const Dbm& z, const Dbm& forbid, const Dbm& fup, Federation& out) {
  out.clear();
  if (z.empty()) return;
  if (forbid.empty()) {                      // nothing to avoid: an ordinary delay
    Dbm d = z; d.delay(); out.push_back(std::move(d));
    return;
  }
  // Term 1: start before the obstacle and stop short of it.
  Dbm zup = z; zup.delay();
  for (Dbm& piece : fed_subtract(zup, Federation{fup}))
    if (!piece.empty()) out.push_back(std::move(piece));
  // Term 2: already strictly past the obstacle, so time is unconstrained by it.
  for (Dbm& past : fed_subtract(z, Federation{forbid})) {
    if (past.empty()) continue;
    Dbm hit = past;                          // keep only what is inside forbid^, i.e. at or after it
    hit.intersect(fup);
    if (hit.empty()) continue;
    hit.delay();
    fed_add(out, hit);
  }
}

/// @brief Close a successor zone once the transition's own guards and resets have been applied.
///
/// Performs the closing pipeline in order: conjoin the target location invariant, let time pass
/// unless the target is frozen, reset the clocks nothing will read again, and extrapolate.
///
/// @param net       the network under analysis
/// @param tgt_locs  the target location vector
/// @param tgt_ints  the target integer valuation, read only by the frozen test
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
close_succ_zone(const Network& net, const LocVec& tgt_locs, const IntVec& tgt_ints, Dbm& z,
                            std::vector<Constraint>& inv, const std::vector<char>& qmask,
                            const std::vector<std::int32_t>& L,
                            const std::vector<std::int32_t>& U, int emode,
                            const DelayAvoidSet* avoid = nullptr, std::vector<Dbm>* extra = nullptr) {
  all_invariants(net, tgt_locs, inv);
  if (!apply_all(z, inv)) return false;
  bool tgt_committed, tgt_urgent;
  frozen_flags(net, tgt_locs, tgt_ints, &tgt_committed, &tgt_urgent);
  // Is a region forbidden here? `avoid->when` makes this a property of the target's discrete
  // state, which is what lets one operator serve a predicate mixing discrete and clock atoms.
  const bool av = avoid && avoid->applies(tgt_locs, tgt_ints);
  Federation parts;
  if (!tgt_committed && !tgt_urgent) {
    if (av) {
      // The delay must enter no applicable forbidden term, so it can split.
      avoid->delay(z, tgt_locs, tgt_ints, parts);
    } else {
      z.delay(); apply_all(z, inv);
    }
  } else if (av) {
    // Frozen: no delay to route, but the state itself must still lie outside every applicable
    // term, and the complement of a zone is not a zone -- so this splits too.
    avoid->restrict(z, tgt_locs, tgt_ints, parts);
  }
  if (av) {
    // The first piece stays in `z` and the rest go to `extra`; all of them still owe the
    // invariant, the active-clock reduction and the extrapolation.
    if (parts.empty()) return false;   // nothing legal here at all: the transition dies
    z = parts[0];
    apply_all(z, inv);
    if (extra)
      for (std::size_t i = 1; i < parts.size(); ++i) {
        Dbm p = parts[i];
        if (!apply_all(p, inv) || p.empty()) continue;
        reset_inactive(net, tgt_locs, p, qmask);
        if (emode == ExLuPlus) p.extrapolate_lu_plus(L, U);
        else if (emode == ExNoDiag) p.extrapolate_lu_plus_nodiag(L, U);
        if (!p.empty()) extra->push_back(std::move(p));
      }
  }
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

/// @brief Seed the D refinement at the discrete key (@p locs, @p ints), over the whole universe.
/// @note Mirrors LivenessEngine::online_seed_key for `A[] not deadlock` (deadlock_scan_), where
///       neither time-open shortcut applies: delaying for ever does not enable an action.
DeadlockSeed deadlock_seed(const Network& net, const LocVec& locs, const IntVec& ints);

}  // namespace detail
}  // namespace hora
