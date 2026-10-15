/// @file src/reachability/engine.cpp
/// @brief The forward zone-graph search: successors, the passed set, and the CPU drivers.
///
/// One symbolic state is a (location vector, integer values, zone) triple. The passed set is
/// keyed by the discrete part and holds an antichain of zones per key; which abstraction bounds
/// that antichain is chosen in hora/config.hpp.
#include "hora/zone.hpp"
#include "hora/config.hpp"
#include "hora/liveness.hpp"
#include "hora/reachability.hpp"
#include "hora/symmetry.hpp"
#include <cstdio>
#include <array>
#include <chrono>
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif
#ifdef __linux__
#include <malloc.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <queue>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <stdexcept>
#include <string>
#include "hora/detail/liveness_engine.hpp"
#include <unordered_map>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "hdbm/dbm_ops_legacy.hpp"
#include "hora/detail/engine_config.hpp"
#include "hora/detail/spin_rw.hpp"
#include "hora/detail/sym_state.hpp"
#include "hora/detail/zone_intern.hpp"
#include "hora/detail/passed_set.hpp"
#include "hora/detail/semantics.hpp"
#include "hora/detail/frontier_gpu.hpp"
#include "hdbm/mindbm.hpp"   // minimal-graph packed zone storage (UDBM mingraph port)

#ifdef HORA_HAVE_OPENMP
#include <omp.h>
#endif

namespace hora {
namespace {
using detail::SpinRW;
using detail::SymState;
using detail::LocVec;
using detail::IntVec;
using detail::ZoneIntern;
using detail::NetPassed;
using detail::NetPassedT;
// The packed discrete key: NarrowKey for a network that fits one word, WideKey for one that does
// not. See detail/disc_key.hpp.
using detail::NarrowKey;
using detail::WideKey;
using detail::KeyHash;
using detail::KeyHashFor;
using detail::key_mix;
using detail::key_budget;
// The transition semantics both engines are built out of (net_semantics.hpp): guards, the
// combined invariant, the frozen test, the active-clock reduction, extrapolation, and the
// broadcast receiver/abstain enumeration.
using detail::ExtrapMode;
using detail::ExNone;
using detail::ExLuPlus;
using detail::ExNoDiag;
using detail::apply_all;
using detail::int_ok;
using detail::all_invariants;
using detail::frozen_flags;
using detail::reset_inactive;
using detail::close_succ_zone;
using detail::disc_key_bytes;
using detail::RecvOpt;
using detail::build_broadcast_parts;
using detail::state_has_deadlock;


#ifdef __linux__
static std::size_t linux_rss_bytes() {
  FILE* f = std::fopen("/proc/self/statm", "r");
  if (!f) return 0;
  unsigned long size = 0, resident = 0;
  const int n = std::fscanf(f, "%lu %lu", &size, &resident);
  std::fclose(f);
  if (n < 2) return 0;
  const long ps = sysconf(_SC_PAGESIZE);
  if (ps <= 0) return 0;
  return static_cast<std::size_t>(resident) * static_cast<std::size_t>(ps);
}

static std::size_t linux_heap_bytes() {
#if defined(__GLIBC__)
  return static_cast<std::size_t>(mallinfo2().uordblks);
#else
  return 0;
#endif
}
#endif

}  // namespace

/// @name General modelling-language guards and updates
/// @{
/// Only edges needing more than `var OP const` carry these; the common case is a null check, so
/// simple models pay nothing for them.
///
/// @warning Both have **external** linkage on purpose: every backend must apply exactly these
///          semantics. Duplicating them is how a backend once diverged silently — an expander
///          that dropped the general update explored roughly half of viking's state space and
///          still reported a verdict. One body, every backend. Declared in @ref hora/netreach.hpp.
/// @}

/// @brief Evaluate @p e's general guard.
///
/// @param net   the network, for its function table
/// @param e     the edge whose guard is evaluated
/// @param ints  the integer variables of the source state
/// @param n     length of @p ints
/// @return @c true iff @p e has no general guard, or the guard evaluates non-zero without
///         faulting
///
/// @post @p ints is unchanged. The guard runs against a thread-local copy, so a guard that calls
///       a function with side effects cannot perturb the live state.
bool gen_ok(const Network& net, const NEdge& e, const long* ints, std::size_t n) {
  if (!e.gen_guard) return true;
  thread_local std::vector<long> scratch;
  scratch.assign(ints, ints + n);
  EvalCtx cx;
  cx.cells = scratch.data();
  cx.funcs = &net.funcs;
  const long v = eval(*e.gen_guard, cx);
  return !cx.fault && v != 0;
}

/// @brief Run @p e's general update over @p ints.
///
/// @param net       the network, for its function table and the variables' declared ranges
/// @param e         the edge whose update is executed
/// @param[in,out] ints  the integer variables; written only on success
/// @param n         length of @p ints
/// @return @c false if the update faulted (bad index, division by zero, runaway loop) or if a
///         write left a variable's declared range
///
/// @post On @c false, @p ints is unchanged: the update runs against a thread-local copy and is
///       committed only once every value has been range-checked.
/// @warning An out-of-range write must be refused, not clamped or stored. ::DiscKeyCodec gives
///          each variable exactly enough bits for its declared range, so an out-of-range value
///          would alias a different discrete state in the packed key — silent state-space
///          corruption rather than a detectable error.
bool gen_apply(const Network& net, const NEdge& e, long* ints, std::size_t n) {
  if (!e.gen_update) return true;
  thread_local std::vector<long> buf;
  buf.assign(ints, ints + n);
  EvalCtx cx;
  cx.cells = buf.data();
  cx.funcs = &net.funcs;
  cx.locals.assign(static_cast<std::size_t>(e.gen_locals) + 4, 0);
  bool returned = false;
  long ret = 0;
  exec(*e.gen_update, cx, &returned, &ret);
  if (cx.fault) return false;
  for (std::size_t i = 0; i < buf.size(); ++i) {
    if (buf[i] < net.intvars[i].lo || buf[i] > net.intvars[i].hi) return false;
    ints[i] = buf[i];
  }
  return true;
}

/// @brief Does @p net compare two clocks anywhere?
///
/// @param net  the network to scan; both invariants and edge guards are examined
/// @return @c true iff some constraint has both indices non-zero, i.e. is diagonal
///
/// @note The LU family of abstractions is unsound on diagonal constraints (Bouyer,
///       *Forward analysis of updatable timed automata*, FMSD 24(3), 2004), so a network for
///       which this returns @c true must keep diagonals exact: inclusion covering with the
///       diagonal-preserving extrapolation, never a_LU.
/// @note External linkage on purpose: the query router asks the same question before choosing an
///       engine. One body.
bool network_has_diagonal(const Network& net) {
  auto diag = [](const std::vector<Constraint>& cs) {
    for (const auto& c : cs) if (c.i > 0 && c.j > 0) return true;
    return false;
  };
  for (const auto& p : net.procs) {
    for (const auto& l : p.locations) if (diag(l.invariant)) return true;
    for (const auto& e : p.edges) if (diag(e.clock_guard)) return true;
  }
  return false;
}

namespace {

/// @brief Generate every symbolic successor of @p s under one step of the network.
///
/// Enumerates the three synchronisation forms of the modelling language in turn: internal edges
/// (no channel), binary rendezvous (`a!` in one process with `a?` in another), and broadcast
/// (`a!` fires together with *all* currently enabled `a?`, including none at all).
///
/// @param net    the network under analysis
/// @param s      the source state; only its @c locs and @c ints are read
/// @param szone  the source zone, supplied separately from @p s
/// @param L      lower-bound array per clock, for the extrapolation
/// @param U      upper-bound array per clock, for the extrapolation
/// @param qmask  clocks the query mentions, which the active-clock reduction must not reset
/// @param emode  extrapolation mode applied when closing each successor zone
/// @param[out] out  successors are appended; existing contents are kept
///
/// @pre  @p s.locs has one entry per process and @p s.ints one per declared integer variable.
/// @post Every appended state has a non-empty zone; empty successors are dropped here rather
///       than by the caller.
///
/// @note @p s.zone is ignored. The zone travels as a separate parameter so that callers already
///       holding it elsewhere — notably the shared-passed path, where it lives in the intern
///       table — can pass a reference rather than copy a full DBM into the frontier state on
///       every expansion.
/// @note When the source is committed, only edges leaving a committed location may fire, and a
///       synchronisation qualifies if *either* participant is committed. This is the standard
///       committed-location semantics: such a state must be left before time can pass.
void successors_net(const Network& net, const SymState& s, const Dbm& szone,
                    const std::vector<std::int32_t>& L,
                    const std::vector<std::int32_t>& U, const std::vector<char>& qmask, int emode, std::vector<SymState>& out,
                    const detail::DelayAvoidSet* avoid = nullptr) {
  thread_local std::vector<Constraint> inv;  // reused invariant scratch (see all_invariants)
  // Extra pieces produced when a delay had to route around a forbidden region and split; empty,
  // and never even looked at, unless `avoid` is non-null.
  std::vector<Dbm> split;
  auto extra = [&]() -> std::vector<Dbm>* { return avoid ? (split.clear(), &split) : nullptr; };
  // Emit `ns` carrying `z1`, plus one clone per extra piece. The clones differ only in the zone:
  // the pieces partition one delay of one transition, so they share its target locations and
  // integers exactly.
  auto emit = [&](SymState& ns, Dbm& z1) {
    for (Dbm& p : split) { SymState c; c.locs = ns.locs; c.ints = ns.ints; c.zone = std::move(p); out.push_back(std::move(c)); }
    split.clear();
    if (z1.empty()) return;
    ns.zone = std::move(z1);
    out.push_back(std::move(ns));
  };
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, s.ints, &src_committed, &src_urgent);
  for (int p = 0; p < net.nprocs(); ++p) {
    if (src_committed && !net.procs[p].locations[s.locs[p]].committed) continue;
    for (int ei : net.procs[p].edges_of(s.locs[p])) {
      const NEdge& e = net.procs[p].edges[ei];
      if (e.sync_chan >= 0) continue;
      if (!int_ok(e.int_guard, s.ints) || !gen_ok(net, e, s.ints.data(), s.ints.size())) continue;
      Dbm z1 = szone;
      if (!apply_all(z1, e.clock_guard)) continue;
      for (int r : e.clock_resets) z1.reset(r);

      SymState ns;
      ns.locs = s.locs;
      ns.locs[p] = e.tgt;
      ns.ints = s.ints;
      for (const auto& a : e.int_updates) ns.ints[a.var] = a.val;
      if (!gen_apply(net, e, ns.ints.data(), ns.ints.size())) continue;

      if (!close_succ_zone(net, ns.locs, ns.ints, z1, inv, qmask, L, U, emode, avoid, extra())) continue;
      emit(ns, z1);
    }
  }

  // ---- binary channel synchronisation (a! in proc i  ||  a? in proc j) ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1) continue;  // sender a!
      if (net.chan_broadcast[se.sync_chan]) continue;      // broadcast handled below
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      const bool i_comm = net.procs[i].locations[s.locs[i]].committed;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan != se.sync_chan || re.sync_dir != -1) continue;  // receiver a?
          if (src_committed && !(i_comm || net.procs[j].locations[s.locs[j]].committed)) continue;
          if (!int_ok(re.int_guard, s.ints) || !gen_ok(net, re, s.ints.data(), s.ints.size())) continue;
          Dbm z1 = szone;
          if (!apply_all(z1, se.clock_guard)) continue;
          if (!apply_all(z1, re.clock_guard)) continue;
          for (int r : se.clock_resets) z1.reset(r);
          for (int r : re.clock_resets) z1.reset(r);
          SymState ns;
          ns.locs = s.locs;
          ns.locs[i] = se.tgt;
          ns.locs[j] = re.tgt;
          ns.ints = s.ints;
          for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;  // sender then receiver
          for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
          if (!gen_apply(net, se, ns.ints.data(), ns.ints.size()) || !gen_apply(net, re, ns.ints.data(), ns.ints.size())) continue;
          if (!close_succ_zone(net, ns.locs, ns.ints, z1, inv, qmask, L, U, emode, avoid, extra())) continue;
          emit(ns, z1);
        }
      }
    }
  }

  // ---- broadcast synchronisation: a! fires with all enabled a? (0 or more) ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1 || !net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      std::vector<std::pair<int, std::vector<RecvOpt>>> parts;   // see build_broadcast_parts
      build_broadcast_parts(net, s, se, i, parts);
      std::vector<std::size_t> pick(parts.size(), 0);
      for (;;) {
        bool part_committed = net.procs[i].locations[s.locs[i]].committed;
        for (std::size_t t = 0; t < parts.size(); ++t)
          if (parts[t].second[pick[t]].edge >= 0 &&
              net.procs[parts[t].first].locations[s.locs[parts[t].first]].committed) part_committed = true;
        if (!src_committed || part_committed) {
          Dbm z1 = szone;
          bool ok = apply_all(z1, se.clock_guard);
          for (std::size_t t = 0; ok && t < parts.size(); ++t) {
            const RecvOpt& o = parts[t].second[pick[t]];
            ok = o.edge >= 0 ? apply_all(z1, net.procs[parts[t].first].edges[o.edge].clock_guard)
                             : apply_all(z1, o.extra);
          }
          if (ok) {
            SymState ns;
            ns.locs = s.locs; ns.ints = s.ints;
            ns.locs[i] = se.tgt;
            for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;
            for (int r : se.clock_resets) z1.reset(r);
            for (std::size_t t = 0; t < parts.size(); ++t) {
              const RecvOpt& o = parts[t].second[pick[t]];
              if (o.edge < 0) continue;                       // abstained: nothing moves
              const NEdge& re = net.procs[parts[t].first].edges[o.edge];
              ns.locs[parts[t].first] = re.tgt;
              for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
              for (int r : re.clock_resets) z1.reset(r);
            }
            if (close_succ_zone(net, ns.locs, ns.ints, z1, inv, qmask, L, U, emode, avoid, extra()))
              emit(ns, z1);
          }
        }
        if (parts.empty()) break;
        std::size_t k = 0;
        for (; k < parts.size(); ++k) { if (++pick[k] < parts[k].second.size()) break; pick[k] = 0; }
        if (k == parts.size()) break;
      }
    }
  }
}

/// @name Transition memo
/// @{
/// Caches the successor-zone pipeline, which is a pure function of
/// (source zone, edge, target-location data). Where a model reaches many discrete states but
/// few *distinct* zones, the same O(dim^3) delay/constrain/close otherwise runs a very large
/// number of times over identical input.
///
/// The tables are thread-local, so lookups take no lock and workers share nothing.
///
/// @note Enabled by `--memo`; see hora::set_memo.
/// @see docs/design.md §8.13 for why the adaptive rule tests reuse density and not hit rate.
/// @}

/// @brief Key of the transition memo: everything the successor pipeline reads, and nothing more.
///
/// @invariant @c l is an *interned* signature, not a fixed-width hash of the target locations.
///            A colliding key would return a zone belonging to another transition, which is
///            silent state-space corruption rather than a detectable error, so the memo trades
///            memory for exactness here instead of accepting a collision probability.
struct MemoKey {
  int z;           ///< interned source-zone index, carried by the frontier already
  std::uint32_t t; ///< edge identity, (process, edge index) packed
  std::uint32_t l; ///< interned target-location signature; see ::memo_locsig
  bool operator==(const MemoKey& o) const { return z == o.z && t == o.t && l == o.l; }
};
/// @brief FNV-1a hash of a ::MemoKey.
struct MemoKeyHash {
  static constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;   // FNV-1a, 64-bit
  static constexpr std::uint64_t kFnvPrime  = 1099511628211ull;
  std::size_t operator()(const MemoKey& k) const {
    std::uint64_t h = kFnvOffset;
    h ^= (std::uint64_t)(std::uint32_t)k.z; h *= kFnvPrime;
    h ^= k.t; h *= kFnvPrime;
    h ^= k.l; h *= kFnvPrime;
    return (std::size_t)h;
  }
};
/// @brief One worker's transition memo, together with the counters driving the adaptive rule.
///
/// A sliding window measures recent behaviour. A bad window switches the memo off and doubles a
/// backoff; once the backoff elapses the memo re-probes for one window. The window starts small
/// and grows while the memo is paying, so a model with no reuse is detected within a few hundred
/// lookups and a model with heavy reuse is re-checked ever less often.
///
/// @invariant All thresholds are dimensionless ratios; nothing here is tuned to model size.
struct SuccMemo {
  std::unordered_map<MemoKey, Dbm, MemoKeyHash> m;      ///< an EMPTY Dbm means "this transition dies"
  std::unordered_map<std::string, std::uint32_t> lsig;  ///< location signature -> interned id
  std::string lk;                                       ///< reused signature buffer; avoids allocating per lookup
  long winLookups = 0, winHits = 0;  ///< current window's lookups and hits
  long window = 512, backoff = 512, skipped = 0;  ///< window length, current backoff, lookups skipped while off
  bool on = true;    ///< is the memo currently consulted?
  long lookups = 0;  ///< lifetime lookups, the numerator of reuse density
  static constexpr long kWindowMax = 65536;
  static constexpr long kMaxBackoff = 1L << 22;
  static constexpr double kMinHit = 0.30;      ///< window hit rate below which the memo is disabled
  static constexpr double kMinDensity = 32.0;  ///< lookups per entry below which the table is not worth it
};
/// @brief The calling thread's transition memo.
/// @return a reference to this thread's ::SuccMemo, constructed on first use
inline SuccMemo& tls_succ_memo() { static thread_local SuccMemo sm; return sm; }

/// @brief Is the transition memo enabled for this run?
/// @return the `--memo` option, read once and cached
inline bool memo_enabled() {
  static const bool v = detail::options().memo;
  return v;
}

/// @brief Intern the part of a location vector that the successor pipeline can observe.
///
/// The pipeline reads the target locations only through the combined invariant, the
/// urgent/committed flags and the active-clock mask. Location vectors agreeing on those three
/// are interchangeable for the memo, and collapsing them is where the reuse comes from.
///
/// @param net  the network under analysis
/// @param l    the target location vector
/// @param[in,out] sm  the worker's memo, whose signature table is extended on a miss
/// @return a dense id, equal for two location vectors exactly when the pipeline cannot tell
///         them apart
///
/// @note Walks every process but emits only for those whose location carries an invariant or a
///       frozen flag, which is normally a small minority.
inline std::uint32_t memo_locsig(const Network& net, const LocVec& l, SuccMemo& sm) {
  sm.lk.clear();
  auto enc = [&](std::uint32_t v) {
    sm.lk += char(v & 0xff); sm.lk += char((v >> 8) & 0xff);
    sm.lk += char((v >> 16) & 0xff); sm.lk += char((v >> 24) & 0xff);
  };
  for (int p = 0; p < net.nprocs(); ++p) {
    const Location& loc = net.procs[p].locations[l[p]];
    if (!loc.invariant.empty()) { enc((std::uint32_t)p); enc((std::uint32_t)l[p]); }
    if (loc.urgent || loc.committed) enc(0x80000000u | (std::uint32_t)p);
  }
  if (!net.active_mask.empty()) {
    std::uint64_t a = 0;
    for (int p = 0; p < net.nprocs(); ++p) a |= net.active_mask[p][l[p]];
    enc((std::uint32_t)(a & 0xffffffffu)); enc((std::uint32_t)(a >> 32));
  } else if (!net.active.empty()) {
    for (int c = 1; c <= net.nclocks; ++c) {
      bool live = false;
      for (int p = 0; p < net.nprocs() && !live; ++p) if (net.active[p][l[p]][c]) live = true;
      if (live) enc((std::uint32_t)c);
    }
  }
  auto it = sm.lsig.find(sm.lk);
  if (it != sm.lsig.end()) return it->second;
  const std::uint32_t id = (std::uint32_t)sm.lsig.size();
  sm.lsig.emplace(sm.lk, id);
  return id;
}

/// @brief Advance the adaptive window and decide whether the memo is live.
///
/// @param[in,out] sm  the worker's memo
/// @return @c true if the memo should be consulted for this lookup
/// @post When the backoff reaches its maximum the table is released, on the grounds that a memo
///       this consistently unhelpful will not become useful later in the same run.
inline bool memo_tick(SuccMemo& sm) {
  if (!sm.on) {
    if (++sm.skipped >= sm.backoff) { sm.on = true; sm.skipped = 0; sm.winLookups = sm.winHits = 0; }
    return sm.on;
  }
  if (sm.winLookups >= sm.window) {
    // Two criteria, because hit rate alone admits tables that are large and thinly reused: a
    // high hit rate spread over very many entries still costs more in lookups and memory
    // pressure than the pipeline it saves. Reuse density (lookups per entry) separates those
    // cases where hit rate does not. See docs/design.md 8.13.
    const double density = sm.m.empty() ? 0.0 : double(sm.lookups) / double(sm.m.size());
    const bool thin = sm.m.size() > 4096 && density < SuccMemo::kMinDensity;
    if (thin || double(sm.winHits) / double(sm.winLookups) < SuccMemo::kMinHit) {
      sm.on = false;
      sm.backoff = sm.backoff < SuccMemo::kMaxBackoff ? sm.backoff * 2 : SuccMemo::kMaxBackoff;
      sm.window = 512;
      if (sm.backoff >= SuccMemo::kMaxBackoff) {
        std::unordered_map<MemoKey, Dbm, MemoKeyHash>().swap(sm.m);   // confident: release
      }
    } else {
      sm.backoff = 512;
      if (sm.window < SuccMemo::kWindowMax) sm.window *= 2;
    }
    sm.winLookups = sm.winHits = 0;
  }
  return sm.on;
}

/// @brief Slot-reusing successor generation for the shared-passed hot path.
///
/// Semantically identical to ::successors_net, but builds each successor in place in a reused
/// slot of @p out instead of constructing and appending a fresh state per successor. The passed
/// set interns each zone by copy and never takes ownership of the state, so the slots stay valid
/// across the caller's loop and are reused on the next frontier state; after a short warm-up
/// this generates successors without allocating.
///
/// @param net    the network under analysis
/// @param s      the source state; only its @c locs and @c ints are read
/// @param szone  the source zone, supplied separately from @p s
/// @param L      lower-bound array per clock, for the extrapolation
/// @param U      upper-bound array per clock, for the extrapolation
/// @param qmask  clocks the query mentions, which the active-clock reduction must not reset
/// @param emode  extrapolation mode applied when closing each successor zone
/// @param[in,out] out  successor slots, reused across calls and grown as needed
/// @param szidx  interned index of @p szone, or a negative value if it is not interned
/// @return the number of successors written, i.e. the valid range is `out[0 .. return)`
///
/// @warning @p out is **not** cleared. Entries beyond the returned count are stale successors
///          from an earlier call and must not be read.
/// @note The memo is consulted only when @p szidx is non-negative, since the interned index is
///       part of the memo key.
/// @note ::successors_net remains for the sequential and single-automaton paths, which
///       accumulate successors rather than reuse slots.
std::size_t gen_succ_reuse(const Network& net, const SymState& s, const Dbm& szone,
                           const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U,
                           const std::vector<char>& qmask, int emode, std::vector<SymState>& out,
                           int szidx) {
  thread_local std::vector<Constraint> inv;
  SuccMemo& sm = tls_succ_memo();
  // The memo key is (source zone, edge, target location signature) — it does not carry the
  // integer valuation, which was sound while the closing pipeline read only the target locations
  // and the zone. With urgent channels the delay in close_succ_zone additionally depends on the
  // target integer valuation, so two states agreeing on all three key components but differing
  // in their integers would alias to one entry with the wrong delay. Widening the key would tax
  // every model; disabling the memo taxes only models that declare an urgent channel.
  const bool memo = memo_enabled() && szidx >= 0 && !net.has_urgent_chan;
  std::size_t on = 0;
  auto slot = [&]() -> SymState& { if (on == out.size()) out.emplace_back(); return out[on]; };
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, s.ints, &src_committed, &src_urgent);

  // ---- location edges ----
  for (int p = 0; p < net.nprocs(); ++p) {
    if (src_committed && !net.procs[p].locations[s.locs[p]].committed) continue;
    for (int ei : net.procs[p].edges_of(s.locs[p])) {
      const NEdge& e = net.procs[p].edges[ei];
      if (e.sync_chan >= 0) continue;
      if (!int_ok(e.int_guard, s.ints) || !gen_ok(net, e, s.ints.data(), s.ints.size())) continue;
      SymState& ns = slot();
      // The discrete half first, so the memo can be consulted before any DBM work. This is a
      // pure reordering: every `continue` below is a discrete decision that does not read the
      // zone, and on the miss path the zone pipeline runs exactly as before.
      ns.locs.assign(s.locs.begin(), s.locs.end());
      ns.locs[p] = e.tgt;
      ns.ints.assign(s.ints.begin(), s.ints.end());
      for (const auto& a : e.int_updates) ns.ints[a.var] = a.val;
      if (!gen_apply(net, e, ns.ints.data(), ns.ints.size())) continue;
      if (memo && memo_tick(sm)) {
        const MemoKey mk{szidx, (std::uint32_t)((p << 16) | ei), memo_locsig(net, ns.locs, sm)};
        ++sm.lookups; ++sm.winLookups;
        auto mit = sm.m.find(mk);
        if (mit != sm.m.end()) {
          ++sm.winHits;
          if (mit->second.empty()) continue;             // memoised: this transition dies
          ns.zone = mit->second;
          ++on;
          continue;
        }
        // miss: run the pipeline, then record. `memo_pend` marks that we owe an insert.
        ns.zone = szone;
        if (!apply_all(ns.zone, e.clock_guard)) { sm.m.emplace(mk, Dbm()); continue; }
        for (int r : e.clock_resets) ns.zone.reset(r);
        if (!close_succ_zone(net, ns.locs, ns.ints, ns.zone, inv, qmask, L, U, emode)) {
          sm.m.emplace(mk, Dbm()); continue;
        }
        sm.m.emplace(mk, ns.zone);                       // records empty zones too
        if (ns.zone.empty()) continue;
        ++on;
        continue;
      }
      ns.zone = szone;                                    // into reused capacity
      if (!apply_all(ns.zone, e.clock_guard)) continue;
      for (int r : e.clock_resets) ns.zone.reset(r);
      if (!close_succ_zone(net, ns.locs, ns.ints, ns.zone, inv, qmask, L, U, emode)) continue;
      if (ns.zone.empty()) continue;
      ++on;
    }
  }

  // ---- binary channel synchronisation ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1) continue;
      if (net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      const bool i_comm = net.procs[i].locations[s.locs[i]].committed;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan != se.sync_chan || re.sync_dir != -1) continue;
          if (src_committed && !(i_comm || net.procs[j].locations[s.locs[j]].committed)) continue;
          if (!int_ok(re.int_guard, s.ints) || !gen_ok(net, re, s.ints.data(), s.ints.size())) continue;
          SymState& ns = slot();
          // Discrete half first, so the memo can be consulted before any DBM work. Channel
          // models (viking, phil) take almost all their transitions through this path, so
          // memoising only the local-edge site above leaves them untouched.
          ns.locs.assign(s.locs.begin(), s.locs.end());
          ns.locs[i] = se.tgt;
          ns.locs[j] = re.tgt;
          ns.ints.assign(s.ints.begin(), s.ints.end());
          for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;
          for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
          if (!gen_apply(net, se, ns.ints.data(), ns.ints.size()) || !gen_apply(net, re, ns.ints.data(), ns.ints.size())) continue;
          if (memo && memo_tick(sm)) {
            const std::uint32_t tsig =
                (std::uint32_t)(((i * 1000003 + ei) * 31u) ^ (std::uint32_t)(j * 1000003 + ej));
            const MemoKey mk{szidx, tsig, memo_locsig(net, ns.locs, sm)};
            ++sm.lookups; ++sm.winLookups;
            auto mit = sm.m.find(mk);
            if (mit != sm.m.end()) {
              ++sm.winHits;
              if (mit->second.empty()) continue;
              ns.zone = mit->second; ++on; continue;
            }
            ns.zone = szone;
            if (!apply_all(ns.zone, se.clock_guard)) { sm.m.emplace(mk, Dbm()); continue; }
            if (!apply_all(ns.zone, re.clock_guard)) { sm.m.emplace(mk, Dbm()); continue; }
            for (int r : se.clock_resets) ns.zone.reset(r);
            for (int r : re.clock_resets) ns.zone.reset(r);
            if (!close_succ_zone(net, ns.locs, ns.ints, ns.zone, inv, qmask, L, U, emode)) {
              sm.m.emplace(mk, Dbm()); continue;
            }
            sm.m.emplace(mk, ns.zone);
            if (ns.zone.empty()) continue;
            ++on; continue;
          }
          ns.zone = szone;
          if (!apply_all(ns.zone, se.clock_guard)) continue;
          if (!apply_all(ns.zone, re.clock_guard)) continue;
          for (int r : se.clock_resets) ns.zone.reset(r);
          for (int r : re.clock_resets) ns.zone.reset(r);
          if (!close_succ_zone(net, ns.locs, ns.ints, ns.zone, inv, qmask, L, U, emode)) continue;
          if (ns.zone.empty()) continue;
          ++on;
        }
      }
    }
  }

  // ---- broadcast synchronisation ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1 || !net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      std::vector<std::pair<int, std::vector<RecvOpt>>> parts;   // see build_broadcast_parts
      build_broadcast_parts(net, s, se, i, parts);
      std::vector<std::size_t> pick(parts.size(), 0);
      for (;;) {
        bool part_committed = net.procs[i].locations[s.locs[i]].committed;
        for (std::size_t t = 0; t < parts.size(); ++t)
          if (parts[t].second[pick[t]].edge >= 0 &&
              net.procs[parts[t].first].locations[s.locs[parts[t].first]].committed) part_committed = true;
        if (!src_committed || part_committed) {
          SymState& ns = slot();
          ns.zone = szone;
          bool ok = apply_all(ns.zone, se.clock_guard);
          for (std::size_t t = 0; ok && t < parts.size(); ++t) {
            const RecvOpt& o = parts[t].second[pick[t]];
            ok = o.edge >= 0 ? apply_all(ns.zone, net.procs[parts[t].first].edges[o.edge].clock_guard)
                             : apply_all(ns.zone, o.extra);
          }
          if (ok) {
            ns.locs.assign(s.locs.begin(), s.locs.end());
            ns.ints.assign(s.ints.begin(), s.ints.end());
            ns.locs[i] = se.tgt;
            for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;
            for (int r : se.clock_resets) ns.zone.reset(r);
            for (std::size_t t = 0; t < parts.size(); ++t) {
              const RecvOpt& o = parts[t].second[pick[t]];
              if (o.edge < 0) continue;                     // abstained: nothing moves
              const NEdge& re = net.procs[parts[t].first].edges[o.edge];
              ns.locs[parts[t].first] = re.tgt;
              for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
              for (int r : re.clock_resets) ns.zone.reset(r);
            }
            if (close_succ_zone(net, ns.locs, ns.ints, ns.zone, inv, qmask, L, U, emode) &&
                !ns.zone.empty()) ++on;
          }
        }
        if (parts.empty()) break;
        std::size_t k = 0;
        for (; k < parts.size(); ++k) { if (++pick[k] < parts[k].second.size()) break; pick[k] = 0; }
        if (k == parts.size()) break;
      }
    }
  }
  return on;
}

/// @brief Widen the LU bound arrays so they also dominate the constraints in @p cs.
///
/// The extrapolation is parameterised by the largest constant each clock is compared against.
/// A query adds constraints that the network itself never states, so its constants must be
/// folded into the bounds before exploration or the extrapolation could merge two states the
/// query distinguishes.
///
/// @param[in,out] L  per-clock lower bounds, widened in place
/// @param[in,out] U  per-clock upper bounds, widened in place
/// @param cs         constraints to absorb
///
/// @pre  @p L and @p U are indexed by clock and are at least as long as the largest index in
///       @p cs.
/// @post @p L and @p U are pointwise greater than or equal to their previous values; the
///       function only ever widens, so it is safe to call repeatedly.
/// @note A diagonal constraint bounds neither clock on its own, so its magnitude is applied to
///       both clocks in both directions — the conservative choice that keeps the abstraction
///       sound whichever of the two the extrapolation later tests.
void extend_bounds(std::vector<std::int32_t>& L, std::vector<std::int32_t>& U,
                   const std::vector<Constraint>& cs) {
  for (const auto& c : cs) {
    const int v = raw_bound(c.bound);
    if (c.i > 0 && c.j == 0) { if (v > U[c.i]) U[c.i] = v; }
    else if (c.i == 0 && c.j > 0) { const int lb = -v; if (lb > L[c.j]) L[c.j] = lb; }
    else if (c.i > 0 && c.j > 0) {
      const int a = std::abs(v);
      if (a > U[c.i]) U[c.i] = a;
      if (a > U[c.j]) U[c.j] = a;
      if (a > L[c.i]) L[c.i] = a;
      if (a > L[c.j]) L[c.j] = a;
    }
  }
}

}  // namespace

/// @brief Rewrite a symbolic state to the canonical representative of its symmetry orbit.
///
/// Chooses the group element yielding the lexicographically smallest location vector and, among
/// those, the smallest zone matrix. Exploring only representatives shrinks the state space by up
/// to a factor of |G|.
///
/// @param G    the automorphism group of the network
/// @param net  the network under analysis
/// @param[in,out] s  the state, rewritten in place
///
/// @pre  Every element of @p G is an automorphism of @p net, so that applying it preserves the
///       transition relation.
/// @post @p s is the canonical representative of its orbit: two states in the same orbit are
///       rewritten to identical (locs, ints, zone) triples.
///
/// @note The location pass decides the winner in almost every case, so the zone is permuted only
///       for the few candidates that tie; the cost is |G| x nprocs integer comparisons per state.
/// @note Permuting a canonical DBM by a clock permutation leaves it canonical — the triangle
///       inequality is preserved under relabelling — so no re-closure is needed afterwards.
static void canonicalize(const std::vector<Automorphism>& G, const Network& net, SymState& s) {
  const int np = static_cast<int>(s.locs.size());
  const int ni = static_cast<int>(s.ints.size());
  const int dim = s.zone.dim();
  // The discrete key is (locations, integer values) and both must move together. Permuting only
  // the locations is not a weaker canonicalisation but an unsound one, because an integer
  // variable may itself hold a process index: rotating the location vector while such a variable
  // still names the old process yields a state outside the orbit, reachable in the reduced graph
  // and unreachable in the real one.
  thread_local std::vector<int> cur, best, win;
  cur.resize(np + ni); best.resize(np + ni);
  auto image = [&](const Automorphism& a, std::vector<int>& out) {
    permute_locs(a, s.locs.data(), np, out.data());
    for (int v = 0; v < ni; ++v)
      out[np + v] = static_cast<int>(a.map_val(v, s.ints[v], net.intvars[v].lo));
  };
  image(G[0], best);
  win.assign(1, 0);
  for (std::size_t g = 1; g < G.size(); ++g) {
    image(G[g], cur);
    const int c = std::memcmp(cur.data(), best.data(), (np + ni) * sizeof(int));
    if (c < 0) { best.swap(cur); win.assign(1, static_cast<int>(g)); }
    else if (c == 0) win.push_back(static_cast<int>(g));
  }
  thread_local std::vector<raw_t> bz, cz;
  if (win.size() > 1 || win[0] != 0) {
    const std::size_t d2 = static_cast<std::size_t>(dim) * dim;
    bz.resize(d2); cz.resize(d2);
    bool first = true;
    for (const int g : win) {
      const auto& clk = G[g].clk;
      const raw_t* z = s.zone.data().data();
      for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) cz[clk[i] * dim + clk[j]] = z[i * dim + j];
      if (first || std::memcmp(cz.data(), bz.data(), d2 * sizeof(raw_t)) < 0) { bz.swap(cz); first = false; }
    }
    std::copy(bz.begin(), bz.end(), s.zone.data().begin());
    for (int p = 0; p < np; ++p) s.locs[p] = best[p];
    for (int v = 0; v < ni; ++v) s.ints[v] = best[np + v];
  }
}

/// @brief Report that the network's discrete state does not fit the packed key, and throw.
///
/// @param net         the network under analysis
/// @param bits_needed total width the packed key would require
/// @param field_bits  the same width function ::DiscKeyCodec uses, so the breakdown agrees with
///                    the packing that failed
/// @throws std::runtime_error always
///
/// @note The message lists the widest fields because the offender is usually a small number of
///       integer variables declared over the language's default range, and narrowing those is
///       the only part of this the user can act on.
/// @see hora::set_narrow_ranges, which derives tighter ranges and often avoids this.
/// @brief Bits needed to distinguish @p nvalues values.
/// @param nvalues number of distinct values the field must hold
/// @return the smallest @c b with `2^b >= nvalues`; @c 0 for a single-valued field
static int disc_field_bits(long nvalues) { int b = 0; while ((1LL << b) < nvalues) ++b; return b; }

/// @brief Total width the packed discrete key needs for @p n, in bits.
///
/// @param n the network under analysis
/// @return the summed field widths, one field per process and per integer variable
///
/// @note Computed without building a codec, because ::reachable_net must consult it to CHOOSE the
///       key type before any codec exists. It must agree with ::DiscKeyCodecT's own layout, which
///       is why both go through ::disc_field_bits.
static int disc_key_bits(const Network& n) {
  int off = 0;
  for (int p = 0; p < n.nprocs(); ++p)
    off += disc_field_bits(static_cast<long>(n.procs[p].locations.size()));
  for (std::size_t v = 0; v < n.intvars.size(); ++v)
    off += disc_field_bits(n.intvars[v].hi - n.intvars[v].lo + 1);
  return off;
}

[[noreturn]] static void throw_disc_key_too_wide(const Network& net, int bits_needed, int budget,
                                                 int (*field_bits)(long)) {
  std::vector<std::pair<int, std::string>> widest;
  for (int p = 0; p < net.nprocs(); ++p)
    widest.emplace_back(field_bits(static_cast<long>(net.procs[p].locations.size())),
                        "proc[" + std::to_string(p) + "] (" +
                        std::to_string(net.procs[p].locations.size()) + " locs)");
  for (std::size_t v = 0; v < net.intvars.size(); ++v)
    widest.emplace_back(field_bits(net.intvars[v].hi - net.intvars[v].lo + 1),
                        "int " + net.intvars[v].name + " [" + std::to_string(net.intvars[v].lo) +
                        "," + std::to_string(net.intvars[v].hi) + "]");
  std::stable_sort(widest.begin(), widest.end(),
                   [](const auto& a, const auto& b) { return a.first > b.first; });
  std::string msg = "reachable_net: discrete state needs " + std::to_string(bits_needed) +
                    " bits but the packed key holds " + std::to_string(budget) +
                    " (network too wide).\n"
                    "  " + std::to_string(net.nprocs()) + " processes + " +
                    std::to_string(net.intvars.size()) + " int vars. Widest fields:";
  for (std::size_t i = 0; i < widest.size() && i < 8; ++i)
    msg += "\n    " + std::to_string(widest[i].first) + " bits  " + widest[i].second;
  throw std::runtime_error(msg);
}

/// @brief Lossless packing of a symbolic state's discrete half into a single integer key.
///
/// The location vector and the integer variables occupy one fixed-width bit-field each, sized
/// from the number of locations of each process and from each variable's declared range. Keys
/// are then compared and hashed as plain integers rather than as formatted strings.
///
/// @invariant The encoding is injective: two discrete states share a key only if they are equal.
///            This is what lets the passed set be keyed on the packed value alone.
/// @invariant Total width is at most `key_budget<Key>()` bits, so every field offset is strictly
///            below the key's width and a zero-width trailing field cannot produce an undefined
///            shift.
///
/// @note Built once per search and thereafter read-only, so it is safe to share across the
///       parallel level expansion without synchronisation.
/// @warning A network whose fields do not fit is rejected outright by ::throw_disc_key_too_wide
///          rather than being packed lossily; an aliased key would silently merge distinct
///          states.
/// @tparam Key the integer type the packing lands in; see detail/disc_key.hpp.
template <class Key>
struct DiscKeyCodecT {
  const Network& net;
  std::vector<int> shift;  ///< bit offset of each field: procs first, then int vars
  std::vector<Key> mask;   ///< per-field value mask, for decoding a key back apart

  static int field_bits(long nvalues) { return disc_field_bits(nvalues); }

  /// @brief Lay out one bit-field per process and per integer variable.
  /// @param n the network whose discrete states are to be packed
  /// @throws std::runtime_error if the fields need 64 bits or more
  explicit DiscKeyCodecT(const Network& n) : net(n) {
    const int nfields = n.nprocs() + static_cast<int>(n.intvars.size());
    shift.resize(nfields);
    mask.resize(nfields);
    int off = 0;
    auto set = [&](int f, int w) {
      shift[f] = off;
      mask[f] = (w >= static_cast<int>(sizeof(Key) * 8))
                    ? ~static_cast<Key>(0) : ((static_cast<Key>(1) << w) - 1);
      off += w;
    };
    for (int p = 0; p < n.nprocs(); ++p)
      set(p, field_bits(static_cast<long>(n.procs[p].locations.size())));
    for (std::size_t v = 0; v < n.intvars.size(); ++v)
      set(n.nprocs() + static_cast<int>(v), field_bits(n.intvars[v].hi - n.intvars[v].lo + 1));
    // The budget is one bit short of the key's width so that every field offset stays strictly
    // below it; a trailing zero-width field would otherwise be shifted by exactly the width,
    // which is undefined.
    if (off > key_budget<Key>()) throw_disc_key_too_wide(n, off, key_budget<Key>(), &field_bits);
  }

  /// @brief Pack a discrete state into its key.
  /// @param locs the location vector, one entry per process
  /// @param ints the integer variables, one entry per declared variable
  /// @return the packed key
  /// @pre Every value lies in its field's declared range; ::gen_apply enforces this for updates.
  Key encode(const LocVec& locs, const IntVec& ints) const {
    Key k = 0;
    for (int p = 0; p < static_cast<int>(locs.size()); ++p)
      k |= static_cast<Key>(static_cast<unsigned>(locs[p])) << shift[p];
    for (std::size_t v = 0; v < ints.size(); ++v)
      k |= static_cast<Key>(ints[v] - net.intvars[v].lo) << shift[net.nprocs() + v];
    return k;
  }

  /// @brief Unpack a key back into a discrete state.
  /// @param dk        a key previously produced by ::DiscKeyCodec::encode
  /// @param[out] locs receives one location per process
  /// @param[out] ints receives one value per integer variable
  /// @pre @p locs and @p ints have room for @c nprocs and @c intvars.size() entries.
  /// @post `decode(encode(l, i))` reproduces @p l and @p i exactly.
  void decode(Key dk, int* locs, long* ints) const {
    for (int p = 0; p < net.nprocs(); ++p)
      locs[p] = static_cast<int>((dk >> shift[p]) & mask[p]);
    for (std::size_t v = 0; v < net.intvars.size(); ++v)
      ints[v] = static_cast<long>((dk >> shift[net.nprocs() + v]) & mask[net.nprocs() + v]) +
                net.intvars[v].lo;
  }
};

/// @brief The codec for a network whose discrete state fits one word.
using DiscKeyCodec = DiscKeyCodecT<NarrowKey>;

/// @brief Maps a packed discrete key to a lock stripe and to a work bucket.
///
/// *Stripes* are the lock granularity of the shared passed table. A given discrete state always
/// lands in the same stripe, so a worker updating it locks that stripe alone, and stripes greatly
/// outnumber threads so collisions are rare.
///
/// *Buckets* are the work granularity of the per-level survivor collection.
///
/// @invariant Both indices come from the high bits of the same multiplicative mix, so each bucket
///            owns a contiguous range of stripes and its scan stays within one slice of the
///            stripe array rather than scattering across all of it.
/// @note Sized at roughly four buckets per thread, which keeps a dynamic schedule balanced
///       without making the per-bucket work too small to amortise its bookkeeping.
/// @note Read-only once built, so it is shared across workers without synchronisation.
struct StripeLayout {
  static constexpr std::uint64_t kMix = 0x9e3779b97f4a7c15ULL;  ///< Fibonacci hashing multiplier

  std::size_t nstripes = 0;
  int stripe_shift = 0;
  int nbuckets = 1;
  int bucket_shift = 0;

  /// @brief Size the stripe and bucket arrays for a given worker count.
  /// @param nthreads number of workers that will share the passed table
  /// @post Both counts are powers of two, so the index functions need only a shift.
  explicit StripeLayout(int nthreads) {
    // The 16384 floor is a SPEED floor, not a contention one. Scaling the count to nthreads*64
    // saves real memory (at 8 threads the floor is 32x more stripes than collisions need, and
    // for arbiter12 it was 70% of accounted memory) but measured SLOWER, so the floor stays:
    // more stripes also means shorter per-stripe chains to scan on every lookup, and that scan
    // is on the hot path where the memory is not.
    int stripe_bits = 10;                                    // floor; grown below past nthreads
    while ((1 << stripe_bits) < std::max(16384, nthreads * 64)) ++stripe_bits;
    nstripes = static_cast<std::size_t>(1) << stripe_bits;
    stripe_shift = 64 - stripe_bits;

    int bucket_bits = 0;
    while ((1 << bucket_bits) < nthreads * 4) ++bucket_bits;
    nbuckets = bucket_bits ? (1 << bucket_bits) : 1;
    bucket_shift = 64 - bucket_bits;
  }

  /// @brief Lock stripe owning a discrete key.
  /// @param dk the packed discrete key
  /// @return an index in `[0, nstripes)`
  template <class Key>
  std::size_t stripe_of(Key dk) const {
    return static_cast<std::size_t>(key_mix(dk) >> stripe_shift);
  }
  /// @brief Work bucket owning a discrete key.
  /// @param dk the packed discrete key
  /// @return an index in `[0, nbuckets)`
  /// @pre `nbuckets > 1`. With a single bucket @c bucket_shift is 64 and the shift is undefined;
  ///      callers take a separate single-bucket path instead.
  template <class Key>
  std::size_t bucket_of(Key dk) const {
    return static_cast<std::size_t>(key_mix(dk) >> bucket_shift);
  }
};

/// @brief Mark-and-sweep the zone interner, reclaiming zones nothing references any more.
///
/// Zones become garbage when the passed set subsumes or merges out the node that held them.
/// Survivors are those referenced by a live arena node or by the next frontier; everything else
/// returns to the interner's free list for reuse in place.
///
/// @param[in,out] zi  the zone interner to sweep
/// @param smap        the striped passed set, scanned for live arena nodes
/// @param frontier    the next frontier, whose zone indices are also roots
/// @param nthreads    workers to use for the mark phase
/// @param floor_      lower bound on the returned threshold, so sweeps cannot become too frequent
/// @return the allocation count at which to sweep again
///
/// @pre No worker is mutating @p smap or @p zi for the duration of the call.
/// @post Every zone index still reachable from @p smap or @p frontier remains valid; all others
///       are freed.
///
/// @note The mark phase runs in parallel and every writer stores the same value, with nothing
///       reading the marks until the sweep, so racing writers cannot disagree. The store is a
///       relaxed atomic to keep that formally race-free; on the supported architectures it
///       compiles to a plain byte store.
template <class Key>
static std::size_t sweep_dead_zones(ZoneIntern& zi, std::vector<NetPassedT<Key>>& smap,
                                    const std::vector<std::pair<Key, int>>& frontier,
                                    int nthreads, std::size_t floor_) {
  std::vector<std::uint8_t> alive(zi.index_bound(), 0);
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 8) num_threads(nthreads)
#endif
  for (long s = 0; s < static_cast<long>(smap.size()); ++s)
    for (const auto& nd : smap[s].arena)
      if (!nd.dead) __atomic_store_n(&alive[nd.zidx], std::uint8_t(1), __ATOMIC_RELAXED);
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
  for (long k = 0; k < static_cast<long>(frontier.size()); ++k)
    __atomic_store_n(&alive[frontier[k].second], std::uint8_t(1), __ATOMIC_RELAXED);
  zi.sweep(alive);
  return std::max<std::size_t>(floor_, zi.allocated() * 2);
}

/// @brief Collect one level's survivors into the next frontier.
///
/// Gathers the passed-set entries stamped with @p level as (discrete key, zone index) pairs. No
/// symbolic state is built and no zone is copied here; that happens lazily at the next expansion.
///
/// @param[in,out] smap  the striped passed set
/// @param lay           stripe and bucket layout, supplying the bucket assignment
/// @param[in,out] tls_touched  per-worker sets of discrete keys touched during the expansion
/// @param nthreads      worker count
/// @param single        take the single-bucket path, for layouts with only one bucket
/// @param level         the level stamp identifying this level's survivors
/// @return the next frontier
///
/// @post Each discrete key appears at most once in the result.
///
/// @note Each discrete key must be scanned exactly once, so the per-worker touched sets have to
///       be merged. Keys are bucketed by hash and each bucket deduplicated independently: a key
///       hashes to exactly one bucket, so that bucket observes every copy of it and the result
///       equals the serial union. Only the order differs, and no caller depends on it.
/// @note Deduplication and the survivor scan share one parallel loop, so the merged key set is
///       never materialised.
template <class Key>
static std::vector<std::pair<Key, int>> collect_next_frontier(
    std::vector<NetPassedT<Key>>& smap, const StripeLayout& lay,
    std::vector<std::unordered_set<Key, typename KeyHashFor<Key>::type>>& tls_touched,
    int nthreads, bool single, int level) {
  const int nbuk = single ? 1 : lay.nbuckets;
  // part[t * nbuk + b]: thread t's touched keys destined for bucket b.
  std::vector<std::vector<Key>> part(static_cast<std::size_t>(nthreads) * nbuk);
  if (single) {
    part[0].assign(tls_touched[0].begin(), tls_touched[0].end());
  } else {
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
    for (long t = 0; t < static_cast<long>(nthreads); ++t) {
      const std::size_t base = static_cast<std::size_t>(t) * nbuk;
      for (const Key dk : tls_touched[t]) part[base + lay.bucket_of(dk)].push_back(dk);
    }
  }
  std::vector<std::vector<std::pair<Key, int>>> nf(nbuk);
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
#endif
  for (long b = 0; b < static_cast<long>(nbuk); ++b) {
    auto& buf = nf[b];
    // `single` already deduplicated inside the one touched set, so no second pass is
    // needed there; the parallel path can see the same dk from several threads.
    if (single) {
      for (const Key dk : part[b])
        smap[lay.stripe_of(dk)].for_level(dk, level, [&](int zidx) { buf.push_back({dk, zidx}); });
      continue;
    }
    std::unordered_set<Key, typename KeyHashFor<Key>::type> seen;
    for (int t = 0; t < nthreads; ++t) {
      for (const Key dk : part[static_cast<std::size_t>(t) * nbuk + b]) {
        if (!seen.insert(dk).second) continue;
        smap[lay.stripe_of(dk)].for_level(dk, level, [&](int zidx) { buf.push_back({dk, zidx}); });
      }
    }
  }
  std::vector<std::pair<Key, int>> next;
  std::size_t ntot = 0;
  for (auto& v : nf) ntot += v.size();
  next.reserve(ntot);                    // one allocation instead of a realloc chain
  for (auto& v : nf) next.insert(next.end(), v.begin(), v.end());
  return next;
}

/// @brief Build the network's initial symbolic state.
///
/// Every process sits in its initial location and every integer variable holds its declared
/// initial value. The zero zone is then opened up by the same closing pipeline every successor
/// goes through: combined invariant, delay unless the state is frozen, active-clock reduction and
/// extrapolation.
///
/// @param net    the network under analysis
/// @param qmask  clocks the query reads, which the active-clock reduction must preserve
/// @param L      per-clock lower bounds
/// @param U      per-clock upper bounds
/// @param emode  extrapolation mode, applied here exactly as to every successor
/// @param[out] s0  receives the initial state; meaningful only on @c true
/// @return @c false when the initial invariant is unsatisfiable, i.e. the model has no initial
///         state at all
///
/// @post On @c true, `s0->zone` is non-empty and closed.
/// @note Using the successor pipeline rather than a separate construction is what keeps the root
///       comparable with everything the search later stores; a root closed differently could fail
///       to subsume states it ought to.
static bool initial_symbolic_state(const Network& net, const std::vector<char>& qmask,
                                   const std::vector<std::int32_t>& L,
                                   const std::vector<std::int32_t>& U, int emode, SymState* s0,
                                   const detail::DelayAvoidSet* avoid = nullptr,
                                   std::vector<Dbm>* extra = nullptr) {
  s0->locs.resize(net.nprocs());
  for (int p = 0; p < net.nprocs(); ++p) s0->locs[p] = net.procs[p].initial;
  s0->ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0->ints[v] = net.intvars[v].init;
  s0->zone = Dbm(net.nclocks);
  std::vector<Constraint> inv;
  return close_succ_zone(net, s0->locs, s0->ints, s0->zone, inv, qmask, L, U, emode, avoid, extra);
}

/// @brief The shared symbolic-reachability engine.
///
/// One engine behind every "is a state satisfying X reachable?" query in the build. It owns the
/// sharded passed list, the zone interner, the parallel level expansion and the interner garbage
/// collection; callers supply only what genuinely differs between queries.
///
/// @tparam TargetFn  predicate on a symbolic state
///
/// @param net       the network under analysis
/// @param target    the query's target, needed for its clock constants — which floor the a_LU
///                  bounds and build @c qmask
/// @param is_target what makes a symbolic state a witness. A predicate rather than a
///                  hora::NetTarget so that a target may inspect the *zone*, which deadlock
///                  must
/// @param force_exact  compare stored zones by plain inclusion instead of the a_LU test
/// @param backend   which expansion backend to run
/// @param[out] stats  optional statistics; unchanged when null
/// @param sym       optional automorphism group enabling symmetry reduction
/// @param orbit     optional per-orbit targets, used with @p sym
/// @param cancel    optional cancellation flag, polled once per BFS level
/// @return @c true if a state satisfying @p is_target is reachable; @c false if none is, **or**
///         if the search was cancelled
///
/// @warning @p force_exact must be set whenever the property is per-*valuation*. a_LU is a
///          location-reachability abstraction and may drop concrete valuations, so a
///          valuation-sensitive target such as deadlock would be unsound without it. It also
///          changes how finiteness is obtained; see the extrapolation mode below.
/// @warning Because cancellation and exhaustion both return @c false, the result of a search the
///          caller cancelled carries no information and must not be consulted.
///
/// @note @p cancel is polled at the level boundary rather than per state, so it costs one relaxed
///       load per level. Abandoning between levels leaves nothing half-built, since the caller
///       discards the whole engine.
/// @see docs/design.md §4 for why stored zones must be exact and what that costs.
/// @brief Would canonical-order insertion pay for itself on @p net at @p nthreads workers?
///
/// Striping the passed table only helps when there are more discrete keys to spread over than
/// there are workers: below that, every worker contends for the same antichain, so parallel
/// insertion wins no concurrency while still paying the convex-union merge's order-sensitivity.
/// `single` is the extreme -- one location, no integer variables, so a single discrete key holds
/// the entire state space -- and there the trade is not close: canonical insertion explores 3,494
/// states in 1.7 s where the racing one explores 39,899 in 47 s.
///
/// The count is a STATIC property of the model, so the decision cannot vary between runs; that
/// matters, because a switch driven by observed counts would be driven by the very nondeterminism
/// it exists to remove. It saturates rather than overflowing -- any model past the threshold is
/// equally "large enough".
static bool auto_deterministic(const Network& net, int nthreads) {
  unsigned long long keys = 1;
  const unsigned long long cap = 1ull << 40;
  for (const auto& p : net.procs) {
    keys *= std::max<std::size_t>(1, p.locations.size());
    if (keys >= cap) return false;
  }
  for (const auto& v : net.intvars) {
    keys *= static_cast<unsigned long long>(std::max<long>(1, v.hi - v.lo + 1));
    if (keys >= cap) return false;
  }
  return keys <= static_cast<unsigned long long>(std::max(1, nthreads));
}

/// @brief A successor held back for canonical-order insertion under --deterministic.
/// @note `i`/`sk` are the frontier index and the successor index within that state -- together
///       the position the sequential engine would have inserted this successor at.
template <class Key>
struct PendingT {
  std::uint32_t i, sk;
  Key dk;
  SymState s;
};

template <class Key, class TargetFn>
static bool reach_engine(const Network& net, const NetTarget& target, TargetFn&& is_target,
                         bool force_exact, Backend backend, ReachStats* stats,
                         const std::vector<Automorphism>* sym = nullptr,
                         const std::vector<NetTarget>* orbit = nullptr,
                         const std::atomic<bool>* cancel = nullptr,
                         const detail::DeadlockHooks* dlh = nullptr, bool* declined = nullptr) {
  const bool use_sym = sym && sym->size() > 1;
  std::vector<std::int32_t> L = net.lbound, U = net.ubound;
  extend_bounds(L, U, target.clock_cond);
  L[0] = U[0] = 0;  // reference clock; mandatory for the a_LU subsumption test
  // With diagonal constraints a_LU is unsound (Bouyer 2004), so such networks fall back to exact
  // zones with diagonals preserved, compared by plain inclusion.
  const bool has_diag = network_has_diagonal(net);
  const bool exact_sub = has_diag || force_exact;
  // Finiteness comes from either a_LU subsumption over exact zones (the default) or, when zones
  // must be compared exactly, from LU-extrapolation. The two are not interchangeable: a_LU only
  // prunes, dropping a zone when a stored one simulates it, whereas Extra_LU+ widens, mapping
  // many exact zones onto one representative. Where the reachable zones are mutually
  // a_LU-incomparable, pruning achieves nothing and widening still collapses them.
  int emode = has_diag ? ExNoDiag : (force_exact ? ExLuPlus : ExNone);
  // Which of the two wins is a property of the model, so the engine starts exact and switches
  // extrapolation on once zones per discrete state cross the threshold. A ratio near 1 means
  // a_LU is already collapsing the antichains and widening would only cost time; a climbing
  // ratio means they are filling with mutually incomparable zones, which only widening removes.
  //
  // Switching mid-run leaves a mixed passed list, and both directions stay sound: a stored exact
  // zone that fails to subsume a new widened one costs one extra stored state, and a stored
  // widened zone that subsumes a new exact one genuinely contains it. Finiteness holds from the
  // switch onwards. See docs/design.md 4.
  constexpr double kAdaptRatio = 4.0;  ///< zones per discrete state at which widening turns on
  std::vector<char> qmask(net.nclocks + 1, 0);  // clocks the query reads; kept active
  for (const auto& c : target.clock_cond) { if (c.i > 0) qmask[c.i] = 1; if (c.j > 0) qmask[c.j] = 1; }

  // The query's clock constants floor every state's a_LU bounds, because the property may hold
  // at any state and the abstraction must not merge two states the query distinguishes.
  std::vector<std::int32_t> Lq(net.nclocks + 1, kAluNegInf), Uq(net.nclocks + 1, kAluNegInf);
  for (const auto& c : target.clock_cond) {
    const int v = raw_bound(c.bound);
    if (c.i > 0 && c.j == 0) Uq[c.i] = std::max(Uq[c.i], v);
    else if (c.i == 0 && c.j > 0) Lq[c.j] = std::max(Lq[c.j], -v);
    else if (c.i > 0 && c.j > 0) {
      const int a = std::abs(v);
      Uq[c.i] = std::max(Uq[c.i], a); Uq[c.j] = std::max(Uq[c.j], a);
      Lq[c.i] = std::max(Lq[c.i], a); Lq[c.j] = std::max(Lq[c.j], a);
    }
  }
  // Per-state a_LU bounds: the per-location bounds maximised over processes, floored by the
  // query's own constants.
  auto state_bounds = [&](const int* locs, std::vector<std::int32_t>& L2,
                          std::vector<std::int32_t>& U2, std::vector<raw_t>& Lr,
                          std::vector<raw_t>& Ur) {
    // The G arrays are the same per-location fixpoint with strictness retained. Combining them
    // across processes by raw max implements the "weakest wins" rule the G-simulation requires
    // (Gastin-Mukherjee-Srivathsan, CAV 2019, Cor. 3.6), matching the pointwise-max composition
    // used for LU.
    if (detail::options().gsim && !net.uraw_loc.empty()) {
      Lr.assign(net.nclocks + 1, kGNone);
      Ur.assign(net.nclocks + 1, kGNone);
      for (int p = 0; p < net.nprocs(); ++p)
        for (int c = 1; c <= net.nclocks; ++c) {
          const raw_t lr = net.lraw_loc[p][locs[p]][c], ur = net.uraw_loc[p][locs[p]][c];
          if (lr != kGNone && (Lr[c] == kGNone || lr > Lr[c])) Lr[c] = lr;
          if (ur != kGNone && (Ur[c] == kGNone || ur > Ur[c])) Ur[c] = ur;
        }
      // The query's constants must floor these bounds exactly as they do L and U, or a state
      // satisfying the property could be abstracted away. Non-strict is the safe direction.
      for (int c = 1; c <= net.nclocks; ++c) {
        if (Uq[c] != kAluNegInf && (Ur[c] == kGNone || le(Uq[c]) > Ur[c])) Ur[c] = le(Uq[c]);
        if (Lq[c] != kAluNegInf && (Lr[c] == kGNone || le(-Lq[c]) > Lr[c])) Lr[c] = le(-Lq[c]);
      }
      // The reference clock must be (<=, 0) here, never kGNone. The G-simulation test skips
      // kGNone entries, so leaving one at index 0 would delete the witness families at x = 0 and
      // y = 0. Missing witnesses can only flip the test from false to true, i.e. prune zones that
      // are not in fact simulated, which is the unsound direction. This mirrors the
      // L[0] = U[0] = 0 requirement of the a_LU test.
      Lr[0] = Ur[0] = kZeroLe;
    }
    if (net.lbound_loc.empty()) { L2 = L; U2 = U; return; }  // fall back to global bounds
    L2.assign(net.nclocks + 1, kAluNegInf);
    U2.assign(net.nclocks + 1, kAluNegInf);
    for (int p = 0; p < net.nprocs(); ++p)
      for (int c = 1; c <= net.nclocks; ++c) {
        L2[c] = std::max(L2[c], net.lbound_loc[p][locs[p]][c]);
        U2[c] = std::max(U2[c], net.ubound_loc[p][locs[p]][c]);
      }
    for (int c = 1; c <= net.nclocks; ++c) { L2[c] = std::max(L2[c], Lq[c]); U2[c] = std::max(U2[c], Uq[c]); }
    L2[0] = U2[0] = 0;
  };


  // Discrete liveness reduction: overwrite integer cells that no process can read before
  // writing, and that the query does not mention, with their low value. Two states differing
  // only in such cells have the same future, so this merges them -- the discrete analogue of
  // reset_inactive zeroing dead clocks, and placed at the same point in the successor path.
  // `qlive` is the query's own read set, the qmask analogue: a cell the property inspects is
  // live everywhere regardless of what the processes do with it.
  std::vector<char> qlive(net.intvars.size(), 0);
  for (const auto& ic : target.int_cond)
    if (ic.var >= 0 && ic.var < static_cast<int>(qlive.size())) qlive[ic.var] = 1;
  const bool do_slice = detail::options().slice && !net.intvars.empty() && !net.iread.empty();
  auto canon_ints = [&](const LocVec& locs, IntVec& ints) {
    if (!do_slice) return;
    const int nv = static_cast<int>(net.intvars.size());
    if (!net.iread_mask.empty()) {              // bit path: nprocs OR-s, then a popcount loop
      std::uint64_t live = 0;
      for (int p = 0; p < net.nprocs(); ++p) live |= net.iread_mask[p][locs[p]];
      for (int v = 0; v < nv; ++v) if (qlive[v]) live |= std::uint64_t(1) << v;
      std::uint64_t dead = ~live & ((nv >= 64) ? ~std::uint64_t(0) : ((std::uint64_t(1) << nv) - 1));
      while (dead) {
        const int v = __builtin_ctzll(dead);
        dead &= dead - 1;
        ints[v] = net.intvars[v].lo;
      }
      return;
    }
    for (int v = 0; v < nv; ++v) {
      if (qlive[v]) continue;
      bool live = false;
      for (int p = 0; p < net.nprocs() && !live; ++p) live = net.iread[p][locs[p]][v];
      if (!live) ints[v] = net.intvars[v].lo;
    }
  };

  bool parallel = false;
  int nthreads = 1;
#ifdef HORA_HAVE_OPENMP
  parallel = (backend != Backend::Seq);
  if (parallel) nthreads = omp_get_max_threads();
#endif

  // The passed list is sharded by discrete-state hash, so a given discrete state always lands in
  // the same shard and each shard is a self-contained antichain owned by exactly one worker
  // during the merge. That is what lets the O(bucket^2 * dim^3) convex merge run without locks.
  const DiscKeyCodecT<Key> dkey(net);

  SymState s0;
  if (!initial_symbolic_state(net, qmask, L, U, emode, &s0)) return false;
  if (is_target(s0)) return true;
  if (use_sym) canonicalize(*sym, net, s0);
  canon_ints(s0.locs, s0.ints);

#if defined(HORA_HAVE_METAL) || defined(HORA_HAVE_CUDA)
  // The frontier-batched GPU engine (src/metal/frontier_metal.mm, src/cuda/frontier_cuda.cpp). It covers the a_LU search over
  // a plain target: exact-zone and predicate targets, symmetry reduction and diagonal constraints
  // stay on the CPU, as does a key too wide for one word. It declines -- returning -1 -- when no
  // device is present, and the CPU engine below then runs instead.
  if constexpr (std::is_same_v<Key, NarrowKey>) {
    if ((backend == Backend::Metal || backend == Backend::Cuda) && !force_exact && !use_sym && !has_diag) {
      detail::FrontierProblem fp;
      fp.net = &net;
      fp.target = &target;
      fp.keys.nprocs = net.nprocs();
      fp.keys.shift = dkey.shift;
      fp.keys.mask.assign(dkey.mask.begin(), dkey.mask.end());
      for (const auto& v : net.intvars) fp.keys.lo.push_back(v.lo);
      fp.key_bits = disc_key_bits(net);
      fp.qmask = qmask;
      // The per-state bounds of state_bounds, as tables: per-location maxima floored by the
      // query's constants, or the global bounds when the model has no per-location ones.
      if (net.lbound_loc.empty()) {
        fp.Lq = L;
        fp.Uq = U;
      } else {
        fp.Lloc = net.lbound_loc;
        fp.Uloc = net.ubound_loc;
        fp.Lq = Lq;
        fp.Uq = Uq;
      }
      fp.Lx = L;
      fp.Ux = U;
      // G-simulation, as state_bounds builds it: per-location raw bounds and the query's floors.
      // Asked for by --sim g, and by default for a CUDA deadlock search, where it is coarser than
      // a_LU at the same soundness; HORA_GPU_SIM=lu|g overrides either.
      const char* simenv = std::getenv("HORA_GPU_SIM");
      const bool want_g = simenv ? std::string(simenv) == "g" : (detail::options().gsim || dlh);
      if (want_g && !net.uraw_loc.empty()) {
        fp.gsim = true;
        fp.Lgloc = net.lraw_loc;
        fp.Ugloc = net.uraw_loc;
        fp.Lgq.assign(net.nclocks + 1, kGNone);
        fp.Ugq.assign(net.nclocks + 1, kGNone);
        for (int c = 1; c <= net.nclocks; ++c) {
          if (Uq[c] != kAluNegInf) fp.Ugq[c] = le(Uq[c]);
          if (Lq[c] != kAluNegInf) fp.Lgq[c] = le(-Lq[c]);
        }
      }
      fp.adapt_ratio = kAdaptRatio;
      fp.canon = [&](const LocVec& l, IntVec& i) { canon_ints(l, i); };
      fp.s0 = s0;
      fp.stats = stats;
      fp.cancel = cancel;
      fp.deadlock = dlh;
      int r = -1;
#ifdef HORA_HAVE_METAL
      if (backend == Backend::Metal) r = detail::frontier_reach_metal(fp);
#endif
#ifdef HORA_HAVE_CUDA
      if (backend == Backend::Cuda) r = detail::frontier_reach_cuda(fp);
#endif
      if (r >= 0) return r == 1;
    }
  }
#endif
  // A deadlock query handed to the GPU path is decided there or not at all: the plain target it
  // carries is the unsatisfiable one, which the CPU engine below would answer `false` to.
  if (dlh) {
    if (declined) *declined = true;
    return false;
  }

  // ======================================================================
  // Shared passed set: one stripe-locked table any worker may touch, locking only the stripe of
  // the discrete state it is updating — the "shared table with fine-grained locking and no static
  // partition" scheme of LTSmin (Laarman, van de Pol and Weber, 2011). The per-bucket update is
  // unchanged: a_LU subsumption followed by the convex-union merge.
  //
  // Expansion and insertion are fused, so there is no serial per-level routing barrier, while
  // strict BFS level synchronisation is retained because it minimises the stored-state count.
  //
  // @invariant At one thread this path is bit-identical to the sequential one: same per-bucket
  //            merge, same insertion order, same counts.
    const StripeLayout lay(nthreads);
    const int dim2 = (net.nclocks + 1) * (net.nclocks + 1);
    ZoneIntern zi(dim2, nthreads);                     // ONE shared distinct-zone table
    std::vector<NetPassedT<Key>> smap(lay.nstripes);   // flat per-stripe passed tables
    for (auto& mp : smap) mp.zi = &zi;
    // The mirror budget is process-wide while these tables are per query, so it must be reset
    // here; otherwise a second query in the same process inherits the first one's charge and
    // never promotes anything.
    detail::hot_bytes().store(0, std::memory_order_relaxed);
    // A reader/writer lock per stripe: the read-only kill probe below runs concurrently within a
    // stripe, and only a genuine insert or merge needs exclusivity.
    std::vector<SpinRW> smtx(lay.nstripes);
    // Explicit --deterministic always wins; otherwise the engine takes it when the model cannot
    // use parallel insertion anyway (see ::auto_deterministic).
    const bool det = detail::options().deterministic || auto_deterministic(net, nthreads);
    //
    // Per-thread local antichains were tried here and removed: they are correct but consistently
    // slower, because a private antichain cannot consult the shared read-only kill probe that
    // answers most successors. See docs/design.md 8.9 before reconsidering.
    //
    // Dynamic-schedule chunk for the frontier expansion, loop-invariant so OpenMP accepts a
    // runtime value.
    const int expand_chunk = std::max(1, omp_chunk());

    // The frontier holds only (discrete key, zone index) per state rather than a full SymState.
    // Locations and integers are recovered from the key and the zone from the intern table, both
    // reconstructed into reused per-thread scratch at expansion time. This avoids copying a full
    // DBM into every frontier state and keeps the resident frontier small enough to stay
    // cache-resident during expansion.
    const int np_ = net.nprocs(), ni_ = static_cast<int>(net.intvars.size());
    std::vector<std::pair<Key, int>> frontier;
    {
      const Key dk = dkey.encode(s0.locs, s0.ints);
      const int z0 = zi.intern(s0.zone, 0);
      smap[lay.stripe_of(dk)].add(std::move(s0), dk, 0, exact_sub, state_bounds, 0);
      frontier.push_back({dk, z0});
    }

    int level = 0;
    std::size_t nexpanded = 0;
    std::atomic<bool> found{false};
    const bool single = (nthreads == 1);   // one worker: skip the locks and the touched union
    // Zone-interner GC trigger: sweep once allocation reaches `sweep_at`, then set the next
    // target to double the surviving live set.
    constexpr std::size_t kSweepFloor = 50000;   // below this the interner is not worth sweeping
    std::size_t sweep_at = kSweepFloor;
    while (!frontier.empty()) {
      if (cancel && cancel->load(std::memory_order_relaxed)) return false;
      if (stats) stats->explored += frontier.size();
      if (has_diag && (nexpanded += frontier.size()) > 500000)
        throw std::runtime_error("state cap exceeded — model likely has unbounded diagonal clocks");
      ++level;

      // Expand each frontier state and insert its successors directly into the shared table,
      // locking only the touched stripe, so there is no serial routing barrier. Inserting during
      // expansion is also what keeps the convex-union merge order stable enough to reproduce the
      // sequential state counts.
      std::vector<std::unordered_set<Key, typename KeyHashFor<Key>::type>> tls_touched(nthreads);
      std::vector<std::size_t> tls_gen(nthreads, 0);
      // --deterministic: successors are stashed with the (frontier index, successor index) that
      // produced them instead of being inserted as they appear, then inserted in that order once
      // the level has been expanded. That order is exactly the sequential one, so the convex-union
      // merge sees the same sequence of operands at any thread count and the state count matches
      // the one-thread run. Empty and untouched when the flag is off.
      std::vector<std::vector<PendingT<Key>>> tls_pending(det ? nthreads : 0);
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel num_threads(nthreads)
      {
        const int tid = omp_get_thread_num();
        auto& touched = tls_touched[tid];
        std::vector<SymState> buf;
        SymState cur;                    // reused per thread: reconstructed frontier state
        cur.locs.resize(np_);
        cur.ints.resize(ni_);
        // Accumulate the generated-successor count in a local, not directly in tls_gen[tid]:
        // those counters are adjacent, so they share a cache line and a per-successor increment
        // would bounce that line between all cores. Published once per thread at the end.
        std::size_t gen = 0;
#pragma omp for schedule(dynamic, expand_chunk) nowait
        for (long i = 0; i < static_cast<long>(frontier.size()); ++i) {
          if (found.load(std::memory_order_relaxed)) continue;
          dkey.decode(frontier[i].first, cur.locs.data(), cur.ints.data());
          // buf is deliberately not cleared: gen_succ_reuse rebuilds slots in place and returns
          // the count, so the loc/int/zone storage is reused across frontier states.
          const std::size_t ns = gen_succ_reuse(net, cur, zi.get(frontier[i].second), L, U, qmask, emode,
                                                buf, frontier[i].second);
          for (std::size_t sk = 0; sk < ns; ++sk) {
            SymState& s = buf[sk];
            ++gen;
            if (is_target(s)) {
              found.store(true, std::memory_order_relaxed);
              break;
            }
            if (use_sym) canonicalize(*sym, net, s);
            canon_ints(s.locs, s.ints);
            const Key dk = dkey.encode(s.locs, s.ints);
            if (det) {   // canonical order decided below, not by who got here first
              tls_pending[tid].push_back(PendingT<Key>{static_cast<std::uint32_t>(i),
                                                 static_cast<std::uint32_t>(sk), dk, std::move(s)});
              continue;
            }
            const std::size_t si = lay.stripe_of(dk);
            if (single) {
              if (smap[si].add(std::move(s), dk, level, exact_sub, state_bounds, tid) >= 0) touched.insert(dk);
            } else {
              if (!detail::options().no_probe) {   // shared: concurrent readers, no serialization on the common "already covered" case
                smtx[si].lock_shared();
                const bool covered = smap[si].probe_kill(dk, s, exact_sub, state_bounds);
                smtx[si].unlock_shared();
                if (covered) continue;
              }
              smtx[si].lock();
              if (smap[si].add(std::move(s), dk, level, exact_sub, state_bounds, tid) >= 0) touched.insert(dk);
              smtx[si].unlock();
            }
          }
        }
        tls_gen[tid] = gen;
      }
#else
      {
        auto& touched = tls_touched[0];
        std::vector<SymState> buf;
        SymState cur;
        cur.locs.resize(np_);
        cur.ints.resize(ni_);
        for (auto& fr : frontier) {
          dkey.decode(fr.first, cur.locs.data(), cur.ints.data());
          const std::size_t ns = gen_succ_reuse(net, cur, zi.get(fr.second), L, U, qmask, emode, buf, fr.second);
          for (std::size_t sk = 0; sk < ns; ++sk) {
            SymState& s = buf[sk];
            ++tls_gen[0];
            if (is_target(s)) return true;
            if (use_sym) canonicalize(*sym, net, s);
            canon_ints(s.locs, s.ints);
            const Key dk = dkey.encode(s.locs, s.ints);
            if (smap[lay.stripe_of(dk)].add(std::move(s), dk, level, exact_sub, state_bounds, 0) >= 0)
              touched.insert(dk);
          }
        }
      }
#endif
      if (det) {
        std::vector<PendingT<Key>> pend;
        std::size_t tot = 0;
        for (auto& v : tls_pending) tot += v.size();
        pend.reserve(tot);
        for (auto& v : tls_pending) for (auto& e : v) pend.push_back(std::move(e));
        std::sort(pend.begin(), pend.end(), [](const PendingT<Key>& a, const PendingT<Key>& b) {
          return a.i != b.i ? a.i < b.i : a.sk < b.sk;
        });
        auto& touched0 = tls_touched[0];
        for (auto& e : pend)
          if (smap[lay.stripe_of(e.dk)].add(std::move(e.s), e.dk, level, exact_sub,
                                            state_bounds, 0) >= 0)
            touched0.insert(e.dk);
      }
      if (stats) for (std::size_t g : tls_gen) stats->generated += g;
      if (found.load()) return true;

      // The passed tables are read-only here (no writers), so for_level is a safe read.
      std::vector<std::pair<Key, int>> next =
          collect_next_frontier(smap, lay, tls_touched, nthreads, single || det, level);
      // Reordering a level to expand its most general zones first was tried and removed: it is
      // sound but does not pay, because zone retirement is driven by zones arriving levels apart,
      // which no intra-level ordering can reach. See docs/design.md 8.14.
      frontier.swap(next);
      // Re-read the zones-per-discrete-state signal now that the level is complete and every
      // worker has joined, and switch widening on if the antichains are filling up. Once on it
      // stays on: a rising ratio only ever justifies more abstraction, and switching back would
      // forfeit the finiteness guarantee established at the switch.
      if (emode == ExNone) {
        std::size_t nlive = 0, ndisc = 0;
        for (const auto& mp : smap) mp.counts(nlive, ndisc);
        if (ndisc && static_cast<double>(nlive) >= kAdaptRatio * static_cast<double>(ndisc)) {
          emode = ExLuPlus;
        }
      }
      // Reclaim dead arena tombstones: one thread per stripe (exclusive), between levels.
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 8) num_threads(nthreads)
#endif
      for (long s = 0; s < static_cast<long>(smap.size()); ++s) smap[s].maybe_compact();

      // The sweep trigger is growth-based rather than per-level. Sweeping once the allocation
      // level has doubled since the last sweep makes the work amortised O(1) per interned zone —
      // a sweep costing O(A) is paid for by the A/2 fresh interns that triggered it — so a run
      // performs O(log) sweeps instead of one per level. The memory traded away is bounded:
      // garbage never exceeds the live set, so the interner stays within 2x of its live size.
      // A level-triggered sweep instead grows with the state space while the expansion around it
      // stays parallel, which caps the achievable speedup.
      if (zi.allocated() >= sweep_at)
        sweep_at = sweep_dead_zones(zi, smap, frontier, nthreads, kSweepFloor);
    }
    return false;
}

/// @brief Decide `E<> target`, for a target that is a conjunction of location, integer and
///        clock constraints.
///
/// @param net     the network under analysis
/// @param target  the target conjunction
/// @param backend which expansion backend to run
/// @param[out] stats  optional statistics; unchanged when null
/// @return @c true iff some reachable symbolic state satisfies @p target
///
/// @note a_LU subsumption is sound here because the property is evaluated per symbolic state
///       rather than per valuation, so covering preserves it. Contrast
///       hora::deadlock_reachable, which is not.
/// @brief Run the forward engine under the narrowest packed key @p net actually fits.
///
/// @param net the network under analysis; its field widths choose the key
/// @return whatever ::reach_engine returns for the selected key
///
/// A network whose discrete state fits one word runs on ::NarrowKey -- the key every existing
/// measurement was produced with, down to the stripe and bucket indices, since ::key_mix reduces
/// to the same multiply. A wider network used to be rejected by ::throw_disc_key_too_wide; it now
/// runs on ::WideKey instead, which costs a two-word key in the passed set, the frontier and the
/// touched sets, and changes nothing else about the search.

template <class TargetFn>
static bool reach_engine_auto(const Network& net, const NetTarget& target, TargetFn&& is_target,
                              bool force_exact, Backend backend, ReachStats* stats,
                              const std::vector<Automorphism>* sym = nullptr,
                              const std::vector<NetTarget>* orbit = nullptr,
                              const std::atomic<bool>* cancel = nullptr,
                              const detail::DeadlockHooks* dlh = nullptr, bool* declined = nullptr) {
  if (disc_key_bits(net) <= key_budget<NarrowKey>())
    return reach_engine<NarrowKey>(net, target, std::forward<TargetFn>(is_target), force_exact,
                                   backend, stats, sym, orbit, cancel, dlh, declined);
  const Backend cpu = (backend == Backend::Seq) ? Backend::Seq : Backend::Omp;
  return reach_engine<WideKey>(net, target, std::forward<TargetFn>(is_target), force_exact, cpu,
                               stats, sym, orbit, cancel, dlh, declined);
}

bool reachable_net(const Network& net, const NetTarget& target, Backend backend,
                   ReachStats* stats) {
  // Symmetry reduction explores one representative per orbit, so a target that is not itself
  // symmetric must be tested against the whole orbit of the property: on a rotation-symmetric
  // ring, "processes 0 and 1 both eat" becomes "some adjacent pair eats", which is the same
  // question. The target is mapped through every group element once, up front, and each state is
  // tested against every image; no state is ever permuted at test time.
  std::vector<Automorphism> G;
  if (symmetry_on()) G = symmetry_group(net);
  const bool use_sym = G.size() > 1;
  std::vector<NetTarget> orbit;
  NetTarget bounds_target = target;   // clock constants seen by the LU bounds and qmask
  if (use_sym) {
    orbit.reserve(G.size());
    for (const auto& g : G) {
      NetTarget t;
      for (const auto& ic : target.int_cond)
        t.int_cond.push_back(IntAtom{ic.var, ic.op,
                                     g.map_val(ic.var, ic.val, net.intvars[ic.var].lo)});
      t.never = target.never;   // an unsatisfiable target stays unsatisfiable in every image
      for (const auto& lr : target.loc_req) t.loc_req.push_back({g.proc[lr.first], lr.second});
      for (const auto& c : target.clock_cond)
        t.clock_cond.push_back(Constraint{g.clk[c.i], g.clk[c.j], c.bound});
      // Every image's clock constants must reach the abstraction. Otherwise the LU bounds would
      // be asymmetric and two states of the same orbit could be abstracted differently, which is
      // precisely what the reduction may not do.
      bounds_target.clock_cond.insert(bounds_target.clock_cond.end(), t.clock_cond.begin(),
                                      t.clock_cond.end());
      orbit.push_back(std::move(t));
    }
  }
  auto match = [](const NetTarget& t, const SymState& s) {
    if (t.never) return false;   // the literal `false` matches no state
    for (auto& lr : t.loc_req)
      if (s.locs[lr.first] != lr.second) return false;
    if (!int_ok(t.int_cond, s.ints)) return false;
    Dbm z = s.zone;
    return apply_all(z, t.clock_cond);
  };
  auto is_target = [&](const SymState& s) {
    if (!use_sym) return match(target, s);
    for (const auto& t : orbit) if (match(t, s)) return true;
    return false;
  };
  return reach_engine_auto(net, bounds_target, is_target, /*force_exact=*/false, backend, stats,
                           use_sym ? &G : nullptr, use_sym ? &orbit : nullptr);
}

/// @brief Decide whether some infinite run exists along which @p phi holds at every state.
///
/// @param net  the network under analysis
/// @param phi  the invariant predicate, over locations and integer variables only
/// @param backend  ignored; this search is sequential
/// @param[out] stats  optional statistics; unchanged when null
/// @return @c true iff a @p phi-invariant infinite run exists
///
/// @note A clock atom in @p phi is handled by the **psi-restricted semantics**: the conjunction
///       `phi.clocks` is adjoined to every location invariant of a private copy of the network,
///       and the search then runs unchanged on that copy. This is exact rather than merely sound
///       because a conjunction of constraints is convex, and the set reachable by delay while
///       staying in a convex `C` is `(Z /\ C)^ /\ C` — every intermediate point of the segment
///       is a convex combination of its endpoints. Restricting only after the delay would
///       over-approximate: for `Z = {x=0}` and `phi = (x>=5)` it gives `[5,inf)` where the true
///       answer is empty, and the search would report a witness that does not exist.
/// @note Making the atoms invariants also discharges the two side conditions automatically
///       rather than by hand. `compute_bounds` then derives the LU bounds from them, so the
///       extrapolation cannot merge zones across the predicate boundary; and `compute_active`
///       sees the clocks read at every location, so the active-clock reduction cannot zero them.
///       `time_diverges` and `state_has_deadlock` read the invariant too, so both are restricted
///       as well — the first matters, since with `phi = (x<=5)` time diverges nowhere and an
///       unrestricted test would manufacture exactly the maximal run this query looks for.
/// @note Zones must be extrapolated here rather than kept exact, because termination of the
///       cycle search depends on the zone graph being finite; a_LU pruning alone does not give
///       that for a greatest-fixpoint search.
/// @brief Opt IN to per-location LU bounds in ::exists_invariant_run (HORA_EG_LOCLU=1).
///
/// Off by default because it was measured and it loses. Per-location bounds are pointwise coarser
/// than the global pair, so they relax each individual zone further -- but this search dedups
/// nodes by exact zone equality and has no subsumption at all, so what it needs from an
/// abstraction is not that each zone be relaxed further, it is that many zones be mapped onto the
/// same representative. One global L/U puts every zone in one lattice; per-location bounds give
/// each discrete state its own, and zones that would have become identical no longer do.
///
/// Measured, `E[] <clock bound>`, states explored, per-location / global:
///
/// It is behind a switch because the direction of the effect depends on how much the query
/// constrains the clocks; see docs/design.md section 8.
/// per-location bounds actually vary across the location vector, which is worth revisiting.
static bool use_loc_lu() {
  static const bool v = [] { const char* e = std::getenv("HORA_EG_LOCLU"); return e && *e == '1'; }();
  return v;
}


/// @brief Decide whether a state containing a deadlocked valuation is reachable.
///
/// @param net      the network under analysis
/// @param backend  Seq runs sequentially; anything else runs the multi-threaded CPU engine
/// @param[out] stats   optional statistics; unchanged when null
/// @param cancel   optional cancellation flag; see ::reach_engine
/// @return @c true iff some reachable state has a deadlocked valuation, @c false if none has
///         **or** the search was cancelled
///
/// @par Why a bisimulation is required
/// Deadlock is the property "enables nothing", which a simulation does not preserve; it needs a
/// bisimulation (Behrmann, Bouyer, Larsen and Pelánek, STTT 8(3):204–215, 2006, Lemma 3, with
/// Lemma 1 giving the bisimulation for `=_M`). Setting `L = U = M` for `M = max(L, U)` per clock
/// collapses the three clauses of the LU preorder to "equal, or both above M(x)", which is
/// symmetric — and a symmetric simulation is a bisimulation. The ordinary a_LU kernel fed these
/// collapsed bounds is therefore exactly the a_M test, and is deadlock-sound.
///
/// @note The bound arrays are collapsed on a copy of the network rather than a flag being
///       threaded through the engine, so that every downstream consumer of L and U becomes
///       M-flavoured at once. This matters beyond the subsumption test: the adaptive path may
///       turn extrapolation back on mid-search, and `Extra_LU+` with `L = U = M` is `Extra_M+`,
///       which — unlike `Extra_LU+` — can neither invent nor destroy a deadlock.
/// @note The predicate closes over the *original* network. It reads guards, invariants and edges,
///       never the bound arrays, so the collapse does not reach it.
/// @see docs/design.md §3 for the full argument and the measured comparison against exact zones.
bool deadlock_reachable(const Network& net, Backend backend, ReachStats* stats,
                        const std::atomic<bool>* cancel) {
  // A predicate target that reads the zone: the CPU engine -- except on CUDA, below, where the
  // frontier engine reads each level's zones back and runs the same exact test on the host.
  const Backend b = (backend == Backend::Seq) ? Backend::Seq : Backend::Omp;
  const auto is_dl = [&](const SymState& s) { return state_has_deadlock(net, s); };

#ifdef HORA_HAVE_CUDA
  if (backend == Backend::Cuda && !network_has_diagonal(net) &&
      disc_key_bits(net) <= key_budget<NarrowKey>()) {
    const char* dm = std::getenv("HORA_DL_MODE");
    if (!dm || *dm == 'd') {
      NetTarget never;     // the search stops on a deadlock, which only the hooks can see
      never.never = true;
      detail::DeadlockHooks hooks;
      const DiscKeyCodecT<NarrowKey> codec(net);
      hooks.has_deadlock = [&](std::uint64_t key, const std::int32_t* zone) {
        SymState s;
        s.locs.resize(net.nprocs());
        s.ints.resize(net.intvars.size());
        codec.decode(static_cast<NarrowKey>(key), s.locs.data(), s.ints.data());
        s.zone = Dbm(net.nclocks);
        std::copy(zone, zone + s.zone.data().size(), s.zone.data().begin());
        return state_has_deadlock(net, s);
      };
      hooks.seed = [&](std::uint64_t key) {
        LocVec l(net.nprocs());
        IntVec i(net.intvars.size());
        codec.decode(static_cast<NarrowKey>(key), l.data(), i.data());
        return detail::deadlock_seed(net, l, i);
      };
      bool declined = false;
      const bool hit = reach_engine_auto(net, never, is_dl, /*force_exact=*/false, backend, stats,
                                         nullptr, nullptr, cancel, &hooks, &declined);
      if (!declined) return hit;
    }
  }
#endif

  // Diagonals make every LU-family abstraction unsound, and the G-bound arrays carry raw encoded
  // bounds with their own sentinel, so collapsing those correctly is a separate exercise. Both
  // cases fall back to exact zones, which is sound but stores more states.
  if (network_has_diagonal(net) || gsim_on())
    return reach_engine_auto(net, NetTarget{}, is_dl, /*force_exact=*/true, b, stats, nullptr,
                             nullptr, cancel);

  Network am = net;
  auto collapse = [](std::vector<std::int32_t>& lo, std::vector<std::int32_t>& hi) {
    for (std::size_t c = 0; c < lo.size() && c < hi.size(); ++c)
      lo[c] = hi[c] = std::max(lo[c], hi[c]);
  };
  collapse(am.lbound, am.ubound);
  for (std::size_t pr = 0; pr < am.lbound_loc.size(); ++pr)
    for (std::size_t l = 0; l < am.lbound_loc[pr].size(); ++l)
      collapse(am.lbound_loc[pr][l], am.ubound_loc[pr][l]);
  const char* dm = std::getenv("HORA_DL_MODE");
  const char mode = dm ? *dm : 'd';
  if (mode == 'd') {
    // The D-refined G-simulation preserves `Dead`, so it decides this query directly and is
    // coarser than the a_M bisimulation the plain engine has to fall back to. `net` is passed,
    // not the collapsed `am`: the refinement supplies deadlock-soundness itself, so collapsing
    // the bounds on top of it would only throw the LU precision away again.
    DiscPred none;
    none.is_dnf = true;                    // an empty DNF reads as `false`, so every state is !q
    LivenessEngine e(net, none, none, false, cancel);
    GfpStats gs;
    bool ok = false;
    const bool hit = e.deadlock_reachable_refined(&gs, &ok);
    if (ok) {
      // `explored` is every record the search stored, frontier included; `expanded` is the
      // records whose successors were actually computed, which is what verifyta's `States
      // explored` and TChecker's VISITED_STATES count. Reporting only the first overstated this
      // route against both by the unexpanded frontier.
      if (stats) { stats->explored = gs.fwd_nodes; stats->expanded = gs.outer_expands; }
      return hit;
    }
    // The construction declined; fall through to the abstraction that needs no refinement.
  }
  return reach_engine_auto(am, NetTarget{}, is_dl, /*force_exact=*/mode == 'e', b, stats, nullptr,
                           nullptr, cancel);
}

}  // namespace hora
