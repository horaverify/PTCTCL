/// @file netreach.cpp
/// @brief The forward engine: symbolic reachability over a network of timed automata.
///
/// This translation unit implements every "is a state satisfying X reachable?" query in the
/// build — ptctlmc::reachable_net, ptctlmc::exists_invariant_run and
/// ptctlmc::deadlock_reachable, all declared in @ref ptctlmc/netreach.hpp — on top of one
/// engine, ::reach_engine. It also owns ptctlmc::detail::state_has_deadlock, which the liveness
/// engine in @ref netliveness.cpp calls for its stopping term.
///
/// @par Shape of the search
/// Level-synchronous breadth-first exploration of the symbolic state space. A symbolic state is
/// a discrete part (a location per process plus the integer variables) and a zone. The discrete
/// part is packed losslessly into a @c uint64_t key (::DiscKeyCodec); the zone is interned
/// (ptctlmc::detail::ZoneIntern) and referenced by index, so the frontier and the passed set
/// carry integers rather than DBMs. The passed set is sharded into stripes
/// (::StripeLayout) so that workers contend only when they touch the same discrete state.
///
/// @par Covering
/// A newly generated state is discarded when the passed set already holds one that *covers* it.
/// The default relation is a_LU subsumption on **exact** zones; extrapolation is switched on
/// only where finiteness would otherwise depend on it, or by the adaptive rule when the
/// zones-per-discrete-state ratio shows the exact graph is not converging. Diagonal networks
/// cannot use a_LU at all (Bouyer 2004) and fall back to inclusion on extrapolated zones.
///
/// @warning Covering is a simulation, so it preserves existential properties only. Deadlock is a
///          universally quantified negative property and is *not* preserved;
///          ptctlmc::deadlock_reachable therefore runs the same engine over a network whose
///          bound arrays have been collapsed to `L = U = M`, which makes the relation symmetric
///          and hence a bisimulation. See @c DESIGN.md.
///
/// @par Design notes
/// The correctness arguments and the measurements behind the choices here — why stored zones
/// must be exact, why deadlock needs a bisimulation, and what was measured and rejected — are
/// collected in @c DESIGN.md at the repository root.

#include "ptctlmc/netreach.hpp"
#include "ptctlmc/symmetry.hpp"
#include <cstdio>
#include <sys/resource.h>
#include <array>
#include <chrono>
#ifdef __APPLE__
#include <malloc/malloc.h>
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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ptctlmc/fed.hpp"
#include "ptctlmc/dbm_ops_legacy.hpp"
#include "ptctlmc/detail/engine_config.hpp"
#include "ptctlmc/detail/spin_rw.hpp"
#include "ptctlmc/detail/sym_state.hpp"
#include "ptctlmc/detail/zone_intern.hpp"
#include "ptctlmc/detail/gpu_zone_intern.hpp"
#include "ptctlmc/detail/net_passed.hpp"
#include "ptctlmc/detail/net_semantics.hpp"
#include "ptctlmc/mindbm.hpp"   // minimal-graph packed zone storage (UDBM mingraph port)
#include "ptctlmc/net_gpu.hpp"

#ifdef PTCTLMC_HAVE_OPENMP
#include <omp.h>
#endif

namespace ptctlmc {
namespace {
using detail::SpinRW;
using detail::SymState;
using detail::LocVec;
using detail::IntVec;
using detail::ZoneIntern;
using detail::GpuZoneIntern;
using detail::NetPassed;
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
/// @warning Both have **external** linkage on purpose. The GPU expanders in @c src/metal and
///          @c src/cuda must apply exactly these semantics, and duplicating them there is how a
///          backend once diverged silently — a Metal expander that dropped the general update
///          explored roughly half of viking's state space and still reported a verdict. One
///          body, every backend. Declared in @ref ptctlmc/netreach.hpp.
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
///       engine, and the GPU backends ask it through the @c netgpu alias below. One body.
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

/// @brief Alias of ::ptctlmc::network_has_diagonal for the GPU expanders, which reach it through
///        the @c netgpu declaration in @ref ptctlmc/net_gpu_encode.hpp. One body, two names.
namespace netgpu { bool network_has_diagonal(const Network& net) { return ptctlmc::network_has_diagonal(net); } }

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
                    const std::vector<std::int32_t>& U, const std::vector<char>& qmask, int emode, std::vector<SymState>& out) {
  thread_local std::vector<Constraint> inv;  // reused invariant scratch (see all_invariants)
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

      if (!close_succ_zone(net, ns.locs, ns.ints, z1, inv, qmask, L, U, emode)) continue;
      if (z1.empty()) continue;
      ns.zone = std::move(z1);
      out.push_back(std::move(ns));
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
          if (!close_succ_zone(net, ns.locs, ns.ints, z1, inv, qmask, L, U, emode)) continue;
          if (z1.empty()) continue;
          ns.zone = std::move(z1);
          out.push_back(std::move(ns));
        }
      }
    }
  }

  // ---- broadcast synchronisation: a! fires with ALL enabled a? (0 or more) ----
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
            if (close_succ_zone(net, ns.locs, ns.ints, z1, inv, qmask, L, U, emode) && !z1.empty()) {
              ns.zone = std::move(z1);
              out.push_back(std::move(ns));
            }
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
/// @note Enabled by `--memo`; see ptctlmc::set_memo.
/// @see DESIGN.md §8.13 for why the adaptive rule tests reuse density and not hit rate.
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
    // cases where hit rate does not. See DESIGN.md 8.13.
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
  // The memo key is (source zone, edge, target LOCATION signature) — it does not carry the
  // integer valuation, which was sound while the closing pipeline read only the target locations
  // and the zone. With urgent channels the delay in close_succ_zone additionally depends on the
  // target INTEGER valuation, so two states agreeing on all three key components but differing
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
      // The DISCRETE half first, so the memo can be consulted before any DBM work. This is a
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
        // MISS: run the pipeline, then record. `memo_pend` marks that we owe an insert.
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
          // models (viking, phil) take almost ALL their transitions through this path, so
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

#if defined(PTCTLMC_HAVE_METAL) || defined(PTCTLMC_HAVE_CUDA)
/// @brief Host port of the GPU merge kernel, for one discrete-state bucket.
///
/// The GPU merge runs one thread per bucket, so a level made of few but very wide buckets stalls
/// on a handful of slow lanes. This lets the flat driver merge such buckets across all CPU cores
/// instead, which is what ptctlmc::set_adaptive_merge routes between.
///
/// @param[in,out] work_buf  zone storage; the bucket's existing antichain occupies
///                          `work_buf[work_off_b .. work_off_b + exist_cnt_b)`
/// @param work_off_b  offset of this bucket within @p work_buf
/// @param exist_cnt_b number of zones already in the bucket's antichain
/// @param oz          expand output, read in place rather than gathered
/// @param new_idx_b   indices into @p oz of the candidate zones for this bucket
/// @param new_cnt_b   length of @p new_idx_b
/// @param L           per-clock lower bounds for the covering test
/// @param U           per-clock upper bounds for the covering test
/// @param dim         zone dimension, so each matrix is `dim * dim`
/// @param has_diag    whether the network has diagonal constraints, selecting the covering test
/// @param[out] WI     per-survivor "is new" flags: 0 for a pre-existing zone, 1 for an appended one
/// @return the number of zones surviving in the bucket
///
/// @post The surviving antichain is written back in place at @p work_off_b, and no surviving
///       zone covers another.
/// @note Must agree with the GPU kernel zone for zone; the routing decision is a scheduling
///       choice, so the two paths have to be interchangeable.
static int flat_merge_bucket(gz_t* work_buf, int work_off_b, int exist_cnt_b,
                             const gz_t* oz, const int* new_idx_b, int new_cnt_b,
                             const std::int32_t* L, const std::int32_t* U,
                             int dim, bool has_diag, int* WI) {
  const int d2 = dim * dim;
  // The pool keeps NARROW entries; the algebra stays raw_t. dbm_includes / dbm_is_alu_le /
  // dbm_try_union are templated on their read-only operand's entry type (dbm_ops.hpp), so `zw`
  // can point straight into the pool -- no widening pass per bucket.
  gz_t* W = work_buf + static_cast<std::size_t>(work_off_b) * d2;
  raw_t zn[kDbmMaxDim * kDbmMaxDim];
  int cur = exist_cnt_b;                              // WI[0..cur) already 0 (existing zones)
  for (int t = 0; t < new_cnt_b; ++t) {
    const gz_t* src = oz + static_cast<std::size_t>(new_idx_b[t]) * d2;
    for (int k = 0; k < d2; ++k) zn[k] = gz_load(src[k]);   // widen out of the narrow pool
    bool subsumed = false;                            // zn subsumed by a stored zone?
    for (int i = 0; i < cur; ++i) {
      const gz_t* zw = W + static_cast<std::size_t>(i) * d2;
      if (has_diag ? dbm_includes(zw, zn, dim) : dbm_is_alu_le(zn, zw, dim, L, U)) { subsumed = true; break; }
    }
    if (subsumed) continue;
    int w = 0;                                        // drop stored zones subsumed by zn
    for (int i = 0; i < cur; ++i) {
      const gz_t* zw = W + static_cast<std::size_t>(i) * d2;
      const bool killed = has_diag ? dbm_includes(zn, zw, dim) : dbm_is_alu_le(zw, zn, dim, L, U);
      if (!killed) {
        if (w != i) { std::memcpy(W + static_cast<std::size_t>(w)*d2, zw, d2*sizeof(gz_t)); WI[w] = WI[i]; }
        ++w;
      }
    }
    cur = w;
    bool merged = true;                               // convex-union absorb (zn becomes the hull)
    while (merged) {
      merged = false;
      for (int kk = 0; kk < cur; ++kk)
        if (dbm_try_union(zn, W + static_cast<std::size_t>(kk)*d2, dim)) {
          for (int i = kk; i < cur - 1; ++i) {
            std::memcpy(W + static_cast<std::size_t>(i)*d2, W + static_cast<std::size_t>(i+1)*d2, d2*sizeof(gz_t));
            WI[i] = WI[i+1];
          }
          --cur; merged = true; break;
        }
    }
    gz_t* dst = W + static_cast<std::size_t>(cur)*d2;
    for (int k = 0; k < d2; ++k) zput(dst, k, zn[k]);
    WI[cur] = -1; ++cur;
  }
  return cur;
}
#endif

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
/// @see ptctlmc::set_narrow_ranges, which derives tighter ranges and often avoids this.
[[noreturn]] static void throw_disc_key_too_wide(const Network& net, int bits_needed,
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
                    " bits but the packed key holds 63 (network too wide).\n"
                    "  " + std::to_string(net.nprocs()) + " processes + " +
                    std::to_string(net.intvars.size()) + " int vars. Widest fields:";
  for (std::size_t i = 0; i < widest.size() && i < 8; ++i)
    msg += "\n    " + std::to_string(widest[i].first) + " bits  " + widest[i].second;
  throw std::runtime_error(msg);
}

/// @brief Lossless packing of a symbolic state's discrete half into a single @c uint64_t.
///
/// The location vector and the integer variables occupy one fixed-width bit-field each, sized
/// from the number of locations of each process and from each variable's declared range. Keys
/// are then compared and hashed as plain integers rather than as formatted strings.
///
/// @invariant The encoding is injective: two discrete states share a key only if they are equal.
///            This is what lets the passed set be keyed on the packed value alone.
/// @invariant Total width is at most 63 bits, so every field offset is strictly below 64 and a
///            zero-width trailing field cannot produce an undefined shift.
///
/// @note Built once per search and thereafter read-only, so it is safe to share across the
///       parallel level expansion without synchronisation.
/// @warning A network whose fields do not fit is rejected outright by ::throw_disc_key_too_wide
///          rather than being packed lossily; an aliased key would silently merge distinct
///          states.
struct DiscKeyCodec {
  const Network& net;
  std::vector<int> shift;           ///< bit offset of each field: procs first, then int vars
  std::vector<std::uint64_t> mask;  ///< per-field value mask, for decoding a key back apart

  /// @brief Bits needed to distinguish @p nvalues values.
  /// @param nvalues number of distinct values the field must hold
  /// @return the smallest @c b with `2^b >= nvalues`; @c 0 for a single-valued field
  static int field_bits(long nvalues) { int b = 0; while ((1LL << b) < nvalues) ++b; return b; }

  /// @brief Lay out one bit-field per process and per integer variable.
  /// @param n the network whose discrete states are to be packed
  /// @throws std::runtime_error if the fields need 64 bits or more
  explicit DiscKeyCodec(const Network& n) : net(n) {
    const int nfields = n.nprocs() + static_cast<int>(n.intvars.size());
    shift.resize(nfields);
    mask.resize(nfields);
    int off = 0;
    auto set = [&](int f, int w) {
      shift[f] = off;
      mask[f] = (w >= 64) ? ~0ULL : ((1ULL << w) - 1);
      off += w;
    };
    for (int p = 0; p < n.nprocs(); ++p)
      set(p, field_bits(static_cast<long>(n.procs[p].locations.size())));
    for (std::size_t v = 0; v < n.intvars.size(); ++v)
      set(n.nprocs() + static_cast<int>(v), field_bits(n.intvars[v].hi - n.intvars[v].lo + 1));
    // The test is >= 64 rather than > 64 so that every field offset stays strictly below 64;
    // a trailing zero-width field would otherwise be shifted by exactly 64, which is undefined.
    if (off >= 64) throw_disc_key_too_wide(n, off, &field_bits);
  }

  /// @brief Pack a discrete state into its key.
  /// @param locs the location vector, one entry per process
  /// @param ints the integer variables, one entry per declared variable
  /// @return the packed key
  /// @pre Every value lies in its field's declared range; ::gen_apply enforces this for updates.
  std::uint64_t encode(const LocVec& locs, const IntVec& ints) const {
    std::uint64_t k = 0;
    for (int p = 0; p < static_cast<int>(locs.size()); ++p)
      k |= static_cast<std::uint64_t>(static_cast<unsigned>(locs[p])) << shift[p];
    for (std::size_t v = 0; v < ints.size(); ++v)
      k |= static_cast<std::uint64_t>(ints[v] - net.intvars[v].lo) << shift[net.nprocs() + v];
    return k;
  }

  /// @brief Unpack a key back into a discrete state.
  /// @param dk        a key previously produced by ::DiscKeyCodec::encode
  /// @param[out] locs receives one location per process
  /// @param[out] ints receives one value per integer variable
  /// @pre @p locs and @p ints have room for @c nprocs and @c intvars.size() entries.
  /// @post `decode(encode(l, i))` reproduces @p l and @p i exactly.
  void decode(std::uint64_t dk, int* locs, long* ints) const {
    for (int p = 0; p < net.nprocs(); ++p)
      locs[p] = static_cast<int>((dk >> shift[p]) & mask[p]);
    for (std::size_t v = 0; v < net.intvars.size(); ++v)
      ints[v] = static_cast<long>((dk >> shift[net.nprocs() + v]) & mask[net.nprocs() + v]) +
                net.intvars[v].lo;
  }
};

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
  std::size_t stripe_of(std::uint64_t dk) const {
    return static_cast<std::size_t>((dk * kMix) >> stripe_shift);
  }
  /// @brief Work bucket owning a discrete key.
  /// @param dk the packed discrete key
  /// @return an index in `[0, nbuckets)`
  /// @pre `nbuckets > 1`. With a single bucket @c bucket_shift is 64 and the shift is undefined;
  ///      callers take a separate single-bucket path instead.
  std::size_t bucket_of(std::uint64_t dk) const {
    return static_cast<std::size_t>((dk * kMix) >> bucket_shift);
  }
};

#if defined(PTCTLMC_HAVE_METAL) || defined(PTCTLMC_HAVE_CUDA)
// ---- flat GPU driver helpers ----------------------------------------------------------

/// @brief Per-level map from packed discrete key to bucket id, for the flat GPU driver.
///
/// An open-addressing table with a generation stamp, so the whole run reuses one allocation and
/// clearing between levels is O(1) rather than proportional to the table. A node-based map would
/// instead allocate once per distinct key and rebuild every level, which is significant because
/// this index is rebuilt on each of many BFS levels.
///
/// @invariant Capacity is always a power of two, so probing masks rather than divides.
/// @invariant An entry is live exactly when its stamp equals the current generation; entries
///            from earlier levels are therefore invisible without being erased.
struct DkIndex {
  std::vector<std::uint64_t> key;    ///< probe slots; meaningful only where stamp == gen
  std::vector<int> val;              ///< bucket id assigned to the corresponding key
  std::vector<std::uint32_t> stamp;  ///< generation that last wrote the slot
  std::uint32_t gen = 0;             ///< current generation; bumped once per level
  std::size_t mask = 0, fill = 0, limit = 0;  ///< capacity-1, live entries, and the grow threshold

  /// @brief splitmix64 finaliser, to spread keys whose low bits are highly structured.
  static std::uint64_t mix(std::uint64_t k) {
    k ^= k >> 30; k *= 0xbf58476d1ce4e5b9ULL;
    k ^= k >> 27; k *= 0x94d049bb133111ebULL;
    return k ^ (k >> 31);
  }
  /// @brief Reallocate to @p cap slots and reset to an empty first generation.
  /// @param cap new capacity; must be a power of two
  /// @post The table is empty and its load limit is 75% of @p cap.
  void resize_to(std::size_t cap) {
    key.assign(cap, 0); val.assign(cap, 0); stamp.assign(cap, 0);
    mask = cap - 1; gen = 1; fill = 0; limit = cap - cap / 4;   // 75% load factor
  }
  /// @brief Start a new level, logically clearing the table.
  ///
  /// @param expect expected number of distinct keys, taken from the previous level because BFS
  ///               level widths change smoothly; an underestimate costs a rehash, not
  ///               correctness
  /// @post All entries from earlier levels are unreachable.
  /// @note When the generation counter wraps, the stamps are cleared explicitly so that a stale
  ///       slot cannot be mistaken for a live one.
  void begin_level(std::size_t expect) {
    std::size_t cap = 1024;
    while (cap - cap / 4 < expect + 1) cap <<= 1;
    if (cap > key.size()) { resize_to(cap); return; }
    mask = key.size() - 1; limit = key.size() - key.size() / 4; fill = 0;
    if (++gen == 0) { std::fill(stamp.begin(), stamp.end(), 0); gen = 1; }
  }
  /// @brief Double the capacity, reinserting the current generation's live entries.
  /// @post Existing key-to-id assignments are preserved.
  void grow() {
    const std::vector<std::uint64_t> ok = key;
    const std::vector<int> ov = val;
    const std::vector<std::uint32_t> os = stamp;
    const std::uint32_t og = gen;
    resize_to(key.size() * 2);
    for (std::size_t i = 0; i < ok.size(); ++i)
      if (os[i] == og) { (void)find_or_add(ok[i], ov[i]); }
  }
  /// @brief Look up @p k, inserting it if absent.
  /// @param k    the packed discrete key
  /// @param next id to assign if @p k is not present
  /// @return the id of @p k, which equals @p next exactly when the key was newly inserted — the
  ///         caller detects insertion by that comparison
  int find_or_add(std::uint64_t k, int next) {
    if (fill >= limit) grow();
    std::size_t i = mix(k) & mask;
    for (;;) {
      if (stamp[i] != gen) { stamp[i] = gen; key[i] = k; val[i] = next; ++fill; return next; }
      if (key[i] == k) return val[i];
      i = (i + 1) & mask;
    }
  }
};

/// @brief Drop the first @p nfr states of the current level, keeping the remainder.
///
/// Used by the chunked frontier driver once a batch has been expanded.
///
/// @param[in,out] l   location components, @p np_ entries per state
/// @param[in,out] iv  integer components, @p niw_ entries per state
/// @param[in,out] z   interned zone indices, one per state
/// @param nfr  number of leading states to remove
/// @param np_  location entries per state
/// @param niw_ integer entries per state
///
/// @pre Each array holds at least @p nfr states' worth of entries.
/// @post The arrays hold the surviving states, still in order.
/// @note Erasing from the front shifts the remainder, so a level split into many batches pays
///       that shift once per batch. Chunk sizes are large enough that this is not the dominant
///       cost, but it does bound how small a useful chunk can be.
static void drop_batch(std::vector<int>& l, std::vector<long>& iv, std::vector<int>& z,
                       int nfr, int np_, int niw_, int) {
  l.erase(l.begin(), l.begin() + static_cast<std::size_t>(nfr) * np_);
  iv.erase(iv.begin(), iv.begin() + static_cast<std::size_t>(nfr) * niw_);
  z.erase(z.begin(), z.begin() + static_cast<std::size_t>(nfr));
}
#endif

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
static std::size_t sweep_dead_zones(ZoneIntern& zi, std::vector<NetPassed>& smap,
                                    const std::vector<std::pair<std::uint64_t, int>>& frontier,
                                    int nthreads, std::size_t floor_) {
  std::vector<std::uint8_t> alive(zi.index_bound(), 0);
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 8) num_threads(nthreads)
#endif
  for (long s = 0; s < static_cast<long>(smap.size()); ++s)
    for (const auto& nd : smap[s].arena)
      if (!nd.dead) __atomic_store_n(&alive[nd.zidx], std::uint8_t(1), __ATOMIC_RELAXED);
#ifdef PTCTLMC_HAVE_OPENMP
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
static std::vector<std::pair<std::uint64_t, int>> collect_next_frontier(
    std::vector<NetPassed>& smap, const StripeLayout& lay,
    std::vector<std::unordered_set<std::uint64_t>>& tls_touched,
    int nthreads, bool single, int level) {
  const int nbuk = single ? 1 : lay.nbuckets;
  // part[t * nbuk + b]: thread t's touched keys destined for bucket b.
  std::vector<std::vector<std::uint64_t>> part(static_cast<std::size_t>(nthreads) * nbuk);
  if (single) {
    part[0].assign(tls_touched[0].begin(), tls_touched[0].end());
  } else {
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
    for (long t = 0; t < static_cast<long>(nthreads); ++t) {
      const std::size_t base = static_cast<std::size_t>(t) * nbuk;
      for (const std::uint64_t dk : tls_touched[t]) part[base + lay.bucket_of(dk)].push_back(dk);
    }
  }
  std::vector<std::vector<std::pair<std::uint64_t, int>>> nf(nbuk);
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
#endif
  for (long b = 0; b < static_cast<long>(nbuk); ++b) {
    auto& buf = nf[b];
    // `single` already deduplicated inside the one touched set, so no second pass is
    // needed there; the parallel path can see the same dk from several threads.
    if (single) {
      for (const std::uint64_t dk : part[b])
        smap[lay.stripe_of(dk)].for_level(dk, level, [&](int zidx) { buf.push_back({dk, zidx}); });
      continue;
    }
    std::unordered_set<std::uint64_t> seen;
    for (int t = 0; t < nthreads; ++t) {
      for (const std::uint64_t dk : part[static_cast<std::size_t>(t) * nbuk + b]) {
        if (!seen.insert(dk).second) continue;
        smap[lay.stripe_of(dk)].for_level(dk, level, [&](int zidx) { buf.push_back({dk, zidx}); });
      }
    }
  }
  std::vector<std::pair<std::uint64_t, int>> next;
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
                                   const std::vector<std::int32_t>& U, int emode, SymState* s0) {
  s0->locs.resize(net.nprocs());
  for (int p = 0; p < net.nprocs(); ++p) s0->locs[p] = net.procs[p].initial;
  s0->ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0->ints[v] = net.intvars[v].init;
  s0->zone = Dbm(net.nclocks);
  std::vector<Constraint> inv;
  return close_succ_zone(net, s0->locs, s0->ints, s0->zone, inv, qmask, L, U, emode);
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
///                  bounds and build @c qmask — and for the GPU driver's flat target test
/// @param is_target what makes a symbolic state a witness. A predicate rather than a
///                  ptctlmc::NetTarget so that a target may inspect the *zone*, which deadlock
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
/// @warning A caller passing a predicate target must pass a default-constructed @p target and
///          keep the GPU path off: an empty ptctlmc::NetTarget matches every state, so the GPU
///          driver's flat test would report a witness immediately.
/// @warning Because cancellation and exhaustion both return @c false, the result of a search the
///          caller cancelled carries no information and must not be consulted.
///
/// @note @p cancel is polled at the level boundary rather than per state, so it costs one relaxed
///       load per level. Abandoning between levels leaves nothing half-built, since the caller
///       discards the whole engine.
/// @see DESIGN.md §4 for why stored zones must be exact and what that costs.
template <class TargetFn>
static bool reach_engine(const Network& net, const NetTarget& target, TargetFn&& is_target,
                         bool force_exact, Backend backend, ReachStats* stats,
                         const std::vector<Automorphism>* sym = nullptr,
                         const std::vector<NetTarget>* orbit = nullptr,
                         const std::atomic<bool>* cancel = nullptr) {
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
  // switch onwards. See DESIGN.md 4.
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

  // Decided after the GPU factory below: a GPU backend that falls back to the CPU path, because
  // the network is out of scope or no device is present, must still use every core. Testing for
  // Backend::Omp alone would silently run such a fallback single-threaded.
  bool parallel = false;
  int nthreads = 1;

  // GPU network backend (Metal or CUDA): the flat GPU-resident driver below runs the whole
  // level loop on the GPU (expand + per-bucket subsumption/convex-union merge). make_net_gpu_*
  // returns null for out-of-scope networks (diagonal constraints) or no device, so we fall back
  // to the CPU path transparently.
  std::unique_ptr<NetGpuExpander> gpu;
  bool use_gpu = false;
  // Two families of query stay on the CPU engine. force_exact callers pass a default NetTarget,
  // which the GPU's flat target test treats as matching everything, and the GPU merge selects its
  // subsumption from has_diag alone, so it cannot honour force_exact at all. Symmetry reduction
  // rewrites each successor to its orbit representative on the host, and the GPU drivers have no
  // such step. The condition is repeated per backend rather than wrapped around the block because
  // the two #ifdefs are independent and a wrapping brace would not pair.
#ifdef PTCTLMC_HAVE_METAL
  if (!force_exact && !use_sym && backend == Backend::Metal) gpu = make_net_gpu_metal(net, qmask);
#endif
#ifdef PTCTLMC_HAVE_CUDA
  if (!force_exact && !use_sym && backend == Backend::Cuda) gpu = make_net_gpu_cuda(net, qmask);
#endif
  use_gpu = (gpu != nullptr);
#ifdef PTCTLMC_HAVE_OPENMP
  // The CPU path runs multi-threaded for --backend omp AND for a GPU backend that fell back.
  parallel = (backend == Backend::Omp) || (!use_gpu && backend != Backend::Seq);
  if (parallel) nthreads = omp_get_max_threads();
#endif

  // The passed list is sharded by discrete-state hash, so a given discrete state always lands in
  // the same shard and each shard is a self-contained antichain owned by exactly one worker
  // during the merge. That is what lets the O(bucket^2 * dim^3) convex merge run without locks.
  const DiscKeyCodec dkey(net);

  SymState s0;
  if (!initial_symbolic_state(net, qmask, L, U, emode, &s0)) return false;
  if (is_target(s0)) return true;
  if (use_sym) canonicalize(*sym, net, s0);
  canon_ints(s0.locs, s0.ints);

#if defined(PTCTLMC_HAVE_METAL) || defined(PTCTLMC_HAVE_CUDA)
  // ---- Flat GPU-resident driver (Metal / CUDA) -------------------------------------------
  // State stays in flat zone arrays across levels: no round-trip through SymState and no
  // per-level rebuild of a map of DBM copies. The GPU performs the expansion and the per-bucket
  // subsumption and convex-union merge; the host only groups by discrete key, which is integer
  // hashing, and marshals flat zone spans.
  if (use_gpu) {
    // The GPU factories decline any network with more than 64 processes or 32 integer variables,
    // so within this block the fixed-size `dl`/`di` decode buffers below cannot overflow.
    // Structure-of-arrays work pool for the split GPU merge: the host gather and rebuild
    // transpose into and out of this layout, and the AoS CPU-merge fallback is disabled because
    // it cannot read it. Declared inside this block because every reader is GPU-only.
    const int np = net.nprocs(), ni = static_cast<int>(net.intvars.size()), dim = net.nclocks + 1, d2 = dim * dim;
    // The passed set holds zone indices into a shared dense pool rather than a private dense copy
    // per (discrete state, zone). With --gpu-intern the pool deduplicates, which is a large win
    // wherever few distinct zones are reachable; without it the pool simply appends through the
    // same indexed access path, so the flag isolates deduplication from every other difference.
    //
    // The arena narrows to 16-bit entries when the model's constants make overflow impossible.
    // The bound is 2*M + 1 for the largest clock constant M anywhere in the model or the query;
    // the margin applied below is deliberately generous. See ptctlmc::detail::GpuZoneIntern.
    std::int32_t maxc = 0;
    for (int c = 0; c <= net.nclocks; ++c) {
      if (L[c] != kAluNegInf) maxc = std::max(maxc, L[c]);
      if (U[c] != kAluNegInf) maxc = std::max(maxc, U[c]);
    }
    for (const auto& p_ : net.procs) {
      for (const auto& l : p_.locations) for (const auto& c : l.invariant) maxc = std::max(maxc, std::abs(raw_bound(c.bound)));
      for (const auto& e : p_.edges) for (const auto& c : e.clock_guard) maxc = std::max(maxc, std::abs(raw_bound(c.bound)));
    }
    for (const auto& c : target.clock_cond) maxc = std::max(maxc, std::abs(raw_bound(c.bound)));
    // Narrow (int16) arena entries are safe while (2M+1)<<1 stays well inside int16 for the
    // largest clock constant M anywhere in the model or the query; the margin is deliberately
    // generous.
    constexpr std::int32_t kNarrowZoneMaxConst = 4000;
    const bool narrow_zones = (maxc < kNarrowZoneMaxConst);
    GpuZoneIntern zi(d2, narrow_zones);
    zi.set_mode(detail::options().gpu_intern ? GpuZoneIntern::Mode::Dedup : GpuZoneIntern::Mode::Append);
    std::unordered_map<std::uint64_t, std::vector<int>> passed;   // dk -> antichain zone indices
    std::vector<int> fl(s0.locs.begin(), s0.locs.end());
    std::vector<long> fi(s0.ints.begin(), s0.ints.end());
    std::vector<int> fz(1, zi.intern(s0.zone.data().data()));
    passed[dkey.encode(s0.locs, s0.ints)].assign(1, fz[0]);
    double gpu_rate = -1, cpu_rate = -1;   // adaptive merge routing: work-units (sum bucket^2) per ms
    auto _now = [] { return std::chrono::steady_clock::now(); };
    auto _ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    DkIndex bidx;
    // Seeds only: both are replaced by measurement after the first level. prev_nb sizes the
    // next level's key index, fanout the next batch.
    constexpr std::size_t kInitialBucketEstimate = 1024;
    constexpr double kInitialFanout = 4.0;
    std::size_t prev_nb = kInitialBucketEstimate;
    double fanout = kInitialFanout;   // running MAX of successors per frontier state

    {   // let the expand kernel pack the discrete key itself (see net_gpu.hpp)
      std::vector<int> ilo(net.intvars.size());
      for (std::size_t v = 0; v < net.intvars.size(); ++v) ilo[v] = static_cast<int>(net.intvars[v].lo);
      gpu->set_key_layout(dkey.shift, ilo);
    }
    // Orbit of the target: with symmetry every state is stored as its orbit representative, so a
    // target that is not itself symmetric has to be tested against every image of the property.
    // Each image's required LOCATIONS are a fixed bit pattern in the packed key, so the test stays
    // |G| integer compares against the key (and only a key that matches pays for decoding ints).
    struct OrbitTest { std::uint64_t mask, val; const NetTarget* t; };
    std::vector<OrbitTest> tgt_tests;
    {
      const std::vector<NetTarget>* src = orbit && !orbit->empty() ? orbit : nullptr;
      const std::size_t n = src ? src->size() : 1;
      for (std::size_t i = 0; i < n; ++i) {
        const NetTarget& t = src ? (*src)[i] : target;
        OrbitTest ot{0, 0, &t};
        for (const auto& lr : t.loc_req) {
          ot.mask |= dkey.mask[lr.first] << dkey.shift[lr.first];
          ot.val  |= static_cast<std::uint64_t>(static_cast<unsigned>(lr.second)) << dkey.shift[lr.first];
        }
        tgt_tests.push_back(ot);
      }
    }
    const int niw = ni ? ni : 1;   // ints stride (kept >=1 so the vector is never zero-sized)
    // --chunking: expand the level in bounded BATCHES so the GPU pools are sized to a batch
    // rather than to the widest BFS level (which is what makes peak memory blow up: the expand
    // output pool is nwork * dim^2 words, and nwork is ~12x the frontier on fischer).
    //
    // The batch boundary must NOT move states between BFS levels. `fl` holds what is left of the
    // CURRENT level and `nf*` accumulates the NEXT one; a batch is taken from the front of `fl`
    // and its survivors go to `nf*`, which only becomes the frontier once `fl` is exhausted.
    // Appending the un-expanded tail to the survivors and swapping would mix level k and level
    // k+1 states into one frontier, losing strict BFS and inflating the explored-state count.
    //
    // A stored zone is a 31-bit index into the zone interner plus one flag bit meaning "added
    // during the level currently being expanded". That flag is what the next frontier is read off
    // at level end, and packing it into the index costs no extra memory.
    static constexpr int kNewBit = 1 << 30;
    std::vector<std::uint64_t> touched;   // buckets changed during the current level
    // Per-batch scratch, hoisted out of the loop. Each of these is sized by the batch, and with
    // --chunking a level is many batches, so allocating them per batch churns the allocator badly
    // enough to cancel the peak-memory saving chunking exists to deliver. Reused buffers keep
    // their capacity instead.
    std::vector<std::uint64_t> bdk;
    std::vector<int> bcnt, w_bucket, work_off, exist_cnt, new_off, new_cnt, new_idx, cursor;
    std::vector<int> fr_off, fr_cnt, host_isnew_v, host_count, out_work_off, out_count;
    std::vector<std::int32_t> Lm, Um;
    std::vector<std::vector<int>*> pzp;
    std::vector<std::size_t> foff;
    std::vector<int>   cfl; std::vector<long>  cfi; std::vector<raw_t> cfz;
    for (;;) {
      if (fl.empty()) {                       // current level exhausted: advance
        // Collect the next frontier: every zone still flagged "added during this level" in a
        // bucket the level touched. Doing this at level end rather than per batch is what keeps
        // batching exact, since a zone a later batch absorbed has had its flag cleared with it.
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        std::vector<int>().swap(fl); std::vector<long>().swap(fi); std::vector<int>().swap(fz);
        zi.flush_released();          // the frontier that could have named these is gone now
        foff.assign(touched.size() + 1, 0);
        for (std::size_t t = 0; t < touched.size(); ++t) {
          std::size_t c = 0;
          for (int idx : passed.find(touched[t])->second) if (idx & kNewBit) ++c;
          foff[t + 1] = foff[t] + c;
        }
        const std::size_t ftot = foff.back();
        if (ftot == 0) break;
        fl.resize(ftot * np); fi.resize(ftot * niw); fz.resize(ftot);
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
        for (long t = 0; t < static_cast<long>(touched.size()); ++t) {
          int dl[64]; long di[32];
          dkey.decode(touched[t], dl, di);
          std::size_t f = foff[t];
          auto& pz = passed.find(touched[t])->second;   // present: inserted earlier this level
          for (std::size_t i = 0; i < pz.size(); ++i) {
            if (!(pz[i] & kNewBit)) continue;
            pz[i] &= ~kNewBit;                                   // consumed
            for (int p = 0; p < np; ++p) fl[f * np + p] = dl[p];
            for (int v = 0; v < ni; ++v) fi[f * niw + v] = di[v];
            fz[f] = pz[i];       // the zone stays in the arena; the frontier names it
            ++f;
          }
        }
        touched.clear();
      }
      const int total = static_cast<int>(fz.size());
      // Batch size: either the fixed --chunking N, or the largest batch whose GPU pools fit the
      // budget.
      int nfr = total;
      if (detail::options().chunking) {
        if (detail::options().chunk_states > 0) {
          nfr = std::min(total, detail::options().chunk_states);
        } else {
          // Three words per successor per matrix entry: the expand output zones, the merge work
          // pool, and the frontier and staging arrays. The fan-out is the running maximum rather
          // than an average, because under-predicting it lets a batch overshoot the budget, and
          // such an overshoot is exactly the peak the budget exists to bound.
          const double per_state = fanout * static_cast<double>(d2) * sizeof(raw_t) * 3.0;
          const double cap = static_cast<double>(detail::chunk_budget_bytes()) / std::max(1.0, per_state);
          nfr = static_cast<int>(std::min<double>(total, std::max(4096.0, cap)));
        }
      }
      const bool whole = (nfr == total);
      if (stats) stats->explored += nfr;
      const int* ev = nullptr; const int* ol = nullptr; const int* oi = nullptr; const gz_t* oz = nullptr;
      const std::uint64_t* odk = nullptr;   // device-packed discrete keys (null => pack host-side)
      std::size_t nwork;
      {
        // Materialise only this batch's zones, widened out of the arena. The frontier itself
        // holds arena indices rather than dense matrices, because the zones are already in the
        // passed set and a level-wide dense copy would store every one of them twice.
        gz_t* fzbuf = nullptr;
        gpu->stage_frontier(static_cast<std::size_t>(nfr), d2, fzbuf);
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long i = 0; i < static_cast<long>(nfr); ++i)   // arena is already narrow: a memcpy
          zi.load_gz(fz[static_cast<std::size_t>(i)], fzbuf + static_cast<std::size_t>(i) * d2);
        cfz.clear();                              // "already staged" (see net_gpu.hpp)
        if (whole) {
          nwork = gpu->expand(nfr, fl, fi, cfz, ev, ol, oi, oz, odk);
        } else {
          cfl.assign(fl.begin(), fl.begin() + static_cast<std::size_t>(nfr) * np);
          cfi.assign(fi.begin(), fi.begin() + static_cast<std::size_t>(nfr) * niw);
          nwork = gpu->expand(nfr, cfl, cfi, cfz, ev, ol, oi, oz, odk);
        }
      }
      if (nfr > 0) fanout = std::max(fanout, static_cast<double>(nwork) / nfr);
      if (nwork == 0) {
        // this batch produced nothing: drop it and continue with what is left of the level
        drop_batch(fl, fi, fz, nfr, np, niw, d2);
        continue;
      }
      // The GPU now holds its own copy of the frontier in a persistent device buffer, so the host
      // copies are dead until the end-of-level swap. Releasing them here rather than later
      // matters because they would otherwise sit alongside the next frontier during the rebuild,
      // a full extra copy at exactly the moment memory peaks. Only safe when the whole level was
      // expanded; when batching, the tail is still needed.
      if (whole) { std::vector<int>().swap(fl); std::vector<long>().swap(fi); std::vector<int>().swap(fz); }
      else drop_batch(fl, fi, fz, nfr, np, niw, d2);   // batch consumed; keep the rest of the level
      // Group the valid successors by discrete key, reading straight from the GPU buffers and
      // testing the target on the fly. The layout is flat (CSR): a per-work-item bucket id and a
      // per-bucket count, followed by a prefix sum. A vector-of-vectors would instead allocate one
      // small vector per bucket per level, whose fragmentation dominates resident memory.
      //
      // This loop is serial by measurement rather than by omission: it is bound by streaming the
      // expand output out of unified memory while the GPU competes for the same bandwidth, not by
      // the hashing, so parallelising the hashing moves little. See DESIGN.md 8.10.
      bidx.begin_level(prev_nb);
      bdk.clear(); bcnt.clear();
      w_bucket.assign(nwork, -1);                         // work item -> bucket id (-1 = not a successor)
      for (std::size_t w = 0; w < nwork; ++w) {
        if (!ev[w]) continue;
        if (stats) ++stats->generated;
        std::uint64_t dk;
        if (odk) {
          dk = odk[w];
        } else {                                          // backend without device-side packing
          const int* wl = ol + w * np;
          const int* wi = oi + w * ni;
          dk = 0;
          for (int p = 0; p < np; ++p) dk |= static_cast<std::uint64_t>(static_cast<unsigned>(wl[p])) << dkey.shift[p];
          for (int v = 0; v < ni; ++v) dk |= static_cast<std::uint64_t>(wi[v] - net.intvars[v].lo) << dkey.shift[np + v];
        }
        // Target test straight off the key: the required locations are a fixed bit pattern in it,
        // so the common (non-target) case is two integer ops instead of a walk over nprocs
        // locations. Only a key that already matches pays for decoding the integers.
        for (const auto& ot : tgt_tests) {
          if ((dk & ot.mask) != ot.val) continue;
          int dl2[64]; long di2[32];
          dkey.decode(dk, dl2, di2);
          bool tgt = true;
          for (const auto& a : ot.t->int_cond) {
            const long x = di2[a.var]; bool r;
            switch (a.op) { case IntOp::Eq: r = x==a.val; break; case IntOp::Ne: r = x!=a.val; break;
              case IntOp::Lt: r = x<a.val; break; case IntOp::Le: r = x<=a.val; break;
              case IntOp::Gt: r = x>a.val; break; default: r = x>=a.val; }
            if (!r) { tgt = false; break; }
          }
          if (!tgt) continue;
          Dbm td(net.nclocks);
          for (int k = 0; k < d2; ++k) td.data()[k] = gz_load(oz[w * d2 + k]);
          if (apply_all(td, ot.t->clock_cond)) return true;
        }
        const int bi = bidx.find_or_add(dk, static_cast<int>(bdk.size()));
        if (bi == static_cast<int>(bdk.size())) { bdk.push_back(dk); bcnt.push_back(0); }
        w_bucket[w] = bi; ++bcnt[bi];
      }
      const int nb = static_cast<int>(bdk.size());
      prev_nb = static_cast<std::size_t>(nb);
      work_off.resize(nb); exist_cnt.resize(nb); new_off.resize(nb); new_cnt.resize(nb);
      Lm.resize(static_cast<std::size_t>(nb) * dim); Um.resize(static_cast<std::size_t>(nb) * dim);
      // Counts and prefix sums. New zones are never copied: the merge reads them in place from
      // the expand output by work-item index, so only the index list is flattened.
      std::size_t ntot = 0, wtot = 0;  // total new zones; total work-pool zones (exist+new)
      for (int b = 0; b < nb; ++b) {
        new_cnt[b] = bcnt[b];
        auto pit = passed.find(bdk[b]);
        exist_cnt[b] = (pit == passed.end()) ? 0 : static_cast<int>(pit->second.size());
        new_off[b] = static_cast<int>(ntot); ntot += new_cnt[b];
        work_off[b] = static_cast<int>(wtot); wtot += exist_cnt[b] + new_cnt[b];
      }
      // Scatter the work-item indices into the flat CSR array. Ascending w preserves the
      // per-bucket ordering the merge relies on, which matters because the convex union is
      // order-sensitive.
      new_idx.resize(ntot);
      {
        cursor = new_off;
        for (std::size_t w = 0; w < nwork; ++w) {
          const int bi = w_bucket[w];
          if (bi >= 0) new_idx[cursor[bi]++] = static_cast<int>(w);
        }
      }
      gz_t* work_buf = nullptr; int* work_isnew = nullptr;
      gpu->stage_merge(wtot, d2, work_buf, work_isnew);
      // Gather only the existing antichains into the unified-memory work buffer; successor zones
      // stay in the expand output and are read in place by the merge kernel.
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 32)
#endif
      for (int b = 0; b < nb; ++b) {
        auto pit = passed.find(bdk[b]);
        if (pit != passed.end()) {
          gz_t* dst = work_buf + static_cast<std::size_t>(work_off[b]) * d2;
          for (int i = 0; i < exist_cnt[b]; ++i)
            zi.load_gz(pit->second[i] & ~kNewBit, dst + static_cast<std::size_t>(i) * d2);
          // Seed each already-stored zone's tag with its passed-set index, kNewBit included. The
          // merge carries a kept zone's tag through its compaction and writes -1 for zones it
          // creates, so the rebuild can distinguish "this output slot is stored zone j" from
          // "this output slot is new", and a kept zone then needs neither copy nor re-intern.
          // Carrying kNewBit is required, not incidental: a zone added by an earlier batch of
          // this same level is still a frontier state, and dropping the flag would lose it.
          int* wi = work_isnew + work_off[b];
          for (int i = 0; i < exist_cnt[b]; ++i) wi[i] = pit->second[i];
        }
        int dl[64]; long di[32];
        std::vector<std::int32_t> L2, U2;
        std::vector<raw_t> Lr_, Ur_;   // unused: the GPU merge implements a_LU only
        dkey.decode(bdk[b], dl, di);
        state_bounds(dl, L2, U2, Lr_, Ur_);
        for (int c = 0; c < dim; ++c) { Lm[static_cast<std::size_t>(b) * dim + c] = L2[c]; Um[static_cast<std::size_t>(b) * dim + c] = U2[c]; }
      }
      out_work_off.clear(); out_count.clear();
      const gz_t* out_zones = nullptr;
      const int* out_isnew = nullptr;
      // Merge routing. In naive mode the GPU one-thread-per-bucket merge always runs. In adaptive
      // mode each path's throughput — work, taken as the sum of squared bucket sizes, per
      // millisecond — is calibrated on early levels, and each later level is routed to the
      // predicted-faster path. This recovers merge-heavy models, where one thread per bucket is
      // the wrong shape, without affecting models on which the GPU already wins.
      long long sum2 = 0;
      for (int b = 0; b < nb; ++b) { const long long s = exist_cnt[b] + new_cnt[b]; sum2 += s*s; }
      bool cpu_merge = false;
      if (detail::options().adaptive_merge) {
        if (cpu_rate < 0) cpu_merge = true;         // calibrate the CPU path first: it is never catastrophic
        else if (gpu_rate < 0) cpu_merge = false;   // then calibrate the GPU path
        else cpu_merge = (cpu_rate >= gpu_rate);    // route to the higher-throughput path
      }
      const auto _tm = _now();
      if (cpu_merge) {
        host_count.clear();
        host_isnew_v.assign(wtot, -1);         // zones this merge creates keep the -1 sentinel
        for (int b = 0; b < nb; ++b) {         // existing zones carry their passed index (see gather)
          auto pit = passed.find(bdk[b]);
          if (pit == passed.end()) continue;
          for (int i = 0; i < exist_cnt[b]; ++i)
            host_isnew_v[static_cast<std::size_t>(work_off[b]) + i] = pit->second[i];
        }
        host_count.assign(nb, 0);
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
        for (int b = 0; b < nb; ++b)
          host_count[b] = flat_merge_bucket(work_buf, work_off[b], exist_cnt[b], oz,
                                            new_idx.data() + new_off[b], new_cnt[b],
                                            Lm.data() + static_cast<std::size_t>(b)*dim,
                                            Um.data() + static_cast<std::size_t>(b)*dim,
                                            dim, has_diag, host_isnew_v.data() + work_off[b]);
        out_work_off = work_off; out_count = std::move(host_count);
        out_zones = work_buf; out_isnew = host_isnew_v.data();
      } else {
        gpu->merge(nb, dim, has_diag, work_off, exist_cnt, new_idx, new_off, new_cnt, Lm, Um,
                   out_work_off, out_count, out_zones, out_isnew);
      }
      const double _dtm = _ms(_tm, _now());
      if (detail::options().adaptive_merge && sum2 > 0 && _dtm > 0.5) {   // update throughput calibration (skip trivial levels)
        const double rate = static_cast<double>(sum2) / _dtm;
        if (cpu_merge) cpu_rate = (cpu_rate < 0) ? rate : 0.5 * cpu_rate + 0.5 * rate;
        else           gpu_rate = (gpu_rate < 0) ? rate : 0.5 * gpu_rate + 0.5 * rate;
      }
      // Rebuild the passed set and assemble the flat next frontier: count the per-bucket
      // survivors, prefix-sum the counts, pre-create the passed nodes, then fill in parallel.
      //
      // The count and the offset prefix-sum stay fused and serial. Buckets hold only a handful of
      // zones on average, so the scan is memory-bound and the per-bucket parallel overhead, plus
      // false sharing on fr_cnt, costs more than the scan itself.
      fr_off.resize(nb); fr_cnt.resize(nb);
      std::size_t frtot = 0;
      for (int b = 0; b < nb; ++b) {
        int c = 0;
        for (int i = 0; i < out_count[b]; ++i) {
          const int tag = out_isnew[static_cast<std::size_t>(out_work_off[b]) + i];
          if (tag < 0 || (tag & kNewBit)) ++c;
        }
        fr_cnt[b] = c; fr_off[b] = static_cast<int>(frtot); frtot += c;
      }
      // A bucket with no survivor had all its new zones subsumed, so its antichain is unchanged
      // and `passed[dk]` is already correct; its rebuild copy is skipped entirely. Only changed
      // buckets are cleared and refilled, and their element pointers stay stable across later
      // inserts.
      pzp.assign(nb, nullptr);
      for (int b = 0; b < nb; ++b) if (fr_cnt[b] > 0) { pzp[b] = &passed[bdk[b]]; touched.push_back(bdk[b]); }
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 32)
#endif
      for (int b = 0; b < nb; ++b) {
        if (fr_cnt[b] == 0) continue;   // unchanged bucket — nothing to do
        int dl[64]; long di[32];
        dkey.decode(bdk[b], dl, di);
        auto& pz = *pzp[b];
        // This bucket's slots before the merge; those the merge drops return to the arena free
        // list below. Thread-local rather than a fixed-size array because antichains can reach
        // several hundred zones, and a fixed bound would silently leak every slot past it.
        thread_local std::vector<int> oldv;
        oldv.clear();
        for (int idx : pz) oldv.push_back(idx & ~kNewBit);
        pz.resize(static_cast<std::size_t>(out_count[b]));
        int fpos = 0;
        std::vector<raw_t> tmp(d2);        // one merged zone, reused across this bucket
        for (int i = 0; i < out_count[b]; ++i) {
          const std::size_t slot = static_cast<std::size_t>(out_work_off[b]) + i;
          const int tag = out_isnew[slot];
          if (tag >= 0) { pz[i] = tag; continue; }   // kept: already stored, already interned
          raw_t* pzi = tmp.data();
          for (int k = 0; k < d2; ++k) pzi[k] = zget(out_zones + slot * d2, k);
          // The high bit of the stored index records "added during the CURRENT level". The
          // frontier is read off these flags once the level is finished (see the level-advance
          // step), NOT here -- because a later batch of the same level can still absorb this zone
          // into a convex union, and emitting it now would put a state in the frontier that the
          // level's own merge went on to delete. That is what made batching cost fischer12 2.5x
          // its state count.
          pz[i] = zi.intern(pzi) | kNewBit;
        }
        // Slots the merge dropped (subsumed, or absorbed into a convex union) are garbage now.
        // Antichains are small, so a linear membership test beats any bookkeeping.
        for (int o : oldv) {
          bool reused = false;
          for (int i = 0; i < out_count[b] && !reused; ++i) reused = (pz[i] & ~kNewBit) == o;
          if (!reused) zi.release(o);
        }
        (void)dl; (void)di; (void)fpos;
      }
#ifdef __APPLE__
#endif
    }
    return false;
  }
#endif

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
    std::vector<NetPassed> smap(lay.nstripes);         // flat per-stripe passed tables
    for (auto& mp : smap) mp.zi = &zi;
    // The mirror budget is process-wide while these tables are per query, so it must be reset
    // here; otherwise a second query in the same process inherits the first one's charge and
    // never promotes anything.
    detail::hot_bytes().store(0, std::memory_order_relaxed);
    // A reader/writer lock per stripe: the read-only kill probe below runs concurrently within a
    // stripe, and only a genuine insert or merge needs exclusivity.
    std::vector<SpinRW> smtx(lay.nstripes);
    //
    // Per-thread local antichains were tried here and removed: they are correct but consistently
    // slower, because a private antichain cannot consult the shared read-only kill probe that
    // answers most successors. See DESIGN.md 8.9 before reconsidering.
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
    std::vector<std::pair<std::uint64_t, int>> frontier;
    {
      const std::uint64_t dk = dkey.encode(s0.locs, s0.ints);
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
      std::vector<std::unordered_set<std::uint64_t>> tls_touched(nthreads);
      std::vector<std::size_t> tls_gen(nthreads, 0);
#ifdef PTCTLMC_HAVE_OPENMP
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
            const std::uint64_t dk = dkey.encode(s.locs, s.ints);
            const std::size_t si = lay.stripe_of(dk);
            if (single) {
              if (smap[si].add(std::move(s), dk, level, exact_sub, state_bounds, tid) >= 0) touched.insert(dk);
            } else {
              {   // shared: concurrent readers, no serialization on the common "already covered" case
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
            const std::uint64_t dk = dkey.encode(s.locs, s.ints);
            if (smap[lay.stripe_of(dk)].add(std::move(s), dk, level, exact_sub, state_bounds, 0) >= 0)
              touched.insert(dk);
          }
        }
      }
#endif
      if (stats) for (std::size_t g : tls_gen) stats->generated += g;
      if (found.load()) return true;

      // The passed tables are read-only here (no writers), so for_level is a safe read.
      std::vector<std::pair<std::uint64_t, int>> next =
          collect_next_frontier(smap, lay, tls_touched, nthreads, single, level);
      // Reordering a level to expand its most general zones first was tried and removed: it is
      // sound but does not pay, because zone retirement is driven by zones arriving levels apart,
      // which no intra-level ordering can reach. See DESIGN.md 8.14.
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
#ifdef PTCTLMC_HAVE_OPENMP
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
///       ptctlmc::deadlock_reachable, which is not.
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
  return reach_engine(net, bounds_target, is_target, /*force_exact=*/false, backend, stats,
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
///       STAYING in a convex `C` is `(Z /\ C)^ /\ C` — every intermediate point of the segment
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
bool exists_invariant_run(const Network& net_in, const DiscPred& phi, Backend backend,
                          ReachStats* stats) {
  (void)backend;  // the cycle detection below is sequential

  // The psi-restricted network. Built only when the predicate mentions a clock, so a purely
  // discrete predicate runs on `net` itself and is unaffected down to the last stored zone.
  Network restricted;
  if (!phi.clocks.empty()) {
    restricted = net_in;
    for (auto& proc : restricted.procs)
      for (auto& loc : proc.locations)
        loc.invariant.insert(loc.invariant.end(), phi.clocks.begin(), phi.clocks.end());
    // Recomputed, not patched: these analyses are exactly what the two side conditions ask for,
    // and deriving them from the restricted invariants is what keeps them consistent with it.
    restricted.compute_bounds();
    restricted.compute_loc_bounds();
    restricted.compute_active();
    restricted.compute_int_live();
  }
  const Network& net = phi.clocks.empty() ? net_in : restricted;

  std::vector<std::int32_t> L = net.lbound, U = net.ubound;
  const int emode = network_has_diagonal(net) ? ExNoDiag : ExLuPlus;  // diagonals: keep exact
  // Clocks the predicate reads. `compute_active` above already marks them active at every
  // location, since they now appear in every invariant; this is the belt to that braces, and
  // stays all-zero -- maximal reduction -- for a purely discrete predicate.
  std::vector<char> qmask(net.nclocks + 1, 0);
  for (const auto& c : phi.clocks) { if (c.i) qmask[c.i] = 1; if (c.j) qmask[c.j] = 1; }

  auto sat = [&](const SymState& s) -> bool {
    for (const auto& r : phi.locs) {
      const bool eq = (s.locs[r.proc] == r.loc);
      if (eq != r.equal) return false;
    }
    return int_ok(phi.ints, s.ints);
  };
  auto key = [&](const SymState& s) -> std::string {
    std::string k = disc_key_bytes(s.locs, s.ints);
    const auto& d = s.zone.data();
    k.append(reinterpret_cast<const char*>(d.data()), d.size() * sizeof(raw_t));
    return k;
  };

  SymState s0;
  if (!initial_symbolic_state(net, qmask, L, U, emode, &s0)) return false;
  if (!sat(s0)) return false;  // no run from the initial state can stay in phi

  // Build the phi-restricted extrapolated zone graph (nodes deduped by exact zone).
  std::unordered_map<std::string, int> id;
  std::vector<SymState> node;
  std::vector<std::vector<int>> adj;
  auto intern = [&](SymState&& s) -> int {
    std::string k = key(s);
    auto it = id.find(k);
    if (it != id.end()) return it->second;
    const int n = static_cast<int>(node.size());
    id.emplace(std::move(k), n);
    node.push_back(std::move(s));
    adj.emplace_back();
    return n;
  };
  // A location where every clock can grow without bound (after delay + invariant) lets
  // time diverge => an infinite run that stays put — an E[] witness on its own.
  auto time_diverges = [&](const SymState& s) -> bool {
    if (net.nclocks == 0) return false;
    // FROZEN LOCATIONS DO NOT DELAY. Delaying unconditionally reports divergence at a
    // committed/urgent location, where no positive delay is legal at all -- inventing a
    // maximal-run witness out of nothing.
    { bool fc, fu; frozen_flags(net, s.locs, s.ints, &fc, &fu); if (fc || fu) return false; }
    Dbm z = s.zone;
    z.delay();
    std::vector<Constraint> inv;
    all_invariants(net, s.locs, inv);
    apply_all(z, inv);
    if (z.empty()) return false;
    for (int i = 1; i <= net.nclocks; ++i)
      if (z.get(i, 0) != kInf) return false;
    return true;
  };

  intern(std::move(s0));
  for (int u = 0; u < static_cast<int>(node.size()); ++u) {
    if (stats) ++stats->explored;
    if (time_diverges(node[u])) return true;
    std::vector<SymState> succ;
    successors_net(net, node[u], node[u].zone, L, U, qmask, emode, succ);
    for (auto& ns : succ) {
      if (!sat(ns)) continue;
      const int v = intern(std::move(ns));
      adj[u].push_back(v);
    }
  }

  // Reachable cycle? Iterative three-colour DFS; a back edge (to a GREY node) = cycle.
  enum { White, Grey, Black };
  std::vector<char> color(node.size(), White);
  std::vector<std::pair<int, std::size_t>> stk;
  for (int root = 0; root < static_cast<int>(node.size()); ++root) {
    if (color[root] != White) continue;
    stk.push_back({root, 0});
    color[root] = Grey;
    while (!stk.empty()) {
      const int u = stk.back().first;
      const std::size_t idx = stk.back().second;
      if (idx < adj[u].size()) {
        stk.back().second = idx + 1;
        const int v = adj[u][idx];
        if (color[v] == Grey) return true;
        if (color[v] == White) { color[v] = Grey; stk.push_back({v, 0}); }
      } else {
        color[u] = Black;
        stk.pop_back();
      }
    }
  }
  return false;
}

/// @brief Does the symbolic state @p s contain a valuation with no outgoing action transition?
///
/// A zone with no successor at all is the easy case. The hard case is a *partial* deadlock,
/// where the zone has successors but some sub-region of it can never reach any of them — for
/// instance where an invariant caps time before the only guard opens. This computes the partial
/// case exactly rather than approximating it.
///
/// The three transition kinds of ::successors_net are mirrored. For each transition that can
/// fire, its *source* region within `s.zone` is collected: the pre-reset valuations from which it
/// fires and lands in a legal target, i.e. the guards intersected with the target invariant
/// pulled back through the resets. The union of the time-pasts of those source regions is the set
/// of non-deadlocked valuations, and whatever remains of the zone after subtracting it is
/// deadlocked.
///
/// @param net  the network under analysis
/// @param s    the symbolic state to test
/// @param[out] stuck    if non-null, receives the deadlocked sub-region; cleared when there is none
/// @param[out] escapes  if non-null, receives the escape regions themselves, for callers that
///                      need to know *which* constraints decide stuckness rather than only the
///                      residue
/// @return @c true iff some valuation of @p s is deadlocked
///
/// @pre  For an exact answer, `s.zone` must already satisfy the invariant of `s.locs`. States
///       produced by the forward engine satisfy this, since ::close_succ_zone applies it.
/// @post `*escapes` is inclusion-minimal; a region contained in a sibling is never added.
///
/// @warning At a **frozen** state — committed or urgent — neither the delay nor the invariant is
///          applied, so the zone subtracted from is `s.zone` as given. A caller that passes the
///          bare universe for such a state therefore receives `Dead` computed against the
///          universe rather than against `Inv(q)`, which over-approximates. This is the
///          conservative direction and costs precision only, but such a caller must intersect the
///          result with `Inv(q)` itself. See KNOWN_LIMITATION.md.
/// @note Enabledness is invariant under LU-extrapolation, so working on the extrapolated zone is
///       sound — the same assumption the rest of the engine relies on.
/// @see DESIGN.md §2 for the deadlock completion this feeds.
bool detail::state_has_deadlock(const Network& net, const SymState& s,
                                Federation* stuck, Federation* escapes) {
  if (s.zone.empty()) return false;
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, s.ints, &src_committed, &src_urgent);
  const bool frozen = src_committed || src_urgent;  // no delay allowed from this state

  // Rebuild the true delay-closure of this state's zone. The stored zone carries the
  // active-clock reduction, which pins dead clocks to 0 with an upper bound and so is no longer
  // delay-closed; without this step down() could not let time pass those clocks and would invent
  // spurious deadlocks. Re-opening every upper bound and re-imposing the real location invariant
  // restores the closure. Dead clocks become free, which is sound because no guard or invariant
  // reachable from here reads them.
  //
  // The invariant is applied whether or not the location is frozen; only the delay is skipped.
  // A frozen location still constrains its clocks, and omitting the invariant left @p Z as the
  // bare universe, which reports valuations stuck that the invariant forbids from arising at all.
  // That was conservative rather than wrong -- a spurious residue only ever costs precision --
  // but it made the caller decline on keys that cannot deadlock.
  Dbm Z = s.zone;
  {
    if (!frozen) Z.delay();
    std::vector<Constraint> cinv;
    all_invariants(net, s.locs, cinv);
    if (!apply_all(Z, cinv)) return false;  // real invariant unsatisfiable: no concrete state
  }

  std::vector<Constraint> tinv;  // target-invariant scratch
  Federation live;

  // `guarded` is Z intersected with the participants' clock guards, before any reset. It is
  // pulled back through the target invariant under the transition's resets, and if a non-empty
  // source region survives it joins `live`. In that pullback a constraint on a reset clock
  // becomes a constraint on clock 0, its post-reset value, so it either constrains the surviving
  // source clocks or, when it reduces to `0 ~ bound`, gates the whole transition on whether that
  // constant is satisfiable.
  auto add_src = [&](Dbm guarded, const LocVec& tgt_locs,
                     const std::vector<const std::vector<int>*>& reset_lists) {
    if (guarded.empty()) return;
    auto is_reset = [&](int c) {
      for (auto* rl : reset_lists) for (int r : *rl) if (r == c) return true;
      return false;
    };
    all_invariants(net, tgt_locs, tinv);
    for (const auto& c : tinv) {
      const int i = is_reset(c.i) ? 0 : c.i;
      const int j = is_reset(c.j) ? 0 : c.j;
      if (i == j) { if (c.bound < kZeroLe) return; continue; }  // 0 <| bound: constant gate
      if (!guarded.constrain(i, j, c.bound)) return;
    }
    // Kept inclusion-minimal rather than simply appended. `live` is consumed only through its
    // union — by the subtraction below, and by `escapes` for the atom construction — so a member
    // contained in a sibling denotes nothing new. It is not free either: the subtraction peels
    // the remainder once per member and the piece count multiplies, so a redundant escape costs
    // an extra peel of an already-large arrangement. Two escapes that agree on their clock guard
    // and differ only in which process moved are the common case.
    //
    // Pruning before down() is sound because W subseteq W' implies down(W) subseteq down(W'), so
    // the union of the time-pasts is unchanged.
    if (!guarded.empty()) fed_add(live, guarded);
  };

  // ---- local (unsynchronised) edges ----
  for (int p = 0; p < net.nprocs(); ++p) {
    if (src_committed && !net.procs[p].locations[s.locs[p]].committed) continue;
    for (int ei : net.procs[p].edges_of(s.locs[p])) {
      const NEdge& e = net.procs[p].edges[ei];
      if (e.sync_chan >= 0) continue;
      if (!int_ok(e.int_guard, s.ints) || !gen_ok(net, e, s.ints.data(), s.ints.size())) continue;
      IntVec nints = s.ints;
      for (const auto& a : e.int_updates) nints[a.var] = a.val;
      if (!gen_apply(net, e, nints.data(), nints.size())) continue;
      Dbm g = Z;
      if (!apply_all(g, e.clock_guard)) continue;
      LocVec tl = s.locs; tl[p] = e.tgt;
      add_src(std::move(g), tl, {&e.clock_resets});
    }
  }

  // ---- binary channel synchronisation (a! || a?) ----
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
          IntVec nints = s.ints;
          for (const auto& a : se.int_updates) nints[a.var] = a.val;
          for (const auto& a : re.int_updates) nints[a.var] = a.val;
          if (!gen_apply(net, se, nints.data(), nints.size()) ||
              !gen_apply(net, re, nints.data(), nints.size())) continue;
          Dbm g = Z;
          if (!apply_all(g, se.clock_guard) || !apply_all(g, re.clock_guard)) continue;
          LocVec tl = s.locs; tl[i] = se.tgt; tl[j] = re.tgt;
          add_src(std::move(g), tl, {&se.clock_resets, &re.clock_resets});
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
          Dbm g = Z;
          bool ok = apply_all(g, se.clock_guard);
          std::vector<const std::vector<int>*> rls{&se.clock_resets};
          LocVec tl = s.locs; tl[i] = se.tgt;
          for (std::size_t t = 0; ok && t < parts.size(); ++t) {
            const RecvOpt& o = parts[t].second[pick[t]];
            if (o.edge < 0) { ok = apply_all(g, o.extra); continue; }   // abstained: nothing moves
            const NEdge& re = net.procs[parts[t].first].edges[o.edge];
            ok = apply_all(g, re.clock_guard);
            rls.push_back(&re.clock_resets);
            tl[parts[t].first] = re.tgt;
          }
          if (ok) add_src(std::move(g), tl, rls);
        }
        if (parts.empty()) break;
        std::size_t k = 0;
        for (; k < parts.size(); ++k) { if (++pick[k] < parts[k].second.size()) break; pick[k] = 0; }
        if (k == parts.size()) break;
      }
    }
  }

  // Non-deadlocked = valuations that can (delay into, unless frozen) some enabled source
  // region. Subtract that from the zone; a non-empty remainder is a set of deadlock states.
  if (!frozen)
    for (auto& W : live) W.down();
  // The escape regions themselves, for callers needing to know which constraints decide
  // stuckness rather than only the residue. Since Dead = Z \ union(live), membership is decided
  // by the atoms of each escape — one convex zone per outgoing transition — whereas the residue
  // is an arrangement whose piece count, and hence atom count, is far larger for the same set.
  if (escapes) *escapes = live;
  // Short circuit when a single escape already covers Z, which is the common case of a location
  // with one unguarded outgoing edge. The subtraction peels Z once per escape and the piece count
  // multiplies, so the arrangement it builds dominates the sweep; one DBM inclusion test per
  // escape avoids building it at all.
  //
  // This is exact, not a heuristic: Z subseteq W subseteq union(live) means Dead(q) is empty,
  // which is precisely what the subtraction would have reported.
  for (const Dbm& W : live)
    if (W.includes(Z)) {
      if (stuck) stuck->clear();
      return false;
    }
  Federation d = fed_subtract(Z, live);      // exactly the valuations with nothing enabled
  if (stuck) *stuck = d;
  return !d.empty();
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
/// @see DESIGN.md §3 for the full argument and the measured comparison against exact zones.
bool deadlock_reachable(const Network& net, Backend backend, ReachStats* stats,
                        const std::atomic<bool>* cancel) {
  // The GPU drivers cannot express this target. deadlock_reachable never asks for a GPU backend,
  // so the CPU engine is selected regardless of force_exact.
  const Backend b = (backend == Backend::Seq) ? Backend::Seq : Backend::Omp;
  const auto is_dl = [&](const SymState& s) { return state_has_deadlock(net, s); };

  // Diagonals make every LU-family abstraction unsound, and the G-bound arrays carry raw encoded
  // bounds with their own sentinel, so collapsing those correctly is a separate exercise. Both
  // cases fall back to exact zones, which is sound but stores more states.
  if (network_has_diagonal(net) || gsim_on())
    return reach_engine(net, NetTarget{}, is_dl, /*force_exact=*/true, b, stats, nullptr, nullptr,
                        cancel);

  Network am = net;
  auto collapse = [](std::vector<std::int32_t>& lo, std::vector<std::int32_t>& hi) {
    for (std::size_t c = 0; c < lo.size() && c < hi.size(); ++c)
      lo[c] = hi[c] = std::max(lo[c], hi[c]);
  };
  collapse(am.lbound, am.ubound);
  for (std::size_t pr = 0; pr < am.lbound_loc.size(); ++pr)
    for (std::size_t l = 0; l < am.lbound_loc[pr].size(); ++l)
      collapse(am.lbound_loc[pr][l], am.ubound_loc[pr][l]);
  return reach_engine(am, NetTarget{}, is_dl, /*force_exact=*/false, b, stats, nullptr, nullptr,
                      cancel);
}



}  // namespace ptctlmc
