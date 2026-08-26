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

// ---- general modelling-language guard/update -------------------------------
// Only edges that need more than `var OP const` carry these (see NEdge). The
// common case is a null check, so this costs nothing for Fischer/viking.
//
// These two are deliberately OUTSIDE the anonymous namespace: the GPU engines
// (src/metal, src/cuda) must apply exactly the same semantics as the CPU path,
// and duplicating them there is how the backends silently diverged before --
// the Metal expander dropped `gen_update` entirely and explored roughly half of
// viking's state space. Declared in netreach.hpp.

// Scratch buffer so a guard that calls a function cannot perturb the live state.
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

// Runs the general update in place. Returns false if it faulted (bad index,
// division by zero, runaway loop) or if a write left a variable's declared
// range -- the latter would silently collide in the packed discrete key, so it
// is refused rather than mis-encoded.
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

// Does the network compare two clocks anywhere (a diagonal constraint)? If so, the
// LU/aLU abstraction is unsound and we fall back to exact zones (diagonals kept exact).
// EXTERNAL linkage on purpose -- declared in netreach.hpp, so the query router in main.cpp
// can ask the same question before choosing an engine.
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

// The GPU engines (netengine_metal.mm / netengine_cuda.cu) ask the same question through the
// `netgpu` declaration in net_gpu_encode.hpp. One body, two names.
namespace netgpu { bool network_has_diagonal(const Network& net) { return ptctlmc::network_has_diagonal(net); } }

namespace {

// The parent zone is passed SEPARATELY (szone) rather than read from s.zone, so
// callers that already hold the zone elsewhere -- notably the shared-passed path,
// where it lives in the intern table -- can pass it by reference instead of
// copying a 576 B DBM into the frontier state for every expansion (millions of
// pointless copies on large viking). s.zone is ignored; s carries only locs/ints.
void successors_net(const Network& net, const SymState& s, const Dbm& szone,
                    const std::vector<std::int32_t>& L,
                    const std::vector<std::int32_t>& U, const std::vector<char>& qmask, int emode, std::vector<SymState>& out) {
  thread_local std::vector<Constraint> inv;  // reused invariant scratch (see all_invariants)
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, &src_committed, &src_urgent);
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

      if (!close_succ_zone(net, ns.locs, z1, inv, qmask, L, U, emode)) continue;
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
          if (!close_succ_zone(net, ns.locs, z1, inv, qmask, L, U, emode)) continue;
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
            if (close_succ_zone(net, ns.locs, z1, inv, qmask, L, U, emode) && !z1.empty()) {
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

// Slot-reusing successor generation for the shared-passed hot path. Semantically

// ---------------------------------------------------------------------------------------
// TRANSITION MEMO (opt-in: --memo).
//
// The successor zone is a function of (source zone, edge, target-location data), and on some
// models very few DISTINCT zones exist -- viking stores 13.6M states drawn from 120 zones -- so
// the same O(dim^3) delay/constrain/close is recomputed enormous numbers of times. The engine
// already noticed this; this caches the result.
//
// KEY. Everything Post reads, and nothing more:
//   * source zone id      -- free: the frontier already carries the interned zidx
//   * edge identity       -- (proc, edge index), packed
//   * target-location sig -- Post reads the target locations ONLY through the combined
//                            invariant, the urgent/committed flags, and the active-clock mask.
//                            Many location vectors share one signature; that sharing is the
//                            whole effect. The signature is INTERNED (not hashed to a fixed
//                            width) so there is no collision risk -- a wrong hit would be
//                            silent corruption, which is not an acceptable trade for a cache.
//
// THREAD-LOCAL, so no locks and no sharing between workers. The tables are tiny where the memo
// pays (220 entries for viking17's 13.6M states), so per-thread duplication costs nothing.
//
// ADAPTIVITY. A sliding window measures the RECENT hit rate; a bad window disables the memo and
// doubles a backoff; after the backoff elapses it re-probes for one window. The window starts
// small and grows while healthy, so a model with no reuse is detected in ~512 lookups and a
// model with reuse checks less and less often. Nothing is tuned to model size.
struct MemoKey {
  int z; std::uint32_t t; std::uint32_t l;
  bool operator==(const MemoKey& o) const { return z == o.z && t == o.t && l == o.l; }
};
struct MemoKeyHash {
  std::size_t operator()(const MemoKey& k) const {
    std::uint64_t h = 1469598103934665603ull;
    h ^= (std::uint64_t)(std::uint32_t)k.z; h *= 1099511628211ull;
    h ^= k.t; h *= 1099511628211ull;
    h ^= k.l; h *= 1099511628211ull;
    return (std::size_t)h;
  }
};
struct SuccMemo {
  std::unordered_map<MemoKey, Dbm, MemoKeyHash> m;   // an EMPTY Dbm means "this transition dies"
  std::unordered_map<std::string, std::uint32_t> lsig;
  std::string lk;                                    // reused signature buffer: no allocation
  long winLookups = 0, winHits = 0;
  long window = 512, backoff = 512, skipped = 0;
  bool on = true;
  long lookups = 0;
  static constexpr long kWindowMax = 65536;
  static constexpr long kMaxBackoff = 1L << 22;
  static constexpr double kMinHit = 0.30;
  static constexpr double kMinDensity = 32.0;  ///< lookups per entry below which the table is not worth it
};
inline SuccMemo& tls_succ_memo() { static thread_local SuccMemo sm; return sm; }

inline bool memo_enabled() {
  static const bool v = detail::options().memo;
  return v;
}

// Interned target-location signature. Cheap: it walks nprocs but only emits for processes whose
// location actually carries an invariant or a frozen flag, which is a small minority.
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

// Evaluate the sliding window; called once per lookup. Returns whether the memo is live.
inline bool memo_tick(SuccMemo& sm) {
  if (!sm.on) {
    if (++sm.skipped >= sm.backoff) { sm.on = true; sm.skipped = 0; sm.winLookups = sm.winHits = 0; }
    return sm.on;
  }
  if (sm.winLookups >= sm.window) {
    // TWO criteria, because hit rate alone is the wrong one. phil12 sustains a 78% hit rate and
    // is still 33% SLOWER, because its table holds 117 629 entries per thread: each entry is
    // consulted only ~4.6 times, so the lookup and memory-pressure cost exceeds the Post it
    // saves. viking15 consults each of its 933 entries ~855 times. REUSE DENSITY
    // (lookups / entries) separates these cleanly where hit rate does not, and it is
    // dimensionless -- nothing is tied to model size.
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

// identical to successors_net above, but instead of building a fresh SymState
// (two pooled loc/int vectors + a DBM) per successor and push_back-ing it -- ~3
// allocations for every one of tens of millions of successors, all discarded by
// add() which keeps only the interned zone -- it BUILDS EACH SUCCESSOR IN PLACE in
// a reused `out` slot. The caller does not clear `out`; it reads out[0..return).
// Because add() interns the zone by copy (never moves the SymState out), the slots
// survive the caller's loop and are reused on the next frontier state, so after a
// short warm-up the per-successor allocation is ZERO. (The legacy successors_net
// stays for the seq/legacy/single-TA paths, which accumulate rather than reuse and
// are slated for removal.)
std::size_t gen_succ_reuse(const Network& net, const SymState& s, const Dbm& szone,
                           const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U,
                           const std::vector<char>& qmask, int emode, std::vector<SymState>& out,
                           int szidx) {
  thread_local std::vector<Constraint> inv;
  SuccMemo& sm = tls_succ_memo();
  const bool memo = memo_enabled() && szidx >= 0;
  std::size_t on = 0;
  auto slot = [&]() -> SymState& { if (on == out.size()) out.emplace_back(); return out[on]; };
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, &src_committed, &src_urgent);

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
        if (!close_succ_zone(net, ns.locs, ns.zone, inv, qmask, L, U, emode)) {
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
      if (!close_succ_zone(net, ns.locs, ns.zone, inv, qmask, L, U, emode)) continue;
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
            if (!close_succ_zone(net, ns.locs, ns.zone, inv, qmask, L, U, emode)) {
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
          if (!close_succ_zone(net, ns.locs, ns.zone, inv, qmask, L, U, emode)) continue;
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
            if (close_succ_zone(net, ns.locs, ns.zone, inv, qmask, L, U, emode) &&
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

// GPU merge routing mode (see netreach.hpp). Default naive = always the one-thread-per-bucket GPU
// merge (the original, published behavior); --adaptive turns on the calibrated CPU/GPU routing.
// Rewrite (locs, zone) to its orbit's canonical representative: the group element giving the
// lexicographically smallest location vector and, among those, the smallest zone matrix.
//
// The location pass alone decides it in almost every case (the zone is only permuted for the
// winners), so the cost is |G| x nprocs integer compares per successor -- against a state space
// |G| times smaller. Permuting a canonical DBM by a clock permutation leaves it canonical: the
// triangle inequality is preserved under relabelling, so no re-closure is needed.
static void canonicalize(const std::vector<Automorphism>& G, const Network& net, SymState& s) {
  const int np = static_cast<int>(s.locs.size());
  const int ni = static_cast<int>(s.ints.size());
  const int dim = s.zone.dim();
  // The DISCRETE key is (locations, integer values) and BOTH move. Permuting only the locations
  // is not a partial canonicalisation, it is a wrong one: Fischer's `id` holds a process index,
  // so a state whose location vector has been rotated while `id` still names the old process is
  // not in the orbit at all. It denotes a configuration where the lock is held by a process that
  // is not in `req` -- reachable in the reduced graph, unreachable in the real one, and it
  // reports mutual exclusion VIOLATED on a model that satisfies it.
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
// Host port of the net_merge GPU kernel body (dbm_ops.hpp host functions), for ONE discrete-state
// bucket. The GPU runs one thread per bucket, so a level with few/fat buckets stalls on a handful of
// slow lanes; this lets the flat driver merge such buckets across all CPU cores instead. Existing
// antichain is at work_buf[work_off_b .. +exist_cnt_b) (already gathered); new zones are read in
// place from the expand output `oz` at indices new_idx_b[0..new_cnt_b). Writes the final antichain
// back in place, sets WI (isnew flags: existing=0, appended new=1), returns the surviving count.
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

// The network is wider than the packed key can hold. Cold, and deliberately verbose: the widest
// fields are almost always a couple of int arrays, and the breakdown is the only part of this the
// user can act on.
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

// PACKED DISCRETE KEY. A symbolic state's discrete half -- the location vector plus the integer
// variables -- packed losslessly into one uint64 as a fixed bit-field per process and per int
// variable. This replaces the per-state std::string key, whose alloc/format/hash dominated the
// profile: keys are now compared and hashed as plain integers, which is what UPPAAL does with its
// "compressed integer" discrete states.
//
// Built once per search and then read-only, so it is safe to share across the parallel level
// expansion. `encode` runs once per generated successor and `decode` once per frontier state and
// per level bucket, so both stay in the header of this translation unit and inline into their
// callers exactly as the lambdas they replaced did.
struct DiscKeyCodec {
  const Network& net;
  std::vector<int> shift;           ///< bit offset of each field: procs first, then int vars
  std::vector<std::uint64_t> mask;  ///< per-field value mask, for decoding a key back apart

  static int field_bits(long nvalues) { int b = 0; while ((1LL << b) < nvalues) ++b; return b; }

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
    // >= 64 (not > 64): keeps every field offset < 64, so a trailing zero-width field
    // (single-value int/location) can never trigger a `value << 64` UB shift.
    if (off >= 64) throw_disc_key_too_wide(n, off, &field_bits);
  }

  std::uint64_t encode(const LocVec& locs, const IntVec& ints) const {
    std::uint64_t k = 0;
    for (int p = 0; p < static_cast<int>(locs.size()); ++p)
      k |= static_cast<std::uint64_t>(static_cast<unsigned>(locs[p])) << shift[p];
    for (std::size_t v = 0; v < ints.size(); ++v)
      k |= static_cast<std::uint64_t>(ints[v] - net.intvars[v].lo) << shift[net.nprocs() + v];
    return k;
  }

  void decode(std::uint64_t dk, int* locs, long* ints) const {
    for (int p = 0; p < net.nprocs(); ++p)
      locs[p] = static_cast<int>((dk >> shift[p]) & mask[p]);
    for (std::size_t v = 0; v < net.intvars.size(); ++v)
      ints[v] = static_cast<long>((dk >> shift[net.nprocs() + v]) & mask[net.nprocs() + v]) +
                net.intvars[v].lo;
  }
};

// STRIPE AND BUCKET LAYOUT for the shared passed table.
//
// STRIPES are the lock granularity: a discrete state always lands in the same stripe, so a worker
// updating it locks only that stripe. There are far more stripes than threads so contention is
// rare.
//
// BUCKETS are the work granularity of the per-level survivor collection. They are taken from the
// SAME high bits of the SAME mix, so bucket b owns a CONTIGUOUS range of stripes and its scan
// stays inside one slice of the stripe array instead of scattering across all of them. Four
// buckets per thread keeps a dynamic schedule balanced.
//
// Both are read-only once built, and both index functions run per successor, so they stay in this
// translation unit and inline into their callers exactly as the lambdas they replaced did.
struct StripeLayout {
  static constexpr std::uint64_t kMix = 0x9e3779b97f4a7c15ULL;  ///< Fibonacci hashing multiplier

  std::size_t nstripes = 0;
  int stripe_shift = 0;
  int nbuckets = 1;
  int bucket_shift = 0;

  explicit StripeLayout(int nthreads) {
    int stripe_bits = 10;                                    // >= 1024 stripes, and >> nthreads
    while ((1 << stripe_bits) < std::max(16384, nthreads * 64)) ++stripe_bits;
    nstripes = static_cast<std::size_t>(1) << stripe_bits;
    stripe_shift = 64 - stripe_bits;

    int bucket_bits = 0;
    while ((1 << bucket_bits) < nthreads * 4) ++bucket_bits;
    nbuckets = bucket_bits ? (1 << bucket_bits) : 1;
    bucket_shift = 64 - bucket_bits;
  }

  std::size_t stripe_of(std::uint64_t dk) const {
    return static_cast<std::size_t>((dk * kMix) >> stripe_shift);
  }
  // Only valid when nbuckets > 1; a zero bucket_bits would make this a shift by 64.
  std::size_t bucket_of(std::uint64_t dk) const {
    return static_cast<std::size_t>((dk * kMix) >> bucket_shift);
  }
};

#if defined(PTCTLMC_HAVE_METAL) || defined(PTCTLMC_HAVE_CUDA)
// ---- flat GPU driver helpers ----------------------------------------------------------
// Per-level discrete-key -> bucket-id index. This was a std::unordered_map rebuilt every
// level, which costs ONE MALLOC PER DISTINCT KEY (227k of them per level on fischer11, over
// 26 levels) plus a pointer chase on every probe. An open-addressing table with a generation
// stamp reuses one allocation for the whole run and clears in O(1): the level's bucketing
// pass measured 267 ms of a 1424 ms fischer11 wall before this.
struct DkIndex {
  std::vector<std::uint64_t> key;
  std::vector<int> val;
  std::vector<std::uint32_t> stamp;
  std::uint32_t gen = 0;
  std::size_t mask = 0, fill = 0, limit = 0;
  static std::uint64_t mix(std::uint64_t k) {   // splitmix64 finalizer
    k ^= k >> 30; k *= 0xbf58476d1ce4e5b9ULL;
    k ^= k >> 27; k *= 0x94d049bb133111ebULL;
    return k ^ (k >> 31);
  }
  void resize_to(std::size_t cap) {
    key.assign(cap, 0); val.assign(cap, 0); stamp.assign(cap, 0);
    mask = cap - 1; gen = 1; fill = 0; limit = cap - cap / 4;   // 75% load factor
  }
  // Size for the PREVIOUS level's distinct-key count (BFS level widths change smoothly), and
  // grow by rehashing if this level turns out wider.
  void begin_level(std::size_t expect) {
    std::size_t cap = 1024;
    while (cap - cap / 4 < expect + 1) cap <<= 1;
    if (cap > key.size()) { resize_to(cap); return; }
    mask = key.size() - 1; limit = key.size() - key.size() / 4; fill = 0;
    if (++gen == 0) { std::fill(stamp.begin(), stamp.end(), 0); gen = 1; }
  }
  void grow() {
    const std::vector<std::uint64_t> ok = key;
    const std::vector<int> ov = val;
    const std::vector<std::uint32_t> os = stamp;
    const std::uint32_t og = gen;
    resize_to(key.size() * 2);
    for (std::size_t i = 0; i < ok.size(); ++i)
      if (os[i] == og) { (void)find_or_add(ok[i], ov[i]); }
  }
  // Return the id of `k`, inserting it with id `next` if absent (caller checks == next).
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

// Consume the first `nfr` states of the current level, keeping the rest. erase() from the
// front would memmove the remainder once per batch -- O(batches * level) over a wide level.
static void drop_batch(std::vector<int>& l, std::vector<long>& iv, std::vector<int>& z,
                       int nfr, int np_, int niw_, int) {
  l.erase(l.begin(), l.begin() + static_cast<std::size_t>(nfr) * np_);
  iv.erase(iv.begin(), iv.begin() + static_cast<std::size_t>(nfr) * niw_);
  z.erase(z.begin(), z.begin() + static_cast<std::size_t>(nfr));
}
#endif

// Reclaim interned zones no live passed-set node references any more (subsumed / merged out):
// mark the survivors -- referenced by a live arena node or by the next frontier -- then sweep the
// rest back to the interner's free-list for in-place reuse. Returns the next allocation level at
// which to sweep again.
//
// MARK IN PARALLEL over stripes. Every writer stores the same value (1) and nothing reads `alive`
// until the sweep, so two threads marking the same zidx cannot disagree -- but concurrent plain
// stores to one byte are still formally a race, so the store is a relaxed atomic. On x86/ARM that
// is the same instruction as a plain byte store (no lock prefix, no fence): the parallelism is
// free. The mark walks every live arena node, which was the serial half of the sweep (0.128s of a
// 3.83s arbiter16 run at 8 threads, and it grows with the state space).
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

// The initial symbolic state: every process in its initial location, every integer at its
// declared init value, and the zero zone opened up by the same closing pipeline every successor
// goes through -- combined invariant, delay unless frozen, active-clock reduction, extrapolation.
// Returns false when the initial invariant is unsatisfiable, i.e. the model has no initial state.
//
// Extrapolation runs only when finiteness depends on it (emode != ExNone). In the default aLU
// mode the passed list stays finite and minimal on exact zones, so extrapolating would only lose
// precision; the successor generators apply the same emode to every successor.
static bool initial_symbolic_state(const Network& net, const std::vector<char>& qmask,
                                   const std::vector<std::int32_t>& L,
                                   const std::vector<std::int32_t>& U, int emode, SymState* s0) {
  s0->locs.resize(net.nprocs());
  for (int p = 0; p < net.nprocs(); ++p) s0->locs[p] = net.procs[p].initial;
  s0->ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0->ints[v] = net.intvars[v].init;
  s0->zone = Dbm(net.nclocks);
  std::vector<Constraint> inv;
  return close_succ_zone(net, s0->locs, s0->zone, inv, qmask, L, U, emode);
}

// ---- the shared symbolic-reachability engine ------------------------------------------------
// ONE engine behind every "is a state satisfying X reachable?" query. It owns the sharded passed
// list, the zone interner, the parallel level expansion and the interner GC; callers supply only
// what actually differs between queries:
//
//   is_target     what makes a symbolic state a witness. A predicate, not a NetTarget, so a
//                 target can inspect the ZONE (deadlock) and not just locations/int vars.
//   force_exact   compare stored zones by plain inclusion instead of the aLU test. MUST be set
//                 when the property is per-VALUATION: aLU is a location-reachability abstraction
//                 and may drop concrete valuations, so `E<> deadlock` would go unsound with it.
//                 It also changes how finiteness is obtained -- see `emode` below.
//   target        still needed for the query's clock constants (aLU bound flooring, qmask) and
//                 by the GPU driver's flat target test. Pass a default-constructed one when the
//                 target is a predicate; then it contributes no constraints. A predicate target
//                 must therefore keep the GPU path off (an empty NetTarget matches everything).
//
// `deadlock_reachable` was a second, independent copy of this search: serial, storing dense Dbm
// copies in an unordered_map keyed by a std::string. Folding it in here is what gives deadlock
// queries the parallel expansion and the packed interned zones.
template <class TargetFn>
// `cancel`, when non-null, is polled once per BFS level: another worker has already settled the
// question and this search is now dead weight. Checked at the level boundary rather than per state
// so it costs one relaxed load per level, and abandoning between levels leaves nothing half-built
// -- the caller discards the whole engine anyway. Returns false when cancelled; the caller must
// therefore only consult the result of a search it did not cancel.
static bool reach_engine(const Network& net, const NetTarget& target, TargetFn&& is_target,
                         bool force_exact, Backend backend, ReachStats* stats,
                         const std::vector<Automorphism>* sym = nullptr,
                         const std::vector<NetTarget>* orbit = nullptr,
                         const std::atomic<bool>* cancel = nullptr) {
  const bool use_sym = sym && sym->size() > 1;
  std::vector<std::int32_t> L = net.lbound, U = net.ubound;
  extend_bounds(L, U, target.clock_cond);
  L[0] = U[0] = 0;  // reference clock: mandatory for the aLU subsumption test
  // With diagonal constraints the aLU abstraction is unsound: fall back to exact zones
  // (diagonals kept exact) + plain inclusion subsumption.
  const bool has_diag = network_has_diagonal(net);
  const bool exact_sub = has_diag || force_exact;
  // Finiteness comes from EITHER aLU subsumption on exact zones (the default) OR, when zones must
  // be compared exactly, from LU-extrapolation.
  //
  // The two are NOT interchangeable, and the comment that used to sit here ("extrapolating too
  // would only coarsen what aLU already covers") was wrong in a way worth stating: aLU only ever
  // PRUNES -- it drops a zone when a stored one simulates it -- whereas Extra_LU+ WIDENS, mapping
  // many exact zones onto one representative. On a model whose reachable zones are mutually
  // aLU-incomparable, pruning does nothing and widening collapses them.
  int emode = has_diag ? ExNoDiag : (force_exact ? ExLuPlus : ExNone);
  // ADAPTIVE: which of the two wins is a property of the model, not a constant, and the engine
  // already measures the distinguishing signal. Zones per discrete state near 1 means aLU is
  // collapsing them and widening would only cost time (phil 1.11, viking 1.00: extrapolation
  // there changes no state count and costs 20-35%). A ratio that climbs means the antichains are
  // filling with mutually incomparable zones, which aLU cannot prune and widening can (single:
  // one location, thousands of zones -- widening takes it from 3873 stored to 3676, which is
  // UPPAAL's 3673). So start exact and switch on once the ratio crosses the threshold.
  //
  // Switching mid-run is sound in both directions of the mixed passed list it leaves behind: a
  // stored EXACT zone that fails to subsume a new widened one only costs an extra stored state,
  // and a stored WIDENED zone that subsumes a new exact one genuinely contains it. Finiteness is
  // guaranteed from the switch onwards.
  constexpr double kAdaptRatio = 4.0;
  std::vector<char> qmask(net.nclocks + 1, 0);  // clocks the query reads -> keep active
  for (const auto& c : target.clock_cond) { if (c.i > 0) qmask[c.i] = 1; if (c.j > 0) qmask[c.j] = 1; }

  // Query clock constants floor every state's aLU bounds (the property may hold anywhere).
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
  // Per-state aLU bounds: max of per-location bounds over processes, floored by the query.
  auto state_bounds = [&](const int* locs, std::vector<std::int32_t>& L2,
                          std::vector<std::int32_t>& U2, std::vector<raw_t>& Lr,
                          std::vector<raw_t>& Ur) {
    // The G arrays are the same per-location fixpoint with strictness kept. Combined over
    // processes by raw max, which is the "weakest wins" rule Cor. 3.6 requires -- the same
    // compositional union upstream TChecker uses for LU with pointwise max.
    if (detail::options().gsim && !net.uraw_loc.empty()) {
      Lr.assign(net.nclocks + 1, kGNone);
      Ur.assign(net.nclocks + 1, kGNone);
      for (int p = 0; p < net.nprocs(); ++p)
        for (int c = 1; c <= net.nclocks; ++c) {
          const raw_t lr = net.lraw_loc[p][locs[p]][c], ur = net.uraw_loc[p][locs[p]][c];
          if (lr != kGNone && (Lr[c] == kGNone || lr > Lr[c])) Lr[c] = lr;
          if (ur != kGNone && (Ur[c] == kGNone || ur > Ur[c])) Ur[c] = ur;
        }
      // The query's own constants must floor the bounds exactly as they do for L/U, or a state
      // satisfying the property could be abstracted away. Non-strict: the safe direction.
      for (int c = 1; c <= net.nclocks; ++c) {
        if (Uq[c] != kAluNegInf && (Ur[c] == kGNone || le(Uq[c]) > Ur[c])) Ur[c] = le(Uq[c]);
        if (Lq[c] != kAluNegInf && (Lr[c] == kGNone || le(-Lq[c]) > Lr[c])) Lr[c] = le(-Lq[c]);
      }
      // (<=,0), NOT kGNone. dbm_is_g_le's loops start at 0 and `continue` on kGNone, so a
      // kGNone here deletes the (U-cut) witness family at y=0 and the (L-unsat') family at x=0.
      // Removing witnesses can only turn the test from FALSE to TRUE, i.e. prune zones that are
      // not actually simulated -- measured at ~19% unsound TRUEs against point-wise ground truth,
      // versus zero for (<=,0). The sibling dbm_is_alu_le documents L[0]=U[0]=0 as MANDATORY for
      // exactly this reason, and the code below does that.
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

  // `parallel` is decided AFTER the GPU factory below, because a GPU backend that falls back to
  // the CPU path (out-of-scope model, no device) must still use every core. It did not: the test
  // was `backend == Omp`, so `--backend metal` on a model the GPU declines ran SINGLE-THREADED --
  // arbiter24 took 93 s that way against `omp`'s 20 s, which reads as "the GPU lost" and is in
  // fact "the fallback used one core".
  bool parallel = false;
  int nthreads = 1;

  // GPU network backend (Metal or CUDA): the flat GPU-resident driver below runs the whole
  // level loop on the GPU (expand + per-bucket subsumption/convex-union merge). make_net_gpu_*
  // returns null for out-of-scope networks (diagonal constraints) or no device, so we fall back
  // to the CPU path transparently.
  std::unique_ptr<NetGpuExpander> gpu;
  bool use_gpu = false;
  // force_exact callers pass a default NetTarget, which the GPU's flat target test would treat
  // as "matches everything"; and the GPU merge selects its subsumption by has_diag alone, so it
  // cannot honour force_exact. Keep such queries on the CPU engine. (Tested per backend rather
  // than around the block: the two #ifdefs are independent, so a wrapping brace would not pair.)
  // Symmetry reduction rewrites each successor to its orbit representative on the host; the GPU
  // drivers have no such step, so the two are mutually exclusive for now.
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

  // Passed list SHARDED by discrete-state hash: a given discrete state always lands
  // in the same shard, so each shard is a self-contained antichain owned by exactly
  // one thread during the merge — the O(bucket^2 * dim^3) convex-merge now runs in
  // parallel with no locks. Per discrete state: antichain + convex-union merge.
  const DiscKeyCodec dkey(net);   // packs (locations, int vars) into one uint64; see above

  SymState s0;
  if (!initial_symbolic_state(net, qmask, L, U, emode, &s0)) return false;
  if (is_target(s0)) return true;
  if (use_sym) canonicalize(*sym, net, s0);
  canon_ints(s0.locs, s0.ints);

#if defined(PTCTLMC_HAVE_METAL) || defined(PTCTLMC_HAVE_CUDA)
  // ---- FLAT GPU-resident driver (Metal / CUDA) -------------------------------------------
  // Everything stays as flat zone arrays across levels: no SymState round-trip, no per-level
  // rebuild of a map of Dbm copies (which the profiler showed dominated the metal wall time).
  // GPU does expand + the per-bucket subsumption/convex-union merge; the host only groups by
  // dk (cheap integer hashing) and marshals flat zone spans.
  if (use_gpu) {
    // SoA work pool (Metal gpusplit merge): host gather/rebuild transpose to/from the transposed
    // layout, and the CPU-merge fallback (AoS) is disabled since it can't read it. Declared here
    // rather than beside use_gpu: every reader is inside this GPU-only block, so at file scope it
    // was an unused variable (and a -Wunused-variable warning) in every CPU-only build.
    const int np = net.nprocs(), ni = static_cast<int>(net.intvars.size()), dim = net.nclocks + 1, d2 = dim * dim;
    // The passed set holds ZONE INDICES into a shared dense pool rather than a private dense copy
    // per (discrete state, zone). With `--gpu-intern` the pool DEDUPLICATES, which is a large win
    // wherever few distinct zones are reachable: viking15 stores 2.77M zones drawn from just 120
    // distinct ones, i.e. GBs of unified memory representing ~120 KB of zone data. Without the
    // flag the pool simply appends, reproducing the old layout's memory with the same indexed
    // access path, so the flag isolates deduplication from everything else.
    // Narrow the arena to 16-bit entries when the model's constants make overflow impossible
    // (see GpuZoneIntern). The bound is 2*M+1 for the largest clock constant M anywhere in the
    // model or the query; the margin below is deliberately generous.
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
    const bool narrow_zones = (maxc < 4000);   // (2M+1)<<1 must stay well inside int16
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
    std::size_t prev_nb = 1024;
    double fanout = 4.0;   // running MAX of successors per frontier state (batch-size estimate)

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
    // (This is the correction to the previous design, which appended the un-expanded tail to the
    // survivors and swapped: that mixes level k and level k+1 states into one frontier, loses
    // strict BFS, and cost 2.5x the explored states on fischer12 -- 4.76M -> 12.0M.)
    // A stored zone is a 31-bit index into the zone interner plus one flag bit: "this zone was
    // added during the level currently being expanded". The flag is what the next frontier is
    // read off at level end, and packing it into the index costs no extra memory.
    static constexpr int kNewBit = 1 << 30;
    std::vector<std::uint64_t> touched;   // buckets changed during the current level
    // Per-BATCH scratch, hoisted: every one of these is sized by the batch and was previously
    // allocated and freed inside the loop. With --chunking a level is many batches, so the churn
    // (hundreds of MB per allocation) left GBs sitting in the allocator -- it turned a 9.5x cut in
    // GPU pool bytes into a net LOSS in peak RSS. Reused buffers keep their capacity instead.
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
        // bucket the level touched. Doing it here rather than per batch is what keeps batching
        // exact -- a zone a later batch absorbed has had its flag cleared with it.
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
      // budget. `fanout` is the measured successors-per-frontier-state of the previous batch;
      // the pools cost about 2.2 words per successor per matrix entry (expand output zones +
      // merge work pool + the discrete arrays).
      int nfr = total;
      if (detail::options().chunking) {
        if (detail::options().chunk_states > 0) {
          nfr = std::min(total, detail::options().chunk_states);
        } else {
          // 3 words per successor per matrix entry: the expand output zones, the merge work
          // pool (exist + new), and the frontier/staging arrays. The fan-out is the running MAX,
          // not an average: under-predicting it is what lets a batch overshoot the budget, and a
          // batch that overshoots is exactly the peak the budget exists to bound.
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
        // Materialise just THIS BATCH's zones, widened out of the arena. The frontier itself is
        // arena indices (4 B per state) rather than dense matrices (dim^2 words): the zones are
        // in the passed set already, so a level-wide dense copy was storing every one of them
        // twice -- 0.24 GB on fischer12 and 0.81 GB on fischer16 -- and it also meant reading
        // them out of the arena at level end only to copy them again here.
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
      // The GPU now holds its own copy of the frontier (persistent device buffer) and the host
      // copies are dead until the end-of-level swap. Release them NOW: otherwise they sit alongside
      // the next frontier being built during the rebuild, i.e. a full extra copy of the frontier at
      // exactly the moment memory peaks.
      // Only safe when the whole level was expanded; when batching, the tail is still needed.
      if (whole) { std::vector<int>().swap(fl); std::vector<long>().swap(fi); std::vector<int>().swap(fz); }
      else drop_batch(fl, fi, fz, nfr, np, niw, d2);   // batch consumed; keep the rest of the level
      // group VALID successors by dk (reading straight from GPU buffers); target check on the fly
      // Bucket the valid successors with a FLAT (CSR) layout: a per-work-item bucket id plus a
      // per-bucket count, then a prefix sum. The obvious `vector<vector<int>>` heap-allocates one
      // small vector PER BUCKET PER LEVEL (millions of alloc/free pairs over a run, and the
      // fragmentation dominated resident memory); this is two flat arrays instead.
      // Bucketing stays SERIAL by measurement, not by omission. A lock-free open-addressing
      // table (one CAS per key, stable counting-sort scatter) was implemented and measured
      // three ways -- oversized table 328ms, nb-sized 467ms, monotonic 336ms -- against this
      // loop's ~200ms. A private-map parallel version (per-thread maps + distinct-key merge +
      // parallel counting-sort scatter) was also measured: 268ms -> 241ms on viking15, i.e. ~1%
      // of wall time, because the loop is bound by streaming the expand output out of unified
      // memory while the GPU is using the same bandwidth -- not by the hashing. Neither variant
      // is worth the complexity here. See docs/GUIDE.md.
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
      // counts + prefix sums. The NEW zones are never copied: the merge reads them in place from
      // the expand output by work-item index, so we only flatten the index list (integers, cheap).
      std::size_t ntot = 0, wtot = 0;  // total new zones; total work-pool zones (exist+new)
      for (int b = 0; b < nb; ++b) {
        new_cnt[b] = bcnt[b];
        auto pit = passed.find(bdk[b]);
        exist_cnt[b] = (pit == passed.end()) ? 0 : static_cast<int>(pit->second.size());
        new_off[b] = static_cast<int>(ntot); ntot += new_cnt[b];
        work_off[b] = static_cast<int>(wtot); wtot += exist_cnt[b] + new_cnt[b];
      }
      // Scatter the work-item indices into the flat CSR array. Iterating w ascending preserves the
      // per-bucket ordering the merge relies on (the convex union is order-sensitive), so this is
      // bit-for-bit equivalent to the old vector-of-vectors.
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
      // PARALLEL: gather only the EXISTING antichains into the work buffer (unified memory);
      // successor zones stay in the expand output and are read in place by the merge kernel.
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 32)
#endif
      for (int b = 0; b < nb; ++b) {
        auto pit = passed.find(bdk[b]);
        if (pit != passed.end()) {
          gz_t* dst = work_buf + static_cast<std::size_t>(work_off[b]) * d2;
          for (int i = 0; i < exist_cnt[b]; ++i)
            zi.load_gz(pit->second[i] & ~kNewBit, dst + static_cast<std::size_t>(i) * d2);
          // Seed each already-stored zone's TAG with its passed-set index (kNewBit included). The
          // merge carries a kept zone's tag through its compaction and writes -1 for the zones it
          // creates, so the rebuild can tell "this output slot IS stored zone j" from "this output
          // slot is new" -- and a kept zone then needs no copy and no re-intern at all. The flag
          // also has to survive because a zone an EARLIER batch of this level added is still a
          // frontier state; without carrying it, batching would silently lose frontier states.
          int* wi = work_isnew + work_off[b];
          for (int i = 0; i < exist_cnt[b]; ++i) wi[i] = pit->second[i];
        }
        int dl[64]; long di[32];
        std::vector<std::int32_t> L2, U2;
        std::vector<raw_t> Lr_, Ur_;   // GPU merge is a_LU only; the G arrays go unused here
        dkey.decode(bdk[b], dl, di);
        state_bounds(dl, L2, U2, Lr_, Ur_);
        for (int c = 0; c < dim; ++c) { Lm[static_cast<std::size_t>(b) * dim + c] = L2[c]; Um[static_cast<std::size_t>(b) * dim + c] = U2[c]; }
      }
      out_work_off.clear(); out_count.clear();
      const gz_t* out_zones = nullptr;
      const int* out_isnew = nullptr;
      // Merge routing. NAIVE (default): always the GPU one-thread-per-bucket merge (original
      // behavior). ADAPTIVE (--adaptive): CALIBRATE each path's throughput (work ~ sum bucket^2, per
      // ms) on early BFS levels, then route each later level to the predicted-faster of {GPU, 12 CPU
      // cores}. This rescues merge-heavy models (e.g. philosophers, where the GPU merge is 93% of the
      // wall time — one thread per bucket is the wrong shape) with no effect where the GPU already wins.
      long long sum2 = 0;
      for (int b = 0; b < nb; ++b) { const long long s = exist_cnt[b] + new_cnt[b]; sum2 += s*s; }
      bool cpu_merge = false;
      if (detail::options().adaptive_merge) {
        if (cpu_rate < 0) cpu_merge = true;         // calibrate the CPU path first (never catastrophic)
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
      // rebuild passed + assemble the flat next frontier (survivors). Prefix-sum the per-bucket
      // survivor counts, pre-create the passed nodes, then fill everything in parallel.
      // Survivor counts, then offsets. Counting scans every merged zone -- O(total zones), the
      // same order as the rebuild copies below -- so it runs in parallel; only the offset
      // prefix-sum stays serial, and that is O(nb) integer adds over data already in cache.
      // (Fusing the two, as this loop used to, forced the whole O(total zones) scan to be
      // serial purely because `frtot` was carried through it.)
      // Survivor counts and offsets. Kept fused and serial deliberately: splitting the count
      // into an omp parallel-for and leaving only the offset prefix-sum serial was measured on
      // arbiter17/Metal and is not faster (rebuild 563ms vs 545ms) -- buckets average ~10 zones,
      // so the scan is memory-bound and the per-bucket parallel overhead plus false sharing on
      // fr_cnt costs more than the scan itself.
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
      // A bucket with NO survivor had all its new zones subsumed — its antichain is unchanged, so
      // `passed[dk]` is already correct. Skip its rebuild copy entirely; only changed buckets are
      // cleared + refilled (their element ptrs stay stable across later inserts).
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
        // This bucket's slots before the merge; the ones the merge drops go back to the arena
        // free list below. Thread-local, not a fixed array: philosophers' antichains reach ~400
        // zones, and a fixed 64 would silently leak every slot past it.
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
  // SHARED-PASSED (hybrid): one stripe-locked passed/waiting table that any
  // worker may touch, locking only the stripe of the discrete state it is
  // updating -- LTSmin's "shared table + fine-grained locking, no static
  // partition" (Alg. 3/4). The per-bucket update is unchanged: PTCTLMC's exact
  // aLU subsumption + convex-union merge (add_to). Expansion and insertion are
  // FUSED, so there is no serial per-level routing barrier; strict-BFS level
  // synchronisation is kept (it minimises the state count). At 1 thread this is
  // bit-identical to the legacy path (same per-bucket merge, same order).
    const StripeLayout lay(nthreads);                  // stripe + bucket index; see above
    const int dim2 = (net.nclocks + 1) * (net.nclocks + 1);
    ZoneIntern zi(dim2, nthreads);                     // ONE shared distinct-zone table
    std::vector<NetPassed> smap(lay.nstripes);         // flat per-stripe passed tables
    for (auto& mp : smap) mp.zi = &zi;
    // The mirror budget is process-wide (one NetPassed per stripe) but these tables are not:
    // a second query in the same process would otherwise inherit the first one's charge and
    // never promote anything.
    detail::hot_bytes().store(0, std::memory_order_relaxed);
    // shared_mutex: the read-only kill probe below runs concurrently per stripe; only a real
    // insert/merge needs exclusivity.
    std::vector<SpinRW> smtx(lay.nstripes);            // one lock per stripe (value-init in place)
    // PER-THREAD LOCAL ANTICHAINS were tried here and REMOVED (see git history). Each thread
    // inserted into its own table during a level and a bucket owner drained them into the shared
    // one at the boundary, so the expand phase took no lock at all. It is correct and it is
    // consistently SLOWER -- 8 threads: phil11 1.48x, viking12 1.29x, twolock10 1.90x,
    // fischer10 1.92x. Removing the lock saves the ~15% of busy thread-time that was spin, but a
    // private antichain cannot see the read-only kill probe that answers 83% of successors, so
    // the drain pays more back than it saves (phil12: 3.2M gathered pairs, 1.5M of them exact
    // duplicates, 19.4s of drain thread-time against a 3.8s wall). Don't retry it without a
    // design that keeps a shared kill probe.
    // Dynamic-schedule chunk for the frontier expansion. Loop-invariant, so OpenMP accepts a
    // runtime value here; --omp-chunk previously reached only reach.cpp's single-TA loop, which
    // made it a no-op for every network (multi-process) model.
    const int expand_chunk = std::max(1, omp_chunk());

    // The frontier holds only (dk, zone-index) per state -- 12 B, not a full
    // SymState. locs/ints come from dk (DiscKeyCodec::decode) and the zone from the intern
    // table, reconstructed into a REUSED per-thread scratch at expansion. This
    // removes the ~1.6 GB of interned-zone copies the old make_survivor did (one
    // full DBM copied into every frontier state) and shrinks the resident
    // frontier ~50× so it stays cache-resident during expansion.
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
    const bool single = (nthreads == 1);   // seq / omp -p1: skip locks + the touched union
    // Fused batched merge: O(1) critical section + one lock-free settle pass per level.
    // Zone-interner GC trigger: sweep once the allocation level reaches `sweep_at`, then set
    // the next target to double the surviving live set (see the sweep site below).
    constexpr std::size_t kSweepFloor = 50000;   // don't bother GCing a small interner
    std::size_t sweep_at = kSweepFloor;
    while (!frontier.empty()) {
      if (cancel && cancel->load(std::memory_order_relaxed)) return false;
      if (stats) stats->explored += frontier.size();
      if (has_diag && (nexpanded += frontier.size()) > 500000)
        throw std::runtime_error("state cap exceeded — model likely has unbounded diagonal clocks");
      ++level;

      // Expand each frontier state and insert its successors DIRECTLY into the
      // shared table, locking only the touched stripe (expansion and insertion
      // → no serial routing barrier). Inserting during expansion is what keeps the
      // convex-union merge order stable enough to match the legacy state counts.
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
        // Count generated successors in a REGISTER, not in tls_gen[tid]: the counters are
        // 8 bytes apart, so all `nthreads` of them share one cache line and the per-successor
        // increment bounced that line between every core on every successor (23M times on
        // phil13). Published once per thread at the end of the region.
        std::size_t gen = 0;
#pragma omp for schedule(dynamic, expand_chunk) nowait
        for (long i = 0; i < static_cast<long>(frontier.size()); ++i) {
          if (found.load(std::memory_order_relaxed)) continue;
          dkey.decode(frontier[i].first, cur.locs.data(), cur.ints.data());
          // buf is NOT cleared: gen_succ_reuse rebuilds slots in place and returns
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

      // Collect this level's survivors (entries stamped `level`) into the next frontier.
      // Each dk must be scanned ONCE, so the per-thread touched sets have to be merged; the
      // tables are read-only here (no writers), so for_level is a safe read.
      //
      // The merge used to be one serial std::unordered_set fed from every thread's set. That
      // was the largest serial phase left in the level loop (0.149s of a 3.83s arbiter16 run
      // at 8 threads) and it grows with level width, so it caps the speedup by Amdahl.
      //
      // Instead dk is BUCKETED by hash and each bucket is deduplicated independently. A given
      // dk hashes to exactly one bucket, so that bucket's set observes every copy of it and
      // the deduplicated key set is identical to the serial union's -- only the order differs,
      // and the order was never fixed anyway (the old for_level pass was a dynamic-schedule
      // loop appending into per-thread buffers, so which thread saw which dk already varied
      // run to run). Dedup and the for_level scan then happen in the SAME parallel loop, which
      // also removes the separate scan pass.
      //
      // The stripe/bucket index `lay` is built once at engine setup; see StripeLayout.
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
      // Survivors go into the next frontier as (dk, zone-index) pairs -- no SymState build,
      // no zone copy here (that happens once, lazily, at the next expansion into `cur`).
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
      for (auto& v : nf)
        next.insert(next.end(), v.begin(), v.end());
      // EXPANSION ORDER was tried here and REMOVED. The idea: a stored zone that a later, more
      // general zone subsumes was expanded for nothing (phil12: 469805 of 1047142 arena nodes,
      // 45%, end up retired), so expanding the most general zones of a level FIRST should let
      // them kill their narrower siblings on insertion instead of a level later. The generality
      // proxy was free -- a minimal constraint system stores a more general zone in a SHORTER
      // packed record -- and ordering a worklist cannot affect the fixpoint, so it was sound.
      // It simply does not pay: 8 threads, most-general-first vs insertion order gave phil12
      // 0.95x (and MORE states, 841637 vs 803845), fischer11 0.98x, twolock11 1.00x, viking15
      // 0.98x. The retirement is driven by zones that arrive levels apart, which no intra-level
      // ordering can reach.
      frontier.swap(next);
      // Adaptive abstraction: re-read the zones-per-discrete-state signal now that the level is
      // complete and every thread has joined, and switch widening on if the antichains are
      // filling up. Once on it stays on -- the ratio only ever justifies more abstraction.
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

      // TRIGGER for the interner sweep below: growth-based, not per-level. The old guard was `zi.n > 50000`, but `n` is a
      // high-water counter that never shrinks, so it latched true at the first big level and
      // ran a full mark+sweep on EVERY level thereafter. Both halves are serial and O(all
      // arena nodes + htab slots), so the cost grew with the state space while the expand
      // phase stayed parallel -- on arbiter17 that was 2.8s of a 9.3s run (30%), and it is
      // what capped the 12-thread speedup at ~5x.
      //
      // Sweeping when the allocation level has DOUBLED since the last sweep makes the work
      // amortized O(1) per interned zone (a sweep costing O(A) is paid for by the A/2 fresh
      // interns that triggered it), so a run does O(log) sweeps instead of one per level. The
      // memory it trades away is bounded: garbage never exceeds the live set, i.e. the
      // interner stays within 2x of its pinned-to-live size.
      if (zi.allocated() >= sweep_at)
        sweep_at = sweep_dead_zones(zi, smap, frontier, nthreads, kSweepFloor);

    }
    return false;
}

// E<> target, where the target is a location/integer/clock-constraint conjunction. aLU
// subsumption applies: the property is evaluated per symbolic state, so the abstraction is sound.
bool reachable_net(const Network& net, const NetTarget& target, Backend backend,
                   ReachStats* stats) {
  // Symmetry reduction explores one representative per orbit, so a target that is NOT itself
  // symmetric has to be tested over the whole ORBIT of the property: `E<> P0.Eat && P1.Eat` on a
  // ring of philosophers becomes "some adjacent pair eats", which is the same question because
  // the network is rotation-symmetric. That is done by mapping the target through every group
  // element ONCE, up front, and testing the state against each image -- no state is permuted.
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
      // Every image's clock constants must reach the abstraction, or the LU bounds would be
      // asymmetric and two states of one orbit could be abstracted differently -- which is
      // exactly what the reduction may not do.
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

bool exists_invariant_run(const Network& net, const DiscPred& phi, Backend backend,
                          ReachStats* stats) {
  (void)backend;  // liveness graph search is sequential (cycle detection); omp TODO
  std::vector<std::int32_t> L = net.lbound, U = net.ubound;
  const int emode = network_has_diagonal(net) ? ExNoDiag : ExLuPlus;  // diagonals: keep exact
  const std::vector<char> qmask(net.nclocks + 1, 0);  // liveness predicate has no clocks

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
    { bool fc, fu; frozen_flags(net, s.locs, &fc, &fu); if (fc || fu) return false; }
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

// Does the (delay-closed, extrapolated) symbolic state `s` contain a concrete valuation
// with no outgoing action transition -- a deadlock?  A whole zone that has no successor is
// the easy case; the hard case is a PARTIAL deadlock, where the zone has successors but a
// sub-region of it can never reach any of them (e.g. an invariant caps time before the only
// guard opens). We compute it exactly: mirror successors_net's three transition kinds, but
// for each transition that CAN fire collect its SOURCE zone within `s.zone` -- the pre-reset
// valuations from which it fires and lands in a legal target (guards ∧ reset^{-1}(target
// invariant)). The union of the time-pasts (down()) of those source zones is the set of
// non-deadlocked valuations (for a frozen source -- committed/urgent -- no time may pass, so
// there is no down()). Anything in `s.zone` left over after subtracting that union is a set
// of deadlock states. Enabledness is invariant under LU-extrapolation, so working on the
// extrapolated zone is sound -- the same assumption the rest of the engine already relies on.
bool detail::state_has_deadlock(const Network& net, const SymState& s,
                                Federation* stuck, Federation* escapes) {
  if (s.zone.empty()) return false;
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, &src_committed, &src_urgent);
  const bool frozen = src_committed || src_urgent;  // no delay allowed from this state

  // Rebuild the true delay-closure of this state's zone. The stored zone has the
  // active-clock reduction applied, which pins DEAD clocks to 0 with an upper bound and so
  // is no longer delay-closed -- down() could not let time pass those clocks, inventing
  // spurious deadlocks. Re-opening every upper bound (delay) and re-imposing only the real
  // location invariant restores the closure; dead clocks become free, which is sound since
  // no guard or invariant here reads them. Frozen locations do not delay, so leave them be.
  Dbm Z = s.zone;
  if (!frozen) {
    Z.delay();
    std::vector<Constraint> cinv;
    all_invariants(net, s.locs, cinv);
    if (!apply_all(Z, cinv)) return false;  // real invariant unsatisfiable: no concrete state
  }

  std::vector<Constraint> tinv;  // target-invariant scratch
  Federation live;

  // `guarded` = Z ∩ (participant clock guards), pre-reset. Pull it back through the target
  // invariant under the transition's resets and, if a non-empty source region survives, add
  // it to `live`. reset^{-1}(inv): a constraint on a reset clock becomes one on clock 0 (its
  // post-reset value), so it either constrains the surviving source clocks or, if it reduces
  // to `0 <| bound`, gates the whole transition on feasibility of that constant.
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
    // INCLUSION-MINIMAL, not append. `live` is consumed only through its UNION -- by the
    // subtraction below and, for the atom construction, by `escapes` -- so a member contained
    // in a sibling denotes nothing new. It is not free, though: fed_subtract peels the
    // remainder once per member and the piece count multiplies, so a redundant escape costs a
    // whole extra peel of an already-exploded arrangement. Two escapes agreeing on their
    // clock guard and differing only in which process moved is the common case (grc4's trains),
    // and it is exactly this case. Pruning BEFORE down() is safe: W subseteq W' implies
    // down(W) subseteq down(W'), so the union of the downs is unchanged too.
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
  // The ESCAPE regions themselves, for callers that need to know WHICH constraints decide
  // stuckness rather than only the residue. Dead = Z \ union(live), so membership is decided by
  // the atoms of each live_e -- |out(q)| convex zones -- whereas the residue is an ARRANGEMENT
  // whose piece count, and hence atom count, is far larger for the same set.
  if (escapes) *escapes = live;
  // ONE-ESCAPE SHORT CIRCUIT. fed_subtract peels Z once per escape and the piece count
  // multiplies, so the arrangement it builds is the dominant cost of the structural sweep --
  // and it is built even when a SINGLE escape already covers Z, which is the overwhelmingly
  // common case (a location with one unguarded outgoing edge). One DBM inclusion test per
  // escape decides that outright. Exact, not a heuristic: Z subseteq W subseteq union(live)
  // means Dead(q) is empty, which is precisely what the subtraction would have reported.
  for (const Dbm& W : live)
    if (W.includes(Z)) {
      if (stuck) stuck->clear();
      return false;
    }
  Federation d = fed_subtract(Z, live);      // exactly the valuations with nothing enabled
  if (stuck) *stuck = d;
  return !d.empty();
}

// E<> deadlock. Same search as reachable_net -- reachability with subsumption -- so it runs on
// the SAME engine; only two things differ, and both are engine parameters:
//
//   * the target inspects the ZONE (a state may be only PARTIALLY deadlocked), so it is a
//     predicate rather than a NetTarget;
//   * subsumption must be exact inclusion. Deadlock is a per-valuation property: a valuation
//     dropped by the aLU abstraction could be the deadlocked one. Exact inclusion IS sound here,
//     because the deadlocked valuations of a subsumed zone lie inside the subsuming zone and are
//     found when that one is checked.
//
// This used to be a second search of its own: single-threaded, with the passed list as an
// unordered_map<std::string, vector<Dbm>> holding DENSE zone copies keyed by a freshly built
// string. On the engine it gets the parallel level expansion, the sharded stripe-locked passed
// list and packed min-graph zone records instead.
// a_M SUBSUMPTION, NOT EXACT ZONES.
//
// This used to pass force_exact=true, which disables the aLU subsumption test outright and
// falls back to plain inclusion on Extra_LU+-widened zones. That is far stronger than the
// property needs, and it is what made `E<> deadlock` cost 167805 states on fischer7 against
// 11951 for plain reachability on the same model.
//
// What deadlock actually needs is a BISIMULATION, not exact zones. aLU is only a simulation and
// so cannot preserve "enables nothing" (Behrmann-Bouyer-Larsen-Pelanek, STTT 8(3):204-215, 2006,
// Lemma 3; Lemma 1 gives the bisimulation for =_M, and their section 3 says in as many words
// that Uppaal's deadlock properties are preserved by bisimulation and not by simulation).
// But set L = U = M with M = max(L, U) per clock and the three clauses of the LU preorder --
// nu'(x)=nu(x), or L(x) < nu'(x) < nu(x), or U(x) < nu(x) < nu'(x) -- collapse to "equal, or
// both above M(x)", which is SYMMETRIC. A symmetric simulation is a bisimulation. So the very
// same dbm_is_alu_le kernel, fed collapsed bounds, IS the a_M test and IS deadlock-sound.
//
// This is also exactly what UPPAAL does: its extrapolation is Lower/Upper for reachability but
// *Local* (per-location max constants) "whenever the query contains either the deadlock keyword
// or involves a liveness property". force_exact was strictly stronger than UPPAAL.
//
// COLLAPSING THE NETWORK'S BOUND ARRAYS, rather than threading a flag through reach_engine, is
// deliberate: every consumer of L/U downstream becomes M-flavoured at once and consistently.
// That matters for more than the subsumption test -- the adaptive path can turn extrapolation
// back on mid-search, and Extra_LU+ with L=U=M is Extra_M+, which (unlike Extra_LU+) can neither
// invent nor destroy a deadlock. Threading a flag would have left that path unsound.
//
// The predicate itself still closes over the ORIGINAL net: it reads guards, invariants and
// edges, never the bound arrays.
bool deadlock_reachable(const Network& net, Backend backend, ReachStats* stats,
                        const std::atomic<bool>* cancel) {
  // The GPU drivers cannot express this target. deadlock_reachable never asks for a GPU backend,
  // so the CPU engine is selected regardless of force_exact.
  const Backend b = (backend == Backend::Seq) ? Backend::Seq : Backend::Omp;
  const auto is_dl = [&](const SymState& s) { return state_has_deadlock(net, s); };

  // Diagonals make every LU-family abstraction unsound, and the G-bound arrays carry raw encoded
  // bounds with their own sentinel, so collapsing them correctly is a separate exercise. Both
  // cases keep the exact behaviour.
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
