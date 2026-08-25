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





bool apply_all(Dbm& z, const std::vector<Constraint>& cs) {
  for (const auto& c : cs)
    if (!z.constrain(c.i, c.j, c.bound)) return false;
  return true;
}

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

namespace {

bool int_ok(const std::vector<IntAtom>& g, const IntVec& ints) {
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
void all_invariants(const Network& net, const LocVec& locs, std::vector<Constraint>& inv) {
  inv.clear();
  for (int p = 0; p < net.nprocs(); ++p)
    for (const auto& c : net.procs[p].locations[locs[p]].invariant) inv.push_back(c);
}

void frozen_flags(const Network& net, const LocVec& locs, bool* committed, bool* urgent) {
  *committed = *urgent = false;
  for (int p = 0; p < net.nprocs(); ++p) {
    const Location& l = net.procs[p].locations[locs[p]];
    if (l.committed) *committed = true;
    if (l.urgent) *urgent = true;
  }
}

// Reset clocks no process will read before resetting AND that the query does not
// read (qmask) — collapses zones that differ only in dead clocks, soundly.
void reset_inactive(const Network& net, const LocVec& locs, Dbm& z,
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

// Does the network compare two clocks anywhere (a diagonal constraint)? If so, the
// LU/aLU abstraction is unsound and we fall back to exact zones (diagonals kept exact).
// EXTERNAL linkage on purpose -- declared in netreach.hpp, so the query router in main.cpp
// can ask the same question before choosing an engine.
}  // namespace

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

std::string disc_key(const LocVec& locs, const IntVec& ints) {
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

static void build_broadcast_parts(const Network& net, const SymState& s, const NEdge& se, int sender,
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

      all_invariants(net, ns.locs, inv);
      if (!apply_all(z1, inv)) continue;
      bool nc, nu;
      frozen_flags(net, ns.locs, &nc, &nu);
      if (!nc && !nu) { z1.delay(); apply_all(z1, inv); }
      reset_inactive(net, ns.locs, z1, qmask);
      if (emode == ExLuPlus) z1.extrapolate_lu_plus(L, U);
      else if (emode == ExNoDiag) z1.extrapolate_lu_plus_nodiag(L, U);
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
          all_invariants(net, ns.locs, inv);
          if (!apply_all(z1, inv)) continue;
          bool nc, nu;
          frozen_flags(net, ns.locs, &nc, &nu);
          if (!nc && !nu) { z1.delay(); apply_all(z1, inv); }
          reset_inactive(net, ns.locs, z1, qmask);
          if (emode == ExLuPlus) z1.extrapolate_lu_plus(L, U);
      else if (emode == ExNoDiag) z1.extrapolate_lu_plus_nodiag(L, U);
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
            all_invariants(net, ns.locs, inv);
            if (apply_all(z1, inv)) {
              bool nc, nu;
              frozen_flags(net, ns.locs, &nc, &nu);
              if (!nc && !nu) { z1.delay(); apply_all(z1, inv); }
              reset_inactive(net, ns.locs, z1, qmask);
              if (emode == ExLuPlus) z1.extrapolate_lu_plus(L, U);
              else if (emode == ExNoDiag) z1.extrapolate_lu_plus_nodiag(L, U);
              if (!z1.empty()) { ns.zone = std::move(z1); out.push_back(std::move(ns)); }
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
// already noticed this (see PTCTLMC_SRCSTATS above); this caches the result.
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
        all_invariants(net, ns.locs, inv);
        if (!apply_all(ns.zone, inv)) { sm.m.emplace(mk, Dbm()); continue; }
        { bool nc2, nu2; frozen_flags(net, ns.locs, &nc2, &nu2);
          if (!nc2 && !nu2) { ns.zone.delay(); apply_all(ns.zone, inv); } }
        reset_inactive(net, ns.locs, ns.zone, qmask);
        if (emode == ExLuPlus) ns.zone.extrapolate_lu_plus(L, U);
        else if (emode == ExNoDiag) ns.zone.extrapolate_lu_plus_nodiag(L, U);
        sm.m.emplace(mk, ns.zone);                       // records empty zones too
        if (ns.zone.empty()) continue;
        ++on;
        continue;
      }
      ns.zone = szone;                                    // into reused capacity
      if (!apply_all(ns.zone, e.clock_guard)) continue;
      for (int r : e.clock_resets) ns.zone.reset(r);
      all_invariants(net, ns.locs, inv);
      if (!apply_all(ns.zone, inv)) continue;
      bool nc, nu;
      frozen_flags(net, ns.locs, &nc, &nu);
      if (!nc && !nu) { ns.zone.delay(); apply_all(ns.zone, inv); }
      reset_inactive(net, ns.locs, ns.zone, qmask);
      if (emode == ExLuPlus) ns.zone.extrapolate_lu_plus(L, U);
      else if (emode == ExNoDiag) ns.zone.extrapolate_lu_plus_nodiag(L, U);
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
            all_invariants(net, ns.locs, inv);
            if (!apply_all(ns.zone, inv)) { sm.m.emplace(mk, Dbm()); continue; }
            { bool nc2, nu2; frozen_flags(net, ns.locs, &nc2, &nu2);
              if (!nc2 && !nu2) { ns.zone.delay(); apply_all(ns.zone, inv); } }
            reset_inactive(net, ns.locs, ns.zone, qmask);
            if (emode == ExLuPlus) ns.zone.extrapolate_lu_plus(L, U);
            else if (emode == ExNoDiag) ns.zone.extrapolate_lu_plus_nodiag(L, U);
            sm.m.emplace(mk, ns.zone);
            if (ns.zone.empty()) continue;
            ++on; continue;
          }
          ns.zone = szone;
          if (!apply_all(ns.zone, se.clock_guard)) continue;
          if (!apply_all(ns.zone, re.clock_guard)) continue;
          for (int r : se.clock_resets) ns.zone.reset(r);
          for (int r : re.clock_resets) ns.zone.reset(r);
          all_invariants(net, ns.locs, inv);
          if (!apply_all(ns.zone, inv)) continue;
          bool nc, nu;
          frozen_flags(net, ns.locs, &nc, &nu);
          if (!nc && !nu) { ns.zone.delay(); apply_all(ns.zone, inv); }
          reset_inactive(net, ns.locs, ns.zone, qmask);
          if (emode == ExLuPlus) ns.zone.extrapolate_lu_plus(L, U);
          else if (emode == ExNoDiag) ns.zone.extrapolate_lu_plus_nodiag(L, U);
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
            all_invariants(net, ns.locs, inv);
            if (apply_all(ns.zone, inv)) {
              bool nc, nu;
              frozen_flags(net, ns.locs, &nc, &nu);
              if (!nc && !nu) { ns.zone.delay(); apply_all(ns.zone, inv); }
              reset_inactive(net, ns.locs, ns.zone, qmask);
              if (emode == ExLuPlus) ns.zone.extrapolate_lu_plus(L, U);
              else if (emode == ExNoDiag) ns.zone.extrapolate_lu_plus_nodiag(L, U);
              if (!ns.zone.empty()) ++on;
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
  // PTCTLMC_EXACTSUB forces plain-inclusion subsumption for A/B-testing the aLU test itself.
  const bool exact_sub = has_diag || force_exact;
  // Finiteness comes from EITHER aLU subsumption on exact zones (the default) OR, when zones must
  // be compared exactly, from LU-extrapolation.
  //
  // The two are NOT interchangeable, and the comment that used to sit here ("extrapolating too
  // would only coarsen what aLU already covers") was wrong in a way worth stating: aLU only ever
  // PRUNES -- it drops a zone when a stored one simulates it -- whereas Extra_LU+ WIDENS, mapping
  // many exact zones onto one representative. On a model whose reachable zones are mutually
  // aLU-incomparable, pruning does nothing and widening collapses them. PTCTLMC_EXTRAP=1 runs
  // both (widen, then subsume with aLU), which is sound: Extra_LU+(Z) is contained in the aLU
  // abstraction of Z, and the query's own constants already floor L and U above.
  static const bool force_extrap = std::getenv("PTCTLMC_EXTRAP") != nullptr;
  int emode = has_diag ? ExNoDiag : ((force_exact || force_extrap) ? ExLuPlus : ExNone);
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
  // guaranteed from the switch onwards. PTCTLMC_ADAPT=0 disables the switch.
  static const bool adapt = [] {
    const char* e = std::getenv("PTCTLMC_ADAPT");
    return !e || std::atol(e) != 0;
  }();
  static const double kAdaptRatio = [] {
    const char* e = std::getenv("PTCTLMC_ADAPTRATIO");
    return e ? std::atof(e) : 4.0;
  }();
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
  if (use_gpu && std::getenv("PTCTLMC_STATS"))
    std::fprintf(stderr, "[gpu] network expander ACTIVE\n");
#ifdef PTCTLMC_HAVE_OPENMP
  // The CPU path runs multi-threaded for --backend omp AND for a GPU backend that fell back.
  parallel = (backend == Backend::Omp) || (!use_gpu && backend != Backend::Seq);
  if (parallel) nthreads = omp_get_max_threads();
#endif

  // Passed list SHARDED by discrete-state hash: a given discrete state always lands
  // in the same shard, so each shard is a self-contained antichain owned by exactly
  // one thread during the merge — the O(bucket^2 * dim^3) convex-merge now runs in
  // parallel with no locks. Per discrete state: antichain + convex-union merge.
  // Discrete-state key packed losslessly into a uint64: a fixed bit-field per process
  // location + per int variable. This replaces the per-state std::string key, whose
  // alloc/format/hash dominated the profile — keys are now compared and hashed as plain
  // integers (what UPPAAL does with its "compressed integer" discrete states).
  const int nfields = net.nprocs() + static_cast<int>(net.intvars.size());
  std::vector<int> dshift(nfields);
  std::vector<std::uint64_t> dmask(nfields);  // per-field value mask (for decoding dk -> locs/ints)
  {
    int off = 0;
    auto bits = [](long n) { int b = 0; while ((1LL << b) < n) ++b; return b; };
    auto set = [&](int f, int w) { dshift[f] = off; dmask[f] = (w >= 64) ? ~0ULL : ((1ULL << w) - 1); off += w; };
    for (int p = 0; p < net.nprocs(); ++p) set(p, bits(static_cast<long>(net.procs[p].locations.size())));
    for (std::size_t v = 0; v < net.intvars.size(); ++v)
      set(net.nprocs() + static_cast<int>(v), bits(net.intvars[v].hi - net.intvars[v].lo + 1));
    // >= 64 (not > 64): keeps every field offset < 64, so a trailing zero-width
    // field (single-value int/location) can never trigger a `value << 64` UB shift.
    if (off >= 64) {
      // Say WHAT overflowed, not just that something did: the widest fields are almost
      // always a few int arrays, and the user can only act on the breakdown.
      std::vector<std::pair<int, std::string>> w;
      for (int p = 0; p < net.nprocs(); ++p)
        w.emplace_back(bits(static_cast<long>(net.procs[p].locations.size())),
                       "proc[" + std::to_string(p) + "] (" +
                       std::to_string(net.procs[p].locations.size()) + " locs)");
      for (std::size_t v = 0; v < net.intvars.size(); ++v)
        w.emplace_back(bits(net.intvars[v].hi - net.intvars[v].lo + 1),
                       "int " + net.intvars[v].name + " [" + std::to_string(net.intvars[v].lo) +
                       "," + std::to_string(net.intvars[v].hi) + "]");
      std::stable_sort(w.begin(), w.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      std::string msg = "reachable_net: discrete state needs " + std::to_string(off) +
                        " bits but the packed key holds 63 (network too wide).\n"
                        "  " + std::to_string(net.nprocs()) + " processes + " +
                        std::to_string(net.intvars.size()) + " int vars. Widest fields:";
      for (std::size_t i = 0; i < w.size() && i < 8; ++i)
        msg += "\n    " + std::to_string(w[i].first) + " bits  " + w[i].second;
      throw std::runtime_error(msg);
    }
  }
  // Decode a packed key back into location + int vectors (inverse of disc_key).
  auto decode_key = [&](std::uint64_t dk, int* locs, long* ints) {
    for (int p = 0; p < net.nprocs(); ++p) locs[p] = static_cast<int>((dk >> dshift[p]) & dmask[p]);
    for (std::size_t v = 0; v < net.intvars.size(); ++v)
      ints[v] = static_cast<long>((dk >> dshift[net.nprocs() + v]) & dmask[net.nprocs() + v]) + net.intvars[v].lo;
  };
  auto disc_key = [&](const LocVec& locs, const IntVec& ints) -> std::uint64_t {
    std::uint64_t k = 0;
    for (int p = 0; p < static_cast<int>(locs.size()); ++p)
      k |= static_cast<std::uint64_t>(static_cast<unsigned>(locs[p])) << dshift[p];
    for (std::size_t v = 0; v < ints.size(); ++v)
      k |= static_cast<std::uint64_t>(ints[v] - net.intvars[v].lo) << dshift[net.nprocs() + v];
    return k;
  };

  // initial symbolic state
  SymState s0;
  s0.locs.resize(net.nprocs());
  for (int p = 0; p < net.nprocs(); ++p) s0.locs[p] = net.procs[p].initial;
  s0.ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0.ints[v] = net.intvars[v].init;
  s0.zone = Dbm(net.nclocks);
  {
    std::vector<Constraint> inv;
    all_invariants(net, s0.locs, inv);
    if (!apply_all(s0.zone, inv)) return false;
    bool c, u;
    frozen_flags(net, s0.locs, &c, &u);
    if (!c && !u) { s0.zone.delay(); apply_all(s0.zone, inv); }
    reset_inactive(net, s0.locs, s0.zone, qmask);
    // Extrapolate only when finiteness depends on it (emode != ExNone). In the default aLU mode
    // the passed list stays finite and minimal on exact zones, so extrapolating would only lose
    // precision; gen_succ_reuse applies the same emode to every successor.
    if (emode == ExLuPlus) s0.zone.extrapolate_lu_plus(L, U);
    else if (emode == ExNoDiag) s0.zone.extrapolate_lu_plus_nodiag(L, U);
  }
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
    passed[disc_key(s0.locs, s0.ints)].assign(1, fz[0]);
    // CPU-expand / GPU-merge hybrid (PTCTLMC_HYBRID=1): each phase on the hardware the
    // measurements say it wins on. Zero-copy via unified memory.
    const bool hybrid = std::getenv("PTCTLMC_HYBRID") != nullptr;
    int hyb_threads = 1;
    std::size_t hyb_ratio = 8;   // successors staged per frontier state (adapts per level)
#ifdef PTCTLMC_HAVE_OPENMP
    hyb_threads = omp_get_max_threads();
#endif
    double gpu_rate = -1, cpu_rate = -1;   // adaptive merge routing: work-units (sum bucket^2) per ms
    auto _now = [] { return std::chrono::steady_clock::now(); };
    auto _ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
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
    } bidx;
    std::size_t prev_nb = 1024;
    double fanout = 4.0;   // running MAX of successors per frontier state (batch-size estimate)

    {   // let the expand kernel pack the discrete key itself (see net_gpu.hpp)
      std::vector<int> ilo(net.intvars.size());
      for (std::size_t v = 0; v < net.intvars.size(); ++v) ilo[v] = static_cast<int>(net.intvars[v].lo);
      gpu->set_key_layout(dshift, ilo);
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
          ot.mask |= dmask[lr.first] << dshift[lr.first];
          ot.val  |= static_cast<std::uint64_t>(static_cast<unsigned>(lr.second)) << dshift[lr.first];
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
    // Consume the first `nfr` states of the current level, keeping the rest. erase() from the
    // front would memmove the remainder once per batch -- O(batches * level) over a wide level.
    auto drop_batch = [](std::vector<int>& l, std::vector<long>& iv, std::vector<int>& z,
                         int nfr, int np_, int niw_, int) {
      l.erase(l.begin(), l.begin() + static_cast<std::size_t>(nfr) * np_);
      iv.erase(iv.begin(), iv.begin() + static_cast<std::size_t>(nfr) * niw_);
      z.erase(z.begin(), z.begin() + static_cast<std::size_t>(nfr));
    };
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
          decode_key(touched[t], dl, di);
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
      if (hybrid) {
        // CPU-EXPAND / GPU-MERGE, ZERO-COPY. Successor generation is branchy,
        // serially-dependent small-matrix work the CPU wins; the per-bucket
        // subsumption/convex-union merge is thousands of independent jobs the GPU wins.
        // Unified memory lets each phase run where it is fastest with NO transfer: the
        // CPU expander writes successors DIRECTLY into the buffers the merge kernel
        // reads. Threads claim slots in blocks (one atomic per BLK successors, not per
        // successor), so the only per-successor cost is the write itself.
        const int NT = hyb_threads;
        constexpr std::size_t BLK = 256;
        const std::size_t cap = std::max<std::size_t>(
            static_cast<std::size_t>(nfr) * hyb_ratio + BLK * NT, 1024);
        int *sv = nullptr, *sl = nullptr, *si2 = nullptr; gz_t* sz = nullptr;
        gpu->stage_expand(cap, np, ni, d2, sv, sl, si2, sz);
        // Zero the flags once instead of tracking each thread's unused block tail: a
        // slot is a successor only if it was written, so unclaimed holes stay invalid.
        std::memset(sv, 0, cap * sizeof(int));
        std::atomic<std::size_t> wnext{0};
        // Overflow spill (only when this level's fan-out beat the estimate).
        std::vector<std::vector<int>>   spl(NT);
        std::vector<std::vector<long>>  spi(NT);
        std::vector<std::vector<raw_t>> spz(NT);
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel num_threads(NT)
#endif
        {
          int tid = 0;
#ifdef PTCTLMC_HAVE_OPENMP
          tid = omp_get_thread_num();
#endif
          std::vector<SymState> buf;
          SymState cur; cur.locs.resize(np); cur.ints.resize(ni);
          Dbm pz(net.nclocks);
          std::size_t mi = 0, me = 0;
          bool spilled = false;
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp for schedule(dynamic, 8) nowait
#endif
          for (long i = 0; i < static_cast<long>(nfr); ++i) {
            for (int p = 0; p < np; ++p) cur.locs[p] = fl[static_cast<std::size_t>(i) * np + p];
            for (int v = 0; v < ni; ++v) cur.ints[v] = fi[static_cast<std::size_t>(i) * niw + v];
            zi.load(fz[static_cast<std::size_t>(i)], pz.data().data());   // frontier holds arena indices
            const std::size_t n = gen_succ_reuse(net, cur, pz, L, U, qmask, emode, buf,
                                                 fz[static_cast<std::size_t>(i)]);
            for (std::size_t k = 0; k < n; ++k) {
              const SymState& ns = buf[k];
              if (!spilled && mi == me) {                       // claim the next block
                const std::size_t b = wnext.fetch_add(BLK, std::memory_order_relaxed);
                if (b + BLK > cap) spilled = true; else { mi = b; me = b + BLK; }
              }
              if (!spilled) {
                const std::size_t w = mi++;
                sv[w] = 1;
                for (int p = 0; p < np; ++p) sl[w * np + p] = ns.locs[p];
                for (int v = 0; v < ni; ++v) si2[w * ni + v] = static_cast<int>(ns.ints[v]);
                for (int k = 0; k < d2; ++k) sz[w * d2 + k] = gz_store(ns.zone.data()[k]);
              } else {
                spl[tid].insert(spl[tid].end(), ns.locs.begin(), ns.locs.end());
                spi[tid].insert(spi[tid].end(), ns.ints.begin(), ns.ints.end());
                spz[tid].insert(spz[tid].end(), ns.zone.data().begin(), ns.zone.data().end());
              }
            }
          }
        }
        const std::size_t direct = std::min<std::size_t>(wnext.load(), cap);
        std::size_t spill = 0;
        for (int t = 0; t < NT; ++t) spill += spz[t].size() / d2;
        if (spill) {   // estimate was low: keep what is written, append the tail
          gpu->stage_expand(direct + spill, np, ni, d2, sv, sl, si2, sz, /*preserve=*/true);
          std::size_t w = direct;
          for (int t = 0; t < NT; ++t) {
            const std::size_t m = spz[t].size() / d2;
            for (std::size_t k = 0; k < m; ++k, ++w) {
              sv[w] = 1;
              for (int p = 0; p < np; ++p) sl[w * np + p] = spl[t][k * np + p];
              for (int v = 0; v < ni; ++v) si2[w * ni + v] = static_cast<int>(spi[t][k * ni + v]);
              for (int q2 = 0; q2 < d2; ++q2) sz[w * d2 + q2] = gz_store(spz[t][k * d2 + q2]);
            }
          }
        }
        nwork = direct + spill;
        // Adapt the estimate to what this level actually produced (x2 headroom), so the
        // spill path stays cold even as the fan-out changes across the search.
        if (nfr) hyb_ratio = std::max<std::size_t>(4, 2 * (nwork / nfr + 1));
        ev = sv; ol = sl; oi = si2; oz = sz;
      } else {
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
          for (int p = 0; p < np; ++p) dk |= static_cast<std::uint64_t>(static_cast<unsigned>(wl[p])) << dshift[p];
          for (int v = 0; v < ni; ++v) dk |= static_cast<std::uint64_t>(wi[v] - net.intvars[v].lo) << dshift[np + v];
        }
        // Target test straight off the key: the required locations are a fixed bit pattern in it,
        // so the common (non-target) case is two integer ops instead of a walk over nprocs
        // locations. Only a key that already matches pays for decoding the integers.
        for (const auto& ot : tgt_tests) {
          if ((dk & ot.mask) != ot.val) continue;
          int dl2[64]; long di2[32];
          decode_key(dk, dl2, di2);
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
        decode_key(bdk[b], dl, di);
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
        decode_key(bdk[b], dl, di);
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
    int nbbits = 10;                                   // >= 1024 stripes, and >> nthreads
    while ((1 << nbbits) < std::max(16384, nthreads * 64)) ++nbbits;
    const std::size_t NB = static_cast<std::size_t>(1) << nbbits;
    const int shift = 64 - nbbits;
    const int dim2 = (net.nclocks + 1) * (net.nclocks + 1);
    ZoneIntern zi(dim2, nthreads);                     // ONE shared distinct-zone table
    std::vector<NetPassed> smap(NB);                   // flat per-stripe passed tables
    for (auto& mp : smap) mp.zi = &zi;
    // The mirror budget is process-wide (one NetPassed per stripe) but these tables are not:
    // a second query in the same process would otherwise inherit the first one's charge and
    // never promote anything.
    detail::hot_bytes().store(0, std::memory_order_relaxed);
    // shared_mutex: the read-only kill probe below runs concurrently per stripe; only a real
    // insert/merge needs exclusivity.
    std::vector<SpinRW> smtx(NB);                      // one lock per stripe (value-init in place)
    // PER-THREAD LOCAL ANTICHAINS were tried here and REMOVED (see git history). Each thread
    // inserted into its own table during a level and a bucket owner drained them into the shared
    // one at the boundary, so the expand phase took no lock at all. It is correct and it is
    // consistently SLOWER -- 8 threads: phil11 1.48x, viking12 1.29x, twolock10 1.90x,
    // fischer10 1.92x. Removing the lock saves the ~15% of busy thread-time that was spin, but a
    // private antichain cannot see the read-only kill probe that answers 83% of successors, so
    // the drain pays more back than it saves (phil12: 3.2M gathered pairs, 1.5M of them exact
    // duplicates, 19.4s of drain thread-time against a 3.8s wall). Don't retry it without a
    // design that keeps a shared kill probe.
    auto stripe_of = [&](std::uint64_t dk) -> std::size_t {
      return static_cast<std::size_t>((dk * 0x9e3779b97f4a7c15ULL) >> shift);
    };
    // Buckets for the per-level survivor collection below (see the level loop). Taken from the
    // SAME high bits of the same mix as stripe_of, so bucket b owns a CONTIGUOUS range of
    // stripes: a bucket's for_level probes stay inside one slice of `smap` instead of scattering
    // across all NB tables. >= 4 buckets per thread keeps a dynamic schedule balanced.
    int nbkbits = 0;
    while ((1 << nbkbits) < nthreads * 4) ++nbkbits;
    const int nbk = nbkbits ? (1 << nbkbits) : 1;
    const int bshift = 64 - nbkbits;
    auto bucket_of = [&](std::uint64_t dk) -> std::size_t {   // only called when nbkbits > 0
      return static_cast<std::size_t>((dk * 0x9e3779b97f4a7c15ULL) >> bshift);
    };
    // Dynamic-schedule chunk for the frontier expansion. Loop-invariant, so OpenMP accepts a
    // runtime value here; --omp-chunk previously reached only reach.cpp's single-TA loop, which
    // made it a no-op for every network (multi-process) model.
    const int expand_chunk = std::max(1, omp_chunk());

    // The frontier holds only (dk, zone-index) per state -- 12 B, not a full
    // SymState. locs/ints come from dk (decode_key) and the zone from the intern
    // table, reconstructed into a REUSED per-thread scratch at expansion. This
    // removes the ~1.6 GB of interned-zone copies the old make_survivor did (one
    // full DBM copied into every frontier state) and shrinks the resident
    // frontier ~50× so it stays cache-resident during expansion.
    const int np_ = net.nprocs(), ni_ = static_cast<int>(net.intvars.size());
    std::vector<std::pair<std::uint64_t, int>> frontier;
    {
      const std::uint64_t dk = disc_key(s0.locs, s0.ints);
      const int z0 = zi.intern(s0.zone, 0);
      smap[stripe_of(dk)].add(std::move(s0), dk, 0, exact_sub, state_bounds, 0);
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
          decode_key(frontier[i].first, cur.locs.data(), cur.ints.data());
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
            const std::uint64_t dk = disc_key(s.locs, s.ints);
            const std::size_t si = stripe_of(dk);
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
          decode_key(fr.first, cur.locs.data(), cur.ints.data());
          const std::size_t ns = gen_succ_reuse(net, cur, zi.get(fr.second), L, U, qmask, emode, buf, fr.second);
          for (std::size_t sk = 0; sk < ns; ++sk) {
            SymState& s = buf[sk];
            ++tls_gen[0];
            if (is_target(s)) return true;
            if (use_sym) canonicalize(*sym, net, s);
            canon_ints(s.locs, s.ints);
            const std::uint64_t dk = disc_key(s.locs, s.ints);
            if (smap[stripe_of(dk)].add(std::move(s), dk, level, exact_sub, state_bounds, 0) >= 0)
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
      // `nbk` / bucket_of are set up next to stripe_of above.
      const int nbuk = single ? 1 : nbk;
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
          for (const std::uint64_t dk : tls_touched[t]) part[base + bucket_of(dk)].push_back(dk);
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
            smap[stripe_of(dk)].for_level(dk, level, [&](int zidx) { buf.push_back({dk, zidx}); });
          continue;
        }
        std::unordered_set<std::uint64_t> seen;
        for (int t = 0; t < nthreads; ++t) {
          for (const std::uint64_t dk : part[static_cast<std::size_t>(t) * nbuk + b]) {
            if (!seen.insert(dk).second) continue;
            smap[stripe_of(dk)].for_level(dk, level, [&](int zidx) { buf.push_back({dk, zidx}); });
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
      if (adapt && emode == ExNone) {
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

      // Reclaim interned zones that no live passed-set node references any more (subsumed /
      // merged out). Mark the survivors -- referenced by a live arena node or the next frontier
      // -- then sweep the rest back to the interner's free-list for in-place reuse.
      //
      // TRIGGER: growth-based, not per-level. The old guard was `zi.n > 50000`, but `n` is a
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
      if (zi.allocated() >= sweep_at) {
        std::vector<std::uint8_t> alive(zi.index_bound(), 0);
        // MARK IN PARALLEL over stripes. Every writer stores the same value (1) and nothing
        // reads `alive` until the sweep below, so two threads marking the same zidx cannot
        // disagree -- but concurrent plain stores to one byte are still formally a race, so
        // the store is a relaxed atomic. On x86/ARM that is the same instruction as a plain
        // byte store (no lock prefix, no fence): the parallelism is free. The mark walks
        // every live arena node, which was the serial half of the sweep (0.128s of a 3.83s
        // arbiter16 run at 8 threads, and it grows with the state space).
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
        sweep_at = std::max<std::size_t>(kSweepFloor, zi.allocated() * 2);
      }

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
    std::string k = disc_key(s.locs, s.ints);
    const auto& d = s.zone.data();
    k.append(reinterpret_cast<const char*>(d.data()), d.size() * sizeof(raw_t));
    return k;
  };

  // initial symbolic state (same construction as reachable_net)
  SymState s0;
  s0.locs.resize(net.nprocs());
  for (int p = 0; p < net.nprocs(); ++p) s0.locs[p] = net.procs[p].initial;
  s0.ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0.ints[v] = net.intvars[v].init;
  s0.zone = Dbm(net.nclocks);
  {
    std::vector<Constraint> inv;
    all_invariants(net, s0.locs, inv);
    if (!apply_all(s0.zone, inv)) return false;
    bool c, u;
    frozen_flags(net, s0.locs, &c, &u);
    if (!c && !u) { s0.zone.delay(); apply_all(s0.zone, inv); }
    reset_inactive(net, s0.locs, s0.zone, qmask);
    if (emode == ExLuPlus) s0.zone.extrapolate_lu_plus(L, U);
    else if (emode == ExNoDiag) s0.zone.extrapolate_lu_plus_nodiag(L, U);
  }
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

// Exact per-valuation deadlock test, defined below and used by `E<> deadlock`. Declared here
// because leads_to_net needs the same notion: a maximal run can end inside a state whose zone is
// only PARTIALLY stuck, and the whole-zone test `adj[u].empty()` cannot see that.
// `stuck`, when non-null, receives the EXACT set of valuations of s.zone from which no
// transition is ever enabled -- the quantity this routine already computes internally and
// then discards. It is what an exact `stop` term needs; see Otf::stop_set.
static bool state_has_deadlock(const Network& net, const SymState& s,
                               Federation* stuck = nullptr, Federation* escapes = nullptr);


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
static bool state_has_deadlock(const Network& net, const SymState& s,
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
  static const bool selfcheck = std::getenv("PTCTLMC_DLSELFCHECK") != nullptr;
  for (const Dbm& W : live)
    if (W.includes(Z)) {
      if (selfcheck && !fed_subtract(Z, live).empty()) {
        std::fprintf(stderr, "[selfcheck] one-escape shortcut claimed Dead=empty, residue is not\n");
        std::abort();
      }
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
  // cases keep the old exact behaviour. PTCTLMC_DLEXACT restores it everywhere, for A/B.
  static const bool dl_exact = std::getenv("PTCTLMC_DLEXACT") != nullptr;
  if (dl_exact || network_has_diagonal(net) || gsim_on())
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


// ============================================================================
// PROTOTYPE (candidate C1): backward greatest-fixpoint liveness on the
// subsumption-closed forward graph.
//
// NOT ON ANY PRODUCTION PATH. This file is a COPY of src/netreach.cpp used only
// by the standalone gfp_proto binary; the shipping engine is untouched.
// See docs/superpowers/specs/parallel-search-novelty.md, candidate C1.
//
// Phase 1  forward BFS with inclusion subsumption -> ONE zone per reachable
//          discrete state on these families (the measured floor), with every
//          transition TAGGED by the edge that produced it.
// Phase 2  restrict to !q.
// Phase 3  chaotic backward greatest fixpoint  W <- W  /\  ( NoSucc \/ Pre(W) ),
//          one federation per node.
//
// The key point the candidate rests on: a SUBSUMPTION EDGE (target zone stored
// is strictly bigger than the real successor) is FATAL for forward cycle
// detection -- it closes loops that the concrete system cannot run -- but it is
// HARMLESS here, because the backward operator only ever uses the target's W as
// an upper bound, and over-approximating the target keeps Omega subseteq W,
// which is the direction soundness needs.
// ============================================================================

// A transition is a SET of participating edges: one for a local edge, two for a binary
// synchronisation, and 1 + n for a broadcast (sender plus every receiver that fired). Keeping
// them in one list means gfp_pre has a single code path for all three.
struct GfpTag {
  std::vector<std::pair<int,int>> parts;   // (process, edge index)
  // BROADCAST ABSTENTION, the backward-direction half of build_broadcast_parts.
  //
  // A receiver that does NOT fire still constrains the transition: it stays put precisely
  // because its own clock guard fails, so the step is enabled only where the COMPLEMENT of
  // that guard holds. `parts` cannot say that -- an abstaining receiver contributes no edge --
  // so the constraints ride here. They are a pure guard on the SOURCE: no reset, no location
  // change, no integer update, which is why gfp_pre can apply them at exactly the same point
  // as the participants' guards and nowhere else. Empty for local edges, binary sync, and
  // every broadcast whose receivers all carry unguarded a? edges (csma).
  std::vector<Constraint> abstain;
};

// GfpStats now lives in include/ptctlmc/netreach.hpp, alongside the three leads-to entry
// points it instruments. It used to be declared here AND hand-copied into main_gfp.cpp,
// which was an ODR hazard waiting to fire the moment a field was added on one side only.

// Tagged successor generation. Mirrors successors_net for LOCAL edges, BINARY synchronisation
// and BROADCAST, and additionally reports which edges produced each successor.
static void gfp_succ_tagged(const Network& net, const SymState& s,
                            const std::vector<std::int32_t>& L,
                            const std::vector<std::int32_t>& U,
                            const std::vector<char>& qmask, int emode,
                            std::vector<SymState>& out, std::vector<GfpTag>& tags) {
  std::vector<Constraint> inv;
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, &src_committed, &src_urgent);

  // ---- local edges ----
  for (int p = 0; p < net.nprocs(); ++p) {
    if (src_committed && !net.procs[p].locations[s.locs[p]].committed) continue;
    for (int ei : net.procs[p].edges_of(s.locs[p])) {
      const NEdge& e = net.procs[p].edges[ei];
      if (e.sync_chan >= 0) continue;
      if (!int_ok(e.int_guard, s.ints) || !gen_ok(net, e, s.ints.data(), s.ints.size())) continue;
      Dbm z1 = s.zone;
      if (!apply_all(z1, e.clock_guard)) continue;
      for (int r : e.clock_resets) z1.reset(r);
      SymState ns;
      ns.locs = s.locs; ns.locs[p] = e.tgt;
      ns.ints = s.ints;
      for (const auto& a : e.int_updates) ns.ints[a.var] = a.val;
      if (!gen_apply(net, e, ns.ints.data(), ns.ints.size())) continue;
      all_invariants(net, ns.locs, inv);
      if (!apply_all(z1, inv)) continue;
      bool nc, nu; frozen_flags(net, ns.locs, &nc, &nu);
      if (!nc && !nu) { z1.delay(); apply_all(z1, inv); }
      reset_inactive(net, ns.locs, z1, qmask);
      if (emode == ExLuPlus) z1.extrapolate_lu_plus(L, U);
      else if (emode == ExNoDiag) z1.extrapolate_lu_plus_nodiag(L, U);
      if (z1.empty()) continue;
      ns.zone = std::move(z1);
      out.push_back(std::move(ns));
      GfpTag t; t.parts.push_back({p, ei}); tags.push_back(std::move(t));
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
          Dbm z1 = s.zone;
          if (!apply_all(z1, se.clock_guard)) continue;
          if (!apply_all(z1, re.clock_guard)) continue;
          for (int r : se.clock_resets) z1.reset(r);
          for (int r : re.clock_resets) z1.reset(r);
          SymState ns;
          ns.locs = s.locs; ns.locs[i] = se.tgt; ns.locs[j] = re.tgt;
          ns.ints = s.ints;
          for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;
          for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
          if (!gen_apply(net, se, ns.ints.data(), ns.ints.size()) ||
              !gen_apply(net, re, ns.ints.data(), ns.ints.size())) continue;
          all_invariants(net, ns.locs, inv);
          if (!apply_all(z1, inv)) continue;
          bool nc, nu; frozen_flags(net, ns.locs, &nc, &nu);
          if (!nc && !nu) { z1.delay(); apply_all(z1, inv); }
          reset_inactive(net, ns.locs, z1, qmask);
          if (emode == ExLuPlus) z1.extrapolate_lu_plus(L, U);
          else if (emode == ExNoDiag) z1.extrapolate_lu_plus_nodiag(L, U);
          if (z1.empty()) continue;
          ns.zone = std::move(z1);
          out.push_back(std::move(ns));
          GfpTag t; t.parts.push_back({i, ei}); t.parts.push_back({j, ej}); tags.push_back(std::move(t));
        }
      }
    }
  }

  // ---- broadcast: a! fires together with EVERY enabled a? (0 or more receivers) ----
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
        for (std::size_t t2 = 0; t2 < parts.size(); ++t2)
          if (parts[t2].second[pick[t2]].edge >= 0 &&
              net.procs[parts[t2].first].locations[s.locs[parts[t2].first]].committed) part_committed = true;
        if (!src_committed || part_committed) {
          Dbm z1 = s.zone;
          bool ok = apply_all(z1, se.clock_guard);
          for (std::size_t t2 = 0; ok && t2 < parts.size(); ++t2) {
            const RecvOpt& o = parts[t2].second[pick[t2]];
            ok = o.edge >= 0 ? apply_all(z1, net.procs[parts[t2].first].edges[o.edge].clock_guard)
                             : apply_all(z1, o.extra);
          }
          if (ok) {
            SymState ns;
            ns.locs = s.locs; ns.ints = s.ints;
            ns.locs[i] = se.tgt;
            for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;
            for (int r : se.clock_resets) z1.reset(r);
            GfpTag tg; tg.parts.push_back({i, ei});
            for (std::size_t t2 = 0; t2 < parts.size(); ++t2) {
              const RecvOpt& o = parts[t2].second[pick[t2]];
              if (o.edge < 0) {          // abstained: nothing moves, but the guard's complement
                                         // is part of what enables this step -- keep it for Pre
                tg.abstain.insert(tg.abstain.end(), o.extra.begin(), o.extra.end());
                continue;
              }
              const int pj = parts[t2].first, ej = o.edge;
              const NEdge& re = net.procs[pj].edges[ej];
              ns.locs[pj] = re.tgt;
              for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
              for (int r : re.clock_resets) z1.reset(r);
              tg.parts.push_back({pj, ej});
            }
            all_invariants(net, ns.locs, inv);
            if (apply_all(z1, inv)) {
              bool nc, nu; frozen_flags(net, ns.locs, &nc, &nu);
              if (!nc && !nu) { z1.delay(); apply_all(z1, inv); }
              reset_inactive(net, ns.locs, z1, qmask);
              if (emode == ExLuPlus) z1.extrapolate_lu_plus(L, U);
              else if (emode == ExNoDiag) z1.extrapolate_lu_plus_nodiag(L, U);
              if (!z1.empty()) {
                ns.zone = std::move(z1);
                out.push_back(std::move(ns));
                tags.push_back(std::move(tg));
              }
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

// Exact concrete predecessor of one tagged transition. One code path for local, binary and
// broadcast: every participating edge contributes its guard and its resets.
//
//   forward:  z1 = Z /\ (all guards) ; reset (all resets) ; /\ inv' ;
//             if !frozen' { delay ; /\ inv' }
//   backward: undo the active-clock reduction at the target ; /\ inv' ; down ; /\ inv' ;
//             /\ (x_r = 0 for every reset r) ; free those r ; /\ (all guards) ;
//             down (the delay BEFORE the action) ; /\ inv(src)
//
// No extrapolation and no reset_inactive appear in the forward direction here: this is the exact
// concrete Pre. The abstraction enters only through the caller's confinement W_v subseteq Z_v.
// An EMPTY zone OF THE RIGHT DIMENSION.
//
// `Dbm()` is documented as "a zero-clock placeholder, so a Dbm can live in a container": it is
// a 1x1 matrix holding kZeroLe, and `dbm_empty` tests the DIAGONAL for a negative entry, so
// `Dbm().empty()` is FALSE. Returning it from gfp_pre to mean "the predecessor is empty"
// therefore hands the caller a non-empty 1x1 DBM, which `if (!pz.empty())` duly pushes into a
// federation of (nclocks+1)-dimensional ones. The next `dbm_includes` inside gfp_fed_compact
// reads dim*dim entries out of a 4-byte allocation.
//
// Caught by AddressSanitizer as a heap-buffer-overflow, after `PTCTLMC_GFPMODE=d6` on grc4
// returned SATISFIED on 4 of 10 identical runs and UNUSABLE on the other 6: the verdict was a
// function of whatever heap garbage followed the placeholder. The bug is in gfp_pre and is
// therefore shared by the published leads_to_gfp, not introduced by the on-the-fly work; the
// hull abstraction merely takes the early-return paths often enough to make it fire.
static Dbm gfp_empty(int nclocks) {
  Dbm z(nclocks);
  z.set(0, 0, lt(0));      // 0 - 0 < 0 is unsatisfiable, so dbm_empty() is true
  return z;
}

static Dbm gfp_pre(const Network& net, const GfpTag& t, const SymState& src, const SymState& tgt,
                   const Dbm& X) {
  Dbm Y = X;

  // Undo the active-clock reduction at the target. Those clocks are in no edge's reset set, so
  // without this Pre silently demands they also be 0 at the SOURCE -- an over-constrained Pre
  // that collapses the fixpoint and MISSES violations (measured on phil4/6, grc4, arbiter4/5).
  // Freeing them is the exact inverse and is sound because they are dead: nothing reads them.
  if (!net.active.empty()) {
    for (int c = 1; c <= net.nclocks; ++c) {
      bool live = false;
      for (int pr = 0; pr < net.nprocs() && !live; ++pr)
        if (net.active[pr][tgt.locs[pr]][c]) live = true;
      if (!live) Y.free(c);
    }
  }

  std::vector<Constraint> inv;
  all_invariants(net, tgt.locs, inv);
  bool nc, nu; frozen_flags(net, tgt.locs, &nc, &nu);
  if (!apply_all(Y, inv)) return gfp_empty(net.nclocks);
  if (!nc && !nu) { Y.down(); if (!apply_all(Y, inv)) return gfp_empty(net.nclocks); }

  // Undo every participant's resets: a reset clock reads 0 after the step, and its value before
  // the step is unconstrained by the step itself.
  for (const auto& pe : t.parts)
    for (int r : net.procs[pe.first].edges[pe.second].clock_resets) {
      if (!Y.constrain(r, 0, le(0))) return gfp_empty(net.nclocks);
      if (!Y.constrain(0, r, le(0))) return gfp_empty(net.nclocks);
    }
  for (const auto& pe : t.parts)
    for (int r : net.procs[pe.first].edges[pe.second].clock_resets) Y.free(r);

  for (const auto& pe : t.parts)
    if (!apply_all(Y, net.procs[pe.first].edges[pe.second].clock_guard)) return gfp_empty(net.nclocks);
  // The abstaining receivers' guard complements, applied here and not elsewhere: like a
  // participant's guard they constrain the source valuation, and unlike one they undo nothing,
  // so they belong after the resets have been freed and before the pre-action delay.
  if (!t.abstain.empty() && !apply_all(Y, t.abstain)) return gfp_empty(net.nclocks);

  // Delay BEFORE the action. A stored zone is delay-closed as a SET, which is why the forward
  // operator may write `Z /\ g`; a single valuation may still have to let time pass first.
  std::vector<Constraint> sinv;
  all_invariants(net, src.locs, sinv);
  bool sc, su; frozen_flags(net, src.locs, &sc, &su);
  if (!sc && !su) { Y.down(); }
  if (!apply_all(Y, sinv)) return gfp_empty(net.nclocks);
  return Y;
}

// ---- small federation helpers (Federation = std::vector<Dbm>, union semantics) ----
static bool gfp_fed_empty(const Federation& f) {
  for (const auto& z : f) if (!z.empty()) return false;
  return true;
}
// Is every point of `a` inside the union `b`?
static bool gfp_fed_subset(const Federation& a, const Federation& b) {
  for (const auto& z : a) {
    if (z.empty()) continue;
    if (!fed_subtract(z, b).empty()) return false;
  }
  return true;
}
// Drop empties, drop members covered by one already kept, and MERGE any pair whose union is
// exactly convex. The convex-union merge is the important part and it is free of charge in
// both directions: dbm_try_union succeeds only when the union really is a single zone, so it
// adds no point and removes none. Without it the federations grow monotonically with every
// Pre and the width cap becomes the binding constraint on the whole algorithm.
//
// (The published version also built a `rest` federation per member and then discarded it
// without reading it -- quadratic work for nothing. Removed.)
static void gfp_fed_compact(Federation& f, std::size_t cap, bool* capped) {
  Federation out;
  for (auto& z : f) {
    if (z.empty()) continue;
    bool absorbed = false;
    for (auto& o : out) {
      if (o.includes(z)) { absorbed = true; break; }          // already covered
      if (o.try_union(z)) { absorbed = true; break; }         // exactly convex union: merge
    }
    if (!absorbed) out.push_back(std::move(z));
    if (out.size() > cap) { *capped = true; break; }
  }
  // One settling pass: a merge can make an earlier member redundant against a later one.
  if (!*capped && out.size() > 1) {
    Federation settled;
    for (auto& z : out) {
      bool absorbed = false;
      for (auto& o : settled) {
        if (o.includes(z)) { absorbed = true; break; }
        if (o.try_union(z)) { absorbed = true; break; }
      }
      if (!absorbed) settled.push_back(std::move(z));
    }
    out.swap(settled);
  }
  f.swap(out);
}



// ============================================================================
// ON-THE-FLY LIVENESS  (branch `liveness-on-the-fly`).
//
// `leads_to_gfp` above is three strictly sequential phases: build the whole aLU-subsumed
// reachable graph, restrict to !q, then run a chaotic backward greatest fixpoint over it.
// Its cost is a function of the MODEL and never of the QUERY -- a `p --> q` violated three
// transitions from the initial state still pays for every reachable symbolic state.
//
// WHY THIS LOOKS HARD, AND WHY IT IS NOT.
//
// EG is a GREATEST fixpoint, so the textbook iteration starts at TOP and DESCENDS, and an
// intermediate iterate is only a PRE-fixpoint (W >= F(W)). A pre-fixpoint is an
// over-approximation and therefore says nothing: "W(v) is non-empty" cannot settle a
// violation, so there is no early exit and no obvious way to interleave with construction.
// That is the real content of "the values are no longer necessarily descending, and you
// need a substantially different convergence/correctness proof".
//
// But a violation never needed the fixpoint. It needs a CERTIFICATE, and by Knaster-Tarski
// the certificates of a greatest fixpoint are exactly its POST-fixpoints:
//
//        X  subseteq  F(X)        ==>        X  subseteq  nu F.                        (*)
//
// So take ANY set S of nodes explored so far, pin every node outside S to EMPTY, and run
// the ordinary descending iteration confined to S. Its limit nu F_S is the exact greatest
// fixpoint of the restricted operator. Restricting only DELETES disjuncts, so F_S <= F
// pointwise, and monotonicity gives nu F_S subseteq nu F: the limit of a pass over a
// FRAGMENT is a post-fixpoint of the operator on the FINISHED graph. By (*) it is a sound
// under-approximation of the final answer, computed without ever seeing the final graph.
//
// Two consequences, and they are the entire design:
//
//   * nu F_S(v) non-empty at a p /\ !q node v  =>  the query is VIOLATED, with exactly the
//     verdict the finished graph would have produced. Stop, and never build the rest.
//   * S = every expanded !q node of the FINISHED graph gives F_S = F, so the last pass IS
//     the published algorithm. Nothing is given up when the property holds.
//
// Under (*) no value has to descend across passes: each pass is a self-contained
// certificate search, and monotonicity in the graph comes for free. The four directions
// below are implemented against that yardstick and measured against each other.
//
// THE ONE RESTRICTION, and it is the only one. A node may enter S only if it is EXPANDED.
// The stopping term of the backward operator needs the ABSENCE of successors
// (`adj[v].empty()`), and an unexpanded node has an empty adjacency for the wrong reason.
// Everything else survives a partial graph untouched: the Pre disjuncts only ever need
// edges that EXIST, never the absence of one. This is what makes the scheme on-the-fly at
// all -- backward work on a half-built graph is legitimate for every node whose own
// successors have been computed, no matter what the rest of the graph is doing.
// ============================================================================


struct OtfEdge {
  int tgt;
  GfpTag tag;
};

// ---------------------------------------------------------------------------
// THE GENERAL DEADLOCK-COMPLETION CONSTRUCTION (default; PTCTLMC_OTFNOATOMS turns it off).
//
// Otf::dtheta_empty decides only the DEGENERATE case: if Dead(q) is empty, or all of Inv(q),
// or Inv(q) is time-open at a non-frozen key, the deadlock completion's constraint-map seed is
// empty and plain a_LU covering is already deadlock-sound. When some key has a genuinely
// PARTIAL Dead(q) the seed is non-empty and the map has to CARRY the atoms of Dead(q), closed
// under pre. This is that map.
//
// An atom is a half-space x_i - x_j <| b. Two kinds, and they are discharged differently:
//
//   DIAGONAL (i > 0 and j > 0). Invariant under time elapse, so agreement on it survives
//   delay, so it can be carried as a CELL TAG: every stored zone is split until it lies wholly
//   inside or wholly outside the atom, the resulting bit is appended to the discrete key, and
//   zones in different cells are never compared. The covering test itself is untouched.
//
//   NON-DIAGONAL (i == 0 or j == 0), i.e. a bound on one clock. NOT delay invariant: two
//   valuations agreeing on `x <= 5` need not agree on it after delaying, so no cell tag can
//   carry it and splitting would not restore the property. What DOES carry it is the a_LU
//   preorder itself -- with L(x) >= c and U(x) >= c its three clauses collapse to "equal, or
//   both above c", which decides every constraint on x with constant <= c the same way for
//   both valuations. So a non-diagonal atom is discharged by RAISING that one clock's bounds:
//   a_M on that clock alone, rather than a_M on the whole model.
//
// With both in place the relation
//     nu R nu'  iff  nu <=_LU nu' (raised bounds)  and  nu, nu' agree on the diagonal atoms
// is a simulation for the COMPLETED automaton A^\perp, because (a) <=_LU is a simulation for A,
// (b) diagonal agreement is preserved by delay and, via the pre-closure below, by every edge,
// and (c) Dead(q) is a boolean combination of Inv(q) (which every stored zone satisfies), the
// diagonal atoms and the non-diagonal ones -- so nu |= Dead(q) iff nu' |= Dead(q). Deadlock is
// then an EXISTENTIAL reachability of the sink, which R preserves.
struct AtomC {
  int i, j;
  raw_t b;                                   // x_i - x_j <| b
  bool operator==(const AtomC& o) const { return i == o.i && j == o.j && b == o.b; }
};

static constexpr std::int32_t kNoBump = -1;  // clock constants are >= 0, so -1 is "unbumped"

struct AtomMap {
  std::unordered_map<std::string, std::vector<AtomC>> diag;  // discrete key -> cell-tag atoms
  std::vector<std::int32_t> bump;   // clock -> constant L(x),U(x) must reach (kNoBump = none)
  std::vector<char> live;           // clocks the atoms read: reset_inactive must not zero them
  bool usable = false;
  std::size_t seed_keys = 0, seed_atoms = 0, nondiag_atoms = 0, closure_atoms = 0;
  std::size_t keys_tagged = 0, rounds = 0, bumped = 0;
  // Atoms the minimisation removed: `dropped_vacuous` = escape regions that miss Inv(q)
  // entirely; `dropped_disjoint` = atoms Inv(q) refutes, so their tag bit is constantly 0.
  std::size_t dropped_vacuous = 0, dropped_disjoint = 0;
};

class Otf {
 public:
  Otf(const Network& n, const DiscPred& pp, const DiscPred& qq,
      bool force_exact = false, const std::atomic<bool>* cancel = nullptr)
      : net(n), p(pp), q(qq), force_exact_(force_exact), cancel_(cancel) {}

  // Returns true iff `p --> q` HOLDS. Meaningless if was_cancelled().
  bool run(GfpStats* gs);

  // THE FORWARD HALF ALONE, and the backward half separately.
  //
  // The atom construction reads only the finished forward GRAPH -- the reachable discrete keys
  // and the key-edge relation with their resets -- and then, if the map is non-empty, throws
  // phase 1 away and searches again under the refined covering. Phase 1's backward fixpoint is
  // ~80% of its wall clock (grc4: 20ms of 25ms) and none of it survives that. So build the
  // graph, decide the map, and pay the fixpoint exactly once: here if the map came out empty
  // (the certificate case, where phase 1's own verdict IS the answer), in the refined search
  // otherwise. Base mode only -- the candidate-confirm modes settle mid-build, which is
  // precisely the case build_atom_map may not read.
  void build_forward(GfpStats* gs);
  bool finish_verdict(GfpStats* gs);          // true iff `p --> q` HOLDS; needs build_forward

  // Structural certificate: is the deadlock-completion's constraint-map seed empty at every
  // reachable key? If so, plain a_LU covering was already deadlock-sound. Needs a built graph.
  bool dtheta_empty();

  // A cancelled run abandoned its exploration part-way, so its verdict is not a verdict.
  // The racing caller in leads_to_otf must consult this before publishing a result.
  bool was_cancelled() const { return cancelled_; }

  // Derive the deadlock-completion constraint map from THIS run's finished
  // graph (so it needs run() to have happened), and install one on a fresh run.
  bool build_atom_map(AtomMap& am);
  void set_atoms(const AtomMap* a) { amap_ = a; }
  void set_am(bool on) { force_am_ = on; }     // a_M covering: bisimulation, deadlock-sound
  std::size_t atom_straddles() const { return straddles_; }
  // (live atoms, total atoms) over the tagged keys; see tagseen_.
  std::pair<std::size_t, std::size_t> atom_liveness(const AtomMap& am) const {
    std::size_t live = 0, tot = 0;
    for (const auto& kv : am.diag) {
      const std::size_t n = kv.second.size();
      tot += n;
      auto it = tagseen_.find(kv.first);
      if (it == tagseen_.end()) continue;              // key carries atoms but stored no zone
      const std::uint64_t varies = it->second.first & ~it->second.second;   // OR & ~AND
      for (std::size_t a = 0; a < n && a < 64; ++a)
        if ((varies >> a) & 1u) ++live;
    }
    return {live, tot};
  }
  std::size_t atom_splitters() const {
    std::size_t n = 0;
    for (const auto& kv : straddled_) n += std::size_t(__builtin_popcountll(kv.second));
    return n;
  }
  std::size_t atom_splits() const { return splits_; }
  std::size_t atom_cand_zones() const { return cand_zones_; }
  std::size_t atom_split_zones() const { return split_zones_; }
  // (probes, probes that left the base case, recursive calls, probes killed by the prefilter)
  std::array<std::size_t, 4> cover_probe_stats() const {
    return {cov_probes_, cov_recursed_, cov_rec_calls_, cov_prefilter_};
  }
  std::array<std::size_t, 3> lazy_stats() const {
    return {lazy_splits_, lazy_redos_, lazy_activated_};
  }

 private:
  const Network& net;
  const DiscPred& p;
  const DiscPred& q;
  bool force_exact_ = false;                   // gate lane B: inclusion covering, no aLU
  bool use_am_ = false;                        // collapse LU to M: bisimulation, deadlock-sound
  const std::atomic<bool>* cancel_ = nullptr;  // set by the winner of the race
  bool cancelled_ = false;

  // Latched, so one observation is enough: nothing downstream of the first check may see the
  // flag flip back and resume a half-abandoned search.
  bool cancelled() {
    if (cancelled_) return true;
    if (cancel_ && cancel_->load(std::memory_order_relaxed)) cancelled_ = true;
    return cancelled_;
  }

  std::vector<std::int32_t> L, U, aL, aU, bL, bU;
  int emode = 0;
  std::vector<char> qmask;
  bool use_alu = false;
  std::size_t fedcap = 4096;
  bool capped = false;
  std::size_t passes = 0, gfp_iters = 0, expand_calls = 0, pre_calls = 0;
  std::size_t fed_max = 0, nonempty = 0;
  double fed_avg = 0;
  bool settled_early = false;
  // OUTER vs INNER accounting. verifyta's leads-to counters report only the OUTER
  // reachability up to the first violating p-state -- measured: on fischer4/csma4/phil4 the
  // `p --> q` figures are byte-identical to `E<> p`, so the nested inevitability check is not
  // counted at all. Ours counts every interned state including the inner DFS's. To compare
  // like with like, split them.
  std::size_t outer_expands = 0, outer_nodes = 0;

  std::vector<SymState> node;
  std::vector<std::vector<OtfEdge>> adj;
  std::vector<std::vector<int>> pred;          // maintained for D1's closure test
  std::unordered_map<std::string, std::vector<int>> bucket;
  std::vector<char> expanded;
  std::vector<char> pc, qc;                    // p/q caches, 2 = unknown
  bool intern_grew = false;
  const AtomMap* amap_ = nullptr;              // null = ordinary a_LU, no cells
  bool force_am_ = false;                      // set by the atom-map predictor
  // Per-node cell tag: bit a is set iff node's zone is contained in atom a of its key. Only
  // meaningful when amap_ is installed and the key carries atoms; 0 otherwise.
  std::vector<std::uint64_t> ntag_;
  // SPLIT-AT-COMPARISON (PTCTLMC_OTFSPLITCMP). Stored zones are NOT atom-homogeneous, so one
  // "contained" bit no longer classifies them: `ntag_out_` carries the second bit of the
  // three-valued classification (In = ntag_, Out = ntag_out_, Straddle = neither). Unused and
  // left all-zero in the default split-at-creation mode.
  std::vector<std::uint64_t> ntag_out_;
  // Probe accounting for the A/B: how many covering probes were asked, how many of those had
  // to leave the O(1)-per-atom base case, and how many recursive calls that cost.
  std::size_t cov_probes_ = 0, cov_recursed_ = 0, cov_rec_calls_ = 0, cov_prefilter_ = 0;
  // LAZY ARM. Nodes whose adjacency a permanent split invalidated and which therefore have to
  // be (re-)expanded; base_forward drains this. `cur_expand_` is the node expand() is in the
  // middle of, which may not be detached under its own feet -- it is redone at the end of the
  // call instead, so its adjacency is rebuilt from scratch and never accumulates duplicates.
  std::vector<int> relist_;
  int cur_expand_ = -1;
  bool expand_invalidated_ = false;
  std::size_t lazy_splits_ = 0, lazy_redos_ = 0, lazy_activated_ = 0;
  // Per discrete key, the atoms a covering decision has actually turned on. split_by_atoms cuts
  // a NEW zone only along these -- so a key whose atoms never discriminate is never split at
  // all, which is the whole point of the lazy arm.
  std::unordered_map<std::string, std::uint64_t> active_;
  // How much the cell refinement actually costs. cand_zones_ counts zones GENERATED at a key
  // that carries atoms -- the denominator of the straddle rate, which is the whole empirical
  // question: a zone that lies wholly inside or wholly outside every atom is tagged for free.
  std::size_t straddles_ = 0, splits_ = 0, cand_zones_ = 0, split_zones_ = 0;
  // WHICH ATOMS ACTUALLY DO ANYTHING (PTCTLMC_OTFATOMSTAT). For each tagged key, the OR and
  // the AND of the tags of every zone stored there, and the set of atoms that ever forced a
  // split. An atom whose bit is 0 in the OR is false on every stored zone; one whose bit is 1
  // in the AND is true on every stored zone; either way it separated nothing and could have
  // been left out of the map. This is the empirical version of the redundancy question --
  // strictly sharper than any syntactic test against Inv(q), because it is measured against
  // the zones the search really stored rather than against every zone Inv(q) permits.
  std::unordered_map<std::string, std::pair<std::uint64_t, std::uint64_t>> tagseen_;
  std::unordered_map<std::string, std::uint64_t> straddled_;

  void setup();
  bool holds(const DiscPred& pr, const SymState& s) const;
  bool is_p(int i);
  bool is_nq(int i);
  void grow();
  int intern(SymState&& s);
  void expand(int u);
  bool time_diverges(const SymState& s) const;
  bool stop_term(int u);                       // whole-zone NoSucc; needs expanded[u]
  Federation stop_set(int u);                  // EXACT NoSucc; needs expanded[u]

  void gfp_on(const std::vector<int>& S, std::vector<Federation>& W);
  bool certificate_in(const std::vector<int>& S);
  std::vector<int> accepting_closure();

  void split_by_atoms(SymState&& s, std::vector<SymState>& out);
  // Split-at-comparison support; see the block comment above Otf::cover_rec.
  void classify_atoms(const Dbm& z, const std::vector<AtomC>& as,
                      std::uint64_t* in, std::uint64_t* out) const;
  bool base_cover(const Dbm& a, const Dbm& b) const {
    return use_alu ? a.alu_subsumed_by(b, bL, bU) : b.includes(a);
  }
  bool cover_rec(const Dbm& a, std::uint64_t ain, const Dbm& b, std::uint64_t bout,
                 const std::vector<AtomC>& as, int depth);
  // LAZY ARM ONLY. `detach` is the in-place-enlargement invalidation recipe, lifted out:
  // drop x's adjacency, unhook it from its targets' pred lists and queue x for re-expansion.
  void detach(int x);
  void lazy_split(std::vector<int>& bk, int j, const AtomC& at, const std::vector<AtomC>& as);
  void base_forward();                         // Base mode's forward half; setup() done already
  void fill_stats(GfpStats* gs);
  bool fwd_ok_ = false;                        // build_all() found a satisfiable initial state
  bool build_all();                            // full forward BFS; true if a certificate fired
  bool final_verdict_violated();               // one exact pass over the finished graph
 public:
  void zdbg();                                 // PTCTLMC_OTFZDBG: antichain-width diagnostic
};

// ---------------------------------------------------------------------------
// DIAGNOSTIC ONLY (env PTCTLMC_OTFZDBG). Not used by any algorithm; it answers
// "why are there 14.5 zones per discrete key on grc4 and 1.0 everywhere else?".
// For every discrete key it reports the antichain width and, for the widest keys,
// the actual zones and WHICH constraint separates them.
static std::string dbg_bound(raw_t r) {
  if (r >= kInf) return "inf";
  char b[32];
  std::snprintf(b, sizeof b, "%s%d", (r & 1) ? "<=" : "<", raw_bound(r));
  return b;
}
void Otf::zdbg() {
  const int topn = std::getenv("PTCTLMC_OTFZTOP")
      ? (int)std::strtol(std::getenv("PTCTLMC_OTFZTOP"), nullptr, 10) : 3;
  std::vector<std::pair<std::size_t, const std::vector<int>*>> keys;
  for (const auto& kv : bucket) keys.push_back({kv.second.size(), &kv.second});
  std::sort(keys.begin(), keys.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  std::map<std::size_t, int> hist;
  for (const auto& k : keys) hist[k.first]++;
  std::fprintf(stderr, "[zdbg] keys=%zu nodes=%zu  width-histogram:", bucket.size(), node.size());
  for (const auto& h : hist) std::fprintf(stderr, " w%zu:%d", h.first, h.second);
  std::fprintf(stderr, "\n");
  // How much of the width would each stronger merge rule remove?
  std::size_t n_incl = 0, n_alu_glob = 0, n_alu_loc = 0, n_convex = 0, n_pairs = 0;
  std::vector<std::int32_t> gL = L, gU = U;
  gL[0] = gU[0] = 0;
  for (const auto& kv : bucket) {
    const auto& b = *(&kv.second);
    for (std::size_t a = 0; a < b.size(); ++a)
      for (std::size_t c = 0; c < b.size(); ++c) {
        if (a == c) continue;
        ++n_pairs;
        const SymState& A = node[b[a]];
        const SymState& C = node[b[c]];
        if (A.zone.dim() != C.zone.dim()) continue;
        if (C.zone.includes(A.zone)) ++n_incl;
        if (A.zone.alu_subsumed_by(C.zone, gL, gU)) ++n_alu_glob;
        std::vector<std::int32_t> lL(net.nclocks + 1, kAluNegInf), lU(net.nclocks + 1, kAluNegInf);
        if (!net.lbound_loc.empty()) {
          for (int pr = 0; pr < net.nprocs(); ++pr)
            for (int cc = 1; cc <= net.nclocks; ++cc) {
              lL[cc] = std::max(lL[cc], net.lbound_loc[pr][A.locs[pr]][cc]);
              lU[cc] = std::max(lU[cc], net.ubound_loc[pr][A.locs[pr]][cc]);
            }
        } else { lL = gL; lU = gU; }
        lL[0] = lU[0] = 0;
        if (A.zone.alu_subsumed_by(C.zone, lL, lU)) ++n_alu_loc;
        if (a < c) { Dbm h = A.zone; if (h.try_union(C.zone)) ++n_convex; }
      }
  }
  std::fprintf(stderr,
      "[zdbg] ordered pairs=%zu  covered-by-inclusion=%zu  aLU(global)=%zu  aLU(per-loc)=%zu"
      "  convex-unionable(unordered)=%zu\n",
      n_pairs, n_incl, n_alu_glob, n_alu_loc, n_convex);
  for (int t = 0; t < topn && t < (int)keys.size(); ++t) {
    const auto& b = *keys[t].second;
    if (b.empty()) continue;
    const SymState& s0 = node[b[0]];
    std::fprintf(stderr, "[zdbg] --- key width=%zu  locs=", b.size());
    for (int pr = 0; pr < net.nprocs(); ++pr)
      std::fprintf(stderr, "%s.%s ", net.procs[pr].name.c_str(),
                   net.procs[pr].locations[s0.locs[pr]].name.c_str());
    std::fprintf(stderr, "\n");
    for (std::size_t i = 0; i < b.size(); ++i) {
      const Dbm& z = node[b[i]].zone;
      std::string line;
      for (int r = 0; r < z.dim(); ++r)
        for (int c2 = 0; c2 < z.dim(); ++c2) {
          if (r == c2) continue;
          const raw_t e = z.get(r, c2);
          if (e >= kInf) continue;
          if (r == 0 && e == kZeroLe) continue;          // x_c >= 0, trivial
          const std::string rn = r == 0 ? "0" : net.clock_names[r];
          const std::string cn = c2 == 0 ? "0" : net.clock_names[c2];
          line += rn + "-" + cn + dbg_bound(e) + " ";
        }
      std::fprintf(stderr, "[zdbg]   z%02zu id=%d: %s\n", i, b[i], line.c_str());
    }
  }
}

void Otf::setup() {
  L = net.lbound;
  U = net.ubound;
  // FINITENESS COMES FROM aLU SUBSUMPTION ON EXACT ZONES, NOT FROM EXTRAPOLATION.
  //
  // These two settings are ONE decision and must move together. Plain inclusion does not bound
  // the zone graph, so with `use_alu` off the only thing that terminates this search is
  // Extra_LU+; and with Extra_LU+ on, every stored zone is an OVER-APPROXIMATION of the
  // reachable set, which is fatal for the `stop` term specifically.
  //
  // WHY IT IS FATAL. Otf::stop_set computes an exact per-valuation deadlock residue, but it
  // computes it from node[u].zone. Extra_LU+(Z) is contained in a_LU(Z), so every valuation
  // extrapolation ADDS does at most what some real valuation does -- enabledness is MONOTONE,
  // not invariant. Hence "some valuation of the extrapolated zone enables e" transfers back to
  // Z, but "some valuation of the extrapolated zone enables NOTHING" does not. stop_set asks
  // the second question. Concretely (diagonal-free, two clocks, one edge):
  //
  //     location q, invariant x <= 10;  edge q -> q2 guarded y >= 3
  //     => L(x)=0, U(x)=10, L(y)=3, U(y)=0
  //     reach q with y reset 6 after x:  Z = { x - y = 6, 6 <= x <= 10 }
  //
  // Z is deadlock-free: from anywhere, delay until y >= 3, i.e. x >= 9, still under the
  // invariant. Extra_LU+ drops the DIAGONAL x - y <= 6 because 6 > L(x) = 0, leaving
  // { x >= 6, y >= 0, x - y >= 6 }, which contains (x=10, y=0) -- that needs d >= 3 for the
  // guard and d <= 0 for the invariant, so it is permanently stuck. Nothing in Z is.
  // state_has_deadlock's delay-closure repair does NOT save this: the dropped constraint is a
  // diagonal, and in a diagonal-free automaton no invariant ever restores a diagonal.
  //
  // `stop` occurs POSITIVELY in the fixpoint operator, so a spurious stop valuation ENLARGES
  // the greatest fixpoint and fabricates a certificate: `p --> q` reported VIOLATED when it
  // holds. That is the opposite error direction from aLU covering, which can only ever lose a
  // witness, and unlike that one it is not recoverable by any later check.
  //
  // On exact zones the whole picture closes: stored zones are genuinely reachable, so stop_set
  // is exact; cycle and divergence witnesses transfer under simulation unconditionally; and the
  // only residual hole is that aLU COVERING may drop a deadlock witness -- one-sided, and
  // discharged by the deadlock-freedom gate (see leads_to_net, src/netreach.cpp:3459).
  //
  // Diagonal models cannot use aLU at all, so they keep ExNoDiag + inclusion and extrapolation
  // stays load-bearing there; their stop term is still exposed. G-simulation is the fix for
  // that case and would land it in this same shape (exact zones + simulation covering); see
  // docs/superpowers/specs/g-simulation-extraction.md.
  //
  // PTCTLMC_GFPNOALU restores the previous behaviour (inclusion covering + Extra_LU+) for A/B.
  // PTCTLMC_GFPEXTRAP keeps aLU covering but restores Extra_LU+, which isolates the
  // extrapolation change on its own -- it is the exact configuration every otf_*.json
  // baseline in this directory was recorded under (run_otf.py sets PTCTLMC_GFPALU=1).
  static const bool no_alu   = std::getenv("PTCTLMC_GFPNOALU")  != nullptr;
  static const bool want_ext = std::getenv("PTCTLMC_GFPEXTRAP") != nullptr;
  use_alu = !no_alu && !force_exact_ && !network_has_diagonal(net);
  if (force_am_ || std::getenv("PTCTLMC_OTFAM")) use_am_ = use_alu;
  emode = network_has_diagonal(net) ? ExNoDiag
                                    : ((use_alu && !want_ext) ? ExNone : ExLuPlus);
  // qmask is the set of clocks the QUERY forces to stay live; a leads-to predicate has no
  // clock constraints, so it is all-zero and the active-clock reduction runs in FULL.
  // (leads_to_gfp above carries a comment claiming this disables the reduction. It does the
  // opposite -- reset_inactive treats qmask as "additionally keep alive", so all-zero is
  // maximal reduction. What actually makes the reduction safe for a BACKWARD analysis is the
  // repair inside gfp_pre: it FREES every clock that is dead at the target before undoing
  // the step. Without that, Pre silently demands the dead clocks be 0 at the SOURCE too --
  // an over-constrained Pre that collapses the fixpoint and MISSES violations.)
  qmask.assign(net.nclocks + 1, 0);
  aL = net.lbound;
  aU = net.ubound;
  aL[0] = aU[0] = 0;
  // THE CONSTRAINT MAP'S TWO GLOBAL EFFECTS (see AtomMap).
  //
  // (1) Non-diagonal atoms are discharged by raising that clock's LU bounds. Raising L or U
  //     only ever makes a_LU FINER, so it is sound wherever it is applied; applying it to the
  //     global arrays (and, in intern, to the per-location ones) keeps it consistent along
  //     every path, which is what the simulation argument needs.
  // (2) reset_inactive pins clocks that no guard will read again to 0. The atoms DO read them
  //     -- that is the whole point of Dead(q) -- so every clock occurring in the map is added
  //     to qmask, which reset_inactive treats as "additionally keep alive". Without this the
  //     reduction erases exactly the differences the cell tag is meant to observe.
  //
  // PTCTLMC_OTFATOMSNOLIVE drops (2) -- DIAGNOSTIC, and possibly unsound. It exists because on
  // csma the pinning is the ENTIRE bill: the bucket count and the straddle count are both
  // unchanged there, so every extra stored node comes from clocks the reduction would otherwise
  // have zeroed. Whether the pinning is actually owed for SEED atoms is open: a clock read by
  // Esc(q) is read by an outgoing guard or a target invariant at q, hence active at q by
  // construction, and a diagonal that survives pullback involves only clocks some reachable
  // guard still reads. That argument is NOT a proof -- `active` is a static over-approximation
  // computed elsewhere -- so the pinning stays on by default and this only prices it.
  if (amap_) {
    static const bool no_live = std::getenv("PTCTLMC_OTFATOMSNOLIVE") != nullptr;
    for (int c = 1; c <= net.nclocks && c < (int)amap_->bump.size(); ++c) {
      if (amap_->bump[c] != kNoBump) {
        aL[c] = std::max(aL[c], amap_->bump[c]);
        aU[c] = std::max(aU[c], amap_->bump[c]);
      }
      if (amap_->live[c] && !no_live) qmask[c] = 1;
    }
  }
  // use_alu / emode are set together at the top of setup(); see the note there.
  if (const char* c = std::getenv("PTCTLMC_GFPFEDCAP")) fedcap = std::strtoul(c, nullptr, 10);
}

bool Otf::holds(const DiscPred& pr, const SymState& s) const {
  for (const auto& r : pr.locs) {
    const bool eq = (s.locs[r.proc] == r.loc);
    if (eq != r.equal) return false;
  }
  return int_ok(pr.ints, s.ints);
}

void Otf::grow() {
  const std::size_t n = node.size();
  if (expanded.size() < n) expanded.resize(n, 0);
  if (pc.size() < n) pc.resize(n, 2);
  if (qc.size() < n) qc.resize(n, 2);
  if (adj.size() < n) adj.resize(n);
  if (pred.size() < n) pred.resize(n);
  if (ntag_.size() < n) ntag_.resize(n, 0);
  if (ntag_out_.size() < n) ntag_out_.resize(n, 0);
}

bool Otf::is_p(int i) {
  grow();
  if (pc[i] == 2) pc[i] = holds(p, node[i]) ? 1 : 0;
  return pc[i] != 0;
}

bool Otf::is_nq(int i) {
  grow();
  if (qc[i] == 2) qc[i] = holds(q, node[i]) ? 1 : 0;
  return qc[i] == 0;
}

// SPLIT AT COMPARISON (PTCTLMC_OTFSPLITCMP), the alternative to split_by_atoms' split at
// creation. Stored zones are whole, so they may STRADDLE an atom, and the split -- if it is
// needed at all -- happens inside the covering test.
//
// THE THREE-VALUED CLASSIFICATION. With homogeneous zones one bit per atom suffices; with
// straddling zones it does not, and reading a straddle as "disjoint from phi" is the one
// mistake that makes this unsound. Both bits come off the canonical DBM in O(1):
//
//     Z.get(i,j) <= b                       =>  Z subseteq phi          (In)
//     raw_add(Z.get(j,i), b) < kZeroLe      =>  Z /\ phi = empty        (Out)
//     otherwise                                                          (Straddle)
//
// WHAT THE COVERING TEST OWES. `a covered by b` must witness, for every nu in a, some nu' in b
// with nu <=_LU nu' AND, for every atom phi of the key, `nu' |= phi => nu |= phi`. (That
// direction only, and it is the same obligation the homogeneous tag rule discharges -- see the
// note in Otf::intern. It is NOT the g-simulation relation of the forward diagonal work.)
//
// The base case is the EXISTING test, unchanged: `base_cover` (a_LU, or plain inclusion on
// diagonal models) plus, per atom, the requirement that the obligation hold for ALL pairs at
// once -- which it does exactly when `b` is Out (no nu' satisfies phi) or `a` is In (every nu
// does). An atom that is neither is BLOCKED, and only blocked atoms are split. On homogeneous
// operands "blocked" degenerates to "b In and a Out", i.e. bit-for-bit the implication rule.
//
// THE RECURSION (Gastin-Mukherjee-Srivathsan, CAV 2019, Lemma 17), adapted to the implication
// obligation rather than to agreement:
//
//     a <= b   <==   (a /\ phi <= b)  and  (a /\ ~phi <= b /\ ~phi)
//
// nu |= phi discharges the phi-obligation for any nu' whatever, so the first conjunct may keep
// the whole of b; nu !|= phi needs nu' !|= phi, so the second must narrow b as well. Each
// conjunct resolves phi (In on the left, Out on both sides on the right), so the blocked set
// strictly shrinks and the recursion terminates in at most 2^|blocked| leaves.
//
// The whole-operand base test is also a NECESSARY condition for the conjunction -- both
// conjuncts test sub-zones of a against sub-zones of b -- so it doubles as a prefilter that
// kills a hopeless probe before any zone is copied. That is what keeps the common reject path
// the same price it is in the split-at-creation mode.
// THE THREE ARMS, and why split-at-COMPARISON is the default.
//
// Split-at-creation cuts every zone along every atom at its key the moment it is produced, so
// stored zones are atom-homogeneous and covering is one mask AND. Split-at-comparison stores
// zones whole and splits transiently inside the covering test. PTCTLMC_OTFSPLITLAZY also stores
// whole but, when a covering decision genuinely turns on a stored zone's straddle, splits that
// node PERMANENTLY -- once per zone rather than once per probe, and only on atoms that actually
// arose (Herbreteau-Srivathsan-Walukiewicz's lazy abstraction, refine where the property is at
// stake). Lazy is split-at-comparison plus that rewrite, so it inherits its correctness whole.
//
// MEASURED, all three arms, zero verdict differences across five configurations:
//
//   model  a_LU floor  creation      cmp       lazy
//   grc4          883      1179      921        921
//   grc5        13939     14571    12554      12554
//   csma7        1696      2368     2368       2368
//
// Comparison wins and is never slower. The reason is not that its test is cheaper -- it is that
// the recursion is NEVER ENTERED: 0 of grc4's 216 blocked probes and 0 of grc5's 2660 recurse.
// At the keys that carry atoms, the a_LU antichain is already wide enough that no covering
// succeeds at all, so the atoms never refuse a merge that would otherwise have happened, and
// the eager split's only effect is fragmentation. Two measurements pin that: with
// PTCTLMC_NOMERGE, creation goes 1179 -> 2579 on grc4 while comparison is unchanged at 921 --
// the convex-union merge exists to glue back what the eager split broke -- and comparison plus
// PTCTLMC_OTFATOMSNOLIVE reaches 883 on grc4, exactly the a_LU floor.
//
// LAZY IS A NULL RESULT: 0 splits and 0 activations on every model here, so it is bit-identical
// to comparison. Forced to fire (PTCTLMC_GFPNOALU) it stores MORE than either, because
// activating an atom re-enables eager splitting for that whole key thereafter.
//
// Do not read the forward-reachability result as contradicting this. There the same recursion
// costs 5x, because coverings at atom-carrying keys DO succeed there and it genuinely fires.
// Which arm wins is a property of whether the atoms ever change a covering decision.
//
// PTCTLMC_OTFSPLITEAGER restores split-at-creation.
static bool otf_splitcmp() {
  static const bool v = std::getenv("PTCTLMC_OTFSPLITEAGER") == nullptr ||
                        std::getenv("PTCTLMC_OTFSPLITLAZY") != nullptr;
  return v;
}
static bool otf_splitlazy() {
  static const bool v = std::getenv("PTCTLMC_OTFSPLITLAZY") != nullptr;
  return v;
}

void Otf::classify_atoms(const Dbm& z, const std::vector<AtomC>& as,
                         std::uint64_t* in, std::uint64_t* out) const {
  std::uint64_t i_ = 0, o_ = 0;
  const std::size_t n = std::min<std::size_t>(as.size(), 64);
  for (std::size_t a = 0; a < n; ++a) {
    const std::uint64_t bit = std::uint64_t(1) << a;
    if (z.get(as[a].i, as[a].j) <= as[a].b) { i_ |= bit; continue; }
    if (raw_add(z.get(as[a].j, as[a].i), as[a].b) < kZeroLe) o_ |= bit;
  }
  *in = i_;
  *out = o_;
}

// `ain` is the In-mask of `a` and `bout` the Out-mask of `b`; they are the only two of the four
// masks the blocked set reads. At depth 0 they come from the caller's cache (the newcomer's own
// classification, and ntag_out_[j] for the stored node), so an unblocked probe costs exactly
// one mask op plus the base test -- the same price as the homogeneous tag rule.
bool Otf::cover_rec(const Dbm& a, std::uint64_t ain, const Dbm& b, std::uint64_t bout,
                    const std::vector<AtomC>& as, int depth) {
  ++cov_rec_calls_;
  const std::size_t n = std::min<std::size_t>(as.size(), 64);
  const std::uint64_t all = n >= 64 ? ~std::uint64_t(0) : ((std::uint64_t(1) << n) - 1);
  const std::uint64_t blocked = all & ~bout & ~ain;
  // Necessary condition for both conjuncts, so it is safe to ask it first and it doubles as
  // the reject prefilter.
  if (!base_cover(a, b)) { if (depth == 0 && blocked) ++cov_prefilter_; return false; }
  if (blocked == 0) return true;
  if (depth == 0) ++cov_recursed_;
  const int k = __builtin_ctzll(blocked);
  const AtomC& at = as[k];
  const Constraint nc = negate(Constraint{at.i, at.j, at.b});
  std::uint64_t xin, xout;
  {
    Dbm ap = a;                                              // a /\ phi  <=  b
    if (ap.constrain(at.i, at.j, at.b)) {
      classify_atoms(ap, as, &xin, &xout);
      if (!cover_rec(ap, xin, b, bout, as, depth + 1)) return false;
    }
  }
  Dbm an = a;                                                // a /\ ~phi  <=  b /\ ~phi
  if (!an.constrain(nc.i, nc.j, nc.bound)) return true;      // a lies wholly inside phi
  Dbm bn = b;
  if (!bn.constrain(nc.i, nc.j, nc.bound)) return false;     // ...but b does not: no witness
  std::uint64_t nin, nout;
  classify_atoms(an, as, &nin, &nout);
  classify_atoms(bn, as, &xin, &xout);
  return cover_rec(an, nin, bn, xout, as, depth + 1);
}

void Otf::detach(int x) {
  for (const auto& e : adj[x]) {
    auto& pv = pred[e.tgt];
    pv.erase(std::remove(pv.begin(), pv.end(), x), pv.end());
  }
  adj[x].clear();
  expanded[x] = 0;
  relist_.push_back(x);
}

// PERMANENTLY SPLIT A STORED NODE (lazy arm). `j` straddles `at`; replace it by its two pieces.
//
// WHAT THE REWRITE OWES. node[j]'s zone SHRINKS, so every edge INTO j may now under-represent
// its source's real successors -- and a missing successor loses runs, which for a greatest
// fixpoint means losing a violation and reporting the query satisfied when it is not. That is
// the one direction that must not happen, so every predecessor of j is detached and re-expanded;
// its successors are then re-interned against the split bucket and land on whichever piece
// covers them. j itself is detached for the same reason (its own adjacency was computed from
// the larger zone) and the new piece starts un-expanded.
//
// Nodes are never deleted, so the zones stored during a discarded expansion stay. They are
// genuinely reachable zones -- generated by gfp_succ from a real stored zone -- so keeping them
// costs a node and nothing else; it cannot fabricate a run the system does not have.
//
// NODE 0 IS NEVER SPLIT: it is the initial state and the search's entry point, and it has no
// predecessor to regenerate the piece that would be cut away from it.
void Otf::lazy_split(std::vector<int>& bk, int j, const AtomC& at,
                     const std::vector<AtomC>& as) {
  Dbm p1 = node[j].zone, p2 = node[j].zone;
  const Constraint nc = negate(Constraint{at.i, at.j, at.b});
  if (!p1.constrain(at.i, at.j, at.b)) return;          // not a straddle after all
  if (!p2.constrain(nc.i, nc.j, nc.bound)) return;
  ++lazy_splits_;
  const int j2 = static_cast<int>(node.size());
  {
    SymState t;
    t.locs = node[j].locs;
    t.ints = node[j].ints;
    t.zone = std::move(p2);
    node.push_back(std::move(t));
  }
  grow();
  node[j].zone = std::move(p1);
  classify_atoms(node[j].zone, as, &ntag_[j], &ntag_out_[j]);
  classify_atoms(node[j2].zone, as, &ntag_[j2], &ntag_out_[j2]);
  bk.push_back(j2);
  ++outer_nodes;
  std::vector<int> ps = pred[j];
  pred[j].clear();
  for (int u : ps) {
    if (u == cur_expand_) { expand_invalidated_ = true; continue; }
    detach(u);
  }
  if (cur_expand_ == j) expand_invalidated_ = true; else detach(j);
  expanded[j2] = 0;
  relist_.push_back(j2);
}

// aLU subsumption, exactly as leads_to_gfp's phase 1, plus the APPEND-ONLY switch.
//
// APPEND-ONLY (D1, D2). The published intern ENLARGES a stored node in place when a
// newcomer covers it -- `node[j].zone = s.zone` -- and that silently invalidates work:
// adj[j] was computed from the smaller zone, so j has to be re-expanded and any backward
// value already computed for j refers to a zone that no longer exists. Append-only refuses
// the enlargement and stores the newcomer as a separate node. Every stored node is then
// immutable from birth, which is exactly the property "start backward work only on
// vertices known not to change" needs -- and it is why D1 cannot be implemented WITHOUT
// D2's append-only rule: with in-place enlargement, "this node will never change" is not
// establishable before the forward phase terminates. The price is the antichain: the
// bucket keeps zones the published version would have collapsed.
int Otf::intern(SymState&& s) {
  intern_grew = false;
  std::string bkey = disc_key(s.locs, s.ints);
  // CELL TAG. split_by_atoms has already made every zone that reaches here atom-HOMOGENEOUS,
  // so `Z subseteq phi` is decided by one canonical DBM entry and the tag is well defined.
  //
  // WHICH TAG RULE. The tag is what stops a_LU covering from dropping a deadlock: if the
  // covering zone Z satisfies an escape atom phi and the covered Z' does not, the simulator
  // escapes where the simulated one is stuck, and Dead is lost. Spelt out: nu' in Z simulates
  // nu in Z', and Dead(q) = Inv(q) /\ AND_e !Esc_e with each Esc_e a CONJUNCTION of atoms, so
  // nu |= Dead needs, per e, some atom of Esc_e that nu falsifies to also be falsified by nu'.
  // The obligation is therefore the IMPLICATION `Z |= phi  =>  Z' |= phi`, in that direction
  // only -- equivalently tag(Z) & ~tag(Z') == 0. It is strictly coarser than same-cell
  // equality, which needlessly refuses the (Z disjoint from phi, Z' contained in phi) pair.
  //
  // The two REWRITING paths still need equality, and that is not a detail: in-place
  // enlargement and the convex-union merge both replace a stored zone, and a zone assembled
  // out of two different cells would STRADDLE phi, destroying the homogeneity the tag is read
  // from. Guarded below.
  //
  // PTCTLMC_OTFATOMSCELLEQ appends the tag to the discrete key instead, so different cells
  // land in different buckets and never meet. Sound (equality implies the implication) but
  // finer; kept to price the implication rule against it.
  // SPLIT-AT-COMPARISON. Under PTCTLMC_OTFSPLITCMP none of the homogeneity above holds:
  // split_by_atoms emits the zone whole, so `tag` is the In-mask, `tag_out` the Out-mask, and
  // everything a straddle makes undecidable is deferred to Otf::cover_rec. cell_eq is a
  // homogeneous-only diagnostic (a straddling zone has no cell to key by) and is ignored here.
  const bool splitcmp = otf_splitcmp();
  static const bool cell_eq = std::getenv("PTCTLMC_OTFATOMSCELLEQ") != nullptr && !otf_splitcmp();
  std::uint64_t tag = 0, tag_out = 0, allmask = 0;
  bool tagged = false;
  const std::vector<AtomC>* atoms = nullptr;
  if (amap_) {
    auto it = amap_->diag.find(bkey);
    if (it != amap_->diag.end() && !it->second.empty()) {
      const std::vector<AtomC>& as = it->second;
      tagged = true;
      atoms = &as;
      const std::size_t na = std::min<std::size_t>(as.size(), 64);
      allmask = na >= 64 ? ~std::uint64_t(0) : ((std::uint64_t(1) << na) - 1);
      if (splitcmp) {
        classify_atoms(s.zone, as, &tag, &tag_out);
      } else {
        for (std::size_t a = 0; a < as.size() && a < 64; ++a)
          if (s.zone.get(as[a].i, as[a].j) <= as[a].b) tag |= std::uint64_t(1) << a;
      }
      if (cell_eq) bkey.append(reinterpret_cast<const char*>(&tag), sizeof(tag));
    }
  }
  auto& b = bucket[bkey];
  if (use_alu) {
    bL.assign(net.nclocks + 1, kAluNegInf);
    bU.assign(net.nclocks + 1, kAluNegInf);
    if (net.lbound_loc.empty()) {
      bL = aL;
      bU = aU;
    } else {
      for (int pr = 0; pr < net.nprocs(); ++pr)
        for (int c = 1; c <= net.nclocks; ++c) {
          bL[c] = std::max(bL[c], net.lbound_loc[pr][s.locs[pr]][c]);
          bU[c] = std::max(bU[c], net.ubound_loc[pr][s.locs[pr]][c]);
        }
      bL[0] = bU[0] = 0;
    }
    // a_M MODE: collapse the two bound arrays to M = max(L, U) per clock.
    //
    // With L = U = M the three clauses of the LU preorder -- nu'(x)=nu(x), or
    // L(x) < nu'(x) < nu(x), or U(x) < nu(x) < nu'(x) -- collapse to "equal, or both above
    // M(x)", which is SYMMETRIC. A symmetric simulation is a bisimulation, and deadlock IS
    // preserved by bisimulation. So the same dbm_is_alu_le kernel, fed collapsed bounds,
    // is a deadlock-sound covering test and needs no gate at all.
    //
    // Behrmann-Bouyer-Larsen-Pelanek, STTT 8(3):204-215 (2006): Lemma 1 (=_M is a
    // bisimulation), Lemma 3 (<=_LU is only a simulation), and the unnumbered prose in the
    // "Simulation & Bisimulation" paragraph of section 3 stating that Uppaal's deadlock
    // properties are not preserved by simulation but are by bisimulation. This is also what
    // UPPAAL itself does: it falls back from LU to LOCAL MAX-CONSTANTS for deadlock and
    // liveness queries, not to exact zones.
    // Per-location LU bounds must carry the non-diagonal atoms' constants too; see setup().
    if (amap_)
      for (int c = 1; c <= net.nclocks && c < (int)amap_->bump.size(); ++c)
        if (amap_->bump[c] != kNoBump) {
          bL[c] = std::max(bL[c], amap_->bump[c]);
          bU[c] = std::max(bU[c], amap_->bump[c]);
        }
    if (use_am_)
      for (int c = 1; c <= net.nclocks; ++c) bL[c] = bU[c] = std::max(bL[c], bU[c]);
  }
  const int dim2 = s.zone.dim();
  // With the tag in the bucket key (cell_eq) every j in b already agrees, so both guards are
  // vacuously true and this costs one predictable branch.
  const bool tagcheck = tagged && !cell_eq;
  const bool reccheck = tagcheck && splitcmp;
  auto covered = [&](int j) {
    if (reccheck) {                                         // straddling operands: recurse
      ++cov_probes_;
      return cover_rec(s.zone, tag, node[j].zone, ntag_out_[j], *atoms, 0);
    }
    if (tagcheck && (ntag_[j] & ~tag) != 0) return false;   // Z |= phi but Z' does not: refuse
    return use_alu ? s.zone.alu_subsumed_by(node[j].zone, bL, bU) : node[j].zone.includes(s.zone);
  };
  // THE REWRITING PATHS UNDER SPLIT-AT-COMPARISON. The equality guard exists only to keep the
  // stored zone HOMOGENEOUS, which is not a property this mode has or wants. What each path
  // actually owes is:
  //
  //   in-place enlargement -- the stored zone is REPLACED by the newcomer, which under a_LU is
  //     not a superset of it, so everything already routed to j must still be covered. That is
  //     transitivity of the covering relation, and the relation IS transitive (<=_LU is a
  //     preorder and `nu'|=phi => nu|=phi` composes), so the ordinary three-valued test in the
  //     opposite direction is exactly and only what is owed.
  //
  //   convex-union merge -- try_union succeeds only when the union is EXACTLY the union, so the
  //     stored zone afterwards is a set SUPERSET of both operands. Growing the right-hand side
  //     of an exists-witness can never invalidate a witness, and the newcomer is covered by the
  //     result via the identity witness. So the merge owes NOTHING here: no tag condition at
  //     all, and it becomes strictly more powerful than in the homogeneous mode, where a
  //     cross-cell union was refused outright.
  //
  // Both paths must REFRESH the stored masks, since the stored zone changed.
  auto covers = [&](int j) {
    if (reccheck) {
      ++cov_probes_;
      return cover_rec(node[j].zone, ntag_[j], s.zone, tag_out, *atoms, 0);
    }
    if (tagcheck && ntag_[j] != tag) return false;          // rewriting: equality, see above
    return use_alu ? node[j].zone.alu_subsumed_by(s.zone, bL, bU) : s.zone.includes(node[j].zone);
  };
  // LAZY REFINEMENT, before any covering decision is taken.
  //
  // The only straddle a permanent split of the STORED zone can resolve is the one where the
  // newcomer is homogeneous and the stored zone is not: newcomer OUT of phi, stored straddling
  // it. (Newcomer IN never blocks; newcomer straddling would need the NEWCOMER cut, and cutting
  // the newcomer is what split-at-creation does -- it costs a node here and saves none, since
  // storing it whole is one node either way.) So refine exactly that case, and only when the
  // base test already says the pair would otherwise merge: that is what "the decision genuinely
  // turns on a straddled atom" means operationally. Each split removes that atom's straddle from
  // both resulting pieces, so the loop makes progress and stops.
  if (reccheck && otf_splitlazy()) {
    // Each refinement retires one atom at this key (the local `done` mask stops it being
    // re-picked, and a split leaves neither piece straddling it), so the loop makes progress.
    // The budget is insurance against a pathological key, not a semantic knob: declining a
    // refinement only leaves cover_rec to decide the pair transiently, which it does correctly
    // either way.
    std::uint64_t done = 0;
    int budget = 64;
    bool changed = true;
    while (changed && budget > 0) {
      changed = false;
      for (std::size_t k = 0; k < b.size(); ++k) {
        const int j = b[k];
        if (j == 0 || node[j].zone.dim() != dim2) continue;
        const std::uint64_t blocked = allmask & ~ntag_out_[j] & ~tag & ~done;
        if (!blocked) continue;
        const std::uint64_t sb = ~ntag_[j] & ~ntag_out_[j];        // stored straddles
        const std::uint64_t sa = ~tag & ~tag_out;                  // newcomer straddles
        const std::uint64_t cand = blocked & (sb | sa);
        if (!cand) continue;                    // stored In, newcomer Out: a genuine refusal
        if (!base_cover(s.zone, node[j].zone)) continue;   // would not merge anyway
        const int a = __builtin_ctzll(cand);
        done |= std::uint64_t(1) << a;
        // ACTIVATE the atom for this key: from here on split_by_atoms cuts new zones along it,
        // so the case this probe could not resolve (both operands straddling) stops recurring.
        auto& act = active_[bkey];
        if (!(act & (std::uint64_t(1) << a))) { act |= std::uint64_t(1) << a; ++lazy_activated_; }
        if (sb & (std::uint64_t(1) << a)) lazy_split(b, j, (*atoms)[a], *atoms);
        changed = true;
        --budget;
        break;                       // b may have grown; restart the scan
      }
    }
  }
  for (int j : b)
    if (covered(j)) return j;
  for (std::size_t k = 0; k < b.size(); ++k)
      if (covers(b[k])) {
        node[b[k]].zone = s.zone;
        if (reccheck) { ntag_[b[k]] = tag; ntag_out_[b[k]] = tag_out; }
        intern_grew = true;
        return b[k];
      }
  // EXACT CONVEX-UNION MERGE, on by default (PTCTLMC_OTFNOMERGE disables).
  //
  // The liveness graph inherited "no merge" from leads_to_net, where it is opt-in and was
  // measured a net loss -- but that measurement was made on an EXACT-ZONE graph, not on this
  // aLU-subsumed one, and the reachability engine has the merge ON by default precisely
  // because it collapses antichains. grc4 stores 14.5 zones per discrete key here against
  // 1.00-1.44 on every other family, which is exactly the shape of an antichain full of zones
  // whose union happens to be convex.
  //
  // dbm_try_union succeeds ONLY when the union of the two zones really is a single zone, so
  // the stored zone afterwards denotes exactly the same set of valuations as the two it
  // replaced: no point is added and none is removed. That is what makes it usable in the
  // UNDER-approximating lane as well -- unlike the hull, it is not an over-approximation, so
  // it cannot invent a certificate.
  static const bool no_merge = std::getenv("PTCTLMC_OTFNOMERGE") != nullptr;
  if (!no_merge) {
    for (int j : b) {
      if (node[j].zone.dim() != dim2) continue;
      // EQUALITY, not implication. The union of a zone contained in phi with one disjoint from
      // phi STRADDLES phi, and a straddling stored zone has no well-defined tag -- the cheap
      // test would stop being exact and, with it, a wqo. Same-cell unions stay homogeneous:
      // both inside phi unions to inside, both outside unions to outside.
      if (tagcheck && !splitcmp && ntag_[j] != tag) continue;
      Dbm t = node[j].zone;
      if (t.try_union(s.zone)) {          // union is exactly convex: absorb, no approximation
        node[j].zone = std::move(t);
        if (reccheck) classify_atoms(node[j].zone, *atoms, &ntag_[j], &ntag_out_[j]);
        intern_grew = true;
        return j;
      }
    }
  }
  const int id = static_cast<int>(node.size());
  static const bool astat_live = std::getenv("PTCTLMC_OTFATOMSTAT") != nullptr;
  if (tagged && astat_live) {
    auto& e = tagseen_[disc_key(s.locs, s.ints)];
    if (e.second == 0 && e.first == 0) e.second = ~std::uint64_t(0);   // AND starts at all-ones
    e.first |= tag;
    e.second &= tag;
  }
  node.push_back(std::move(s));
  if (ntag_.size() <= static_cast<std::size_t>(id)) ntag_.resize(id + 1, 0);
  if (ntag_out_.size() <= static_cast<std::size_t>(id)) ntag_out_.resize(id + 1, 0);
  ntag_[id] = tag;
  ntag_out_[id] = tag_out;
  b.push_back(id);
  ++outer_nodes;
  grow();
  return id;
}

// SPLIT A GENERATED ZONE INTO ATOM-HOMOGENEOUS PIECES.
//
// Splitting is set-theoretic: the pieces partition the zone exactly, so post, pre and stop_set
// all distribute over them and the greatest fixpoint is unchanged. It is also what makes the
// cheap cell tag EXACT -- a straddling zone has no well-defined tag, so the tagged covering
// test would stop being a well-quasi-order and the search could loop.
//
// stop_set stays correct on a piece even though a piece need not be delay-closed:
// state_has_deadlock rebuilds the delay closure Up(Z) /\ Inv(q) before testing, which is the
// set the escape regions must be measured against, and gfp_on intersects the result back with
// node[v].zone. (Up(Z) /\ Inv) \ live, intersected with Z, is exactly Z \ live.
void Otf::split_by_atoms(SymState&& s, std::vector<SymState>& out) {
  out.clear();
  const std::vector<AtomC>* at = nullptr;
  if (amap_) {
    auto it = amap_->diag.find(disc_key(s.locs, s.ints));
    if (it != amap_->diag.end() && !it->second.empty()) at = &it->second;
  }
  static const bool astat_split = std::getenv("PTCTLMC_OTFATOMSTAT") != nullptr;
  const std::string skey = astat_split ? disc_key(s.locs, s.ints) : std::string();
  const bool splitcmp = otf_splitcmp();
  out.push_back(std::move(s));
  if (!at) return;
  ++cand_zones_;
  const std::size_t str0 = straddles_;
  // SPLIT AT COMPARISON: emit the zone WHOLE. The classification is still counted -- it is the
  // straddle rate the A/B is measured against -- but only when the statistics are asked for,
  // so the timed configuration pays nothing for it here. Every deferred obligation is
  // discharged in Otf::cover_rec. The LAZY arm cuts here too, but only along the atoms a
  // covering decision has already turned on at this key -- `active_`, filled by intern.
  std::uint64_t cutmask = ~std::uint64_t(0);          // split-at-creation: cut on every atom
  if (splitcmp) {
    cutmask = 0;
    if (otf_splitlazy()) {
      auto ai = active_.find(disc_key(out[0].locs, out[0].ints));
      if (ai != active_.end()) cutmask = ai->second;   // lazy: only the atoms that mattered
    }
    if (astat_split) {                                 // straddle rate of the zone AS GENERATED
      std::size_t sb = 0;
      for (const AtomC& a : *at) {
        const std::size_t abit = sb++;
        if (out[0].zone.get(a.i, a.j) <= a.b) continue;
        if (raw_add(out[0].zone.get(a.j, a.i), a.b) < kZeroLe) continue;
        ++straddles_;
        if (abit < 64) straddled_[skey] |= std::uint64_t(1) << abit;
      }
      if (straddles_ != str0) ++split_zones_;
    }
    if (!cutmask) return;
  }
  std::size_t aidx = 0;
  for (const AtomC& a : *at) {
    const std::size_t abit = aidx++;
    // The default arm is left bit-for-bit as it was: cutmask is all-ones and the guard is
    // skipped entirely, so an atom past bit 63 -- which the 64-atom cap already forbids -- is
    // still cut exactly as before.
    if (splitcmp && (abit >= 64 || !((cutmask >> abit) & 1u))) continue;
    const std::size_t n = out.size();
    bool emptied = false;
    for (std::size_t k = 0; k < n; ++k) {
      if (out[k].zone.empty()) { emptied = true; continue; }
      if (out[k].zone.get(a.i, a.j) <= a.b) continue;                    // Z subseteq phi
      if (raw_add(out[k].zone.get(a.j, a.i), a.b) < kZeroLe) continue;   // Z disjoint from phi
      ++straddles_;
      if (astat_split && abit < 64) straddled_[skey] |= std::uint64_t(1) << abit;
      SymState t;
      t.locs = out[k].locs;
      t.ints = out[k].ints;
      t.zone = out[k].zone;
      // NARROW out[k] BEFORE pushing: push_back may reallocate `out`, so nothing may hold a
      // reference into it across the insertion.
      const Constraint nc = negate(Constraint{a.i, a.j, a.b});
      if (!out[k].zone.constrain(nc.i, nc.j, nc.bound)) emptied = true;
      if (t.zone.constrain(a.i, a.j, a.b)) out.push_back(std::move(t));
    }
    if (emptied)
      out.erase(std::remove_if(out.begin(), out.end(),
                               [](const SymState& x) { return x.zone.empty(); }), out.end());
  }
  // In the lazy arm split_zones_ was already accounted above, against the zone as GENERATED --
  // which is the straddle rate the three arms are compared on -- so do not count it twice.
  if (!splitcmp && straddles_ != str0) ++split_zones_;
}

void Otf::expand(int u) {
  grow();
  if (cancelled()) return;   // choke point: every forward step goes through here
  if (expanded[u]) return;
  expanded[u] = 1;
  ++expand_calls;
  ++outer_expands;
  const int prev_expand = cur_expand_;
  cur_expand_ = u;
  std::vector<SymState> succ;
  std::vector<GfpTag> tags;
  std::vector<SymState> pieces;
  gfp_succ_tagged(net, node[u], L, U, qmask, emode, succ, tags);
  for (std::size_t k = 0; k < succ.size(); ++k) {
    pieces.clear();
    if (amap_) {
      split_by_atoms(std::move(succ[k]), pieces);
      if (pieces.size() > 1) splits_ += pieces.size() - 1;
    } else {
      pieces.push_back(std::move(succ[k]));
    }
    for (std::size_t pi = 0; pi < pieces.size(); ++pi) {
    const int v = intern(std::move(pieces[pi]));
    grow();
    adj[u].push_back(OtfEdge{v, tags[k]});
    pred[v].push_back(u);
    // WITNESS AT GENERATION, not at expansion.
    // In-place enlargement invalidates the enlarged node's adjacency (it was computed from
    // the smaller zone).
    if (intern_grew) {
      if (v != u) {
        expanded[v] = 0;
        for (const auto& e : adj[v]) {
          auto& pv = pred[e.tgt];
          pv.erase(std::remove(pv.begin(), pv.end(), v), pv.end());
        }
        adj[v].clear();
      } else {
        expanded[u] = 0;
      }
    }
    }
  }
  cur_expand_ = prev_expand;
  // A lazy split hit `u` while it was mid-expansion. Its adjacency is now half-built against a
  // bucket that has since changed, so throw the whole call away and redo it: rebuilding from
  // scratch is what keeps adj[u] from accumulating duplicate edges.
  if (expand_invalidated_) {
    expand_invalidated_ = false;
    grow();
    detach(u);
    ++lazy_redos_;
  }
}

bool Otf::time_diverges(const SymState& s) const {
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
}

// The END-OF-RUN term, whole-zone by necessity, not by laziness -- identical to
// leads_to_gfp's. Stored zones are LU-extrapolated, so they contain points the system
// never reaches; each is LU-SIMULATED BY a reachable one, so "this point has an infinite
// !q run" transfers to the reachable point (sound) while "this point is STUCK" does not
// (unsound). Hence the whole-zone convention, inherited from the shipping engine, together
// with its known incompleteness on partial deadlocks.
//
// The EXACT stop set: the valuations of Z_v from which no transition is ever enabled, or
// from which time diverges. PTCTLMC_OTFEXACTSTOP selects it over the whole-zone convention.
//
// The whole-zone form admits all of Z_v as soon as ONE valuation is stuck, which
// over-approximates and leaves the "non-empty fixpoint implies a witness" direction
// conditional. The exact set is already computed inside state_has_deadlock -- it subtracts
// the union of the enabled source regions from the delay-closure of Z_v -- and merely
// discarded. The two whole-zone cases stay exact: an empty adjacency means every valuation
// is stuck, and time_diverges is a property of the delay-closed zone as a whole.
//
// Historical note, because the naive version of this WAS measured wrong: a per-valuation
// stuck test on the STORED zone reported a false violation of arbiter4 `C1.cs --> C1.idle`,
// witnessed by x0..x3 > 4 -- points Extra_LU+ invents (U = 4). That failure was traced to the
// active-clock reduction leaving the stored zone not delay-closed, which state_has_deadlock
// now repairs before testing. This uses the repaired routine.
Federation Otf::stop_set(int u) {
  static const bool no_pd = std::getenv("PTCTLMC_GFPNOPD") != nullptr;
  Federation f;
  if (adj[u].empty() || time_diverges(node[u])) { f.push_back(node[u].zone); return f; }
  if (!no_pd) {
    Federation d;
    if (state_has_deadlock(net, node[u], &d)) f.swap(d);
  }
  return f;
}

// `adj[u].empty()` is only meaningful once u is EXPANDED. Callers guarantee that.
bool Otf::stop_term(int u) {
  static const bool no_pd = std::getenv("PTCTLMC_GFPNOPD") != nullptr;
  return adj[u].empty() || time_diverges(node[u]) ||
         (!no_pd && state_has_deadlock(net, node[u]));
}

// Descending greatest-fixpoint iteration CONFINED TO S. Nodes outside S are pinned to
// empty, which is what makes the limit a post-fixpoint of the full operator rather than
// merely of this one. S must contain only EXPANDED !q nodes.
void Otf::gfp_on(const std::vector<int>& S, std::vector<Federation>& W) {
  ++passes;
  const std::size_t N = node.size();
  std::vector<char> inS(N, 0);
  for (int v : S) inS[v] = 1;
  W.assign(N, Federation{});
  std::vector<Federation> nosucc(N);
  for (int v : S) {
    W[v].push_back(node[v].zone);
    // Exact by default; PTCTLMC_OTFWHOLEZONESTOP restores the old convention for A/B.
    static const bool whole_zone = std::getenv("PTCTLMC_OTFWHOLEZONESTOP") != nullptr;
    if (!whole_zone) nosucc[v] = stop_set(v);
    else if (stop_term(v)) nosucc[v].push_back(node[v].zone);
  }
  // ROUND-ROBIN, and it is not an oversight -- a worklist was tried and REVERTED.
  //
  // The obvious optimisation is a true chaotic worklist: W(v) reads only its successors, so
  // re-queue v exactly when one of them shrinks (`pred` is maintained for it). Measured, that
  // is a NET LOSS twice over:
  //   * it saves nothing where it should. `fischer7 P0.req --> P0.wait` under the hull goes
  //     825891 -> 824775 Pre calls, 0.1%. Every node genuinely needs recomputing, because the
  //     values do propagate: there is no wasted sweep to remove.
  //   * it REGRESSES grc4 `T0.in --> T0.far` from SATISFIED (128 states) to UNUSABLE. Changing
  //     the order does not change the limit of a monotone descending iteration, but it does
  //     change the intermediate FEDERATION SHAPES, and the width cap is a resource bound on
  //     those. Letting some nodes descend far ahead of their neighbours fragments the
  //     federations enough to trip the cap at 5147 Pre calls where the lockstep sweep converges
  //     at 56892.
  // The lockstep sweep is doing real work as a de-facto widening-free normaliser. Do not
  // "optimise" it again without measuring the cap.
  bool changed = true;
  std::size_t it = 0;
  while (changed && !capped) {
    if (cancelled()) return;
    changed = false;
    ++it;
    for (int v : S) {
      if (gfp_fed_empty(W[v])) continue;
      Federation cand = nosucc[v];
      for (const auto& e : adj[v]) {
        const int j = e.tgt;
        if (!inS[j] || gfp_fed_empty(W[j])) continue;
        for (const auto& Y : W[j]) {
          ++pre_calls;
          Dbm pz = gfp_pre(net, e.tag, node[v], node[j], Y);
          if (!pz.empty()) cand.push_back(std::move(pz));
        }
      }
      gfp_fed_compact(cand, fedcap, &capped);
      // z /\ cand, DIRECTLY.
      //
      // The published form is `fed_subtract(z, fed_subtract(z, cand))`, which denotes the same
      // set but computes it as a double complement: the inner subtraction peels z by every
      // constraint of every member of cand, so its output is an ARRANGEMENT of |cand| zones --
      // its size is not bounded by |cand| at all -- and the outer subtraction then peels z by
      // all of those. The intersection of two zones is a single zone (dbm_intersect is
      // elementwise min plus a close), so `z /\ cand` needs exactly |cand| pieces:
      Federation nw;
      for (const auto& z : W[v]) {
        for (const auto& c : cand) {
          Dbm t = z;
          t.intersect(c);
          if (!t.empty()) nw.push_back(std::move(t));
        }
      }
      gfp_fed_compact(nw, fedcap, &capped);
      if (!gfp_fed_subset(W[v], nw)) changed = true;
      W[v].swap(nw);
      if (capped) break;
    }
    if (it > 100000) break;
  }
  gfp_iters += it;
}

// A certificate over S: some p /\ !q node in S keeps a non-empty value once the confined
// iteration has CONVERGED. Intermediate iterates are pre-fixpoints and say nothing, so the
// test may only be applied to the limit -- which is why gfp_on runs to fixpoint before
// this looks at anything.
//
// A CAPPED pass has not converged, so it proves nothing IN EITHER DIRECTION. Returning
// `false` here would launder "I ran out of federation budget" into "no certificate", i.e.
// into a silent SATISFIED -- the worst possible failure mode for a verification tool. The
// `capped` flag is therefore sticky and the caller must report UNUSABLE rather than a
// verdict; see Otf::run and the printout in main_gfp.cpp.
bool Otf::certificate_in(const std::vector<int>& S) {
  if (S.empty()) return false;
  std::vector<Federation> W;
  gfp_on(S, W);
  std::size_t tot = 0, ne = 0;
  for (int v : S) {
    if (gfp_fed_empty(W[v])) continue;
    ++ne;
    tot += W[v].size();
    if (W[v].size() > fed_max) fed_max = W[v].size();
  }
  nonempty = ne;
  fed_avg = ne ? double(tot) / double(ne) : 0.0;
  if (capped) return false;                      // caller checks `capped` before believing this
  for (int v : S)
    if (is_p(v) && !gfp_fed_empty(W[v])) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Forward construction, shared. Level-synchronous BFS, the same shape as leads_to_gfp's
// phase 1. `trigger` is called after every level and may settle the query.
bool Otf::build_all() {
  SymState s0;
  s0.locs.resize(net.nprocs());
  for (int pr = 0; pr < net.nprocs(); ++pr) s0.locs[pr] = net.procs[pr].initial;
  s0.ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0.ints[v] = net.intvars[v].init;
  s0.zone = Dbm(net.nclocks);
  {
    std::vector<Constraint> inv;
    all_invariants(net, s0.locs, inv);
    if (!apply_all(s0.zone, inv)) return false;     // unreachable init: caller sees no nodes
    bool c, u;
    frozen_flags(net, s0.locs, &c, &u);
    if (!c && !u) {
      s0.zone.delay();
      apply_all(s0.zone, inv);
    }
    reset_inactive(net, s0.locs, s0.zone, qmask);
    if (emode == ExLuPlus) s0.zone.extrapolate_lu_plus(L, U);
    else if (emode == ExNoDiag) s0.zone.extrapolate_lu_plus_nodiag(L, U);
  }
  if (amap_) {
    std::vector<SymState> pieces;
    split_by_atoms(std::move(s0), pieces);
    if (pieces.size() > 1) splits_ += pieces.size() - 1;
    for (auto& pc2 : pieces) intern(std::move(pc2));
  } else {
    intern(std::move(s0));
  }
  return true;
}


// The !q nodes that can actually influence the verdict: those forward-reachable, THROUGH !q
// EDGES ONLY, from a p /\ !q node. F is defined per node from its own successors, so the
// value at an accepting node depends on exactly this set and on nothing else -- restricting
// to it is EXACT, not an approximation, and it is what keeps the final pass from iterating
// over !q states no accepting state can ever reach. (`p --> q` is decided by the accepting
// nodes alone, so nothing outside the closure is ever read.)
std::vector<int> Otf::accepting_closure() {
  grow();
  const std::size_t N = node.size();
  std::vector<char> in(N, 0);
  std::vector<int> stk, out;
  for (std::size_t v = 0; v < N; ++v)
    if (expanded[v] && is_nq((int)v) && is_p((int)v)) { in[v] = 1; stk.push_back((int)v); }
  while (!stk.empty()) {
    const int v = stk.back();
    stk.pop_back();
    out.push_back(v);
    for (const auto& e : adj[v])
      if (!in[e.tgt] && expanded[e.tgt] && is_nq(e.tgt)) { in[e.tgt] = 1; stk.push_back(e.tgt); }
  }
  return out;
}

// One exact pass over the FINISHED graph: S = the accepting closure, on which F_S = F at
// every accepting node, so the limit is the true greatest fixpoint where it is read. This is
// the published algorithm, and every mode falls back to it when no certificate fired early.
bool Otf::final_verdict_violated() {
  return certificate_in(accepting_closure());
}

// THE DEADLOCK-COMPLETION CERTIFICATE.
//
// Deadlock is a negative property, so a_LU covering can lose a deadlock witness. Rather than
// weaken the covering to a_M or pay a second search (the gate/race), put the deadlock INTO the
// automaton: add a sink with an edge guarded by Dead(q) at every q. The completed automaton is
// deadlock-free by construction, so a maximal run that ENDS in a deadlock becomes a run that
// REACHES the sink -- an existential property, which any simulation preserves. The completion is
// never materialised; its only effect is on which atoms the constraint map has to carry, namely
// the atoms of Dead(q), closed under pre.
//
// Those atoms may be dropped at q when Dead(q) is EMPTY (nothing to preserve), when it is ALL of
// Inv(q) (its atoms are already carried by the invariant), or when Inv(q) bounds no clock (time
// is open, so the divergence disjunct of stop already covers the whole zone). If every reachable
// key falls into one of those three cases the seed is empty everywhere, the constraint map is
// the ordinary one, and PLAIN a_LU COVERING IS ALREADY SOUND -- no gate, no second search, and
// no weakening to a_M.
//
// This sweep decides exactly that. It is STRUCTURAL: it asks about Dead(q) over all of Inv(q),
// not over any reachable zone, so it needs no search -- one federation computation per DISCRETE
// KEY (256 of them on arbiter6, against the 28591-state reachability traversal the gate's
// deadlock-freedom lane used to pay). Enumerating the keys from the a_LU graph is sound because
// a simulation never loses a reachable discrete key: the covering zone does at least what the
// covered one does, so every key the covered zone could reach is still generated.
bool Otf::dtheta_empty() {
  std::unordered_set<std::string> seen;
  for (std::size_t i = 0; i < node.size(); ++i) {
    const std::string k = disc_key(node[i].locs, node[i].ints);
    if (!seen.insert(k).second) continue;
    SymState u; u.locs = node[i].locs; u.ints = node[i].ints;
    u.zone = Dbm::universe(net.nclocks);
    Dbm inv = Dbm::universe(net.nclocks);
    bool c2, u2; frozen_flags(net, u.locs, &c2, &u2);
    std::vector<Constraint> ci; all_invariants(net, u.locs, ci);
    if (!c2 && !u2) inv.delay();
    if (!apply_all(inv, ci)) continue;                        // vacuous key
    Federation stuck, esc;
    if (!state_has_deadlock(net, u, &stuck, &esc)) continue;   // Dead(q) = empty
    bool open = true;
    for (int c = 1; c <= net.nclocks; ++c)
      if (inv.get(c, 0) != kInf) { open = false; break; }
    // (iii) is a claim about TIME DIVERGENCE, so it is only available where time can pass. At a
    // committed/urgent key no positive delay is legal, Inv(q) is the undelayed universe and so
    // trivially bounds no clock -- which is exactly when this test would fire, and exactly when
    // it is wrong. Frozen keys fall through to (ii)/PARTIAL.
    if (open && !c2 && !u2) continue;                          // divergence already covers stop
    // Inv \ Dead = Inv ^ union Esc, so "Dead(q) = Inv(q)" is "nothing in Inv(q) escapes" --
    // one DBM intersection per outgoing transition instead of peeling Inv(q) by the residue
    // ARRANGEMENT. Same predicate; see the twin test in build_atom_map for why it matters.
    bool some_escapes = false;
    for (const Dbm& W : esc) {
      Dbm t = W;
      t.intersect(inv);
      if (!t.empty()) { some_escapes = true; break; }
    }
    static const bool selfcheck = std::getenv("PTCTLMC_DLSELFCHECK") != nullptr;
    if (selfcheck && fed_subtract(inv, stuck).empty() == some_escapes) {
      std::fprintf(stderr, "[selfcheck] dtheta_empty Dead=Inv MISMATCH at key %s\n", k.c_str());
      std::abort();
    }
    if (!some_escapes) continue;                               // Dead(q) = Inv(q): atoms implied
    return false;                                              // PARTIAL: real atoms are needed
  }
  return true;
}


// BUILD THE DEADLOCK-COMPLETION CONSTRAINT MAP from this run's finished graph.
//
// SEED. Only a key with a genuinely PARTIAL Dead(q) contributes. The three degenerate cases
// dtheta_empty() already decides carry no information and are skipped: Dead(q) empty (there is
// no sink edge), Dead(q) = Inv(q) (the sink edge is always enabled and every stored zone
// satisfies Inv(q)), and Inv(q) time-open at a NON-frozen key (the divergence disjunct of stop
// already admits the whole zone). The frozen guard is not optional -- at a committed/urgent key
// Inv(q) is the undelayed universe and so trivially bounds no clock, which is exactly where
// that reasoning fails; see Change 5b.
//
// The atoms are read straight off the CANONICAL DBMs of the Dead(q) federation rather than
// derived from the guards. Dead = \/_i /\_j c_ij, so the truth values of the c_ij decide
// membership exactly and every atom family comes for free -- including the lower bound that
// survives Down(), which dbm_down re-derives when it re-closes after zeroing row 0, and which a
// hand-derived formula for Down(K) got wrong in half the cases.
//
// Atoms IMPLIED BY Inv(q) are dropped: every stored zone satisfies Inv(q), so such an atom is
// constantly true on everything the search will ever compare and distinguishes nothing. On a
// canonical DBM the test is one entry comparison.
//
// PRE-CLOSURE. The map must satisfy [R]^-1(G(q')) subseteq G(q) for every edge (q,g,R,q'), or
// two zones sharing a cell at q could step into different cells at q'. The pullback of
// x_i - x_j <| b through a reset set R is the index substitution i,j -> 0 for reset clocks --
// the same rule state_has_deadlock uses for target invariants. A pullback that lands on clock 0
// is a NON-diagonal atom and is discharged by the bound raise instead. Dropping an atom at the
// target only ever RELAXES the clause at the source, so the three seed drops above stay safe
// under closure. Guard atoms need no seeding: a_LU is already a simulation for the ORIGINAL
// automaton, and the only edge this construction adds is the sink's.
bool Otf::build_atom_map(AtomMap& am) {
  const int nc = net.nclocks;
  am.diag.clear();
  am.bump.assign(nc + 1, kNoBump);
  am.live.assign(nc + 1, 0);
  am.usable = false;
  if (nc <= 0 || nc >= 63) return false;

  const char* capenv = std::getenv("PTCTLMC_OTFATOMCAP");
  // HARD 64. The per-node cell tag is a uint64 bitmask, so an atom past bit 63 would be
  // silently dropped from the covering test while still splitting zones -- a lost obligation,
  // i.e. a wrong verdict, not merely a coarseness loss. Exceeding the cap returns "unusable"
  // and falls back to the race, which is always correct.
  //
  // THE DEFAULT IS THE HARD LIMIT, not half of it. It used to be 32, which cost nothing while
  // the construction was expensive -- declining early meant declining into the race, and the
  // race was competitive. It is not competitive any more, and 32 was where grc5 fell out:
  // one key there needs 36 seed atoms, so the map was declared unusable and grc5 ran the race
  // (20.9s) or a_M (18.5s) when the construction does it in 3.4s. There is no reason to
  // decline below the point where the representation actually breaks.
  const std::size_t kMaxAtoms =
      std::min<std::size_t>(capenv ? std::strtoul(capenv, nullptr, 10) : 64, 64);

  std::unordered_map<std::string, Dbm> invof;
  const auto t_seed0 = std::chrono::steady_clock::now();

  // ---- seed ---------------------------------------------------------------
  for (std::size_t i = 0; i < node.size(); ++i) {
    const std::string k = disc_key(node[i].locs, node[i].ints);
    if (invof.count(k)) continue;
    SymState u;
    u.locs = node[i].locs;
    u.ints = node[i].ints;
    u.zone = Dbm::universe(nc);                 // NOT Dbm(nc): that is the ZERO zone (Change 5a)
    Dbm inv = Dbm::universe(nc);
    bool fc, fu;
    frozen_flags(net, u.locs, &fc, &fu);
    std::vector<Constraint> ci;
    all_invariants(net, u.locs, ci);
    if (!fc && !fu) inv.delay();
    if (!apply_all(inv, ci)) continue;                       // vacuous key
    invof.emplace(k, inv);
    Federation stuck, esc;
    if (!state_has_deadlock(net, u, &stuck, &esc)) continue;  // Dead(q) = empty
    bool open = true;
    for (int c = 1; c <= nc; ++c)
      if (inv.get(c, 0) != kInf) { open = false; break; }
    if (open && !fc && !fu) continue;                         // divergence already covers stop
    // Dead(q) = Inv(q)? Written as `fed_subtract(inv, stuck).empty()` this peels Inv(q) once
    // per piece of the RESIDUE, and the residue is an arrangement -- 472 pieces at grc4's one
    // partial key, which is 126ms, two thirds of the whole structural sweep and the single
    // largest term in the atom construction's bill. It never needed the residue at all:
    //     Inv \ Dead = Inv \ (Inv \ union Esc) = Inv ^ union Esc,
    // so the test is "does NO valuation of Inv(q) escape", which is |Esc| independent DBM
    // intersections -- one per outgoing transition, five on that key.
    //
    // THE PREMISE IS INCLUSION, NOT EQUALITY, and the difference is real. `stuck` is
    // `Z \ union esc` for the `Z` state_has_deadlock built, and that `Z` is NOT always this
    // `inv`: at a FROZEN key state_has_deadlock skips both the delay and the invariant, so its
    // Z is the bare universe while `inv` here has the invariant applied. What the identity
    // needs is only `inv subseteq Z` --
    //     x in inv \ (Z \ E)  iff  x in inv and not (x in Z and x notin E)  iff  x in inv ^ E,
    // using x in inv => x in Z -- and that holds both ways round: equal when not frozen, and
    // `inv = universe ^ Inv subseteq universe = Z` when frozen. PTCTLMC_DLSELFCHECK runs the
    // old federation form alongside and aborts on any disagreement; it is clean across the
    // 48-query differential and the three liveness regressions.
    //
    // ESCAPES THAT MISS Inv(q) ARE NOT ESCAPES. Dead(q) = Inv(q) \ union Esc, so a transition
    // whose escape region does not meet Inv(q) subtracts nothing and its atoms decide nothing
    // -- yet every one of its constraints would be seeded, pulled back through the whole key
    // graph, and split against. Drop those members here, where the intersection is being
    // computed for the Dead=Inv test anyway.
    bool some_escapes = false;
    {
      Federation kept;
      kept.reserve(esc.size());
      for (const Dbm& W : esc) {
        Dbm t = W;
        t.intersect(inv);
        if (t.empty()) { ++am.dropped_vacuous; continue; }
        some_escapes = true;
        kept.push_back(W);
      }
      esc.swap(kept);
    }
    // PTCTLMC_DLSELFCHECK: run the old federation form alongside and abort on disagreement.
    // Kept because the rewrite is the kind of set identity that is obviously true on paper and
    // silently wrong if `esc` is ever not the federation `stuck` was subtracted from.
    static const bool selfcheck = std::getenv("PTCTLMC_DLSELFCHECK") != nullptr;
    if (selfcheck) {
      const bool old_eq_inv = fed_subtract(inv, stuck).empty();
      if (old_eq_inv == some_escapes) {
        std::fprintf(stderr, "[selfcheck] Dead=Inv MISMATCH at key %s: old=%d new=%d\n",
                     k.c_str(), int(old_eq_inv), int(!some_escapes));
        std::abort();
      }
    }
    if (!some_escapes) continue;                              // Dead(q) = Inv(q): atoms implied
    ++am.seed_keys;
    // WHICH ZONES TO READ THE ATOMS OFF -- two exact representations of the same set.
    //
    // Dead(q) = Inv(q) \ union Esc(q), and every stored zone satisfies Inv(q). So membership in
    // Dead(q) is decided EITHER by the atoms of the residue's disjoint pieces (Dead = \/_i /\_j
    // c_ij) OR by the atoms of the ESCAPE zones alone -- one convex zone per outgoing transition
    // -- since nu is dead iff it escapes by none of them. Both are exact; neither dominates.
    // The residue's cut constraints are artefacts of the peeling order and blow up on grc4
    // (3234 occurrences against 25 for the escapes), while the escapes carry diagonals that pull
    // back through resets into non-diagonal atoms and cost csma four clock bounds the residue
    // does not. So compute both and keep the cheaper, counting a non-diagonal atom heavier
    // because it is discharged by raising that clock to M GLOBALLY, not by a local cell.
    //
    // MINIMAL GRAPH, not the dense matrix. A canonical DBM's redundant entries are implied by
    // shortest paths through the others, so the non-redundant edges alone decide membership --
    // mindbm's round trip reproduces the canonical matrix exactly. Reading the dense matrix
    // seeds one atom per DERIVED bound and buys nothing.
    const int dim = nc + 1;
    std::vector<std::uint32_t> mbits(mindbm_bits2words(std::uint32_t(dim) * std::uint32_t(dim)));
    std::vector<int> mfirst(dim), mnext(dim);
    auto extract = [&](const Federation& f, std::vector<AtomC>& g,
                       std::vector<std::pair<int, std::int32_t>>& nd) {
      for (const Dbm& d : f) {
        if (d.empty()) continue;
        mindbm_analyze(d.data().data(), dim, mbits.data(), mfirst.data(), mnext.data());
        for (int a = 0; a < dim; ++a)
          for (int b2 = 0; b2 < dim; ++b2) {
            if (a == b2) continue;
            const std::uint32_t idx = std::uint32_t(a) * std::uint32_t(dim) + std::uint32_t(b2);
            if (!((mbits[idx >> 5] >> (idx & 31)) & 1u)) continue;   // redundant: implied
            const raw_t r = d.get(a, b2);
            if (r >= kInf) continue;
            if (a == 0 && r >= kZeroLe) continue;               // x_b2 >= 0: trivial
            // BOTH CONSTANT CASES, not just one. Every stored zone at q satisfies Inv(q), so an
            // atom Inv(q) IMPLIES is true on all of them and an atom Inv(q) is DISJOINT from is
            // false on all of them; either way its tag bit never varies and it distinguishes
            // nothing. Only the first was being dropped. On a canonical DBM both are one entry
            // comparison: implied is inv(a,b2) <= r, disjoint is inv(b2,a) + r < (<=,0).
            if (inv.get(a, b2) <= r) continue;                  // implied by Inv(q): always true
            if (raw_add(inv.get(b2, a), r) < kZeroLe) {         // disjoint from Inv(q): never
              ++am.dropped_disjoint;
              continue;
            }
            if (a > 0 && b2 > 0) {
              const AtomC at{a, b2, r};
              if (std::find(g.begin(), g.end(), at) == g.end()) g.push_back(at);
            } else {
              const int cl = (a == 0) ? b2 : a;
              const std::int32_t cst = (a == 0) ? -raw_bound(r) : raw_bound(r);
              bool seen = false;
              for (auto& e2 : nd)
                if (e2.first == cl) { e2.second = std::max(e2.second, cst); seen = true; break; }
              if (!seen) nd.push_back({cl, cst});
            }
          }
      }
    };
    static const bool force_res = std::getenv("PTCTLMC_OTFATOMSRESIDUE") != nullptr;
    static const bool force_esc = std::getenv("PTCTLMC_OTFATOMSESCAPE") != nullptr;
    std::vector<AtomC> gr, ge;
    std::vector<std::pair<int, std::int32_t>> nr, ne2;
    if (!force_esc) extract(stuck, gr, nr);
    if (!force_res) extract(esc, ge, ne2);
    const std::size_t cr = gr.size() + 4 * nr.size();
    const std::size_t ce = ge.size() + 4 * ne2.size();
    const bool take_esc = force_esc || (!force_res && ce <= cr);
    std::vector<AtomC>& g = take_esc ? ge : gr;
    for (const auto& e2 : (take_esc ? ne2 : nr)) {
      ++am.nondiag_atoms;
      if (e2.second > am.bump[e2.first]) am.bump[e2.first] = e2.second;
    }
    am.seed_atoms += g.size() + (take_esc ? ne2.size() : nr.size());
    if (!g.empty()) {
      if (g.size() > kMaxAtoms) return false;
      am.diag.emplace(k, std::move(g));
    }
  }

  const double ms_seed = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t_seed0).count();

  // NOTHING TO CLOSE. The pre-closure propagates atoms BACKWARDS from keys that carry them, so
  // with no seed anywhere it has nothing to move and the key-edge relation it would move them
  // along need not be built at all. That relation is a string key and a reset mask per graph
  // EDGE -- on fischer7, 9798 nodes' worth -- and building it is the only work this function
  // does that the certificate it replaced did not. Returning here is what makes the nine
  // empty-seed models a literal no-op rather than a no-op plus a sweep.
  {
    bool any_seed = !am.diag.empty();
    for (int c = 1; c <= nc && !any_seed; ++c)
      if (am.bump[c] != kNoBump) any_seed = true;
    if (!any_seed) {
      if (std::getenv("PTCTLMC_OTFATOMSTAT"))
        std::fprintf(stderr, "[atoms] time: seed-sweep=%.1fms (empty seed: no closure)\n", ms_seed);
      am.usable = true;
      return true;
    }
  }

  // ---- the key-edge relation, with the union of each transition's resets --
  struct KE { std::string s, t; std::uint64_t r; };
  std::vector<KE> kedges;
  {
    std::unordered_set<std::string> seenE;
    const std::size_t n = std::min(adj.size(), node.size());
    for (std::size_t u = 0; u < n; ++u) {
      const std::string ks = disc_key(node[u].locs, node[u].ints);
      for (const auto& e : adj[u]) {
        std::uint64_t rm = 0;
        for (const auto& pr : e.tag.parts)
          for (int rc : net.procs[pr.first].edges[pr.second].clock_resets)
            if (rc >= 1 && rc < 64) rm |= std::uint64_t(1) << rc;
        const std::string kt = disc_key(node[e.tgt].locs, node[e.tgt].ints);
        std::string sig = ks;
        sig += '\x01';
        sig += kt;
        sig.append(reinterpret_cast<const char*>(&rm), sizeof(rm));
        if (!seenE.insert(sig).second) continue;
        kedges.push_back(KE{ks, kt, rm});
      }
    }
  }

  // ---- pre-closure fixpoint ----------------------------------------------
  bool changed = true;
  while (changed) {
    changed = false;
    if (++am.rounds > 4096) return false;
    for (const KE& e : kedges) {
      auto it = am.diag.find(e.t);
      if (it == am.diag.end()) continue;
      const std::vector<AtomC> tgt = it->second;   // copy: am.diag may rehash below
      for (const AtomC& a : tgt) {
        const int i2 = (a.i > 0 && ((e.r >> a.i) & 1)) ? 0 : a.i;
        const int j2 = (a.j > 0 && ((e.r >> a.j) & 1)) ? 0 : a.j;
        if (i2 == j2) continue;                    // both reset: a constant, not an atom
        if (i2 > 0 && j2 > 0) {
          auto iv = invof.find(e.s);
          if (iv != invof.end()) {
            if (iv->second.get(i2, j2) <= a.b) continue;                      // Inv(q) implies
            // ...and, symmetrically, Inv(q) refutes: the pullback lands outside the source
            // key's invariant, so no zone stored there can ever satisfy it. This is where the
            // drop earns its keep -- a closure atom is a constraint from SOMEWHERE ELSE's
            // deadlock, and the further it is pulled back the less it has to do with the
            // invariant it arrives at.
            if (raw_add(iv->second.get(j2, i2), a.b) < kZeroLe) {
              ++am.dropped_disjoint;
              continue;
            }
          }
          std::vector<AtomC>& g = am.diag[e.s];
          const AtomC at{i2, j2, a.b};
          if (std::find(g.begin(), g.end(), at) == g.end()) {
            if (g.size() >= kMaxAtoms) return false;
            g.push_back(at);
            ++am.closure_atoms;
            changed = true;
          }
        } else {
          const int cl = (i2 == 0) ? j2 : i2;
          const std::int32_t cst = (i2 == 0) ? -raw_bound(a.b) : raw_bound(a.b);
          if (cst > am.bump[cl]) {
            am.bump[cl] = cst;
            ++am.closure_atoms;
            changed = true;
          }
        }
      }
    }
  }

  // IF EVERY CLOCK ENDED UP AT M, THE CELLS ARE REDUNDANT. With L(x) = U(x) = M(x) for every
  // clock the LU preorder collapses to "equal, or both above M", which is SYMMETRIC, hence a
  // bisimulation, hence deadlock-preserving on its own (Behrmann-Bouyer-Larsen-Pelanek 2006,
  // Lemma 1). Keeping the cell tags on top of that would only split zones a_M already keeps
  // apart. Dropping them here is what stops the construction from ever being WORSE than the
  // a_M fallback it is meant to beat.
  {
    bool all_at_m = true;
    for (int c = 1; c <= nc && all_at_m; ++c) {
      const std::int32_t mc = std::max(net.lbound[c], net.ubound[c]);
      if (am.bump[c] < mc) all_at_m = false;
    }
    if (all_at_m) am.diag.clear();
  }

  // PTCTLMC_OTFATOMSLANE=cell|bump -- DIAGNOSTIC, and each setting is UNSOUND on its own.
  //
  // The map has two lanes and they discharge different atoms: cells carry the diagonals, bumps
  // carry the clock bounds. Running one without the other answers "which lane is actually
  // load-bearing on this model, and which one is paying the bill" -- a question the combined
  // figures cannot answer, because a pre-closure that turns a diagonal into a bump at some
  // predecessor makes the two lanes cover for each other.
  if (const char* lane = std::getenv("PTCTLMC_OTFATOMSLANE")) {
    if (std::string(lane) == "cell") am.bump.assign(nc + 1, kNoBump);
    else if (std::string(lane) == "bump") am.diag.clear();
  }

  // ---- active-clock reduction must keep every clock the map reads ---------
  for (const auto& kv : am.diag)
    for (const AtomC& a : kv.second) {
      if (a.i > 0) am.live[a.i] = 1;
      if (a.j > 0) am.live[a.j] = 1;
    }
  for (int c = 1; c <= nc; ++c)
    if (am.bump[c] != kNoBump) { am.live[c] = 1; ++am.bumped; }
  am.keys_tagged = am.diag.size();
  am.usable = true;
  // HOW CLOSE TO a_M DID THE BUMPS LAND? The bump is "a_M on one clock", so a map that bumps
  // every clock to M has already paid the whole a_M price and the cells on top of it are pure
  // loss. The all-at-M shortcut above catches only the total case; this reports the margin, and
  // it is the number that decides whether the construction can pay off on a given model at all.
  if (std::getenv("PTCTLMC_OTFATOMSTAT")) {
    std::fprintf(stderr, "[atoms] time: seed-sweep=%.1fms key-edges=%zu\n",
                 ms_seed, kedges.size());
    std::fprintf(stderr, "[atoms] clocks=%d bump/M:", nc);
    for (int c = 1; c <= nc; ++c)
      std::fprintf(stderr, " x%d=%d/%d", c, am.bump[c],
                   std::max(net.lbound[c], net.ubound[c]));
    std::fprintf(stderr, "\n");
  }
  return true;
}

// The forward half of Base mode. `setup()` must already have run.
void Otf::base_forward() {
      // Control: builds the whole graph and runs one exact pass, i.e. Algorithm 1 through
      // this code path. Kept so the differential can check that the shared machinery
      // reproduces the published node count.
      fwd_ok_ = build_all();
      if (!fwd_ok_) return;
      // DEDUPLICATED WORKLIST, not a level-synchronous frontier.
      //
      // The level-synchronous form pushes a successor whenever `!seen[v] || !expanded[v]`, with
      // no dedup WITHIN a level. That is harmless while `intern` only ever appends, because
      // `expanded` then goes 0 -> 1 once per node. It is quadratic as soon as `intern` MERGES:
      // every enlargement resets `expanded[v]` to 0, so v is re-queued once per in-edge, every
      // level, and the frontier fills with duplicates of the same handful of nodes. That, and
      // not the abstraction, is most of what made the hull lane slow on `fischer`.
      //
      // An `inq` flag makes each node resident at most once. Nothing about WHAT is explored
      // changes -- the closure is the same, so stored-state counts are unaffected -- only how
      // many times it is queued.
      std::deque<int> work;
      std::vector<char> inq;
      auto push = [&](int v) {
        if (inq.size() < node.size()) inq.resize(node.size(), 0);
        if (!inq[v]) { inq[v] = 1; work.push_back(v); }
      };
      // A permanent lazy split detaches nodes ANYWHERE in the graph, not only successors of the
      // node just expanded, so they cannot be recovered by walking adj[u]. relist_ carries them
      // and is drained here -- the one place where no expansion is in flight.
      auto drain_relist = [&]() {
        while (!relist_.empty()) {
          const int v = relist_.back();
          relist_.pop_back();
          grow();
          if (!expanded[v]) push(v);
        }
      };
      push(0);
      drain_relist();
      while (!work.empty()) {
        const int u = work.front();
        work.pop_front();
        grow();
        if (inq.size() < node.size()) inq.resize(node.size(), 0);
        inq[u] = 0;
        expand(u);                                  // no-op if still expanded
        grow();
        // A successor needs work exactly when it has no adjacency of its own: brand new, or
        // reset because a merge enlarged it.
        for (const auto& e : adj[u])
          if (!expanded[e.tgt]) push(e.tgt);
        drain_relist();
      }
      // PTCTLMC_OTFDLSTAT: how much of the graph is actually deadlock-prone?
      //
      // This measures the HEADROOM for any scheme that pays a_M strength only where a deadlock
      // could really be lost. a_M costs 1.3x-10.3x more nodes than a_LU here and the ratio grows
      // with model size; the whole point of a promotion scheme is to pay that only where it is
      // owed. If nearly every discrete key can deadlock, there is nothing to win and a_M is
      // already optimal. If only a handful can, the prize is most of that 10.3x.
      // PTCTLMC_OTFDLSTRUCT: classify Dead(q) STRUCTURALLY, over all of Inv(q) rather than over
      // the reachable zone. This is what decides the deadlock-completed constraint map: the seed
      // D(q) can be dropped whenever Dead(q) is EMPTY (nothing to preserve), whenever it is ALL
      // of Inv(q) (its atoms are already carried by the invariant), and whenever Inv(q) bounds no
      // clock (time-open, so the divergence disjunct of stop already covers the whole zone).
      // Only a PARTIAL Dead(q) forces new atoms -- and only there does the construction cost
      // anything against plain a_LU.
      if (std::getenv("PTCTLMC_OTFDLSTRUCT")) {
        std::unordered_map<std::string, int> cls;   // 0 empty, 1 full, 2 time-open, 3 PARTIAL
        for (std::size_t i = 0; i < node.size(); ++i) {
          const std::string k = disc_key(node[i].locs, node[i].ints);
          if (cls.count(k)) continue;
          SymState u; u.locs = node[i].locs; u.ints = node[i].ints;
          u.zone = Dbm::universe(net.nclocks);
          // Inv(q): the delay-closure of the universe under this key's real invariants.
          Dbm inv = Dbm::universe(net.nclocks);
          bool c2, u2; frozen_flags(net, u.locs, &c2, &u2);
          std::vector<Constraint> ci; all_invariants(net, u.locs, ci);
          if (!c2 && !u2) { inv.delay(); }
          if (!apply_all(inv, ci)) { cls[k] = 0; continue; }   // vacuous key
          bool open = true;
          for (int c = 1; c <= net.nclocks; ++c) if (inv.get(c, 0) != kInf) { open = false; break; }
          Federation stuck;
          if (!state_has_deadlock(net, u, &stuck)) { cls[k] = 0; continue; }
          if (open) { cls[k] = 2; continue; }
          bool full = true;                                    // Dead(q) covers all of Inv(q)?
          Dbm rem = inv;
          for (const auto& d : stuck) { if (fed_subtract(rem, Federation{d}).empty()) { break; } }
          full = fed_subtract(inv, stuck).empty();
          cls[k] = full ? 1 : 3;
        }
        int e = 0, f = 0, o = 0, pa = 0;
        for (const auto& kv : cls)
          (kv.second == 0 ? e : kv.second == 1 ? f : kv.second == 2 ? o : pa)++;
        std::fprintf(stderr,
                     "[dlstruct] keys=%zu  Dead=empty:%d  Dead=full:%d  time-open:%d  "
                     "PARTIAL:%d (%.1f%%)\n",
                     cls.size(), e, f, o, pa,
                     cls.empty() ? 0.0 : 100.0 * double(pa) / double(cls.size()));
      }
      if (std::getenv("PTCTLMC_OTFDLSTAT")) {
        std::unordered_map<std::string, char> key_dl;   // discrete key -> any node deadlock-prone
        std::size_t dl_nodes = 0, whole = 0, partial = 0;
        for (std::size_t i = 0; i < node.size(); ++i) {
          const bool w = adj[i].empty();
          const bool pd = !w && state_has_deadlock(net, node[i]);
          if (w) ++whole;
          if (pd) ++partial;
          const std::string k = disc_key(node[i].locs, node[i].ints);
          auto it = key_dl.find(k);
          if (it == key_dl.end()) key_dl.emplace(k, (w || pd) ? 1 : 0);
          else if (w || pd) it->second = 1;
          if (w || pd) ++dl_nodes;
        }
        std::size_t dl_keys = 0;
        for (const auto& kv : key_dl) dl_keys += kv.second ? 1 : 0;
        std::fprintf(stderr,
                     "[dlstat] nodes=%zu dl-nodes=%zu (%.1f%%) whole=%zu partial=%zu | "
                     "keys=%zu dl-keys=%zu (%.1f%%)\n",
                     node.size(), dl_nodes,
                     node.empty() ? 0.0 : 100.0 * double(dl_nodes) / double(node.size()),
                     whole, partial, key_dl.size(), dl_keys,
                     key_dl.empty() ? 0.0 : 100.0 * double(dl_keys) / double(key_dl.size()));
      }
}

void Otf::build_forward(GfpStats* gs) {
  setup();
  const auto t0 = std::chrono::steady_clock::now();
  base_forward();
  if (std::getenv("PTCTLMC_OTFATOMSTAT"))
    std::fprintf(stderr, "[atoms]   (forward=%.1fms nodes=%zu)\n",
                 std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0).count(),
                 node.size());
  fill_stats(gs);
}

bool Otf::finish_verdict(GfpStats* gs) {
  const auto t0 = std::chrono::steady_clock::now();
  const bool violated = fwd_ok_ && final_verdict_violated();
  if (std::getenv("PTCTLMC_OTFATOMSTAT"))
    std::fprintf(stderr, "[atoms]   (backward=%.1fms)\n",
                 std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0).count());
  fill_stats(gs);
  return !violated;
}

bool Otf::run(GfpStats* gs) {
  setup();
  base_forward();
  const bool violated = fwd_ok_ && final_verdict_violated();
  fill_stats(gs);
  return !violated;
}

void Otf::fill_stats(GfpStats* gs) {
  if (std::getenv("PTCTLMC_OTFZDBG")) zdbg();
  if (gs) {
    gs->fwd_nodes = node.size();
    gs->disc_keys = bucket.size();
    gs->fwd_edges = 0;
    for (const auto& a : adj) gs->fwd_edges += a.size();
    gs->iters = gfp_iters;
    gs->capped = capped;
    gs->fed_max = fed_max;
    gs->fed_avg = fed_avg;
    gs->nonempty = nonempty;
    gs->expanded_nodes = expand_calls;
    gs->passes = passes;
    gs->pre_calls = pre_calls;
    gs->early = settled_early;
    // D6 over-approximates, so a non-empty fixpoint is NOT a witness -- it may rest entirely on
    // hull points the system never reaches. Only the empty direction concludes.
    // Do NOT count the initial state, matching verifyta. Its `E<> p` figures are exactly our
    // count minus one on every query measured (5 against 6, 6 against 7, 9 against 10), which
    // pins the convention: it reports the states GENERATED, not the root it started from.
    gs->outer_nodes = outer_nodes;
    gs->outer_expands = outer_expands;
  }
}

// Entry point. `mode` comes from PTCTLMC_GFPMODE; returns true iff `p --> q` HOLDS.
// DEADLOCK-FREEDOM GATE, AND THE RACE THAT DISCHARGES IT.
//
// SUPERSEDED AS THE DEFAULT, kept as the fallback. What follows is the original argument and
// it is still exactly why the race is correct -- but the race now runs only when the
// deadlock-completion constraint map cannot be represented (more than 64 atoms at one key).
// The default is: build the forward graph, derive the map, and either return phase 1's own
// verdict (empty map -- the certificate) or search once more under the refined covering. The
// order below, "phase 1 to a verdict, then discharge", is therefore no longer what happens on
// the default path: phase 1 stops after its FORWARD half, because the refined search would
// otherwise repeat its backward fixpoint. See Otf::build_forward.
//
// a_LU covering is a SIMULATION. It preserves the positive witnesses -- an infinite run of
// discrete steps, and time-divergence -- unconditionally, because a simulating valuation does
// at least what the simulated one does. It does NOT preserve the third witness, DEADLOCK,
// which is "enables nothing": a universally quantified negative property that simulation says
// nothing about. So `stop`'s deadlock disjunct may be LOST under a_LU covering.
//
// The error is one-sided, and that is the whole lever. Stored zones are exact (see the note in
// Otf::setup), and pre/stop are exact concrete operations, so a witness FOUND under a_LU sits
// at a genuinely reachable valuation and is genuine. Only the ABSENCE of a witness is
// inconclusive. Hence:
//
//   phase 1  run under a_LU. Violated => done, and the gate is never paid at all. This is the
//            common case for counterexample queries, which is exactly where this engine is
//            far ahead of verifyta -- hoisting the gate in front of it would make every such
//            query pay for a full reachability traversal whose answer it never needs.
//   phase 2  no witness. Two ways to discharge the obligation, deciding the SAME proposition:
//              (A) prove the model deadlock-free -- then nothing could have been dropped, so
//                  the a_LU answer was sound after all;
//              (B) redo the search with exact-inclusion covering, which is sound outright.
//
// NEITHER DOMINATES and which one wins is not predictable: guessing (A) is cheap means
// deciding deadlock-freedom, and guessing (B)'s size is the query itself. Measured symbolic
// states in the shipping engine: fischer7 is 180420 for (A) against 30994 for (B); arbiter6 is
// 28591 against 66393. A factor of 5.8 one way, 2.3 the other.
//
// So run both and take the first CONCLUSIVE answer, cancelling the loser. What the cores buy
// here is not speed -- neither lane runs faster than it would alone -- but the removal of an
// undecidable choice. A sequential tool must gamble and can lose by 5.8x.
//
// BOTH LANES ARE CANCELLABLE, and that is the difference from the shipping engine's version at
// src/netreach.cpp:3486. There, `search()` never reads the stop flag, so `lane_b.join()` waits
// for lane B unconditionally and the wall clock is always lane B's -- lane A can only ever add
// work. Here lane B polls through Otf::cancelled() (checked in expand(), the choke point for
// every forward step, and at the head of each backward fixpoint sweep), so the pair really does
// finish in min(A, B).
//
// ASYMMETRY TO PRESERVE: lane A is conclusive only when it answers "deadlock-free". "A deadlock
// is reachable" says nothing about THIS query -- the dropped witness may or may not have been
// one -- so that outcome must not publish a verdict and must not cancel lane B.
bool leads_to_otf(const Network& net, const DiscPred& p, const DiscPred& q, GfpStats* gs) {
  // Phase 1.
  Otf e1(net, p, q);

  // Diagonal models never used a_LU (Otf::setup forces inclusion), so phase 1 is already
  // exact: there is nothing to discharge and no atom construction to run.
  if (network_has_diagonal(net)) return e1.run(gs);

  // THE GENERAL CONSTRUCTION. It subsumes the certificate -- an empty map IS dtheta_empty's
  // condition -- so where the certificate used to fire, this is the same structural sweep and
  // the same a_LU search, node for node. Where the certificate DECLINED it carries the model
  // instead of falling back to the race. Best of 3, `Bus.Active --> Bus.Idle` and
  // `T0.in --> T0.far`:
  //
  //     model    certificate   THIS     a_M       a_LU floor (unsound)
  //     csma6      0.138       0.018    0.142     0.012
  //     csma7      3.985       0.053    9.729     0.033
  //     grc4       0.085       0.042    0.056     0.031
  //     grc5      21.936       3.535   19.475     5.815
  //
  // grc5 is the one to look at twice: the REFINED covering beats the coarser one it refines
  // (3.5s against the 5.8s floor), because splitting a zone along an atom of Dead(q) also
  // splits the backward federations that pass through it, and narrower federations converge
  // faster than the arrangement one wide zone produces.
  //
  // The race below survives only for a map that exceeds the 64-bit tag cap.
  //
  // FORWARD ONLY, for now. The atom map is read off this graph; if it turns out non-empty the
  // refined search replaces this run wholesale and phase 1's backward fixpoint -- 80% of its
  // wall clock -- would have been dead work. It is paid below, via e1.finish_verdict(), on
  // exactly the paths that still need phase 1's own verdict.
  e1.build_forward(gs);

  // Phase 2: THE ATOM MAP (see AtomMap and Otf::build_atom_map). Where the deadlock-completion
  // certificate only DECIDES whether the completion's seed is empty, this builds the seed,
  // closes it under pre, and redoes the search under a_LU refined by the resulting cells.
  //
  // The verdict is sound either way round. VIOLATED is genuine outright -- zones are exact and
  // pre/stop are exact concrete operations -- and HOLDS is sound because the refined covering
  // preserves the sink's reachability, hence deadlock. If the map exceeds its atom cap the
  // whole block falls through to the certificate and the race, unchanged.
  //
  // NO CLOSURE-DOMINANCE PREDICTOR. There used to be one here: when the pre-closure added more
  // atoms than the seed did, it abandoned the construction and ran a_M instead, because on grc4
  // (139 closure atoms against 25 seed, 100% straddle) the atoms lane measured 0.18s against
  // a_M's 0.05s. That reading was wrong about WHERE grc4's time went. Split by phase, the atoms
  // lane was 0.035s of searching behind 0.126s of structural sweep and 0.025s of a phase-1
  // backward fixpoint that the refined search then repeated. Both of those are now gone -- the
  // sweep by the `Inv ^ Esc` rewrite of the Dead=Inv test, the fixpoint by deferring phase 1's
  // backward half -- and the atoms lane wins outright on the very model the predictor was built
  // to hand to a_M. The straddle rate was never the bill: grc4 splits 100% of its tagged zones,
  // stores 1.34x the a_LU nodes for it, and the refined search still costs the same.
  AtomMap am;
  if (e1.build_atom_map(am) && am.usable) {
    if (am.diag.empty() && am.bumped == 0) {
      // EMPTY MAP: this IS the deadlock-completion certificate, so phase 1's a_LU covering was
      // already deadlock-sound and phase 1's own verdict is the answer. Its backward fixpoint
      // was deferred, not skipped -- pay it here, exactly once, on the graph already built.
      return e1.finish_verdict(gs);
    }
    Otf e2(net, p, q);
    e2.set_atoms(&am);
    GfpStats gs2;
    const bool holds2 = e2.run(&gs2);
    if (gs) *gs = gs2;
    return holds2;
  }

  // The construction cannot run (atom cap), so phase 1's verdict is owed after all: pay the
  // backward pass build_forward deferred, then rejoin the certificate/race path.
  if (!e1.finish_verdict(gs)) return false;        // genuine witness

  // THE DEADLOCK-COMPLETION CERTIFICATE (see Otf::dtheta_empty). If the completion's
  // constraint-map seed is empty at every reachable key, the ordinary constraint map already
  // preserves deadlock and phase 1's a_LU covering was sound as it stands. One federation
  // computation per DISCRETE KEY and no search -- against a full reachability traversal for
  // the deadlock-freedom lane.
  if (e1.dtheta_empty()) return true;

  std::atomic<bool> done{false};
  std::atomic<int> verdict{-1};                     // 0 = holds, 1 = violated
  GfpStats gs_b;

  std::thread lane_b([&] {
    Otf e(net, p, q, /*force_exact=*/true, &done);
    const bool holds = e.run(&gs_b);
    if (e.was_cancelled()) return;                  // abandoned: its verdict is not a verdict
    if (!done.exchange(true)) verdict.store(holds ? 0 : 1, std::memory_order_release);
  });

  {
    ReachStats dl;
    const bool may_dl = deadlock_reachable(net, Backend::Omp, &dl, &done);
    // A cancelled reach_engine returns false, indistinguishable from "no deadlock reachable",
    // so this lane may only be trusted if it was not the one cancelled.
    if (!done.load(std::memory_order_acquire) && !may_dl && !done.exchange(true))
      verdict.store(0, std::memory_order_release);  // a_LU found nothing and was sound
  }

  lane_b.join();
  const int v = verdict.load(std::memory_order_acquire);
  if (v == 1 && gs) *gs = gs_b;                     // lane B won and it disagreed with phase 1
  if (v >= 0) return v == 0;

  // Unreachable in the current logic: lane B is cancelled only by lane A concluding
  // "deadlock-free", which publishes verdict 0. Kept so that a future change to either lane
  // cannot silently turn an inconclusive pair into a wrong answer.
  return Otf(net, p, q, /*force_exact=*/true).run(nullptr);
}

}  // namespace ptctlmc
