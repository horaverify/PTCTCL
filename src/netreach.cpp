#include "ptctlmc/netreach.hpp"
#include "ptctlmc/symmetry.hpp"
#include <cstdio>
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
#ifdef PTCTLMC_DEBUG_KNOBS
  // Ablation switch for this optimization. Compiled out by default: this function runs once per
  // generated successor, and a function-local static with dynamic initialization costs a guarded
  // acquire-load on EVERY call, which is pure overhead in a build that never sets the variable.
  static const bool off = std::getenv("PTCTLMC_NOACTIVE") != nullptr;
  if (off) return;
#endif
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
      // Each other process contributes at most one enabled a? (nondeterministic choice).
      std::vector<std::pair<int, std::vector<int>>> parts;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        std::vector<int> es;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan == se.sync_chan && re.sync_dir == -1 && int_ok(re.int_guard, s.ints) && gen_ok(net, re, s.ints.data(), s.ints.size()))
            es.push_back(ej);
        }
        if (!es.empty()) parts.push_back({j, std::move(es)});
      }
      std::vector<std::size_t> pick(parts.size(), 0);
      for (;;) {
        bool part_committed = net.procs[i].locations[s.locs[i]].committed;
        for (const auto& pr : parts)
          if (net.procs[pr.first].locations[s.locs[pr.first]].committed) part_committed = true;
        if (!src_committed || part_committed) {
          Dbm z1 = szone;
          bool ok = apply_all(z1, se.clock_guard);
          for (std::size_t t = 0; ok && t < parts.size(); ++t)
            ok = apply_all(z1, net.procs[parts[t].first].edges[parts[t].second[pick[t]]].clock_guard);
          if (ok) {
            SymState ns;
            ns.locs = s.locs; ns.ints = s.ints;
            ns.locs[i] = se.tgt;
            for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;
            for (int r : se.clock_resets) z1.reset(r);
            for (std::size_t t = 0; t < parts.size(); ++t) {
              const NEdge& re = net.procs[parts[t].first].edges[parts[t].second[pick[t]]];
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
                           const std::vector<char>& qmask, int emode, std::vector<SymState>& out) {
  thread_local std::vector<Constraint> inv;
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
      ns.zone = szone;                                    // into reused capacity
      if (!apply_all(ns.zone, e.clock_guard)) continue;
      for (int r : e.clock_resets) ns.zone.reset(r);
      ns.locs.assign(s.locs.begin(), s.locs.end());
      ns.locs[p] = e.tgt;
      ns.ints.assign(s.ints.begin(), s.ints.end());
      for (const auto& a : e.int_updates) ns.ints[a.var] = a.val;
      if (!gen_apply(net, e, ns.ints.data(), ns.ints.size())) continue;
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
          ns.zone = szone;
          if (!apply_all(ns.zone, se.clock_guard)) continue;
          if (!apply_all(ns.zone, re.clock_guard)) continue;
          for (int r : se.clock_resets) ns.zone.reset(r);
          for (int r : re.clock_resets) ns.zone.reset(r);
          ns.locs.assign(s.locs.begin(), s.locs.end());
          ns.locs[i] = se.tgt;
          ns.locs[j] = re.tgt;
          ns.ints.assign(s.ints.begin(), s.ints.end());
          for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;
          for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
          if (!gen_apply(net, se, ns.ints.data(), ns.ints.size()) || !gen_apply(net, re, ns.ints.data(), ns.ints.size())) continue;
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
      std::vector<std::pair<int, std::vector<int>>> parts;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        std::vector<int> es;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan == se.sync_chan && re.sync_dir == -1 && int_ok(re.int_guard, s.ints) && gen_ok(net, re, s.ints.data(), s.ints.size()))
            es.push_back(ej);
        }
        if (!es.empty()) parts.push_back({j, std::move(es)});
      }
      std::vector<std::size_t> pick(parts.size(), 0);
      for (;;) {
        bool part_committed = net.procs[i].locations[s.locs[i]].committed;
        for (const auto& pr : parts)
          if (net.procs[pr.first].locations[s.locs[pr.first]].committed) part_committed = true;
        if (!src_committed || part_committed) {
          SymState& ns = slot();
          ns.zone = szone;
          bool ok = apply_all(ns.zone, se.clock_guard);
          for (std::size_t t = 0; ok && t < parts.size(); ++t)
            ok = apply_all(ns.zone, net.procs[parts[t].first].edges[parts[t].second[pick[t]]].clock_guard);
          if (ok) {
            ns.locs.assign(s.locs.begin(), s.locs.end());
            ns.ints.assign(s.ints.begin(), s.ints.end());
            ns.locs[i] = se.tgt;
            for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;
            for (int r : se.clock_resets) ns.zone.reset(r);
            for (std::size_t t = 0; t < parts.size(); ++t) {
              const NEdge& re = net.procs[parts[t].first].edges[parts[t].second[pick[t]]];
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
#ifdef PTCTLMC_DEBUG_KNOBS
// FNV-1a over a packed record, for the PTCTLMC_CHECKGC frontier-root snapshot.
static std::uint64_t dbg_rec_hash(const std::uint8_t* p, std::size_t n) {
  std::uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
  return h;
}
#endif
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
#ifdef PTCTLMC_DEBUG_KNOBS
  static const bool force_exact_env = std::getenv("PTCTLMC_EXACTSUB") != nullptr;
#else
  constexpr bool force_exact_env = false;
#endif
  const bool exact_sub = has_diag || force_exact || force_exact_env;
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
      Lr[0] = Ur[0] = kGNone;
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
    const bool soa = use_gpu && gpu->soa_work_pool();
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
    // Optional phase profiler (PTCTLMC_MTLPROF): host-side timing of the two synchronous GPU
    // calls (expand/merge, which block until the kernel finishes since outputs are read in place)
    // vs. the host marshaling/rebuild. Zero overhead when the env var is unset.
    const bool mtlprof = std::getenv("PTCTLMC_MTLPROF") != nullptr;
    // Memory accounting (PTCTLMC_MEMPROF): where the resident bytes actually go — the passed list
    // (live zone bytes vs vector CAPACITY, i.e. growth slack), the duplicated frontier copy, and the
    // transient per-level pools the GPU buffers are sized to. Zero cost when unset.
    const bool memprof = std::getenv("PTCTLMC_MEMPROF") != nullptr;
    // CPU-expand / GPU-merge hybrid (PTCTLMC_HYBRID=1): each phase on the hardware the
    // measurements say it wins on. Zero-copy via unified memory.
    const bool hybrid = std::getenv("PTCTLMC_HYBRID") != nullptr;
    int hyb_threads = 1;
    std::size_t hyb_ratio = 8;   // successors staged per frontier state (adapts per level)
    int hyb_prev_nb = 1024;      // previous level's distinct-bucket count (sizes the bucket table)
#ifdef PTCTLMC_HAVE_OPENMP
    hyb_threads = omp_get_max_threads();
#endif
    std::size_t peak_nwork = 0, peak_wtot = 0, peak_front = 0;
    std::size_t peak_rss_ = 0, peak_rss_heap_ = 0, peak_rss_gpu_ = 0, peak_rss_gpulive_ = 0;
    double t_expand = 0, t_merge = 0, t_bucket = 0, t_gather = 0, t_rebuild = 0, t_binit = 0, t_bmain = 0, t_bcnt = 0, t_bpfx = 0, t_bscat = 0;
    int lvl_cpu = 0, lvl_gpu = 0;   // merge routing actually taken, per level
    std::size_t tot_work = 0, tot_valid = 0;   // GPU lane utilisation: survivors / work items
    int prof_nb = 0, prof_maxsz = 0; double prof_meansz = 0; long long prof_peak_sum2 = 0;
    double gpu_rate = -1, cpu_rate = -1;   // adaptive merge routing: work-units (sum bucket^2) per ms
    auto _now = [] { return std::chrono::steady_clock::now(); };
    auto _ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    // Rebuild sub-phase attribution (PTCTLMC_RBPROF): copy/transpose vs intern vs frontier write.
    const bool rbprof = std::getenv("PTCTLMC_RBPROF") != nullptr;
    const int RBT = std::max(1, hyb_threads);
    std::vector<double> rb_copy(RBT * 8, 0.0), rb_intern(RBT * 8, 0.0), rb_front(RBT * 8, 0.0);
    auto rbnow = [](bool on) {
      return on ? std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now().time_since_epoch()).count() : 0.0;
    };
    auto ompt = [] {
#ifdef PTCTLMC_HAVE_OPENMP
      return omp_get_thread_num() * 8;   // stride 8 doubles = 64B, one cache line per thread
#else
      return 0;
#endif
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
    const auto _wall0 = _now();
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
      const auto _te = _now();
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
            const std::size_t n = gen_succ_reuse(net, cur, pz, L, U, qmask, emode, buf);
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
      if (mtlprof) {
        const double _dte = _ms(_te, _now());
        t_expand += _dte; tot_work += nwork;
        // Per-level (items, ms) so the fixed-vs-variable split of a dispatch can be REGRESSED
        // from many points within one model, instead of inferred from a two-point fit across
        // models (which is exactly determined, hence unfalsifiable -- and wrong: it mispredicts
        // arbiter17 by 2.7x because per-item cost scales with the zone size).
        if (std::getenv("PTCTLMC_LVLPROF"))
          std::fprintf(stderr, "[lvl] frontier=%d nwork=%zu expand_ms=%.3f\n", nfr, nwork, _dte);
      }
      if (nfr > 0) fanout = std::max(fanout, static_cast<double>(nwork) / nfr);
      if (memprof && nwork > peak_nwork) peak_nwork = nwork;
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
      const auto _tb = _now();
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
        if (mtlprof) ++tot_valid;
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
      const auto _tp = _now();
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
      if (mtlprof) t_bpfx += _ms(_tp, _now());
      new_idx.resize(ntot);
      {
        cursor = new_off;
        for (std::size_t w = 0; w < nwork; ++w) {
          const int bi = w_bucket[w];
          if (bi >= 0) new_idx[cursor[bi]++] = static_cast<int>(w);
        }
      }
      if (mtlprof) t_bucket += _ms(_tb, _now());
      if (memprof && wtot > peak_wtot) peak_wtot = wtot;
      const auto _tg = _now();
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
          if (soa) {   // transpose AoS antichain -> SoA pool: zone i entry k at dst[k*cap + i]
            const std::size_t cap = static_cast<std::size_t>(exist_cnt[b]) + new_cnt[b];
            for (int i = 0; i < exist_cnt[b]; ++i)
              zi.load_gz_strided(pit->second[i] & ~kNewBit, dst + i, cap);
          } else {
            for (int i = 0; i < exist_cnt[b]; ++i)
              zi.load_gz(pit->second[i] & ~kNewBit, dst + static_cast<std::size_t>(i) * d2);
          }
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
      if (mtlprof) t_gather += _ms(_tg, _now());
      out_work_off.clear(); out_count.clear();
      const gz_t* out_zones = nullptr;
      const int* out_isnew = nullptr;
      // Merge routing. NAIVE (default): always the GPU one-thread-per-bucket merge (original
      // behavior). ADAPTIVE (--adaptive): CALIBRATE each path's throughput (work ~ sum bucket^2, per
      // ms) on early BFS levels, then route each later level to the predicted-faster of {GPU, 12 CPU
      // cores}. This rescues merge-heavy models (e.g. philosophers, where the GPU merge is 93% of the
      // wall time — one thread per bucket is the wrong shape) with no effect where the GPU already wins.
      long long sum2 = 0, sumsz = 0; int maxsz = 0;
      for (int b = 0; b < nb; ++b) { const long long s = exist_cnt[b] + new_cnt[b]; sum2 += s*s; sumsz += s; if (s > maxsz) maxsz = (int)s; }
      if (mtlprof && sum2 > prof_peak_sum2) { prof_peak_sum2 = sum2; prof_nb = nb; prof_maxsz = maxsz; prof_meansz = nb ? (double)sumsz/nb : 0; }
      bool cpu_merge = false;
      if (detail::options().adaptive_merge && !soa) {   // SoA pool is unreadable by the AoS CPU merge -> GPU only
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
      if (cpu_merge) ++lvl_cpu; else ++lvl_gpu;
      const double _dtm = _ms(_tm, _now());
      if (mtlprof) t_merge += _dtm;
      if (detail::options().adaptive_merge && sum2 > 0 && _dtm > 0.5) {   // update throughput calibration (skip trivial levels)
        const double rate = static_cast<double>(sum2) / _dtm;
        if (cpu_merge) cpu_rate = (cpu_rate < 0) ? rate : 0.5 * cpu_rate + 0.5 * rate;
        else           gpu_rate = (gpu_rate < 0) ? rate : 0.5 * gpu_rate + 0.5 * rate;
      }
      // rebuild passed + assemble the flat next frontier (survivors). Prefix-sum the per-bucket
      // survivor counts, pre-create the passed nodes, then fill everything in parallel.
      const auto _tr = _now();
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
      if (memprof && frtot > peak_front) peak_front = frtot;
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
        const std::size_t poolbase = static_cast<std::size_t>(out_work_off[b]) * d2;
        const std::size_t cap = static_cast<std::size_t>(exist_cnt[b]) + new_cnt[b];
        int fpos = 0;
        std::vector<raw_t> tmp(d2);        // one merged zone, reused across this bucket
        const int _tid = ompt();
        for (int i = 0; i < out_count[b]; ++i) {
          const std::size_t slot = static_cast<std::size_t>(out_work_off[b]) + i;
          const int tag = out_isnew[slot];
          if (tag >= 0) { pz[i] = tag; continue; }   // kept: already stored, already interned
          raw_t* pzi = tmp.data();
          const auto _r0 = rbnow(rbprof);
          if (soa)   // transpose SoA pool -> AoS zone: entry k at poolbase + k*cap + i
            for (int k = 0; k < d2; ++k) pzi[k] = zget(out_zones, static_cast<int>(poolbase) + k * static_cast<int>(cap) + i);
          else
            for (int k = 0; k < d2; ++k) pzi[k] = zget(out_zones + slot * d2, k);
          const auto _r1 = rbnow(rbprof);
          // The high bit of the stored index records "added during the CURRENT level". The
          // frontier is read off these flags once the level is finished (see the level-advance
          // step), NOT here -- because a later batch of the same level can still absorb this zone
          // into a convex union, and emitting it now would put a state in the frontier that the
          // level's own merge went on to delete. That is what made batching cost fischer12 2.5x
          // its state count.
          pz[i] = zi.intern(pzi) | kNewBit;
          const auto _r2 = rbnow(rbprof);
          if (rbprof) { rb_copy[_tid] += _r1 - _r0; rb_intern[_tid] += _r2 - _r1; }
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
      if (mtlprof) t_rebuild += _ms(_tr, _now());
#ifdef __APPLE__
      if (memprof) {   // sample the TRUE resident peak per level, with its composition
        mach_task_basic_info info{};
        mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &cnt) == KERN_SUCCESS &&
            info.resident_size > peak_rss_) {
          malloc_statistics_t ms{}; malloc_zone_statistics(malloc_default_zone(), &ms);
          peak_rss_ = info.resident_size;
          peak_rss_heap_ = ms.size_in_use;
          peak_rss_gpu_ = gpu->device_bytes();
          peak_rss_gpulive_ = gpu->live_buffer_bytes();
        }
      }
#endif
    }
    if (memprof) {
      std::size_t live = 0, cap = 0;
      for (const auto& kv : passed) { live += kv.second.size(); cap += kv.second.capacity(); }
      const double GB = 1024.0 * 1024.0 * 1024.0, B = sizeof(raw_t);
      // per-bucket container overhead: unordered_map node (~48 B: hash+next+key+vector object) + malloc header
      const double node_ov = passed.size() * 64.0;
      std::fprintf(stderr,
        "[memprof] buckets=%zu  zones=%zu\n"
        "          passed live   = %6.2f GB   (the actual data)\n"
        "          passed capacity= %6.2f GB   (+%.0f%% vector growth slack)\n"
        "          bucket overhead= %6.2f GB   (~64 B/bucket: map node + malloc header)\n"
        "          frontier copy  = %6.2f GB   (peak %zu zones — DUPLICATES passed)\n"
        "          gpu expand pool= %6.2f GB   (peak %zu work items, persistent)\n"
        "          gpu merge pool = %6.2f GB   (peak %zu work zones, persistent)\n"
        "          passed arena   = %6.2f GB  (%d B/entry)\n"
        "          gpu pools LIVE = %6.2f GB  (sum of our own buffers)\n"
        "          gpu DRIVER map = %6.2f GB  (what Metal actually has mapped)\n"
        "          ---- accounted = %6.2f GB\n",
        passed.size(), live / (std::size_t)d2,
        live * B / GB, cap * B / GB, live ? 100.0 * (double)(cap - live) / (double)live : 0.0,
        node_ov / GB,
        peak_front * (double)d2 * B / GB, peak_front,
        peak_nwork * (double)d2 * B / GB, peak_nwork,
        peak_wtot * (double)d2 * B / GB, peak_wtot,
        zi.bytes() / GB, zi.entry_bytes(), gpu->live_buffer_bytes() / GB, gpu->device_bytes() / GB,
        (cap * B + node_ov + peak_front * (double)d2 * B + gpu->device_bytes()) / GB);
#ifdef __APPLE__
      // Separate "bytes the program is using" from "bytes malloc is holding but not returning to
      // the OS". A large gap here means the cost is allocator RETENTION from per-level churn, not
      // live data — a different fix (pooling) than shrinking any one structure.
      {
        malloc_statistics_t st{};
        malloc_zone_statistics(malloc_default_zone(), &st);
        std::fprintf(stderr,
          "          malloc in-use  = %6.2f GB   (live heap)\n"
          "          malloc reserved= %6.2f GB   (held by allocator, not returned to OS)\n"
          "          >>> PEAK RSS   = %6.2f GB   at peak: heap %.2f GB + gpu-mapped %.2f GB "
          "(of which our pools %.2f GB) = %.2f GB\n",
          st.size_in_use / GB, st.size_allocated / GB,
          peak_rss_ / GB, peak_rss_heap_ / GB, peak_rss_gpu_ / GB, peak_rss_gpulive_ / GB,
          (peak_rss_heap_ + peak_rss_gpu_) / GB);
      }
#endif
    }
    if (mtlprof) {
      const double wall = _ms(_wall0, _now());
      std::fprintf(stderr, "[mtlprof] expand=%.0fms merge=%.0fms host=%.0fms wall=%.0fms  "
                   "(merge %.0f%%)  peak-level: nb=%d meansz=%.1f maxsz=%d\n"
                   "[mtlprof]   host split: bucket=%.0fms (passed-lookups %.0fms) gather=%.0fms rebuild=%.0fms other=%.0fms\n"
                   "[mtlprof]   MERGE ROUTING: %d levels CPU, %d levels GPU\n"
                   "[mtlprof]   LANE UTILISATION: %zu work items -> %zu survivors (%.1f%% survive)\n",
                   t_expand, t_merge, wall - t_expand - t_merge, wall,
                   100.0 * t_merge / wall, prof_nb, prof_meansz, prof_maxsz,
                   t_bucket, t_bpfx, t_gather, t_rebuild,
                   wall - t_expand - t_merge - t_bucket - t_gather - t_rebuild, lvl_cpu, lvl_gpu, tot_work, tot_valid,
                   tot_work ? 100.0 * (double)tot_valid / (double)tot_work : 0.0);
      if (rbprof) {
        double c = 0, in = 0, f = 0;
        for (int t = 0; t < RBT; ++t) { c += rb_copy[t*8]; in += rb_intern[t*8]; f += rb_front[t*8]; }
        std::fprintf(stderr, "[rbprof]   rebuild CPU-time split: copy/transpose=%.0fms intern=%.0fms frontier-write=%.0fms\n", c, in, f);
      }
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
#ifdef PTCTLMC_DEBUG_KNOBS
    std::vector<std::uint64_t> fr_snap;   // PTCTLMC_CHECKGC: per-frontier-entry record hash
#endif
    // Peak-RSS composition sampler (PTCTLMC_MEMPROF): sample at each level boundary and
    // remember the composition at the moment RSS was highest.
    const bool memprof = std::getenv("PTCTLMC_MEMPROF") != nullptr;
    std::size_t pk_rss = 0, pk_lvl = 0, pk_front = 0, pk_pack = 0, pk_map = 0, pk_cache = 0,
                pk_arena = 0, pk_tab = 0, pk_heap = 0;
    std::atomic<bool> found{false};
    const bool phprof = std::getenv("PTCTLMC_PHPROF") != nullptr;
    double ph_expand = 0, ph_collect = 0, ph_compact = 0, ph_gc = 0;
    std::vector<double> ph_busy(static_cast<std::size_t>(nthreads) * 8, 0.0);   // padded
    auto ph_now = [] { return std::chrono::steady_clock::now(); };
    auto ph_ms = [](std::chrono::steady_clock::time_point a) {
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    };
    const bool single = (nthreads == 1);   // seq / omp -p1: skip locks + the touched union
    // Debug hooks for the local-antichain drain, compiled out unless PTCTLMC_DEBUG_KNOBS is
    // defined: they sit in the per-successor loop, where even a function-local static costs a
    // guarded acquire-load on every generated state. Same convention as reset_inactive's ablation.
#ifdef PTCTLMC_DEBUG_KNOBS
    static const bool dbg_front = std::getenv("PTCTLMC_DUMPFRONT") != nullptr;
    static const int dbg_lvl = [] {
      const char* e = std::getenv("PTCTLMC_TRACELVL");
      return e ? std::atoi(e) : -1;
    }();
    static const std::uint64_t dbg_dk = [] {
      const char* e = std::getenv("PTCTLMC_DUMPDK");
      return e ? std::strtoull(e, nullptr, 10) : ~std::uint64_t(0);
    }();
#else
    constexpr bool dbg_front = false;
    constexpr int dbg_lvl = -1;
    constexpr std::uint64_t dbg_dk = ~std::uint64_t(0);
#endif
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
      auto _t_exp = ph_now();
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
        const auto _t_busy = std::chrono::steady_clock::now();
#pragma omp for schedule(dynamic, expand_chunk) nowait
        for (long i = 0; i < static_cast<long>(frontier.size()); ++i) {
          if (found.load(std::memory_order_relaxed)) continue;
          decode_key(frontier[i].first, cur.locs.data(), cur.ints.data());
#ifdef PTCTLMC_DEBUG_KNOBS
          // PTCTLMC_SRCSTATS: how many DISTINCT source zones do we expand? The successor zone is a
          // function of (source zone, edge), so a small distinct count means the same O(dim^3)
          // delay/constrain/close is being recomputed many times over.
          {
            static const bool sst = std::getenv("PTCTLMC_SRCSTATS") != nullptr;
            if (sst) {
              static std::mutex m; static std::unordered_set<int> zs; static long tot = 0;
              std::lock_guard<std::mutex> lk(m);
              zs.insert(frontier[i].second); ++tot;
              if (tot % 200000 == 0)
                std::fprintf(stderr, "[srcstats] expansions=%ld distinct-source-zones=%zu (reuse %.1fx)\n",
                             tot, zs.size(), double(tot) / double(zs.size()));
            }
          }
#endif
#ifdef PTCTLMC_DEBUG_KNOBS
          if (!fr_snap.empty()) {
            const std::uint8_t* r = zi.record(frontier[i].second);
            if (dbg_rec_hash(r, mindbm_size(r)) != fr_snap[i]) {
              std::fprintf(stderr,
                           "[GC CLOBBERED A FRONTIER ROOT] level=%d entry=%ld dk=%llu zidx=%d\n",
                           level, i, (unsigned long long)frontier[i].first, frontier[i].second);
              std::abort();
            }
          }
          // Same invariant check, but on the state as it comes BACK OUT of the passed list.
          // A successor that was clean when generated and dirty when re-read means storage
          // (intern / merge / collect), not the successor relation, broke it.
          {
            static const bool chk = std::getenv("PTCTLMC_CHECKINV") != nullptr;
            if (chk) {
              const int zd = net.nclocks + 1;
              const raw_t* fz = zi.get(frontier[i].second).data().data();
              for (int p = 0; p < np_; ++p) {
                for (const auto& c : net.procs[p].locations[cur.locs[p]].invariant) {
                  std::vector<raw_t> t(fz, fz + zd * zd);
                  dbm_constrain(t.data(), zd, c.i, c.j, c.bound);
                  if (!std::equal(t.begin(), t.end(), fz)) {
                    std::fprintf(stderr,
                                 "[FRONTIER INVARIANT BROKEN] level=%d dk=%llu proc=%d loc=%d "
                                 "x%d-x%d bound=%d locs=",
                                 level, (unsigned long long)frontier[i].first, p, static_cast<int>(cur.locs[p]),
                                 c.i, c.j, raw_bound(c.bound));
                    for (int q = 0; q < np_; ++q) std::fprintf(stderr, "%d,", static_cast<int>(cur.locs[q]));
                    std::fprintf(stderr, " zone:");
                    for (int a = 0; a < zd * zd; ++a) std::fprintf(stderr, " %d", fz[a]);
                    std::fprintf(stderr, "\n");
                    std::abort();
                  }
                }
              }
            }
          }
#endif
          if (dbg_dk == frontier[i].first) {
              const raw_t* pz = zi.get(frontier[i].second).data().data();
              const int pd = net.nclocks + 1;
              std::fprintf(stderr, "[zone] level=%d dk=%llu :", level,
                           (unsigned long long)frontier[i].first);
              for (int a = 0; a < pd * pd; ++a) std::fprintf(stderr, " %d", pz[a]);
              std::fprintf(stderr, "\n");
            }
          // buf is NOT cleared: gen_succ_reuse rebuilds slots in place and returns
          // the count, so the loc/int/zone storage is reused across frontier states.
          const std::size_t ns = gen_succ_reuse(net, cur, zi.get(frontier[i].second), L, U, qmask, emode, buf);
          for (std::size_t sk = 0; sk < ns; ++sk) {
            SymState& s = buf[sk];
            ++gen;
            if (is_target(s)) {
              if (dbg_front) {
                std::fprintf(stderr, "[hit] level=%d parent_dk=%llu parent_zidx=%d locs=", level,
                             (unsigned long long)frontier[i].first, frontier[i].second);
                for (int p = 0; p < np_; ++p) std::fprintf(stderr, "%d,", static_cast<int>(cur.locs[p]));
                std::fprintf(stderr, " ints=");
                for (int v = 0; v < ni_; ++v) std::fprintf(stderr, "%d,", static_cast<int>(cur.ints[v]));
                std::fprintf(stderr, "\n[hit] child locs=");
                for (int p = 0; p < np_; ++p) std::fprintf(stderr, "%d,", static_cast<int>(s.locs[p]));
                std::fprintf(stderr, " ints=");
                for (int v = 0; v < ni_; ++v) std::fprintf(stderr, "%d,", static_cast<int>(s.ints[v]));
                std::fprintf(stderr, "\n[hit] child zone=");
                {
                  const int cd = net.nclocks + 1;
                  for (int a = 0; a < cd * cd; ++a)
                    std::fprintf(stderr, "%d ", s.zone.data()[a]);
                }
                std::fprintf(stderr, "\n[hit] parent zone=");
                const raw_t* pz = zi.get(frontier[i].second).data().data();
                const int pd = net.nclocks + 1;
                for (int a = 0; a < pd * pd; ++a) std::fprintf(stderr, "%d ", pz[a]);
                std::fprintf(stderr, "\n");
              }
              found.store(true, std::memory_order_relaxed);
              break;
            }
#ifdef PTCTLMC_DEBUG_KNOBS
            // PTCTLMC_CHECKINV: a symbolic state's zone must satisfy the invariant of EVERY
            // process's current location, not just the one that moved. If tightening by an
            // invariant changes the zone, that invariant was not enforced on this successor.
            {
              static const bool chk = std::getenv("PTCTLMC_CHECKINV") != nullptr;
              if (chk) {
                const int zd = net.nclocks + 1;
                for (int p = 0; p < np_; ++p) {
                  const auto& inv = net.procs[p].locations[s.locs[p]].invariant;
                  for (const auto& c : inv) {
                    std::vector<raw_t> t(s.zone.data().begin(), s.zone.data().end());
                    dbm_constrain(t.data(), zd, c.i, c.j, c.bound);
                    if (!std::equal(t.begin(), t.end(), s.zone.data().begin())) {
                      std::fprintf(stderr,
                                   "[INVARIANT NOT ENFORCED] level=%d proc=%d loc=%d "
                                   "constraint x%d-x%d bound=%d locs=",
                                   level, p, static_cast<int>(s.locs[p]), c.i, c.j, raw_bound(c.bound));
                      for (int q = 0; q < np_; ++q) std::fprintf(stderr, "%d,", static_cast<int>(s.locs[q]));
                      std::fprintf(stderr, " zone:");
                      for (int a = 0; a < zd * zd; ++a)
                        std::fprintf(stderr, " %d", s.zone.data()[a]);
                      std::fprintf(stderr, "\n");
                      std::abort();
                    }
                  }
                }
              }
            }
#endif
            if (use_sym) canonicalize(*sym, net, s);
            canon_ints(s.locs, s.ints);
            const std::uint64_t dk = disc_key(s.locs, s.ints);
            if (dbg_lvl == level)
              std::fprintf(stderr, "[edge] %llu -> %llu\n",
                           (unsigned long long)frontier[i].first, (unsigned long long)dk);
            const std::size_t si = stripe_of(dk);
            if (dbg_dk == dk) {
              std::fprintf(stderr, "[succ] level=%d from=%llu plocs=", level,
                           (unsigned long long)frontier[i].first);
              for (int p = 0; p < np_; ++p) std::fprintf(stderr, "%d,", static_cast<int>(cur.locs[p]));
              std::fprintf(stderr, " pints=");
              for (int v = 0; v < ni_; ++v) std::fprintf(stderr, "%d,", static_cast<int>(cur.ints[v]));
              std::fprintf(stderr, " clocs=");
              for (int p = 0; p < np_; ++p) std::fprintf(stderr, "%d,", static_cast<int>(s.locs[p]));
              std::fprintf(stderr, " cints=");
              for (int v = 0; v < ni_; ++v) std::fprintf(stderr, "%d,", static_cast<int>(s.ints[v]));
              std::fprintf(stderr, " czone:");
              const int _pd = net.nclocks + 1;
              for (int a = 0; a < _pd * _pd; ++a) std::fprintf(stderr, " %d", s.zone.data()[a]);
              std::fprintf(stderr, "\n");
            }
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
        if (phprof)
          ph_busy[static_cast<std::size_t>(tid) * 8] +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - _t_busy).count();
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
          const std::size_t ns = gen_succ_reuse(net, cur, zi.get(fr.second), L, U, qmask, emode, buf);
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
      if (phprof) ph_expand += ph_ms(_t_exp);
      auto _t_col = ph_now();
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
#ifdef PTCTLMC_DEBUG_KNOBS
      // PTCTLMC_CHECKGC: a frontier entry is a GC root, so its interned record must be byte-
      // identical when the level expands it. Snapshot here (before the sweep) and re-check at
      // expansion; a mismatch means the sweep freed and reused an index that was still named.
      static const bool chk_gc = std::getenv("PTCTLMC_CHECKGC") != nullptr;
      if (chk_gc) {
        fr_snap.clear();
        fr_snap.reserve(frontier.size());
        for (const auto& f : frontier) {
          const std::uint8_t* r = zi.record(f.second);
          fr_snap.push_back(dbg_rec_hash(r, mindbm_size(r)));
        }
      }
#endif
      if (dbg_front) {
        // (dk, zone) of every frontier entry, hashed so the two engines' whole levels can be
        // diffed cheaply. Sorted, so the comparison is order-insensitive: the drain builds the
        // frontier in a different order than the shared collect, and order is not a difference.
        const int pd = net.nclocks + 1;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> fz;
        fz.reserve(frontier.size());
        for (const auto& f : frontier) {
          const raw_t* z = zi.get(f.second).data().data();
          std::uint64_t h = 1469598103934665603ULL;
          for (int a = 0; a < pd * pd; ++a) {
            h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(z[a]));
            h *= 1099511628211ULL;
          }
          fz.push_back({f.first, h});
        }
        std::sort(fz.begin(), fz.end());
        fz.erase(std::unique(fz.begin(), fz.end()), fz.end());
        for (const auto& e : fz)
          std::fprintf(stderr, "[fz] %d %llu %llu\n", level, (unsigned long long)e.first,
                       (unsigned long long)e.second);
      }
      // Adaptive abstraction: re-read the zones-per-discrete-state signal now that the level is
      // complete and every thread has joined, and switch widening on if the antichains are
      // filling up. Once on it stays on -- the ratio only ever justifies more abstraction.
      if (adapt && emode == ExNone) {
        std::size_t nlive = 0, ndisc = 0;
        for (const auto& mp : smap) mp.counts(nlive, ndisc);
        if (ndisc && static_cast<double>(nlive) >= kAdaptRatio * static_cast<double>(ndisc)) {
          emode = ExLuPlus;
          if (std::getenv("PTCTLMC_STATS"))
            std::fprintf(stderr, "[adapt] level %d: %zu zones / %zu discrete states -> Extra_LU+ on\n",
                         level, nlive, ndisc);
        }
      }
      if (phprof) ph_collect += ph_ms(_t_col);
      auto _t_cmp = ph_now();
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
      if (phprof) ph_compact += ph_ms(_t_cmp);
      auto _t_gc = ph_now();
#ifdef PTCTLMC_DEBUG_KNOBS
      static const bool no_gc = std::getenv("PTCTLMC_NOGC") != nullptr;   // ablation
#else
      constexpr bool no_gc = false;
#endif
      if (!no_gc && zi.allocated() >= sweep_at) {
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
      if (phprof) ph_gc += ph_ms(_t_gc);

      if (memprof) {
#ifdef __APPLE__
        mach_task_basic_info info{};
        mach_msg_type_number_t c = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &c) == KERN_SUCCESS &&
            info.resident_size > pk_rss) {
          std::size_t pack = 0, mapb = 0, cacheb = 0, an = 0, dn = 0, ab = 0, tb = 0;
          zi.account(pack, mapb, cacheb);
          for (const auto& mp : smap) mp.account(an, dn, ab, tb);
          pk_rss = info.resident_size; pk_lvl = level;
          pk_front = frontier.capacity() * sizeof(std::pair<std::uint64_t, int>);
          pk_pack = pack; pk_map = mapb; pk_cache = cacheb; pk_arena = ab; pk_tab = tb;
          pk_heap = mstats().bytes_used;   // process-wide live heap (all zones)
        }
#elif defined(__linux__)
        const std::size_t rss = linux_rss_bytes();
        if (rss > pk_rss) {
          std::size_t pack = 0, mapb = 0, cacheb = 0, an = 0, dn = 0, ab = 0, tb = 0;
          zi.account(pack, mapb, cacheb);
          for (const auto& mp : smap) mp.account(an, dn, ab, tb);
          pk_rss = rss; pk_lvl = level;
          pk_front = frontier.capacity() * sizeof(std::pair<std::uint64_t, int>);
          pk_pack = pack; pk_map = mapb; pk_cache = cacheb; pk_arena = ab; pk_tab = tb;
          pk_heap = linux_heap_bytes();
        }
#endif
      }
    }
    if (memprof) {
      auto MB = [](std::size_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
      const std::size_t known = pk_front + pk_pack + pk_map + pk_cache + pk_arena + pk_tab;
      std::fprintf(stderr,
                   "[memprof] peak-RSS=%.0fMB @level %zu | heap-in-use=%.0fMB | packed=%.0f map=%.0f "
                   "cache=%.0f arena=%.0f tab=%.0f frontier=%.0f known=%.0f other=%.0f\n",
                   MB(pk_rss), pk_lvl, MB(pk_heap), MB(pk_pack), MB(pk_map), MB(pk_cache),
                   MB(pk_arena), MB(pk_tab), MB(pk_front), MB(known), MB(pk_rss) - MB(known));
      std::fprintf(stderr, "[memprof] distinct-zones=%zu htab-slots=%zu (load=%.2f)\n",
                   zi.distinct(), zi.htab_slots(),
                   zi.htab_slots() ? double(zi.distinct()) / zi.htab_slots() : 0.0);
    }
    if (phprof) {
      const double tot = ph_expand + ph_collect + ph_compact + ph_gc;
      std::fprintf(stderr,
                   "[phase] levels=%d expand=%.0fms(%.1f%%) collect=%.0fms(%.1f%%) "
                   "compact=%.0fms(%.1f%%) gc=%.0fms(%.1f%%)  post-barrier=%.1f%%\n",
                   level, ph_expand, 100*ph_expand/tot, ph_collect, 100*ph_collect/tot,
                   ph_compact, 100*ph_compact/tot, ph_gc, 100*ph_gc/tot,
                   100*(ph_collect+ph_compact+ph_gc)/tot);
      double busy = 0;
      for (int t = 0; t < nthreads; ++t) busy += ph_busy[static_cast<std::size_t>(t) * 8];
      const double cap = ph_expand * nthreads;
      std::fprintf(stderr,
                   "[phase] expand busy=%.0fms of %.0fms thread-time (%.1f%% occupancy) "
                   "-> tail idle = %.1f%% of the run\n",
                   busy, cap, cap > 0 ? 100 * busy / cap : 0.0,
                   cap > 0 ? 100 * (cap - busy) / cap * (ph_expand / tot) : 0.0);
    }
    if (std::getenv("PTCTLMC_STATS")) {
      std::size_t zones = 0, disc = 0;
      for (const auto& mp : smap) mp.stats(disc, zones);
      std::unordered_set<std::uint64_t> distinct;   // distinct-zone probe (FNV-1a of zone bytes)
      for (const auto& mp : smap)
        mp.for_each_live_zone([&](const Dbm& z) {
          const auto& d = z.data();
          std::uint64_t h = 1469598103934665603ULL;
          for (raw_t r : d) { h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(r)); h *= 1099511628211ULL; }
          distinct.insert(h);
        });
      std::fprintf(stderr, "[stats] discrete-states=%zu stored-zones=%zu distinct-zones=%zu generated=%zu (shared-passed, %zu stripes)\n",
                   disc, zones, distinct.size(), stats ? stats->generated : 0, NB);
      std::size_t packedB = 0, mapB = 0, cacheB = 0;
      zi.account(packedB, mapB, cacheB);
      std::size_t arenaNodes = 0, deadNodes = 0, arenaB = 0, tabB = 0;
      for (const auto& mp : smap) mp.account(arenaNodes, deadNodes, arenaB, tabB);
      auto MB = [](std::size_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
      std::size_t rsub = 0, rmrg = 0;
      for (const auto& mp : smap) mp.retire_causes(rsub, rmrg);
      std::fprintf(stderr, "[retire] subsumed-by-newer=%zu absorbed-into-convex-union=%zu\n",
                   rsub, rmrg);
      // ---- PTCTLMC_GAPHIST: how many BFS levels elapse between a zone being STORED and being
      // retired by a later, more general one. Bucket 0 = retired inside its own level (the node
      // never reached the frontier, so it cost no expansion); bucket g>=1 = the node was already
      // expanded g levels ago, so that expansion is the re-exploration tax. Bounds any
      // "hold the state back for W levels and let it grow first" policy: such a policy can only
      // reach retirements with gap <= W.
      if (std::getenv("PTCTLMC_GAPHIST")) {
        std::uint64_t gh[66] = {0};
        for (const auto& mp : smap) mp.gap_hist(gh);
        std::uint64_t tot = 0, cross = 0;
        for (int i = 0; i < 66; ++i) { tot += gh[i]; if (i >= 1) cross += gh[i]; }
        std::fprintf(stderr, "[gaphist] retirements=%llu same-level(g=0)=%llu cross-level(g>=1)=%llu\n",
                     (unsigned long long)tot, (unsigned long long)gh[0], (unsigned long long)cross);
        std::uint64_t cum = 0;
        std::fprintf(stderr, "[gaphist] cumulative %% of cross-level retirements caught by a W-level hold:");
        for (int W = 1; W <= 16 && cross; ++W) {
          cum += gh[W];
          std::fprintf(stderr, " W=%d:%.1f%%", W, 100.0 * (double)cum / (double)cross);
        }
        std::fprintf(stderr, "\n[gaphist] raw:");
        for (int i = 0; i < 66; ++i) if (gh[i]) std::fprintf(stderr, " %d:%llu", i, (unsigned long long)gh[i]);
        std::fprintf(stderr, "\n");
      }
      // ---- PTCTLMC_FEDPROBE: measurement-only headroom for a NON-convex (federation) merge.
      // For each discrete state's antichain, greedily drop any member covered by the UNION of the
      // survivors (fed_covers is exact, so this is a legal representation change: the denoted set
      // is unchanged). The delta is the most a federation representation could ever save.
      if (std::getenv("PTCTLMC_FEDPROBE")) {
        std::size_t tot = 0, red = 0, multi = 0, maxk = 0, capped = 0;
        std::vector<std::size_t> hist(16, 0);
        for (const auto& mp : smap)
          mp.for_each_antichain([&](const std::vector<Dbm>& zs) {
            const std::size_t k = zs.size();
            tot += k;
            if (k >= 2) ++multi;
            maxk = std::max(maxk, k);
            hist[std::min<std::size_t>(k, 15)] += 1;
            if (k < 2) { red += k; return; }
            static const std::size_t kcap = [] {
              const char* e = std::getenv("PTCTLMC_FEDPROBECAP");
              return e ? static_cast<std::size_t>(std::atol(e)) : std::size_t(10);
            }();
            if (k > kcap) { ++capped; red += k; return; }  // fed_subtract blows up; do not guess
            std::vector<bool> gone(k, false);
            std::size_t live = k;
            for (std::size_t i = 0; i < k && live > 1; ++i) {
              Federation rest;
              for (std::size_t j = 0; j < k; ++j)
                if (j != i && !gone[j]) rest.push_back(zs[j]);
              if (fed_covers(rest, zs[i])) { gone[i] = true; --live; }
            }
            red += live;
          });
        std::fprintf(stderr,
                     "[fedprobe] antichains-with>=2=%zu maxk=%zu capped(k>10)=%zu | zones=%zu "
                     "after-federation-reduce=%zu removable=%zu (%.2f%%)\n",
                     multi, maxk, capped, tot, red, tot - red,
                     tot ? 100.0 * static_cast<double>(tot - red) / static_cast<double>(tot) : 0.0);
        std::fprintf(stderr, "[fedprobe] antichain size histogram:");
        for (std::size_t i = 1; i < 16; ++i)
          if (hist[i]) std::fprintf(stderr, " %zu:%zu", i, hist[i]);
        std::fprintf(stderr, "\n");
      }
      std::fprintf(stderr,
                   "[bytes] packed-zones=%.0fMB intern-map=%.0fMB tcaches=%.0fMB | arena=%.0fMB "
                   "(%zu nodes, %zu dead=%.0f%%) tab=%.0fMB\n",
                   MB(packedB), MB(mapB), MB(cacheB), MB(arenaB), arenaNodes, deadNodes,
                   arenaNodes ? 100.0 * deadNodes / arenaNodes : 0.0, MB(tabB));
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

bool leads_to_net(const Network& net, const DiscPred& p, const DiscPred& q, Backend backend,
                  ReachStats* stats) {
  (void)backend;
  std::vector<std::int32_t> L = net.lbound, U = net.ubound;
  const int emode = network_has_diagonal(net) ? ExNoDiag : ExLuPlus;
  const std::vector<char> qmask(net.nclocks + 1, 0);

  auto holds = [&](const DiscPred& pr, const SymState& s) -> bool {
    for (const auto& r : pr.locs) { const bool eq = (s.locs[r.proc] == r.loc); if (eq != r.equal) return false; }
    return int_ok(pr.ints, s.ints);
  };
  auto key = [&](const SymState& s) -> std::string {
    std::string k = disc_key(s.locs, s.ints);
    const auto& d = s.zone.data();
    k.append(reinterpret_cast<const char*>(d.data()), d.size() * sizeof(raw_t));
    return k;
  };
  auto extrap = [&](Dbm& z) {
    if (emode == ExLuPlus) z.extrapolate_lu_plus(L, U);
    else if (emode == ExNoDiag) z.extrapolate_lu_plus_nodiag(L, U);
  };

  SymState s0;
  s0.locs.resize(net.nprocs());
  for (int pr = 0; pr < net.nprocs(); ++pr) s0.locs[pr] = net.procs[pr].initial;
  s0.ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0.ints[v] = net.intvars[v].init;
  s0.zone = Dbm(net.nclocks);
  {
    std::vector<Constraint> inv;
    all_invariants(net, s0.locs, inv);
    if (!apply_all(s0.zone, inv)) return true;  // unreachable init => vacuously holds
    bool c, u; frozen_flags(net, s0.locs, &c, &u);
    if (!c && !u) { s0.zone.delay(); apply_all(s0.zone, inv); }
    reset_inactive(net, s0.locs, s0.zone, qmask);
    extrap(s0.zone);
  }

  // Build the full reachable graph (exact-hash nodes, extrapolated for finiteness).
  // SUBSUMPTION, not exact-zone equality.
  //
  // Nodes used to be deduplicated by an exact hash of (discrete key + raw DBM bytes), so two
  // states at the same discrete state whose zones merely CONTAINED one another were stored twice.
  // That is why the liveness graph dwarfed the reachability passed set for the same model:
  // arbiter6 built 1343096 nodes where verifyta stores 10453.
  //
  // Storing an antichain under zone INCLUSION instead is sound here for a reason specific to this
  // engine: DiscPred forbids clock constraints (see netreach.hpp), so `p` and `q` are functions of
  // the DISCRETE key alone and never of the zone. Hence if Z_u subseteq Z_v at the same discrete
  // state, every run of u is a run of v over the same sequence of discrete states, so u can stay
  // in !q forever only if v can. Dropping u loses no witness, and finding none at v proves none
  // at u.
  //
  // Plain inclusion, NOT the aLU simulation the reachability engine uses. aLU is a
  // location-reachability abstraction that may drop individual valuations, and one dropped
  // valuation is exactly what a deadlock witness is made of -- the same reason `E<> deadlock`
  // forces exact zones.
  // aLU SUBSUMPTION, GATED ON DEADLOCK-FREEDOM.
  //
  // Exact inclusion leaves the liveness graph enormously larger than the reachability passed set
  // for the same model: measured on arbiter6, 256 discrete keys carrying 66522 zones -- 260 per
  // discrete state -- where the reachability engine stores exactly one. aLU is what collapses
  // them, and it is a genuine simulation, so for two of the three witnesses it is sound outright:
  // a simulating zone matches every delay (hence divergence transfers) and every discrete step
  // (hence !q cycles transfer), and p/q read only the discrete key, which subsumption preserves.
  //
  // The third witness is the problem. A maximal run may end by DEADLOCKING in !q, and that is a
  // per-valuation property: if v is deadlocked and v is simulated by v', v' may well have
  // successors, so the witness evaporates. That is the same reason `E<> deadlock` sets
  // force_exact.
  //
  // So decide deadlock-freedom first, with the engine's own parallel deadlock check. If no
  // reachable state has a deadlocked valuation then the third witness cannot arise at all and aLU
  // is sound here; otherwise fall back to exact inclusion, which is always sound and is what this
  // code did before. The check costs one reachability run and buys back two orders of magnitude.
  std::vector<std::int32_t> aL = net.lbound, aU = net.ubound;
  aL[0] = aU[0] = 0;   // reference clock: mandatory for the aLU test
  // Per-state bounds, exactly as reach_engine computes them: the max over processes of the
  // per-location bounds, which is far coarser than the global ones away from guards.
  auto state_bounds = [&](const int* locs, std::vector<std::int32_t>& L2,
                          std::vector<std::int32_t>& U2) {
    if (net.lbound_loc.empty()) { L2 = aL; U2 = aU; return; }
    L2.assign(net.nclocks + 1, kAluNegInf);
    U2.assign(net.nclocks + 1, kAluNegInf);
    for (int pr = 0; pr < net.nprocs(); ++pr)
      for (int c = 1; c <= net.nclocks; ++c) {
        L2[c] = std::max(L2[c], net.lbound_loc[pr][locs[pr]][c]);
        U2[c] = std::max(U2[c], net.ubound_loc[pr][locs[pr]][c]);
      }
    L2[0] = U2[0] = 0;
  };
  // LAZY GATING. The search below runs in one of two subsumption modes and is invoked more than
  // once, so everything it owns is declared inside it.
  auto search = [&](bool use_alu) -> bool {
  std::unordered_map<std::string, std::vector<int>> bucket;   // discrete key -> antichain
  std::vector<SymState> node;
  std::vector<std::vector<int>> adj;
  // Per-node analysis state, declared here because intern() has to consult it: a node may only be
  // retired while nothing has looked at it yet.
  std::vector<char> expanded;         // adj[u] filled in?
  std::vector<int> tidx, low, comp_of, fwd;
  std::vector<char> onstk, escapes;
  std::vector<char> pcache, qcache;
  auto grow = [&] {
    const std::size_t n = node.size();
    expanded.resize(n, 0); pcache.resize(n, 2); qcache.resize(n, 2);
    tidx.resize(n, -1); low.resize(n, 0); comp_of.resize(n, -1);
    onstk.resize(n, 0); escapes.resize(n, 0);
    for (std::size_t k = fwd.size(); k < n; ++k) fwd.push_back(static_cast<int>(k));
  };
  // A retired node forwards to the one that absorbed it. Union-find with path compression, so
  // resolving is effectively O(1) and edges stored as plain indices stay valid forever.
  std::function<int(int)> resolve = [&](int v) -> int {
    while (fwd[v] != v) { fwd[v] = fwd[fwd[v]]; v = fwd[v]; }
    return v;
  };
  std::vector<std::int32_t> bL, bU;
  auto intern = [&](SymState&& s) -> int {
    std::vector<int>& b = bucket[disc_key(s.locs, s.ints)];
    const int dim = s.zone.dim();
    // E0 ablation (PTCTLMC_LTNOSUB): fall back to EXACT ZONE EQUALITY, the identity this code
    // used before subsumption was introduced. That version cannot create a spurious cycle,
    // because it never redirects an edge to a different zone, so any verdict difference between
    // the two modes is a subsumption artefact.
    // NODE IDENTITY IS EXACT ZONE EQUALITY BY DEFAULT; subsumption is opt-in (PTCTLMC_LTSUB),
    // because subsumption here was measured to produce a WRONG VERDICT.
    //
    // On grc4 with `T0.in --> T0.far` the subsuming version answers UNSAT where verifyta and this
    // engine's own pre-subsumption implementation both answer SAT. The mechanism is structural,
    // not a coding slip: dropping u because Z_u subseteq Z_v redirects every edge w -> u into
    // w -> v, so a predecessor whose real successor was the smaller zone now appears to step into
    // the larger one. A cycle through v that uses such an edge need not be realised by any run,
    // and a spurious !q cycle is a spurious violation. Subsumption over-approximates runs, so the
    // error is always in this direction: witnesses are invented, never lost.
    //
    // What is NOT affected: divergence and deadlock witnesses are local properties of a stored
    // node, and every stored node is a genuinely reachable symbolic state, so they stay sound
    // under subsumption. Only the CYCLE witness depends on edges. The principled repair is to mark
    // subsumption edges and refuse a cycle that uses one -- the liveness-compatible subsumption
    // criterion of Herbreteau-Srivathsan-Walukiewicz (FMSD 2012) and Laarman et al. (CAV 2013) --
    // rather than to abandon subsumption. Until that is built and tested, the default is the
    // identity that cannot be wrong.
    static const bool lt_nosub = std::getenv("PTCTLMC_LTSUB") == nullptr;
    if (use_alu && !lt_nosub) state_bounds(s.locs.data(), bL, bU);
    auto covered_by = [&](int j) {
      if (lt_nosub)
        return node[j].zone.dim() == dim &&
               std::equal(node[j].zone.data().begin(), node[j].zone.data().end(),
                          s.zone.data().begin());
      return use_alu ? dbm_is_alu_le(s.zone.data().data(), node[j].zone.data().data(), dim,
                                     bL.data(), bU.data())
                     // dbm_includes(a, b) is [b] subseteq [a]
                     : dbm_includes(node[j].zone.data().data(), s.zone.data().data(), dim);
    };
    for (std::size_t bi = 0; bi < b.size(); ++bi)
      if (covered_by(resolve(b[bi]))) return resolve(b[bi]);
    // The incoming zone may instead CONTAIN stored ones, or form an exact convex union with them.
    // Either way the stored node is redundant and is retired into the newcomer -- but only while
    // it is still untouched. A node that has been expanded has an adjacency computed for its own
    // (smaller) zone, and a node the Tarjan pass has entered carries live state; growing or
    // retiring either would invalidate work already done, so those are left alone and simply
    // remain in the antichain. Skipping a retirement only ever costs one extra node.
    const int n = static_cast<int>(node.size());
    std::size_t w = 0;
    for (std::size_t bi = 0; bi < b.size(); ++bi) {
      const int j = resolve(b[bi]);
      const bool untouched = !expanded[j] && tidx[j] == -1;
      // The convex-union merge is OPT-IN here (PTCTLMC_LTMERGE), unlike in the reachability
      // engine where it is the default and a clear win. Measured on the liveness graph it is a
      // net LOSS on half the models -- fischer6 55623 -> 48336 but arbiter6 66137 -> 75818 and
      // twolock6 35351 -> 41943. Merging replaces two stored zones with their union, which is a
      // strictly larger zone whose successors differ, so it can generate more distinct downstream
      // states than it saves. The same effect is recorded in this project's TCTLSAT CEGAR notes,
      // where dropping the finer cut points cost 251032 nodes against 68624 because those cut
      // points were also acting as a canonicaliser. Retirement by plain inclusion has no such
      // downside -- it never changes a stored zone, only removes a redundant one -- and is
      // therefore always on.
      static const bool lt_merge = std::getenv("PTCTLMC_LTMERGE") != nullptr;
      const bool covers_j =
          lt_nosub ? false :
          use_alu ? dbm_is_alu_le(node[j].zone.data().data(), s.zone.data().data(), dim,
                                  bL.data(), bU.data())
                  : dbm_includes(s.zone.data().data(), node[j].zone.data().data(), dim);
      if (untouched &&
          (covers_j ||
           (lt_merge && dbm_try_union(s.zone.data().data(), node[j].zone.data().data(), dim)))) {
        fwd[j] = n;          // j is absorbed; every edge already pointing at it resolves here
        continue;            // drop j from the bucket
      }
      b[w++] = b[bi];
    }
    b.resize(w);
    b.push_back(n);
    node.push_back(std::move(s));
    adj.emplace_back();
    grow();
    fwd[n] = n;
    return n;
  };
  auto time_diverges = [&](const SymState& s) -> bool {
    if (net.nclocks == 0) return false;
    Dbm z = s.zone; z.delay();
    std::vector<Constraint> inv; all_invariants(net, s.locs, inv); apply_all(z, inv);
    if (z.empty()) return false;
    for (int i = 1; i <= net.nclocks; ++i) if (z.get(i, 0) != kInf) return false;
    return true;
  };
  // COPY, never move: `search` is invoked more than once (aLU first, exact inclusion as the
  // fallback), and moving the initial state would leave the second call interning an empty one.
  { SymState s0copy = s0; intern(std::move(s0copy)); }

  // ---- on-the-fly emptiness for `p --> q` -----------------------------------------------
  //
  // `p --> q` fails exactly when some REACHABLE state satisfying p /\ !q can stay in !q forever.
  // From such a state that means reaching, through !q edges only, one of three things: a state
  // with no successors (a maximal run ends there without ever seeing q), a state where time
  // diverges (an infinite run stays put), or a !q cycle. Write ESC for the !q states that can.
  //
  // ESC is a least fixpoint over the !q subgraph and Tarjan computes it in ONE pass, because it
  // emits strongly connected components in reverse topological order: when a component is popped,
  // every component reachable from it is already finished, so its own value is decidable on the
  // spot from (i) is it non-trivial, (ii) does a member deadlock or diverge, (iii) does a member
  // have an edge into a finished component that escapes.
  //
  // Everything is generated LAZILY. Successors are computed the first time a node is expanded, so
  // the search stops the moment an escaping p /\ !q state is found rather than after building the
  // whole zone graph. That is the entire difference against the previous implementation, which
  // constructed every reachable node before looking at the question: its cost was a function of
  // the MODEL alone -- 1087 nodes on fischer4 and 8806 on fischer5 no matter what was asked --
  // while verifyta answered the same queries in 1 to 56 states.
  std::vector<char> esc_comp;         // per completed component
  std::vector<int> tstk;
  int counter = 0;

  // node/adj grow while we walk them, so every vector indexed by node id has to grow with them and
  // NO reference into any of them may be held across an intern().
  auto is_p = [&](int i) { grow(); if (pcache[i] == 2) pcache[i] = holds(p, node[i]); return pcache[i] != 0; };
  auto is_q = [&](int i) { grow(); if (qcache[i] == 2) qcache[i] = holds(q, node[i]); return qcache[i] != 0; };
  auto expand = [&](int u) {
    grow();
    if (expanded[u]) return;
    expanded[u] = 1;
    if (stats) ++stats->explored;
    std::vector<SymState> succ;
    successors_net(net, node[u], node[u].zone, L, U, qmask, emode, succ);
    for (auto& ns : succ) { const int v = intern(std::move(ns)); adj[u].push_back(v); }
    grow();
  };

  // Iterative Tarjan restricted to the !q subgraph. Returns true as soon as a component that
  // escapes contains a p /\ !q state -- that state is a witness and the verdict is settled.
  struct Frame { int u; std::size_t i; };
  auto neg_scc = [&](int root) -> bool {
    std::vector<Frame> fs;
    // EAGER ESCAPE. Waiting for a component to be popped before deciding that it escapes is
    // correct but far too late: the whole point of going on-the-fly is to stop at the first
    // witness. The DFS stack is, by construction, a !q path from the root to the node being
    // visited, so the instant ANY node is shown to escape, every node on the stack escapes with
    // it -- and if one of those satisfies p, the query is already settled. Without this the
    // search still terminates with the right answer, but only after the component containing the
    // initial state is finished, which on fischer7 means 748k nodes instead of 1.
    auto stack_has_p = [&]() {
      for (const Frame& f : fs) if (is_p(f.u)) return true;
      return false;
    };
    grow();
    tidx[root] = low[root] = counter++;
    tstk.push_back(root); onstk[root] = 1;
    fs.push_back({root, 0});
    while (!fs.empty()) {
      const int u = fs.back().u;
      expand(u);
      // Witness kind 1, local to one state: no successor at all (a maximal run ends here without
      // ever reaching q), or time diverges (an infinite run stays here). Either way u escapes.
      if (fs.back().i == 0 && (adj[u].empty() || time_diverges(node[u]))) {
        escapes[u] = 1;
        if (stack_has_p()) return true;
      }
      bool descended = false;
      while (fs.back().i < adj[u].size()) {
        const int v = resolve(adj[u][fs.back().i++]);
        if (is_q(v)) continue;                       // !q subgraph only
        // Witness kind 2, a !q cycle: v is on the current stack, so v ->* u -> v is a cycle and
        // every node on the stack reaches it.
        if (tidx[v] != -1 && onstk[v] && stack_has_p()) return true;
        // Witness kind 3: an edge into a finished component that already escapes.
        if (tidx[v] != -1 && !onstk[v] && comp_of[v] >= 0 && esc_comp[comp_of[v]] &&
            stack_has_p())
          return true;
        if (tidx[v] == -1) {
          tidx[v] = low[v] = counter++;
          tstk.push_back(v); onstk[v] = 1;
          fs.push_back({v, 0});
          descended = true;
          break;
        }
        if (onstk[v] && tidx[v] < low[u]) low[u] = tidx[v];
      }
      if (descended) continue;
      if (low[u] == tidx[u]) {
        const int cid = static_cast<int>(esc_comp.size());
        std::vector<int> comp;
        int w;
        do { w = tstk.back(); tstk.pop_back(); onstk[w] = 0; comp.push_back(w); comp_of[w] = cid; }
        while (w != u);
        // (i) non-trivial component, or a self-loop: an infinite !q run cycles here.
        bool esc = comp.size() > 1;
        if (!esc) for (int v : adj[u]) if (resolve(v) == u) { esc = true; break; }
        // (ii) a member deadlocks or lets time diverge. Every member has been expanded, so an
        //      empty adj really is "no successors" and not "not looked yet".
        if (!esc)
          for (int m : comp)
            if (adj[m].empty() || time_diverges(node[m])) { esc = true; break; }
        // (iii) a member reaches an already-finished component that escapes. Reverse topological
        //       order is what makes this a single pass rather than a fixpoint iteration.
        if (!esc)
          for (int m : comp) {
            for (std::size_t k = 0; k < adj[m].size() && !esc; ++k) {
              const int v = resolve(adj[m][k]);
              if (is_q(v) || comp_of[v] == cid) continue;
              if (comp_of[v] >= 0 && esc_comp[comp_of[v]]) esc = true;
            }
            if (esc) break;
          }
        esc_comp.push_back(esc ? 1 : 0);
        if (esc) {
          for (int m : comp) escapes[m] = 1;
          for (int m : comp) if (is_p(m)) return true;   // reachable p /\ !q that stays in !q
        }
      }
      fs.pop_back();
      if (!fs.empty() && low[u] < low[fs.back().u]) low[fs.back().u] = low[u];
    }
    return false;
  };

  // Outer discovery over the FULL reachable graph, LEVEL-SYNCHRONOUS AND PARALLEL.
  //
  // A p /\ !q state may only be reachable through q states, so the !q analysis alone would never
  // see it -- the outer search has to cover everything. That search is where the time goes on the
  // exhaustive (property-holds) queries, because each node costs a full successor computation:
  // delay, constrain, reset, canonicalise, extrapolate, per outgoing edge. It parallelises the
  // same way reach_engine's does, and for the same reason: the expensive part happens OUTSIDE the
  // shared structure, and only interning touches it.
  //
  // The inner !q search stays sequential. Tarjan is inherently so, and it does not need to be
  // parallel: it runs only from accepting nodes and, on every query measured, either settles the
  // verdict within a handful of nodes or is never reached at all because the property holds.
  //
  // Early exit is kept at LEVEL granularity rather than per state. Accepting nodes found in a
  // level are examined once the level closes, which is early enough -- violations sit near the
  // initial state -- while letting the whole level expand concurrently.
  const bool ltdbg = std::getenv("PTCTLMC_LTDEBUG") != nullptr;
  std::vector<char> seen;
  std::vector<int> frontier{0};
  seen.assign(node.size(), 0);
  seen[0] = 1;
  std::mutex gmtx;                       // guards node/adj/bucket/fwd and the id-indexed vectors
  std::atomic<bool> lt_found{false};      // a local witness found by any worker ends the search
  int nthreads = 1;
#ifdef PTCTLMC_HAVE_OPENMP
  if (backend != Backend::Seq) nthreads = omp_get_max_threads();
#endif
  while (!frontier.empty()) {
    std::vector<int> accepting;          // p /\ !q nodes seen in this level
    std::vector<int> next;
    // Expand the level. Successor generation is pure and runs unlocked; only interning the
    // results and publishing the frontier take the lock.
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel num_threads(nthreads)
    {
      std::vector<int> loc_next, loc_acc;
      std::vector<SymState> succ;
#pragma omp for schedule(dynamic, 8) nowait
      for (long fi = 0; fi < static_cast<long>(frontier.size()); ++fi) {
        const int u = frontier[fi];
        SymState cur;
        { std::lock_guard<std::mutex> lk(gmtx); cur = node[u]; if (expanded[u]) continue; }
        succ.clear();
        successors_net(net, cur, cur.zone, L, U, qmask, emode, succ);   // the expensive part
        // Cheapest witness, checked WITHOUT the lock: this state alone satisfies p, violates q,
        // and already stays in !q forever -- no successors at all, or time diverges in it. It
        // depends only on this thread's own copy, so waiting for the level to close before
        // noticing would just make the answer later (phil6 9 nodes -> 29 when deferred).
        if (holds(p, cur) && !holds(q, cur) && (succ.empty() || time_diverges(cur)))
          lt_found.store(true, std::memory_order_relaxed);
        if (lt_found.load(std::memory_order_relaxed)) continue;
        std::lock_guard<std::mutex> lk(gmtx);
        if (expanded[u]) continue;
        expanded[u] = 1;
        if (stats) ++stats->explored;
        // Two steps on purpose: intern() appends to `adj`, which can REALLOCATE it, so the
        // reference implied by `adj[u].push_back(intern(...))` may dangle mid-expression.
        for (auto& ns : succ) { const int v = intern(std::move(ns)); adj[u].push_back(v); }
        if (is_p(u) && !is_q(u)) loc_acc.push_back(u);
        for (std::size_t k = 0; k < adj[u].size(); ++k) {
          const int v = resolve(adj[u][k]);
          if (seen.size() < node.size()) seen.resize(node.size(), 0);
          if (!seen[v]) { seen[v] = 1; loc_next.push_back(v); }
        }
      }
#pragma omp critical
      {
        next.insert(next.end(), loc_next.begin(), loc_next.end());
        accepting.insert(accepting.end(), loc_acc.begin(), loc_acc.end());
      }
    }
#else
    {
      std::vector<SymState> succ;
      for (std::size_t fi = 0; fi < frontier.size(); ++fi) {
        const int u = frontier[fi];
        if (expanded[u]) continue;
        expand(u);
        if (is_p(u) && !is_q(u)) accepting.push_back(u);
        for (std::size_t k = 0; k < adj[u].size(); ++k) {
          const int v = resolve(adj[u][k]);
          if (seen.size() < node.size()) seen.resize(node.size(), 0);
          if (!seen[v]) { seen[v] = 1; next.push_back(v); }
        }
      }
    }
#endif
    if (lt_found.load()) return false;
    // Settle the level's accepting nodes. Cheapest witness first: the state alone already stays
    // in !q forever. Otherwise ask whether it can REACH such a state or a !q cycle, memoised
    // across calls through comp_of/esc_comp so every node is analysed at most once overall.
    for (int u : accepting) {
      if (adj[u].empty() || time_diverges(node[u])) return false;
      if (comp_of[u] >= 0) { if (escapes[u]) return false; }
      else if (neg_scc(u)) return false;
      else if (escapes[u]) return false;
    }
    frontier.swap(next);
  }
  if (ltdbg)
    std::fprintf(stderr, "[lt] FINAL nodes=%zu discrete-keys=%zu zones/dk=%.2f\n",
                 node.size(), bucket.size(),
                 bucket.empty() ? 0.0 : double(node.size()) / double(bucket.size()));
  return true;
  };

  // Which mode, and whether the deadlock gate has to be paid at all.
  //
  // aLU is a simulation, so it can only ever LOSE a witness, never invent one -- every witness it
  // reports sits at a real stored node whose own zone was tested. A witness under aLU is
  // therefore genuine and settles the query on the spot. It is the ABSENCE of a witness that is
  // not conclusive, because aLU may have dropped the valuation a deadlock witness is made of.
  //
  // So: search with aLU first, and pay for deadlock-freedom only when the answer would otherwise
  // rest on it. This matters because the gate is a COMPLETE reachability traversal in exact mode
  // -- measured at 169252 states and 3.86s on fischer7, against 6 nodes for the liveness query
  // itself. Running it up front made every counterexample query, which is where this engine is
  // orders of magnitude ahead of verifyta, pay for a traversal whose answer it never needed.
  if (!network_has_diagonal(net)) {
    if (!search(/*use_alu=*/true)) return false;      // genuine witness: done, gate never paid
    // No witness under aLU. Two ways to finish, and they compute the same verdict:
    //
    //   (A) prove the model deadlock-free, which makes the aLU answer above sound;
    //   (B) redo the search with exact inclusion, which is sound unconditionally.
    //
    // NEITHER DOMINATES, and which one wins cannot be predicted -- deciding it is as hard as the
    // query. Measured in symbolic states: fischer7 is 180420 for (A) against 30994 for (B), a
    // factor of 5.8 the one way; arbiter6 is 28591 against 66393, a factor of 2.3 the other.
    //
    // So run both and take whichever finishes first, cancelling the loser. This is where extra
    // cores buy something a sequential tool cannot have at any price: it does not make either
    // strategy faster, it removes the need to choose between them. The race is sound because both
    // branches decide the same proposition -- (A) says the aLU run was conclusive, (B) computes
    // the answer directly -- so whichever reports first is reporting the truth.
    // PTCTLMC_NORACE: take lane A then lane B sequentially, the pre-race behaviour, so the two
    // strategies can be compared in one binary.
    static const bool no_race = std::getenv("PTCTLMC_NORACE") != nullptr;
    if (no_race) {
      ReachStats dl_seq;
      if (!deadlock_reachable(net, backend, &dl_seq)) return true;
      return search(/*use_alu=*/false);
    }
    std::atomic<bool> stop{false};
    std::atomic<int> verdict{-1};        // 0 = property holds, 1 = violated
    std::atomic<bool> gate_says_safe{false}, gate_done{false};

    std::thread lane_b([&] {
      const bool holds = search(/*use_alu=*/false);
      if (!stop.exchange(true)) verdict.store(holds ? 0 : 1, std::memory_order_release);
    });
    {
      ReachStats dl_stats;
      const bool may_dl = deadlock_reachable(net, backend, &dl_stats, &stop);
      // A cancelled reach_engine returns false, which is indistinguishable from "no deadlock
      // reachable" -- so only trust this lane if it was NOT the one cancelled.
      if (!stop.load(std::memory_order_acquire)) {
        gate_says_safe.store(!may_dl);
        gate_done.store(true);
        if (!may_dl && !stop.exchange(true))
          verdict.store(0, std::memory_order_release);   // aLU found no witness and was sound
      }
    }
    lane_b.join();
    const int v = verdict.load(std::memory_order_acquire);
    if (v >= 0) return v == 0;
    // The gate finished first and reported that the model CAN deadlock, so the aLU answer is not
    // usable and lane B was cancelled before finishing. Redo it, uncancelled.
    return search(/*use_alu=*/false);
  }
  // Diagonals: aLU is unusable outright, so exact inclusion is the only option.
  return search(/*use_alu=*/false);
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
static bool state_has_deadlock(const Network& net, const SymState& s) {
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
    if (!guarded.empty()) live.push_back(std::move(guarded));
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
      std::vector<std::pair<int, std::vector<int>>> parts;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        std::vector<int> es;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan == se.sync_chan && re.sync_dir == -1 &&
              int_ok(re.int_guard, s.ints) && gen_ok(net, re, s.ints.data(), s.ints.size()))
            es.push_back(ej);
        }
        if (!es.empty()) parts.push_back({j, std::move(es)});
      }
      std::vector<std::size_t> pick(parts.size(), 0);
      for (;;) {
        bool part_committed = net.procs[i].locations[s.locs[i]].committed;
        for (const auto& pr : parts)
          if (net.procs[pr.first].locations[s.locs[pr.first]].committed) part_committed = true;
        if (!src_committed || part_committed) {
          Dbm g = Z;
          bool ok = apply_all(g, se.clock_guard);
          std::vector<const std::vector<int>*> rls{&se.clock_resets};
          LocVec tl = s.locs; tl[i] = se.tgt;
          for (std::size_t t = 0; ok && t < parts.size(); ++t) {
            const NEdge& re = net.procs[parts[t].first].edges[parts[t].second[pick[t]]];
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
  return !fed_subtract(Z, live).empty();
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
bool deadlock_reachable(const Network& net, Backend backend, ReachStats* stats,
                        const std::atomic<bool>* cancel) {
  // The GPU drivers cannot express this target, and reach_engine keeps force_exact queries on
  // the CPU regardless; ask for the parallel CPU backend rather than silently running serially.
  const Backend b = (backend == Backend::Seq) ? Backend::Seq : Backend::Omp;
  return reach_engine(net, NetTarget{},
                      [&](const SymState& s) { return state_has_deadlock(net, s); },
                      /*force_exact=*/true, b, stats, nullptr, nullptr, cancel);
}

}  // namespace ptctlmc
