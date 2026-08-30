/// @file symmetry.cpp
/// @brief Discovery of the network's process-symmetry automorphism group.
///
/// An automorphism permutes processes, and with them clocks, channels and integer values, while
/// preserving the transition relation. hora::set_symmetry then explores one representative per
/// orbit.
///
/// @par Shape of the search
/// 1. Give every process a **shape signature**: its template with clock and channel identifiers
///    replaced by the order in which they are first used. Two processes can map to one another
///    only if their signatures are equal, and when they are, matching them position by position
///    determines the clock and channel correspondence between them.
/// 2. **Seed** a candidate with "process 0 maps to q" for each q of the same shape, which fixes
///    the clock and channel permutations on everything process 0 touches.
/// 3. **Propagate**: a process whose clocks and channels are already mapped has at most one
///    possible image. Fixing it maps further clocks and channels, which determines further
///    processes, and so on.
/// 4. **Verify** the finished candidate against the network, edge by edge.
///
/// @warning Steps 1–3 are heuristics; step 4 is not. Only ::verify decides whether a candidate is
///          an automorphism, so a search that guesses badly costs a missed reduction and can never
///          produce a wrong answer. Any future shortcut must preserve that division.
#include "hora/symmetry.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace hora {
namespace {

/// @brief Order-of-first-use numbering of clocks, channels and integer constants.
///
/// Shared by the signature builder and the seed matcher, so that the numbering two processes are
/// compared under is built the same way for both.
///
/// @invariant Clock slot 0 is the reference clock and is never permuted.
struct Slots {
  std::vector<int> clk, chan;                      // slot -> id
  std::unordered_map<int, int> clk_of, chan_of;    // id -> slot
  // Per integer variable, the constants it is compared with or assigned, in order of first use.
  // Matching two processes slot by slot then pairs a constant in one with the corresponding
  // constant in the other, which is exactly the value permutation the process permutation
  // induces — necessary because a variable may hold a process index.
  std::vector<std::vector<long>> val;
  std::vector<std::unordered_map<long, int>> val_of;
  void ensure_vars(std::size_t n) { if (val.size() < n) { val.resize(n); val_of.resize(n); } }
  int val_slot(int v, long c) {
    auto it = val_of[v].find(c);
    if (it != val_of[v].end()) return it->second;
    const int s = static_cast<int>(val[v].size());
    val_of[v].emplace(c, s); val[v].push_back(c);
    return s;
  }
  int clk_slot(int c) {
    if (c == 0) return 0;                          // the reference clock is fixed
    auto it = clk_of.find(c);
    if (it != clk_of.end()) return it->second;
    const int s = static_cast<int>(clk.size()) + 1;
    clk_of.emplace(c, s); clk.push_back(c);
    return s;
  }
  int chan_slot(int a) {
    auto it = chan_of.find(a);
    if (it != chan_of.end()) return it->second;
    const int s = static_cast<int>(chan.size());
    chan_of.emplace(a, s); chan.push_back(a);
    return s;
  }
};

/// @brief Append a value to a byte-string key, little-endian and fixed-width.
/// @note Fixed width matters: a variable-width encoding would let two different automorphisms
///       serialise to the same bytes and be deduplicated as one.
void put(std::string& s, long v) {
  s.push_back(static_cast<char>(v & 0xff));
  s.push_back(static_cast<char>((v >> 8) & 0xff));
  s.push_back(static_cast<char>((v >> 16) & 0xff));
  s.push_back(static_cast<char>((v >> 24) & 0xff));
}

/// @brief Build a process's shape signature.
///
/// @param p     the process
/// @param[in,out] sl slot numbering, extended as new clocks, channels and constants are seen
/// @param nvars number of declared integer variables
/// @return a byte string equal for two processes only if they have the same shape
///
/// @note The signature is index-free in clocks and channels but **not** in locations or edges.
///       Templates instantiated from one source enumerate those identically, and requiring that
///       is what keeps the correspondence unambiguous and the search linear. A model whose copies
///       were written in a different order simply yields no symmetry — a missed reduction, not a
///       wrong result.
std::string signature(const NProc& p, Slots& sl, std::size_t nvars) {
  sl.ensure_vars(nvars);
  std::string s;
  put(s, static_cast<long>(p.locations.size()));
  put(s, p.initial);
  for (const auto& l : p.locations) {
    put(s, l.urgent ? 1 : 0);
    put(s, l.committed ? 1 : 0);
    put(s, static_cast<long>(l.invariant.size()));
    for (const auto& c : l.invariant) {
      put(s, sl.clk_slot(c.i)); put(s, sl.clk_slot(c.j)); put(s, c.bound);
    }
  }
  put(s, static_cast<long>(p.edges.size()));
  for (const auto& e : p.edges) {
    put(s, e.src); put(s, e.tgt);
    put(s, static_cast<long>(e.clock_guard.size()));
    for (const auto& c : e.clock_guard) {
      put(s, sl.clk_slot(c.i)); put(s, sl.clk_slot(c.j)); put(s, c.bound);
    }
    put(s, static_cast<long>(e.clock_resets.size()));
    for (int r : e.clock_resets) put(s, sl.clk_slot(r));
    put(s, e.sync_dir);
    put(s, e.sync_chan < 0 ? -1 : sl.chan_slot(e.sync_chan));
    put(s, static_cast<long>(e.int_guard.size()));
    for (const auto& g : e.int_guard) {
      put(s, g.var); put(s, static_cast<long>(g.op)); put(s, sl.val_slot(g.var, g.val));
    }
    put(s, static_cast<long>(e.int_updates.size()));
    for (const auto& u : e.int_updates) { put(s, u.var); put(s, sl.val_slot(u.var, u.val)); }
  }
  return s;
}

/// @brief Is process @p p, relabelled by the given permutations, structurally equal to @p q?
///
/// @param p    the source process
/// @param q    the candidate image
/// @param clk  clock permutation
/// @param chan channel permutation
/// @param aut  the candidate automorphism, for its integer-value mapping
/// @param net  the network, for the variables' declared ranges
/// @return @c true iff every location and every edge corresponds exactly
///
/// @note This reads the finished permutation and accepts nothing it cannot check, which is what
///       makes the heuristic search above safe.
bool same_under(const NProc& p, const NProc& q, const std::vector<int>& clk,
                const std::vector<int>& chan, const Automorphism& aut, const Network& net) {
  if (p.locations.size() != q.locations.size() || p.edges.size() != q.edges.size()) return false;
  if (p.initial != q.initial) return false;
  auto cs_eq = [&](const std::vector<Constraint>& a, const std::vector<Constraint>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
      if (clk[a[i].i] != b[i].i || clk[a[i].j] != b[i].j || a[i].bound != b[i].bound) return false;
    return true;
  };
  for (std::size_t l = 0; l < p.locations.size(); ++l) {
    if (p.locations[l].urgent != q.locations[l].urgent) return false;
    if (p.locations[l].committed != q.locations[l].committed) return false;
    if (!cs_eq(p.locations[l].invariant, q.locations[l].invariant)) return false;
  }
  for (std::size_t i = 0; i < p.edges.size(); ++i) {
    const NEdge& a = p.edges[i];
    const NEdge& b = q.edges[i];
    if (a.src != b.src || a.tgt != b.tgt || a.sync_dir != b.sync_dir) return false;
    if ((a.sync_chan < 0) != (b.sync_chan < 0)) return false;
    if (a.sync_chan >= 0 && chan[a.sync_chan] != b.sync_chan) return false;
    if (!cs_eq(a.clock_guard, b.clock_guard)) return false;
    if (a.clock_resets.size() != b.clock_resets.size()) return false;
    for (std::size_t k = 0; k < a.clock_resets.size(); ++k)
      if (clk[a.clock_resets[k]] != b.clock_resets[k]) return false;
    if (a.int_guard.size() != b.int_guard.size()) return false;
    for (std::size_t k = 0; k < a.int_guard.size(); ++k) {
      const IntAtom& g = a.int_guard[k];
      const IntAtom& h = b.int_guard[k];
      if (g.var != h.var || g.op != h.op) return false;
      if (aut.map_val(g.var, g.val, net.intvars[g.var].lo) != h.val) return false;
    }
    if (a.int_updates.size() != b.int_updates.size()) return false;
    for (std::size_t k = 0; k < a.int_updates.size(); ++k) {
      const IntAsgn& g = a.int_updates[k];
      const IntAsgn& h = b.int_updates[k];
      if (g.var != h.var) return false;
      if (aut.map_val(g.var, g.val, net.intvars[g.var].lo) != h.val) return false;
    }
  }
  return true;
}

/// @brief Decide whether a candidate really is an automorphism of the network.
/// @param net the network
/// @param a   the candidate
/// @return @c true iff @p a is a bijection on processes and preserves every process's structure
/// @post A @c true result is a guarantee, not a heuristic: the caller may reduce by @p a.
bool verify(const Network& net, const Automorphism& a) {
  const int np = net.nprocs();
  std::vector<char> seen(np, 0);
  for (int p = 0; p < np; ++p) {
    if (a.proc[p] < 0 || a.proc[p] >= np || seen[a.proc[p]]) return false;
    seen[a.proc[p]] = 1;
  }
  if (a.clk[0] != 0) return false;
  std::vector<char> cs(net.nclocks + 1, 0);
  for (int c = 1; c <= net.nclocks; ++c) {
    if (a.clk[c] < 1 || a.clk[c] > net.nclocks || cs[a.clk[c]]) return false;
    cs[a.clk[c]] = 1;
  }
  const int nch = static_cast<int>(net.channels.size());
  std::vector<char> hs(nch, 0);
  for (int h = 0; h < nch; ++h) {
    if (a.chan[h] < 0 || a.chan[h] >= nch || hs[a.chan[h]]) return false;
    hs[a.chan[h]] = 1;
    if (net.chan_broadcast[h] != net.chan_broadcast[a.chan[h]]) return false;
  }
  for (std::size_t v = 0; v < net.intvars.size(); ++v) {
    if (v >= a.ival.size() || a.ival[v].empty()) continue;   // left fixed
    const IntVar& iv = net.intvars[v];
    const std::size_t dom = static_cast<std::size_t>(iv.hi - iv.lo + 1);
    if (a.ival[v].size() != dom) return false;
    std::vector<char> hit(dom, 0);
    for (const long x : a.ival[v]) {
      if (x < iv.lo || x > iv.hi) return false;
      if (hit[static_cast<std::size_t>(x - iv.lo)]) return false;   // not a bijection
      hit[static_cast<std::size_t>(x - iv.lo)] = 1;
    }
    // The initial valuation has to be a fixed point, otherwise g(initial) need not be reachable
    // and "s reachable => g(s) reachable" -- the whole basis of exploring one per orbit -- fails.
    if (a.map_val(static_cast<int>(v), iv.init, iv.lo) != iv.init) return false;
  }
  for (int p = 0; p < np; ++p)
    if (!same_under(net.procs[p], net.procs[a.proc[p]], a.clk, a.chan, a, net)) return false;
  return true;
}

/// @brief Try to complete an automorphism from the seed "process 0 maps to q".
///
/// @param net   the network
/// @param sig   per-process shape signatures
/// @param slots per-process slot numbering
/// @param q     the image chosen for process 0
/// @param[out] out receives the automorphism; written only on @c true
/// @return @c false when propagation stalls, contradicts itself, or the result fails ::verify
///
/// @post A @c true result has already passed ::verify, so callers need not re-check it.
bool build_from_seed(const Network& net, const std::vector<std::string>& sig,
                     std::vector<Slots>& slots, int q, Automorphism& out) {
  const int np = net.nprocs();
  const int nch = static_cast<int>(net.channels.size());
  Automorphism a;
  a.proc.assign(np, -1);
  a.clk.assign(net.nclocks + 1, -1);
  a.chan.assign(nch, -1);
  a.clk[0] = 0;
  // Value permutations start as "unbound" (-1 per slot) over each variable's declared domain and
  // are filled in as processes are matched; anything still unbound at the end maps to itself.
  a.ival.resize(net.intvars.size());
  std::vector<std::vector<long>> vmap(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v)
    vmap[v].assign(static_cast<std::size_t>(net.intvars[v].hi - net.intvars[v].lo + 1), -1);

  auto bind_clk = [&](int from, int to) {
    if (a.clk[from] != -1) return a.clk[from] == to;
    a.clk[from] = to;
    return true;
  };
  auto bind_chan = [&](int from, int to) {
    if (a.chan[from] != -1) return a.chan[from] == to;
    a.chan[from] = to;
    return true;
  };
  auto bind_val = [&](int v, long from, long to) {
    const IntVar& iv = net.intvars[v];
    if (from < iv.lo || from > iv.hi || to < iv.lo || to > iv.hi) return false;
    long& slot = vmap[v][static_cast<std::size_t>(from - iv.lo)];
    if (slot != -1) return slot == to;
    slot = to;
    return true;
  };
  // Map process p onto process r; their signatures are equal, so slot k of p corresponds to
  // slot k of r and the clock/channel bindings follow.
  auto bind_proc = [&](int p, int r) {
    if (a.proc[p] != -1) return a.proc[p] == r;
    a.proc[p] = r;
    const Slots& sp = slots[p];
    const Slots& sr = slots[r];
    if (sp.clk.size() != sr.clk.size() || sp.chan.size() != sr.chan.size()) return false;
    for (std::size_t k = 0; k < sp.clk.size(); ++k)
      if (!bind_clk(sp.clk[k], sr.clk[k])) return false;
    for (std::size_t k = 0; k < sp.chan.size(); ++k)
      if (!bind_chan(sp.chan[k], sr.chan[k])) return false;
    for (std::size_t v = 0; v < sp.val.size(); ++v) {
      if (sp.val[v].size() != sr.val[v].size()) return false;
      for (std::size_t k = 0; k < sp.val[v].size(); ++k)
        if (!bind_val(static_cast<int>(v), sp.val[v][k], sr.val[v][k])) return false;
    }
    return true;
  };

  if (sig[0] != sig[q]) return false;
  if (!bind_proc(0, q)) return false;

  // Propagate on partial information: an unmapped process is narrowed by whichever of its clocks
  // and channels already have images, and is fixed as soon as one candidate survives. Requiring
  // every slot to be bound first would stall immediately -- seeding "fork 0 -> fork 1" binds two
  // of a philosopher's four channels, and it is exactly those two that pin down which
  // philosopher it must become.
  std::vector<char> taken(np, 0);
  for (int p = 0; p < np; ++p) if (a.proc[p] != -1) taken[a.proc[p]] = 1;
  for (bool progress = true; progress;) {
    progress = false;
    for (int p = 0; p < np; ++p) {
      if (a.proc[p] != -1) continue;
      const Slots& sp = slots[p];
      int found = -1, ncand = 0;
      for (int r = 0; r < np && ncand < 2; ++r) {
        if (taken[r] || sig[r] != sig[p]) continue;
        const Slots& sr = slots[r];
        if (sr.clk.size() != sp.clk.size() || sr.chan.size() != sp.chan.size()) continue;
        bool ok = true;
        for (std::size_t k = 0; ok && k < sp.clk.size(); ++k)
          if (a.clk[sp.clk[k]] != -1) ok = (a.clk[sp.clk[k]] == sr.clk[k]);
        for (std::size_t k = 0; ok && k < sp.chan.size(); ++k)
          if (a.chan[sp.chan[k]] != -1) ok = (a.chan[sp.chan[k]] == sr.chan[k]);
        for (std::size_t v = 0; ok && v < sp.val.size(); ++v) {
          if (sp.val[v].size() != sr.val[v].size()) { ok = false; break; }
          for (std::size_t k = 0; ok && k < sp.val[v].size(); ++k) {
            const long b = vmap[v][static_cast<std::size_t>(sp.val[v][k] - net.intvars[v].lo)];
            if (b != -1) ok = (b == sr.val[v][k]);
          }
        }
        if (ok) { if (ncand == 0) found = r; ++ncand; }
      }
      if (ncand == 0) return false;      // no consistent image: this seed is not an automorphism
      if (ncand > 1) continue;           // still ambiguous; another binding may resolve it
      if (!bind_proc(p, found)) return false;
      taken[found] = 1;
      progress = true;
    }
  }
  // Propagation alone only finishes when the processes are chained by shared structure -- phil's
  // philosophers and forks share channels, so fixing one fixes the next. Fischer's do not: each
  // process owns a private clock and a private `id` value and touches nothing else, so after the
  // seed every remaining process is equally plausible and the loop above stalls with all of them
  // ambiguous. That ambiguity is the signature of a fully symmetric family, where any completion
  // is as good as any other, so complete it greedily -- lowest unmapped process to lowest
  // consistent image -- and let verify() have the last word.
  for (bool done = false; !done;) {
    done = true;
    for (int p = 0; p < np; ++p) {
      if (a.proc[p] != -1) continue;
      const Slots& sp = slots[p];
      int pick = -1;
      // Candidate order matters for the size of the group we end up with, not its validity. In a
      // fully symmetric family every completion verifies, and taking the lowest free image yields
      // a transposition, whose cyclic closure has order 2. Offering the shifted image first
      // (p + delta, delta fixed by the seed 0 -> q) yields the N-cycle instead, whose closure has
      // order N -- the difference between a 2x and an Nx reduction on Fischer.
      const int delta = q;
      for (int t = 0; t < np && pick < 0; ++t) {
        const int r = (t == 0) ? ((p + delta) % np) : t - (t <= (p + delta) % np ? 1 : 0);
        if (r < 0 || r >= np) continue;
        if (taken[r] || sig[r] != sig[p]) continue;
        const Slots& sr = slots[r];
        if (sr.clk.size() != sp.clk.size() || sr.chan.size() != sp.chan.size()) continue;
        bool ok = true;
        for (std::size_t k = 0; ok && k < sp.clk.size(); ++k)
          if (a.clk[sp.clk[k]] != -1) ok = (a.clk[sp.clk[k]] == sr.clk[k]);
        for (std::size_t k = 0; ok && k < sp.chan.size(); ++k)
          if (a.chan[sp.chan[k]] != -1) ok = (a.chan[sp.chan[k]] == sr.chan[k]);
        for (std::size_t v = 0; ok && v < sp.val.size(); ++v) {
          if (sp.val[v].size() != sr.val[v].size()) { ok = false; break; }
          for (std::size_t k = 0; ok && k < sp.val[v].size(); ++k) {
            const long b = vmap[v][static_cast<std::size_t>(sp.val[v][k] - net.intvars[v].lo)];
            if (b != -1) ok = (b == sr.val[v][k]);
          }
        }
        if (ok) pick = r;
      }
      if (pick < 0) return false;
      if (!bind_proc(p, pick)) return false;
      taken[pick] = 1;
      done = false;
      break;                       // a binding may determine others; re-run propagation first
    }
    if (!done) {
      for (bool progress = true; progress;) {   // re-propagate after the greedy choice
        progress = false;
        for (int p = 0; p < np; ++p) {
          if (a.proc[p] != -1) continue;
          const Slots& sp = slots[p];
          int found = -1, ncand = 0;
          for (int r = 0; r < np && ncand < 2; ++r) {
            if (taken[r] || sig[r] != sig[p]) continue;
            const Slots& sr = slots[r];
            if (sr.clk.size() != sp.clk.size() || sr.chan.size() != sp.chan.size()) continue;
            bool ok = true;
            for (std::size_t k = 0; ok && k < sp.clk.size(); ++k)
              if (a.clk[sp.clk[k]] != -1) ok = (a.clk[sp.clk[k]] == sr.clk[k]);
            for (std::size_t k = 0; ok && k < sp.chan.size(); ++k)
              if (a.chan[sp.chan[k]] != -1) ok = (a.chan[sp.chan[k]] == sr.chan[k]);
            for (std::size_t v = 0; ok && v < sp.val.size(); ++v) {
              if (sp.val[v].size() != sr.val[v].size()) { ok = false; break; }
              for (std::size_t k = 0; ok && k < sp.val[v].size(); ++k) {
                const long b = vmap[v][static_cast<std::size_t>(sp.val[v][k] - net.intvars[v].lo)];
                if (b != -1) ok = (b == sr.val[v][k]);
              }
            }
            if (ok) { if (ncand == 0) found = r; ++ncand; }
          }
          if (ncand == 0) return false;
          if (ncand > 1) continue;
          if (!bind_proc(p, found)) return false;
          taken[found] = 1;
          progress = true;
        }
      }
    }
  }
  for (int p = 0; p < np; ++p) if (a.proc[p] == -1) return false;
  // Unused clocks/channels (none of our models have them) map to themselves.
  for (int c = 1; c <= net.nclocks; ++c) if (a.clk[c] == -1) a.clk[c] = c;
  for (int h = 0; h < nch; ++h) if (a.chan[h] == -1) a.chan[h] = h;
  for (std::size_t v = 0; v < net.intvars.size(); ++v) {
    const IntVar& iv = net.intvars[v];
    bool moved = false;
    for (std::size_t k = 0; k < vmap[v].size(); ++k) {
      if (vmap[v][k] == -1) vmap[v][k] = iv.lo + static_cast<long>(k);   // untouched: fixed
      else if (vmap[v][k] != iv.lo + static_cast<long>(k)) moved = true;
    }
    if (moved) a.ival[v] = vmap[v];   // leave empty when the variable is pointwise fixed
  }
  if (!verify(net, a)) return false;
  out = std::move(a);
  return true;
}

/// @brief Compose two automorphisms, applying @p f first and then @p g.
/// @param f   the automorphism applied first
/// @param g   the automorphism applied second
/// @param net the network, for the variables' declared ranges
/// @return the composite
/// @note Closing the generated set under composition is what makes the result a group, which the
///       orbit canonicalisation requires.
Automorphism compose(const Automorphism& f, const Automorphism& g, const Network& net) {
  Automorphism r;                                                       // g after f
  r.proc.resize(f.proc.size());
  r.clk.resize(f.clk.size());
  r.chan.resize(f.chan.size());
  for (std::size_t i = 0; i < f.proc.size(); ++i) r.proc[i] = g.proc[f.proc[i]];
  for (std::size_t i = 0; i < f.clk.size(); ++i) r.clk[i] = g.clk[f.clk[i]];
  for (std::size_t i = 0; i < f.chan.size(); ++i) r.chan[i] = g.chan[f.chan[i]];
  r.ival.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) {
    const IntVar& iv = net.intvars[v];
    if ((v >= f.ival.size() || f.ival[v].empty()) && (v >= g.ival.size() || g.ival[v].empty()))
      continue;                                                          // both fix v pointwise
    r.ival[v].resize(static_cast<std::size_t>(iv.hi - iv.lo + 1));
    for (std::size_t k = 0; k < r.ival[v].size(); ++k) {
      const long x = iv.lo + static_cast<long>(k);
      r.ival[v][k] = g.map_val(static_cast<int>(v), f.map_val(static_cast<int>(v), x, iv.lo), iv.lo);
    }
  }
  return r;
}

/// @brief Serialise an automorphism to a byte string, for deduplication during closure.
/// @return a key equal for two automorphisms exactly when they act identically
std::string key_of(const Automorphism& a) {
  std::string s;
  for (int v : a.proc) put(s, v);
  for (int v : a.clk) put(s, v);
  for (const auto& m : a.ival) { put(s, static_cast<long>(m.size())); for (long x : m) put(s, x); }
  return s;
}

}  // namespace

/// @brief Apply an automorphism's process permutation to a location vector.
/// @param a    the automorphism
/// @param locs the source location vector
/// @param np   number of processes
/// @param[out] out receives the permuted vector; must not alias @p locs
void permute_locs(const Automorphism& a, const int* locs, int np, int* out) {
  for (int p = 0; p < np; ++p) out[a.proc[p]] = locs[p];
}

/// @brief Compute the network's automorphism group.
///
/// Builds one candidate per admissible seed, keeps those that verify, then closes the survivors
/// under composition.
///
/// @param net the network
/// @param cap maximum group size
/// @return the group, or an **empty** vector if the closure exceeded @p cap
///
/// @post Every returned element has passed ::verify.
/// @note On overflow the result is empty rather than a partial subset, and the caller then runs
///       without symmetry reduction. Returning a partial set would be unsound here: the orbit
///       canonicalisation needs a set closed under composition to pick a well-defined
///       representative, which an arbitrary truncation is not.
std::vector<Automorphism> symmetry_group(const Network& net, std::size_t cap) {
  const int np = net.nprocs();
  Automorphism id;
  id.proc.resize(np);
  for (int p = 0; p < np; ++p) id.proc[p] = p;
  id.clk.resize(net.nclocks + 1);
  for (int c = 0; c <= net.nclocks; ++c) id.clk[c] = c;
  id.chan.resize(net.channels.size());
  for (std::size_t h = 0; h < net.channels.size(); ++h) id.chan[h] = static_cast<int>(h);
  id.ival.resize(net.intvars.size());
  id.identity = true;
  std::vector<Automorphism> group{id};

  // Out of scope: integer variables (a process permutation would generally have to permute
  // index-valued data, e.g. Fischer's `id`, which needs the variable's meaning) and the general
  // expression/function edges (no relabelling defined for them yet).
  if (np < 2 || !net.funcs.empty()) return group;
  for (const auto& p : net.procs)
    for (const auto& e : p.edges) {
      if (e.gen_guard || e.gen_update) return group;
      // Only == / != . Under `<` an arbitrary value permutation is not a symmetry, and nothing
      // here establishes that the permutation is order-preserving.
      for (const auto& g : e.int_guard)
        if (g.op != IntOp::Eq && g.op != IntOp::Ne) return group;
    }

  std::vector<Slots> slots(np);
  std::vector<std::string> sig(np);
  for (int p = 0; p < np; ++p) sig[p] = signature(net.procs[p], slots[p], net.intvars.size());

  // Generators: every image of process 0 that completes to a verified automorphism.
  std::vector<Automorphism> gens;
  for (int q = 1; q < np; ++q) {
    if (sig[q] != sig[0]) continue;
    Automorphism a;
    if (build_from_seed(net, sig, slots, q, a)) gens.push_back(std::move(a));
  }
  if (gens.empty()) return group;

  // Close under composition. Compositions of automorphisms are automorphisms, so the closure
  // needs no re-verification -- but it must be complete. Canonicalisation picks the minimum over
  // G, and if G is missing a composition then rep(g(s)) can differ from rep(s), so a state and
  // its image get explored as different representatives while the target orbit -- also built from
  // G -- no longer contains the element that would recognise the witness. A truncated closure is
  // therefore not "less reduction", it is wrong. Overflow falls back to the largest cyclic
  // subgroup that fits, which is closed by construction.
  auto closure = [&](const std::vector<Automorphism>& gs) {
    std::vector<Automorphism> g{id};
    std::unordered_map<std::string, int> seen;
    seen.emplace(key_of(id), 0);
    for (std::size_t i = 0; i < g.size(); ++i) {
      for (const auto& x : gs) {
        Automorphism c = compose(g[i], x, net);
        if (!seen.emplace(key_of(c), 1).second) continue;
        g.push_back(std::move(c));
        if (g.size() > cap) return std::vector<Automorphism>{};   // overflow: caller falls back
      }
    }
    return g;
  };
  std::vector<Automorphism> full = closure(gens);
  if (!full.empty()) return full;
  std::vector<Automorphism> best = group;                     // {identity}
  for (const auto& g : gens) {
    std::vector<Automorphism> c = closure({g});
    if (c.size() > best.size()) best = std::move(c);
  }
  return best;
}

}  // namespace hora
