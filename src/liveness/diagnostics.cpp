/// @file src/liveness/diagnostics.cpp
/// @brief Reference implementations used to check the online D construction.
///
/// No public query route calls anything here. These derive the same refinement from a finished
/// graph, which is how the online seeding rule is checked against a specification.

#include "hora/zone.hpp"
#include "hora/detail/liveness_engine.hpp"
#ifdef HORA_HAVE_OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#include "hdbm/mindbm.hpp"
#include "hora/detail/semantics.hpp"
#if defined(__APPLE__)
#endif
#include "src/liveness/internal.hpp"

namespace hora {

bool LivenessEngine::deadlock_refinement_is_empty() {
  std::unordered_set<std::string> seen;
  for (std::size_t i = 0; i < nrec(); ++i) {
    const std::string k = disc_key_bytes(node[i].locs, node[i].ints);
    if (!seen.insert(k).second) continue;
    SymState u; u.locs = node[i].locs; u.ints = node[i].ints;
    u.zone = Dbm::universe(net.nclocks);
    Dbm inv = Dbm::universe(net.nclocks);
    bool c2, u2; frozen_flags(*dead_net, u.locs, u.ints, &c2, &u2);
    std::vector<Constraint> ci; all_invariants(*dead_net, u.locs, ci);
    if (!c2 && !u2) inv.delay();
    if (!apply_all(inv, ci)) continue;                        // vacuous key
    Federation stuck, esc;
    if (!state_has_deadlock(*dead_net, u, &stuck, &esc)) continue;   // Dead(q) = empty
    bool open = true;
    for (int c = 1; c <= net.nclocks; ++c)
      if (inv.get(c, 0) != kInf) { open = false; break; }
    // The time-open drop is a claim about divergence, so it is available only where time can
    // pass. At a committed/urgent key no positive delay is legal, so Inv(q) is the undelayed
    // universe and trivially bounds no clock -- exactly when this test would fire and exactly
    // when it would be wrong. Frozen keys fall through to the Dead=Inv test instead.
    if (open && !c2 && !u2) continue;                          // divergence already covers stop
    // Inv \ Dead = Inv ^ union Esc, so "Dead(q) = Inv(q)" is "nothing in Inv(q) escapes": one
    // DBM intersection per outgoing transition, instead of peeling the invariant by a residue.
    bool some_escapes = false;
    for (const Dbm& W : esc) {
      Dbm t = W;
      t.intersect(inv);
      if (!t.empty()) { some_escapes = true; break; }
    }
    if (!some_escapes) continue;                               // Dead(q) = Inv(q): atoms implied
    return false;                                              // PARTIAL: real atoms are needed
  }
  return true;
}


/// @brief Reference-only fixed-map construction from a finished graph.
///
/// No production route calls this routine.  It is retained to compare online seed/pullback
/// behaviour against the old finished-graph formulation.  It derives diagonal cell atoms and
/// non-diagonal LU-bound raises from partial `Dead(q)` regions, then closes them backwards over
/// reset-labelled edges.
///
/// The representation is intentionally limited to 64 diagonal atoms per key because this
/// diagnostic path uses a 64-bit cached classification.  The online path switches to a dynamic
/// classification path and does not inherit that limit.
///
/// @brief Reconstruct a key invariant from its discrete representative.
///
/// Both the online and reference seed rules use an invariant obtained by delay-closing the
/// universe unless the key is frozen, then applying the location invariants.
bool key_invariant(const Network& net, const SymState& rep, int nc, Dbm& out) {
  out = Dbm::universe(nc);
  bool fc, fu;
  frozen_flags(net, rep.locs, rep.ints, &fc, &fu);
  std::vector<Constraint> ci;
  all_invariants(net, rep.locs, ci);
  if (!fc && !fu) out.delay();
  return apply_all(out, ci);
}

bool LivenessEngine::build_atom_map(AtomMap& am) {
  // deadlock IS judged IN @ref dead_net, SO the completion must BE built there too.
  //
  // For `p --> q` the two are the same automaton and this is a no-op. For `E[] psi` they are not:
  // the search runs on the psi-restricted network, psi's clock conjunction having been adjoined to
  // every location invariant, while ::LivenessEngine::stop_set asks
  // hora::detail::state_has_deadlock about the unrestricted one -- maximality is a property of the
  // original system. Deriving Dead(q) from `net` here would complete the covering for the
  // restricted automaton's deadlocks and leave the fixpoint reading the unrestricted automaton's,
  // so the covering could merge two states that differ in exactly the property the completion
  // exists to preserve.

  const int nc = net.nclocks;
  am.diag.clear();
  am.bump.assign(nc + 1, kNoBump);
  am.live.assign(nc + 1, 0);
  am.usable = false;
  if (nc <= 0 || nc >= 63) return false;

  // The reference representation uses a 64-bit atom cache.  It must decline rather than silently
  // omit an atom from the covering relation.
  constexpr std::size_t kMaxAtoms = 64;

  // Store one representative node per key; rebuild its invariant only if backward closure needs
  // it.  The invariant depends on the discrete key, not on the particular reached zone.
  std::unordered_map<std::string, int> invrep;
  std::unordered_map<std::string, Dbm> invmemo;   // rebuilt on demand, only in the pre-closure

  // ---- seed ---------------------------------------------------------------
  for (std::size_t i = 0; i < nrec(); ++i) {
    const std::string k = disc_key_bytes(node[i].locs, node[i].ints);
    if (invrep.count(k)) continue;
    SymState u;
    u.locs = node[i].locs;
    u.ints = node[i].ints;
    u.zone = Dbm::universe(nc);                 // NOT Dbm(nc), which is the ZERO zone
    Dbm inv = Dbm::universe(nc);
    bool fc, fu;
    frozen_flags(*dead_net, u.locs, u.ints, &fc, &fu);
    std::vector<Constraint> ci;
    all_invariants(*dead_net, u.locs, ci);
    if (!fc && !fu) inv.delay();
    if (!apply_all(inv, ci)) continue;                       // vacuous key
    invrep.emplace(k, static_cast<int>(i));
    Federation stuck, esc;
    if (!state_has_deadlock(*dead_net, u, &stuck, &esc)) continue;  // Dead(q) = empty
    bool open = true;
    for (int c = 1; c <= nc; ++c)
      if (inv.get(c, 0) != kInf) { open = false; break; }
    if (open && !fc && !fu) continue;                         // divergence already covers stop
    // `Dead(q) = Inv(q) \ union Esc(q)`.  First detect the constant case using the equivalent
    // question "does any valuation of Inv(q) escape?".  Then discard escape regions disjoint
    // from the invariant: their atoms cannot distinguish a stored valuation at this key.
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
    if (!some_escapes) continue;                              // Dead(q) = Inv(q): atoms implied
    ++am.seed_keys;
    // which zones TO read the atoms off -- two exact representations of the same set, computed both
    // ways and the cheaper kept.
    //
    // Dead(q) = Inv(q) \ union Esc(q), and every stored zone satisfies Inv(q). Membership in
    // Dead(q) is therefore decided either by the atoms of the residue's disjoint pieces, or by the
    // atoms of the escape zones alone -- one convex zone per outgoing transition -- since a
    // valuation is dead iff it escapes by none of them. A non-diagonal atom is weighted four times
    // heavier in the comparison because it is discharged by raising that clock to M globally rather
    // than by a local cell.
    //
    // The atoms come from the minimal graph, not the dense matrix: a canonical DBM's redundant
    // entries are implied by shortest paths through the others, so the non-redundant edges alone
    // decide membership. Reading the dense matrix seeds one atom per derived bound and buys
    // nothing.
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
            // both constant cases. Every stored zone at q satisfies Inv(q), so an atom Inv(q)
            // implies is true on all of them and an atom Inv(q) is disjoint from is false on all
            // of them; either way its tag bit never varies and it distinguishes nothing. On a
            // canonical DBM both are one entry comparison.
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
    std::vector<AtomC> gr, ge;
    std::vector<std::pair<int, std::int32_t>> nr, ne2;
    extract(stuck, gr, nr);
    extract(esc, ge, ne2);
    const std::size_t cr = gr.size() + 4 * nr.size();
    const std::size_t ce = ge.size() + 4 * ne2.size();
    const bool take_esc = ce <= cr;
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

  // No seed means no backward closure obligation.
  {
    bool any_seed = !am.diag.empty();
    for (int c = 1; c <= nc && !any_seed; ++c)
      if (am.bump[c] != kNoBump) any_seed = true;
    if (!any_seed) {
      am.usable = true;
      return true;
    }
  }

  // ---- the key-edge relation, with the union of each transition's resets --
  struct KE { std::string s, t; std::uint64_t r; };
  std::vector<KE> kedges;
  {
    std::unordered_set<std::string> seenE;
    const std::size_t n = std::min(adj.size(), nrec());
    for (std::size_t u = 0; u < n; ++u) {
      const std::string ks = disc_key_bytes(node[u].locs, node[u].ints);
      for (const auto& e : adj[u]) {
        std::uint64_t rm = 0;
        for (const auto& pr : tags_[e.tag].parts)
          for (int rc : net.procs[pr.first].edges[pr.second].clock_resets)
            if (rc >= 1 && rc < 64) rm |= std::uint64_t(1) << rc;
        const std::string kt = disc_key_bytes(node[e.tgt].locs, node[e.tgt].ints);
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
          const Dbm* ivp = nullptr;
          {
            auto mit = invmemo.find(e.s);
            if (mit == invmemo.end()) {
              auto rit = invrep.find(e.s);
              if (rit != invrep.end()) {
                Dbm rebuilt;
                if (key_invariant(*dead_net, node[rit->second], nc, rebuilt))
                  mit = invmemo.emplace(e.s, std::move(rebuilt)).first;
              }
            }
            if (mit != invmemo.end()) ivp = &mit->second;
          }
          if (ivp) {
            const Dbm& iv2 = *ivp;
            if (iv2.get(i2, j2) <= a.b) continue;                             // Inv(q) implies
            // ...and, symmetrically, Inv(q) refutes: the pullback lands outside the source key's
            // invariant, so no zone stored there can ever satisfy it. A closure atom is a
            // constraint from some other key's deadlock, and the further it is pulled back the
            // less it has to do with the invariant it arrives at, so this drop matters more here
            // than at the seed.
            if (raw_add(iv2.get(j2, i2), a.b) < kZeroLe) {
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

  // If every clock reached M, the LU relation itself is symmetric enough to preserve deadlock;
  // keeping cell tags would only add needless splits.
  {
    bool all_at_m = true;
    for (int c = 1; c <= nc && all_at_m; ++c) {
      const std::int32_t mc = std::max(net.lbound[c], net.ubound[c]);
      if (am.bump[c] < mc) all_at_m = false;
    }
    if (all_at_m) am.diag.clear();
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
  // `am.bumped` records how many clocks were raised. A bump is "a_M on one clock", so a map that
  // bumps every clock has already paid the whole a_M price and its cells are pure loss; the
  // all-at-M shortcut above catches the total case, and this counter reports the margin.
  return true;
}

}  // namespace hora
