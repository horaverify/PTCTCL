/// @file src/liveness/engine.cpp
/// @brief LivenessEngine lifetime, configuration, stored zones and the covering test.
///
/// Everything about how a symbolic state is stored and when one covers another. The covering
/// relation and the extrapolation mode must move together; ::LivenessEngine::setup says why.

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

// ---- LivenessEngine lifetime, cache, and stored-zone helpers --------------------------

LivenessEngine::LivenessEngine(const Network& n, const DiscPred& pp, const DiscPred& qq,
         bool force_exact, const std::atomic<bool>* cancel, const Network* dead)
    : net(n), p(pp), q(qq), dead_net(dead ? dead : &n), force_exact_(force_exact),
      cancel_(cancel) {}

bool LivenessEngine::cancelled() {
  if (cancelled_) return true;
  if (cancel_ && cancel_->load(std::memory_order_relaxed)) cancelled_ = true;
  return cancelled_;
}

bool LivenessEngine::is_nq_fast(int i) {
  if (qc[i] == 2) { SymState sc; nstate(i, sc); qc[i] = holds(q, sc) ? 1 : 0; }
  return qc[i] == 0;
}

void LivenessEngine::set_eg_early(bool v) { eg_early_ = v; }

bool LivenessEngine::eg_early_hit() const { return eg_early_hit_; }

bool LivenessEngine::was_cancelled() const { return cancelled_; }

bool LivenessEngine::was_capped() const { return capped; }

void LivenessEngine::set_avoid(const detail::DelayAvoidSet* a, bool forward) {
  avoid_ = a;
  avoid_forward_ = forward;
}

void LivenessEngine::set_atoms(const AtomMap* a) { amap_ = a; }

void LivenessEngine::release() {
  std::vector<SymState>().swap(node);
  std::vector<std::vector<OtfEdge>>().swap(adj);
  std::unordered_map<std::uint64_t, std::vector<int>>().swap(bucket);
  std::vector<GfpTag>().swap(tags_);
  zs_.reset(zdim_);
  std::unordered_map<std::uint64_t, std::vector<int>>().swap(tagidx_);
  std::vector<char>().swap(expanded);
  std::vector<char>().swap(pc);
  std::vector<char>().swap(qc);
  std::vector<std::uint64_t>().swap(ntag_);
  std::vector<std::uint64_t>().swap(ntag_out_);
}

void LivenessEngine::zstore(int i, Dbm&& z) {
  zs_.store(i, z);
  if (!online_d_) node[i].zone = Dbm();   // online records have no SymState to scrub
}

const Dbm& LivenessEngine::zread(int i, Dbm& sc) const {
  zs_.load(i, sc);
  return sc;
}

const SymState& LivenessEngine::sread(int i, SymState& sc) const {
  nstate(i, sc);
  zread(i, sc.zone);
  return sc;
}

/// @brief Extra_LU+(Z_v) under @p v's OWN local bounds, cached; nullptr when it equals Z_v.
///
/// @note The bounds are recomputed from `v`'s locations rather than read from @ref bL / @ref bU.
///       Those members are scratch, configured as a side effect of whichever cover probe ran
///       last, so reading them here made the carrier depend on call order. Using the global
///       arrays instead is deterministic but far too fine -- larger constants widen less, the
///       carrier shrinks, and covering collapses (arbiter12: 0.4s to over 45s). Per-node local
///       bounds are the only choice that is both deterministic and as coarse as the covering
///       test expects, and they are well defined for a comparison because covering only ever
///       relates two nodes that share a discrete key.
/// @brief Raise @p l / @p u by every constant the psi-restriction constrains a clock with.
///
/// Extra_LU+ may not widen a zone back across the psi boundary: the restriction is carried by
/// the avoid set, not by the bounds, so an extrapolation blind to the query's constants merges
/// states the property separates and the fixpoint then reports runs that leave psi. This is the
/// liveness counterpart of ::extend_bounds in the reachability lane.
void LivenessEngine::raise_by_psi(std::vector<std::int32_t>& l, std::vector<std::int32_t>& u) const {
  if (!avoid_) return;
  for (const auto& part : avoid_->parts) {
    const Dbm& f = part.forbid;
    if (f.empty()) continue;
    for (int i = 0; i <= net.nclocks; ++i)
      for (int j = 0; j <= net.nclocks; ++j) {
        if (i == j) continue;
        const raw_t b = f.get(i, j);
        if (b >= kInf) continue;
        const std::int32_t c = raw_bound(b);
        const std::int32_t a = c < 0 ? -c : c;
        if (i >= 1 && i <= net.nclocks) { l[i] = std::max(l[i], a); u[i] = std::max(u[i], a); }
        if (j >= 1 && j <= net.nclocks) { l[j] = std::max(l[j], a); u[j] = std::max(u[j], a); }
      }
  }
}

const Dbm* LivenessEngine::carrier_of(int v) {
  if (carrier_state_.size() < nrec()) { carrier_state_.resize(nrec(), kCarUnknown); carrier_.resize(nrec()); }
  if (carrier_state_[v] == kCarUnknown) {
    Dbm z; zread(v, z);
    Dbm w = z;
    // Forward phase: the caller configured bL/bU for this node's key, and covering only ever
    // compares nodes sharing a key, so the ambient arrays ARE this node's bounds. The discrete
    // half does not exist yet on the online route, so they cannot be recomputed here.
    //
    // Backward phase: NOTHING configures bL/bU, so reading them would extrapolate with whatever
    // key the last cover probe left behind -- non-deterministic, and coarser than this node's
    // own bounds, which is how spurious witnesses appeared on viking. Derive them instead: from
    // the node's locations when the network has per-location bounds, and from the global arrays
    // when it does not.
    std::vector<std::int32_t> l, u;
    if (fc_mode_ || carrier_mode_) {
      if (net.lbound_loc.empty()) {
        l = aL; u = aU;                        // already carry the constraint-map bumps
      } else {
        SymState sc;
        nstate(v, sc);
        local_bounds(sc.locs.data(), l, u);
        if (amap_)
          for (int c = 1; c <= net.nclocks && c < (int)amap_->bump.size(); ++c)
            if (amap_->bump[c] != kNoBump) {
              l[c] = std::max(l[c], amap_->bump[c]);
              u[c] = std::max(u[c], amap_->bump[c]);
            }
      }
    } else {
      l = bL; u = bU;
    }
    raise_by_psi(l, u);
    w.extrapolate_lu_plus(l, u);
    if (w == z) { carrier_state_[v] = kCarSame; carrier_[v] = Dbm(); }
    else { carrier_state_[v] = kCarWide; carrier_[v] = std::move(w); }
  }
  return carrier_state_[v] == kCarWide ? &carrier_[v] : nullptr;
}

bool LivenessEngine::alu_le_stored(const Dbm& z, int j) const {
  if (tight_) {                       // Z subset of Extra_LU+(Z_j); see @ref tight_
    const Dbm* car = const_cast<LivenessEngine*>(this)->carrier_of(j);
    if (!car) return stored_includes(j, z);      // carrier == zone: no copy, as in the baseline
    return car->includes(z);
  }
  return zs_.with(j, [&](const auto* pz) {
    return dbm_is_alu_le(z.data().data(), pz, zdim_, bL.data(), bU.data());
  });
}

bool LivenessEngine::alu_le_new(int j, const Dbm& z) const {
  if (tight_) {                       // Z_j subset of Extra_LU+(Z)
    Dbm car = z;
    std::vector<std::int32_t> l = bL, u = bU; raise_by_psi(l, u);
    car.extrapolate_lu_plus(l, u);   // newcomer shares the key; psi constants raised in
    return includes_stored(car, j);
  }
  return zs_.with(j, [&](const auto* pz) {
    return dbm_is_alu_le(pz, z.data().data(), zdim_, bL.data(), bU.data());
  });
}

bool LivenessEngine::stored_includes(int j, const Dbm& z) const {
  return zs_.with(j, [&](const auto* pz) {
    return dbm_includes(pz, z.data().data(), zdim_);
  });
}

bool LivenessEngine::includes_stored(const Dbm& z, int j) const {
  return zs_.with(j, [&](const auto* pz) {
    return dbm_includes(z.data().data(), pz, zdim_);
  });
}

bool LivenessEngine::base_cover(const Dbm& a, const Dbm& b) const {
  if (use_alu && tight_) {
    Dbm car = b;
    std::vector<std::int32_t> l = bL, u = bU; raise_by_psi(l, u);
    car.extrapolate_lu_plus(l, u);
    return car.includes(a);
  }
  return use_alu ? a.alu_subsumed_by(b, bL, bU) : b.includes(a);
}

/// @brief Choose the covering discipline and the extrapolation mode for this run.
///
/// @post `use_alu` and `emode` are consistent: the search terminates, and every stored zone is
///       either exact (a_LU covering) or extrapolated (inclusion covering) — never both.
/// @post `qmask` marks every clock the constraint map reads, so ::reset_inactive keeps them.
///
/// @invariant **Stored zones are exact whenever a_LU covering is in force.** Finiteness of the
/// zone graph then comes from a_LU subsumption, not from extrapolation, and the two settings
/// must move together: plain inclusion does not bound the zone graph, so with a_LU off the only
/// thing that terminates the search is Extra_LU+, and with Extra_LU+ on every stored zone
/// over-approximates the reachable set. See docs/design.md, "Stored zones must be exact", for the
/// counterexample; in short, ::LivenessEngine::stop_set asks whether some valuation of the stored zone
/// enables *nothing*, enabledness is monotone rather than invariant under extrapolation, and a
/// spurious stop valuation occurs positively in the fixpoint operator, so it fabricates a
/// counterexample rather than losing one.
///
/// @note Diagonal models cannot use a_LU at all (Bouyer 2004), so they keep ExNoDiag with
///       inclusion covering and extrapolation stays load-bearing there.
void LivenessEngine::setup() {
  zdim_ = net.nclocks + 1;
  zs_.reset(zdim_, packed_zones_on());
  L = net.lbound;
  U = net.ubound;
  M.resize(net.nclocks + 1);
  for (int c = 0; c <= net.nclocks; ++c) M[c] = std::max(net.lbound[c], net.ubound[c]);
  M[0] = 0;
  if (std::getenv("HORA_DBG_LU")) {
    std::fprintf(stderr, "[LU] ");
    for (int c = 1; c <= net.nclocks && c < 6; ++c)
      std::fprintf(stderr, "c%d L=%d U=%d  ", c,
                   c < (int)net.lbound.size() ? net.lbound[c] : -9,
                   c < (int)net.ubound.size() ? net.ubound[c] : -9);
    std::fprintf(stderr, "\n");
  }
  use_alu = !force_exact_ && !network_has_diagonal(net);
  tight_ = std::getenv("HORA_TIGHT") != nullptr;
  // ExNone relies on a_LU/D subsumption alone for finiteness.  The lazy refinement loop suspends
  // that subsumption on demand, so on its route the graph needs a bound that survives having
  // covers taken away; see ::LivenessEngine::lazy_mode_.
  // HORA_LAZY_XP=1 puts an Extra_LU+ floor under the lazy route.  OFF by default, and the
  // reason is measured: the floor exists only to bound REFINEMENT, and once the recurrence
  // filter proves a component cannot fire infinitely often the loop materialises nothing at all.
  // On every benchmark row the obligation set is empty, so the floor protects nothing and costs
  // exploration order -- arbiter8 q5 12 -> 15 records, csma10 q5 14 -> 16, phil10 q4 13 -> 14,
  // all early-terminated SAT rows where the first witness simply turns up later.  With it off
  // the lazy route is bit-identical to the default one.
  //
  // It is still the right thing to switch on for a model where the filter leaves a LIVE
  // component, because there refinement can run and `emode == ExNone` leaves a_LU/D subsumption
  // as the only finiteness argument -- which is precisely what refinement suspends.  Deciding
  // that at ::LivenessEngine::setup time needs the syntactic form of the filter, which reads only
  // per-process location graphs and so can run before any graph exists.
  static const bool lazy_xp = std::getenv("HORA_LAZY_XP") != nullptr;
  emode = network_has_diagonal(net) ? ExNoDiag
                                    : ((use_alu && !(lazy_mode_ && lazy_xp)) ? ExNone : ExLuPlus);
  // Clocks the query forces to stay live. A leads-to predicate carries no clock constraints, so
  // this starts all-zero; reset_inactive reads it as "additionally keep alive", so all-zero is
  // maximal reduction, not none. The reduction is safe for a backward analysis because exact_predecessor
  // frees every clock dead at the target before undoing the step.
  qmask.assign(net.nclocks + 1, 0);
  aL = net.lbound;
  aU = net.ubound;
  aL[0] = aU[0] = 0;
  build_local_bound_tables();
  // The constraint map has two global effects (see ::AtomMap):
  //
  // (1) A non-diagonal atom is discharged by raising its clock's L and U. Raising either only
  //     ever makes a_LU finer, so it is sound wherever applied; applying it to the global
  //     arrays here and to the per-location arrays in LivenessEngine::intern keeps it consistent along
  //     every path, which is what the simulation argument needs.
  // (2) reset_inactive pins clocks no guard will read again to 0, but the atoms DO read them --
  //     that is the point of Dead(q) -- so every clock the map mentions joins qmask. Without
  //     this the reduction erases exactly the differences the cell tag exists to observe.
  //
  // The pinning in (2) is kept for seed atoms as well as closure atoms even though seed clocks
  // are arguably active at their own key by construction: `active` is a static
  // over-approximation, so the argument is not a proof and the conservative choice stands.
  if (amap_) {
    for (int c = 1; c <= net.nclocks && c < (int)amap_->bump.size(); ++c) {
      if (amap_->bump[c] != kNoBump) {
        aL[c] = std::max(aL[c], amap_->bump[c]);
        aU[c] = std::max(aU[c], amap_->bump[c]);
      }
      if (amap_->live[c]) qmask[c] = 1;
    }
  }
  // Online D deliberately uses the same active-clock reduction as the ordinary construction.
  // This remains safe while its map grows: a seed atom reads a clock in an invariant or an
  // outgoing guard at that key, so compute_active keeps it. Pullback either resets that clock
  // (making its old value irrelevant) or propagates the target's active bit to the source.
  // Thus a D-relevant value cannot be erased before a later atom needs it.
}

/// @brief Precompute what the per-location LU fold actually depends on.
///
/// `L[c]` at a discrete state is `max_p lbound_loc[p][locs[p]][c]`, an `nprocs x nclocks` fold
/// that ::online_configure_bounds and ::LivenessEngine::intern both need per key. Almost none of it varies:
/// a clock is typically constrained by the one process that owns it, and every other process
/// contributes the same value from every one of its locations.
///
/// So split the fold. A process whose bound for clock `c` is the same at all of its locations
/// contributes a constant, folded once here into @ref base_l_ / @ref base_u_. Only the
/// (process, clock) pairs that really vary stay in @ref vary_clocks_, and the per-key fold walks
/// those alone, for the identical result.
void LivenessEngine::build_local_bound_tables() {
  const int dim = net.nclocks + 1;
  base_l_.assign(dim, kAluNegInf);
  base_u_.assign(dim, kAluNegInf);
  vary_clocks_.assign(net.nprocs(), {});
  if (net.lbound_loc.empty()) {          // no per-location analysis: the global arrays are it
    base_l_.assign(net.lbound.begin(), net.lbound.end());
    base_u_.assign(net.ubound.begin(), net.ubound.end());
    base_l_.resize(dim, kAluNegInf);
    base_u_.resize(dim, kAluNegInf);
    return;
  }
  for (int pr = 0; pr < net.nprocs(); ++pr) {
    const auto& lp = net.lbound_loc[pr];
    const auto& up = net.ubound_loc[pr];
    if (lp.empty()) continue;
    for (int c = 1; c <= net.nclocks; ++c) {
      bool varies = false;
      for (std::size_t l = 1; l < lp.size() && !varies; ++l)
        varies = lp[l][c] != lp[0][c] || up[l][c] != up[0][c];
      if (varies) vary_clocks_[pr].push_back(c);
      else {
        base_l_[c] = std::max(base_l_[c], lp[0][c]);
        base_u_[c] = std::max(base_u_[c], up[0][c]);
      }
    }
  }
  base_l_[0] = base_u_[0] = 0;
}

/// @brief The a_LU bounds at one discrete state, folded through ::build_local_bound_tables.
void LivenessEngine::local_bounds(const int* locs, std::vector<std::int32_t>& outL,
                       std::vector<std::int32_t>& outU) const {
  outL = base_l_;
  outU = base_u_;
  if (net.lbound_loc.empty()) return;
  for (int pr = 0; pr < net.nprocs(); ++pr) {
    const std::vector<int>& vary = vary_clocks_[pr];
    if (vary.empty()) continue;
    const auto& lp = net.lbound_loc[pr][locs[pr]];
    const auto& up = net.ubound_loc[pr][locs[pr]];
    for (int c : vary) {
      outL[c] = std::max(outL[c], lp[c]);
      outU[c] = std::max(outU[c], up[c]);
    }
  }
  outL[0] = outU[0] = 0;
}

bool LivenessEngine::holds(const DiscPred& pr, const SymState& s) const {
  // A predicate carrying a DNF denotes the disjunction of its terms, not a conjunction. Only the
  // E[] entry point builds one, and only ever from discrete literals: it needs `!psi_discrete`,
  // which for an arbitrary boolean `psi` is a disjunction and has no conjunctive form.
  if (pr.is_dnf || !pr.forbid.empty()) {
    for (const auto& t : pr.forbid) {
      // A term constraining a clock forbids only part of the vertex, so it cannot exclude the
      // vertex outright; detail::DelayAvoidSet carries it and cuts the zone instead.
      if (!t.clocks.empty()) continue;
      bool all = true;
      for (const auto& r : t.locs)
        if ((s.locs[r.proc] == r.loc) != r.equal) { all = false; break; }
      if (all && !int_ok(t.ints, s.ints)) all = false;
      if (all) return true;
    }
    return false;
  }
  for (const auto& r : pr.locs) {
    const bool eq = (s.locs[r.proc] == r.loc);
    if (eq != r.equal) return false;
  }
  return int_ok(pr.ints, s.ints);
}

void LivenessEngine::grow() {
  const std::size_t n = nrec();
  if (expanded.size() < n) expanded.resize(n, 0);
  if (online_d_) {
    if (online_active_.size() < n) online_active_.resize(n, 0);
    if (online_active_next_.size() < n) online_active_next_.resize(n, -1);
    if (online_active_prev_.size() < n) online_active_prev_.resize(n, -1);
    if (online_cover_.size() < n) online_cover_.resize(n, -1);
    if (online_raw_slot_.size() < n) online_raw_slot_.resize(n, -1);
    if (online_inq_.size() < n) online_inq_.resize(n, 0);
    if (online_eg_prune_ && qc.size() < n) qc.resize(n, 2);
    return;
  }
  if (pc.size() < n) pc.resize(n, 2);
  if (qc.size() < n) qc.resize(n, 2);
  if (adj.size() < n) adj.resize(n);
  if (ntag_.size() < n) ntag_.resize(n, 0);
  if (ntag_out_.size() < n) ntag_out_.resize(n, 0);
}

/// @brief The discrete over-approximation of `p` at one state; see ::LivenessEngine::is_p.
bool LivenessEngine::p_discrete(const SymState& s) const {
  if (!p.is_dnf && p.forbid.empty()) return holds(p, s);
  for (const auto& t : p.forbid) {
    bool all = true;
    for (const auto& r : t.locs)
      if ((s.locs[r.proc] == r.loc) != r.equal) { all = false; break; }
    if (all && !int_ok(t.ints, s.ints)) all = false;
    if (all) return true;
  }
  return false;
}

/// @brief The clock region of `p` inside @p s; see ::LivenessEngine::p_region.
Federation LivenessEngine::p_region_of(const SymState& s) const {
  Federation out;
  if (p.forbid.empty()) { out.push_back(s.zone); return out; }
  for (const auto& t : p.forbid) {
    bool guard = true;
    for (const auto& r : t.locs)
      if ((s.locs[r.proc] == r.loc) != r.equal) { guard = false; break; }
    if (guard && !int_ok(t.ints, s.ints)) guard = false;
    if (!guard) continue;
    Dbm z = s.zone;
    if (!apply_all(z, t.clocks) || z.empty()) continue;
    fed_add(out, z);
  }
  return out;
}

bool LivenessEngine::is_p(int i) {
  grow();
  if (pc[i] != 2) return pc[i] != 0;
  // The over-approximation: does some term's discrete guard hold, so the node could contain a
  // p-valuation at all? Seeding the accepting closure wide is harmless, because the verdict is
  // decided by the exact region test in ::LivenessEngine::certificate_in.
  //
  // ::LivenessEngine::holds is deliberately not reused. Its DNF branch skips a term that constrains a clock,
  // which is right for `q` -- there the clock part is carried by the forbidden region and a term
  // cannot exclude a whole vertex -- and exactly wrong for `p`, where such a term is the reason
  // the node is interesting. Reusing it made `x >= 0 --> P.wait` hold vacuously: no term counted,
  // so no node seeded the closure, so no violation could be found.
  SymState sc;
  nstate(i, sc);
  pc[i] = p_discrete(sc) ? 1 : 0;
  return pc[i] != 0;
}

/// @brief The region of `p` at node @p i, as a federation; empty when `p` cannot hold there.
///
/// `p` in DNF is a union of conjunctive terms, each a discrete guard plus a convex clock region.
/// At a fixed node the discrete guards are decided, so the region is the union of the clock parts
/// of the terms that survive.
///
/// @note Returns the whole zone when `p` constrains no clock, which is the pre-existing case and
///       makes the intersection in ::LivenessEngine::certificate_in a no-op there.
Federation LivenessEngine::p_region(int i) {
  SymState cst;
  return p_region_of(sread(i, cst));
}

bool LivenessEngine::is_nq(int i) {
  grow();
  if (qc[i] == 2) { SymState sc; nstate(i, sc); qc[i] = holds(q, sc) ? 1 : 0; }
  return qc[i] == 0;
}

/// @brief Classify a zone against each atom of its key: inside, disjoint, or straddling.
///
/// Stored zones are kept whole rather than cut along the atoms, so a zone may *straddle* an
/// atom. One bit per atom cannot express that, and reading a straddle as "disjoint from phi" is
/// the one mistake that makes the covering test unsound, so the classification is three-valued
/// and rides in two masks. Both bits come off the canonical DBM in O(1):
///
///     Z.get(i,j) <= b                   =>  Z subseteq phi        (In)
///     raw_add(Z.get(j,i), b) < kZeroLe  =>  Z /\ phi = empty      (Out)
///     otherwise                                                   (Straddle)
///
/// @param z   the zone to classify
/// @param as  the atoms of @p z's discrete key; at most 64 are read
/// @param[out] in   bit @c a set iff @p z lies wholly inside atom @c a
/// @param[out] out  bit @c a set iff @p z is disjoint from atom @c a
///
/// @pre @p z is canonical (closed).
/// @post `(*in & *out) == 0`; a bit set in neither mask means @p z straddles that atom.
void LivenessEngine::classify_atoms(const Dbm& z, const std::vector<AtomC>& as,
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

/// @brief The covering test refined by the constraint map's atoms.
///
/// `a` is covered by `b` when, for every valuation @c nu in @p a, there is a @c nu' in @p b
/// with `nu <=_LU nu'` **and**, for every atom @c phi of the key, `nu' |= phi => nu |= phi`.
/// That implication direction only: it is what stops a_LU covering from dropping a deadlock. If
/// the covering zone satisfies an escape atom and the covered one does not, the simulator
/// escapes where the simulated valuation is stuck, and the deadlock is lost.
///
/// An atom is discharged for all pairs at once exactly when @p b is Out (no @c nu' satisfies
/// it) or @p a is In (every @c nu does). An atom that is neither is *blocked*, and only blocked
/// atoms are split. The recursion is Gastin–Mukherjee–Srivathsan, CAV 2019, Lemma 17, adapted
/// to the implication obligation rather than to agreement:
///
///     a <= b   <==   (a /\ phi <= b)  and  (a /\ ~phi <= b /\ ~phi)
///
/// `nu |= phi` discharges the phi-obligation for any @c nu' whatever, so the first conjunct may
/// keep the whole of @p b; `nu !|= phi` needs `nu' !|= phi`, so the second must narrow @p b as
/// well.
///
/// @param a,b    the candidate covered and covering zones
/// @param ain    the In-mask of @p a (see ::classify_atoms)
/// @param bout   the Out-mask of @p b
/// @param as     the atoms of the shared discrete key
/// @param depth  recursion depth, for diagnostics only
/// @return @c true iff @p a is covered by @p b under the refined relation
///
/// @pre @p a and @p b have the same discrete key, and @p as are that key's atoms.
/// @post Terminates in at most `2^|blocked|` leaves: each conjunct resolves one blocked atom
///       (In on the left, Out on both sides on the right), so the blocked set strictly shrinks.
/// @note ::base_cover is a *necessary* condition for both conjuncts — both test sub-zones of
///       @p a against sub-zones of @p b — so asking it first is sound and doubles as a
///       prefilter that rejects a hopeless probe before any zone is copied.
/// @note At depth 0 the two masks come from the caller's cache, so an unblocked probe costs one
///       mask operation plus the base test.
bool LivenessEngine::cover_rec(const Dbm& a, std::uint64_t ain, const Dbm& b, std::uint64_t bout,
                    const std::vector<AtomC>& as, int depth) {
  const std::size_t n = std::min<std::size_t>(as.size(), 64);
  const std::uint64_t all = n >= 64 ? ~std::uint64_t(0) : ((std::uint64_t(1) << n) - 1);
  const std::uint64_t blocked = all & ~bout & ~ain;
  if (!base_cover(a, b)) return false;   // necessary for both conjuncts; also the reject prefilter
  if (blocked == 0) return true;
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

}  // namespace hora
