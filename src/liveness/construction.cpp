/// @file src/liveness/construction.cpp
/// @brief The online quotient: interning, covering, activation and compaction.
///
/// Builds the forward graph under the deadlock-preserving D refinement, one construction with no
/// preliminary pass. A provisionally covered successor is kept as a replay recipe rather than a
/// materialised state until its cover is known to survive.

#include "hora/config.hpp"
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

// ---- Online D: persistent, monotonically refined forward quotient -------------------------

void LivenessEngine::online_reset() {
  online_nrec_ = 0;
  node.clear();
  node.shrink_to_fit();
  online_am_ = AtomMap{};
  online_am_.bump.assign(net.nclocks + 1, kNoBump);
  online_am_.live.assign(net.nclocks + 1, 0);
  online_am_.usable = true;
  online_keys_.clear();
  online_key_index_.clear();
  online_packed_key_keys_.clear();
  online_packed_key_values_.clear();
  online_packed_key_mask_ = 0;
  online_packed_key_count_ = 0;
  online_disc_shift_.assign(net.nprocs() + static_cast<int>(net.intvars.size()), 0);
  online_disc_mask_.assign(net.nprocs() + static_cast<int>(net.intvars.size()), 0);
  online_packed_key_usable_ = true;
  int disc_bits = 0;
  const auto field_bits = [](long nvalues) {
    int b = 0;
    std::uint64_t cap = 1;
    while (cap < static_cast<std::uint64_t>(nvalues) && b < 63) { cap <<= 1; ++b; }
    if (cap < static_cast<std::uint64_t>(nvalues)) return 64;
    return b;
  };
  for (int p = 0; p < net.nprocs(); ++p) {
    online_disc_shift_[p] = disc_bits;
    const int width = field_bits(static_cast<long>(net.procs[p].locations.size()));
    online_disc_mask_[p] = width == 0 ? 0 : (width >= 64 ? ~std::uint64_t(0)
                                                        : ((std::uint64_t(1) << width) - 1));
    disc_bits += width;
  }
  for (std::size_t v = 0; v < net.intvars.size(); ++v) {
    const int field = net.nprocs() + static_cast<int>(v);
    online_disc_shift_[field] = disc_bits;
    const int width = field_bits(net.intvars[v].hi - net.intvars[v].lo + 1);
    online_disc_mask_[field] = width == 0 ? 0 : (width >= 64 ? ~std::uint64_t(0)
                                                            : ((std::uint64_t(1) << width) - 1));
    disc_bits += width;
  }
  // Keep every shift strictly below 64.  Wider networks retain the previous exact vector-key
  // lookup; this fast path is optional and never aliases two discrete states.
  if (disc_bits >= 64) {
    online_packed_key_usable_ = false;
    online_disc_shift_.clear();
    online_disc_mask_.clear();
  } else {
    constexpr std::size_t initial_capacity = 1024;
    online_packed_key_keys_.resize(initial_capacity);
    online_packed_key_values_.assign(initial_capacity, -1);
    online_packed_key_mask_ = initial_capacity - 1;
  }
  online_key_by_id_.clear();
  key_locs_.clear();
  key_ints_.clear();
  key_loc_stride_ = net.nprocs();
  key_int_stride_ = static_cast<int>(net.intvars.size());
  online_key_of_.clear();
  online_next_in_key_.clear();
  online_key_edges_.clear();
  online_key_graph_enabled_ = false;
  online_reset_sets_.clear();
  online_tag_reset_class_.clear();
  online_atom_work_.clear();
  online_dirty_keys_.clear();
  online_dirty_queued_.clear();
  online_global_dirty_ = false;
  online_failed_ = false;
  online_bound_key_ = -1;
  online_bound_epoch_ = 0;
  online_bound_cached_epoch_ = std::numeric_limits<std::size_t>::max();
  online_active_.clear();
  online_active_next_.clear();
  online_active_prev_.clear();
  online_cover_.clear();
  online_raw_adj_.clear();
  online_raw_owner_.clear();
  online_raw_slot_.clear();
  online_deferred_index_enabled_ = false;
  online_work_.clear();
  online_inq_.clear();
}

/// @brief Compute the current D-covering bounds at one discrete key.
///
/// `AtomMap::bump` is append-only in online mode.  It is intentionally applied here, at every
/// cover check, rather than once in setup: a later non-diagonal deadlock atom must refine every
/// existing certificate without changing any stored exact zone.
void LivenessEngine::online_configure_bounds(int key) {
  if (!use_alu) return;
  if (online_bound_cached_epoch_ == online_bound_epoch_ && online_bound_key_ == key) return;
  online_bound_key_ = key;
  online_bound_cached_epoch_ = online_bound_epoch_;
  local_bounds(key_locs(key), bL, bU);
  for (int c = 1; c <= net.nclocks && c < (int)online_am_.bump.size(); ++c)
    if (online_am_.bump[c] != kNoBump) {
      bL[c] = std::max(bL[c], online_am_.bump[c]);
      bU[c] = std::max(bU[c], online_am_.bump[c]);
    }
}

/// @brief Does the exact candidate @p a have a current D cover in stored node @p b?
bool LivenessEngine::online_covers(const SymState& a, int b, int key) {
  const bool profile = online_perf.enabled;
  if (profile) ++online_perf.cover_calls;
  std::unique_ptr<OnlinePerfTimer> timer;
  if (profile) timer = std::make_unique<OnlinePerfTimer>(&online_perf.cover_ns);
  online_configure_bounds(key);
  const OnlineKeyInfo& ki = *online_key_by_id_[key];
  // A key with no diagonal atom needs only the base relation, and ::alu_le_stored /
  // ::stored_includes evaluate that against the stored operand at its own entry type. Widening it
  // first, as the atom path must, is a full ZoneStore::load per probe -- and the probe is the
  // hottest call in the forward phase.
  if (!ki.diag)
    return use_alu ? alu_le_stored(a.zone, b) : stored_includes(b, a.zone);
  Dbm bsc;
  const Dbm& bz = zread(b, bsc);
  const std::vector<AtomC>& diag = *ki.diag;
  if (diag.size() > 64) return online_cover_rec(a.zone, bz, diag, 0);
  std::uint64_t ain, aout, bin, bout;
  classify_atoms(a.zone, diag, &ain, &aout);
  classify_atoms(bz, diag, &bin, &bout);
  return cover_rec(a.zone, ain, bz, bout, diag, 0);
}

/// @brief Does stored node @p a have a current D cover in stored node @p b?
///
/// Both records are already known to share a discrete key.  Keeping this form avoids rebuilding
/// a temporary SymState (and copying its location and integer vectors) during compaction and
/// certificate rechecks.
bool LivenessEngine::online_covers(int a, int b, int key) {
  const bool profile = online_perf.enabled;
  if (profile) ++online_perf.cover_calls;
  std::unique_ptr<OnlinePerfTimer> timer;
  if (profile) timer = std::make_unique<OnlinePerfTimer>(&online_perf.cover_ns);
  online_configure_bounds(key);
  const OnlineKeyInfo& ki = *online_key_by_id_[key];
  Dbm asc;
  if (!ki.diag) {
    const Dbm& az0 = zread(a, asc);
    return use_alu ? alu_le_stored(az0, b) : stored_includes(b, az0);
  }
  const Dbm& az = zread(a, asc);
  Dbm bsc;
  const Dbm& bz = zread(b, bsc);
  const std::vector<AtomC>& diag = *ki.diag;
  if (diag.size() > 64) return online_cover_rec(az, bz, diag, 0);
  std::uint64_t ain, aout, bin, bout;
  classify_atoms(az, diag, &ain, &aout);
  classify_atoms(bz, diag, &bin, &bout);
  return cover_rec(az, ain, bz, bout, diag, 0);
}

/// @brief The D-cell obligation without the ordinary engine's 64-bit classification cache.
///
/// The common path uses ::cover_rec, whose masks make atom comparisons essentially free.  A
/// genuinely online construction cannot turn a newly found atom 65 into a restart, though, so
/// this slow-path evaluates the same branch rule directly from canonical DBM entries.  It is
/// reached only for unusually wide key-local atom sets; all atom positions remain append-only.
bool LivenessEngine::online_cover_rec(const Dbm& a, const Dbm& b, const std::vector<AtomC>& as, int depth) {
  if (!base_cover(a, b)) return false;
  for (const AtomC& at : as) {
    const bool ain = a.get(at.i, at.j) <= at.b;  // every valuation of a satisfies phi
    const bool bout = raw_add(b.get(at.j, at.i), at.b) < kZeroLe;  // b is disjoint from phi
    if (ain || bout) continue;

    // Identical to cover_rec's split: on phi, b itself is enough; on !phi, b must be cut too.
    Dbm ap = a;
    if (ap.constrain(at.i, at.j, at.b) && !online_cover_rec(ap, b, as, depth + 1)) return false;
    const Constraint nc = negate(Constraint{at.i, at.j, at.b});
    Dbm an = a;
    if (!an.constrain(nc.i, nc.j, nc.bound)) return true;
    Dbm bn = b;
    if (!bn.constrain(nc.i, nc.j, nc.bound)) return false;
    return online_cover_rec(an, bn, as, depth + 1);
  }
  return true;
}

int LivenessEngine::online_append(SymState&& s, int key) {
  // NO SymState IS pushed. During the online phase the discrete half lives once in the key arenas
  // and the zone lives in the ZoneStore, so `node`'s entry would be three empty vector headers --
  // 80 bytes a record, and the search materialises far more records than it keeps. The vector is
  // built once, at its final size, in ::online_finalize_graph; until then `online_nrec_` is the id
  // counter.
  const int id = static_cast<int>(online_nrec_++);
  Dbm z = std::move(s.zone);
  zstore(id, std::move(z));
  online_key_of_.push_back(key);
  OnlineKeyInfo& ki = *online_key_by_id_[key];
  online_next_in_key_.push_back(ki.node_head);
  ki.node_head = id;
  ++ki.node_count;
  ++outer_nodes;
  grow();
  return id;
}

/// @brief Return a current D coverer for @p v, optionally requiring completed successors.
///
/// A queued active representative is a valid *provisional* coverer: it is guaranteed to be
/// expanded before the forward phase can quiesce, and if a later representative replaces it,
/// that replacement has already been expanded and covers it transitively.  Allowing this avoids
/// activating a whole same-level antichain merely because its first representative has not yet
/// reached the front of the FIFO queue.  Final edge redirection asks for an expanded coverer.
int LivenessEngine::online_coverer(int v, bool require_expanded) {
  if (v < 0 || v >= (int)nrec()) return -1;
  if (online_active_[v] && (!require_expanded || expanded[v])) return v;
  const OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
  if (online_perf.enabled) {
    ++online_perf.coverer_calls;
    online_perf.coverer_active_scan += ki.active_count;
  }
  const int cached = online_cover_[v];
  if (cached >= 0 && cached < (int)nrec() && online_active_[cached] &&
      (!require_expanded || expanded[cached]) && online_covers(v, cached, ki.id))
    return cached;
  for (int j = ki.active_head; j >= 0; j = online_active_next_[j])
    if (j != v && online_active_[j] && (!require_expanded || expanded[j]) &&
        online_covers(v, j, ki.id)) return j;
  return -1;
}

/// @brief Resolve a previously recorded cover without repeating a D comparison.
///
/// The forward worklist rechecks every inactive materialised record after each map refinement.
/// Once quiescent, its cover points to an active representative or through another inactive
/// record to one.  D-covering is transitive, so final edge redirection can chase that chain
/// without repeating a DBM cover test for every raw edge.
int LivenessEngine::online_resolve_cover(int v, bool require_expanded) {
  if (v < 0 || v >= (int)nrec()) return -1;
  int cur = v;
  std::size_t hops = 0;
  while (!online_active_[cur]) {
    const int next = online_cover_[cur];
    if (next < 0 || next >= (int)nrec() || next == cur || ++hops > nrec()) return -1;
    cur = next;
  }
  if (require_expanded && !expanded[cur]) return -1;

  // Path compression makes repeated redirects through an inactive record constant-time.
  while (v != cur && !online_active_[v]) {
    const int next = online_cover_[v];
    online_cover_[v] = cur;
    v = next;
  }
  return cur;
}

/// @brief Remember the current D certificate of an inactive materialised record.
void LivenessEngine::online_set_cover(int v, int cover) {
  if (v < 0 || v >= (int)nrec()) return;
  online_cover_[v] = cover;
}

void LivenessEngine::online_activate(int v) {
  if (v < 0 || v >= (int)nrec()) return;
  if (!online_active_[v]) {
    online_active_[v] = 1;
    online_set_cover(v, -1);
    OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
    online_active_prev_[v] = -1;
    online_active_next_[v] = ki.active_head;
    if (ki.active_head >= 0) online_active_prev_[ki.active_head] = v;
    ki.active_head = v;
    ++ki.active_count;
  }
  if (!expanded[v] && !online_inq_[v]) {
    online_inq_[v] = 1;
    // Before D changes, preserve ordinary FIFO order: it produces smaller aLU antichains on
    // dense keys.  After the first change, prioritise the first record at a new key so a
    // downstream seed can propagate before sibling zones expand under the coarser map.
    const OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
    bool key_seen_semantically = false;
    for (int r = ki.node_head; r >= 0; r = online_next_in_key_[r])
      if (r != v && expanded[r]) { key_seen_semantically = true; break; }
    // `--exploration dfs`: depth-first for `E[] psi`. A cyclic witness closes after one path's
    // worth of records depth-first, and only after the whole breadth under FIFO -- see
    // hora::detail::EngineOptions::eg_dfs for the measurement. Guarded by `eg_early_` so it
    // applies to the E[] construction alone and leaves leads-to on its FIFO order.
    if (eg_dfs_on() && eg_early_) online_work_.push_front(v);
    else if (key_seen_semantically || !online_deferred_index_enabled_) online_work_.push_back(v);
    else online_work_.push_front(v);
  }
}

/// @brief Replace only the *representative status* of zones a fresh active node covers.
///
/// The old records and their raw outgoing edges stay intact.  A future map refinement may wake
/// one again, but no enlarged zone or stale adjacency ever has to be reconstructed.  The new
/// representative is allowed to be queued rather than expanded: its eventual expansion is the
/// simulation witness for the provisional cover, and a refinement can revoke it before then.
void LivenessEngine::online_compact_key(int v) {
  if (!online_active_[v]) return;
  OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
  if (online_perf.enabled) {
    ++online_perf.compact_calls;
    online_perf.compact_active_scan += ki.active_count;
  }
  for (int j = ki.active_head; j >= 0;) {
    const int next = online_active_next_[j];
    // Keep the distinguished initial record active.  Leads-to may quotient it away safely, but
    // E[] reads vertex 0 directly; retaining this one representative keeps the online graph
    // reusable for that query shape too without a special root redirect.
    // An unexpanded record already has an exact incoming edge, but not yet its own semantic
    // adjacency.  Redirecting that incoming edge to a simulator is sufficient for forward
    // reachability, not for the exact backward predecessor used below.  Let it expand once;
    // after that it may leave the forward antichain while remaining as a historical graph
    // vertex with its exact zone and exact outgoing edges.
    if (j == 0 || j == v || !online_active_[j] ||
        (online_eg_repair_ && !expanded[j])) {
      j = next;
      continue;
    }
    if (online_covers(j, v, ki.id)) {
      if (online_perf.enabled) {
        ++online_perf.compact_deact;
        if (!expanded[j]) {
          ++online_perf.compact_deact_unexp;
          Dbm jz;
          zread(j, jz);
          if (stored_includes(v, jz)) ++online_perf.compact_deact_permanent;
        }
      }
      online_active_[j] = 0;
      online_set_cover(j, v);
      const int prev = online_active_prev_[j];
      if (prev >= 0) online_active_next_[prev] = next;
      else ki.active_head = next;
      if (next >= 0) online_active_prev_[next] = prev;
      online_active_next_[j] = online_active_prev_[j] = -1;
      --ki.active_count;
      // A stale deque entry is harmless (the consumer checks online_active_), but its bit must
      // not suppress a later reactivation after a D refinement invalidates this cover.
      online_inq_[j] = 0;
      j = next;
      continue;
    }
    j = next;
  }
}

/// @brief Intern an active candidate or defer a revocably covered successor by replay recipe.
int LivenessEngine::online_intern(SymState&& s, int key, int source, int ordinal, int* deferred) {
  if (deferred) *deferred = -1;
  // online_build_all registers the root, and online_expand registers its whole successor batch
  // and settles the atom worklist *before* it gets here.  Keeping that boundary batched matters:
  // doing a no-op map drain and a no-op inactive-record scan for every successor was pure hot-path
  // overhead, and it could not reveal a D refinement that the preceding batch phase missed.
  OnlineKeyInfo& ki = *online_key_by_id_[key];
  const bool profile = online_perf.enabled;
  if (profile) {
    ++online_perf.intern_calls;
    online_perf.intern_active_scan += ki.active_count;
  }
  std::chrono::steady_clock::time_point cover_start;
  if (profile) cover_start = std::chrono::steady_clock::now();
  int cover = -1;
  for (int j = ki.active_head; j >= 0; j = online_active_next_[j])
    if (online_active_[j] && online_covers(s, j, key)) { cover = j; break; }
  if (profile)
    online_perf.intern_cover_probe_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - cover_start).count();

  // Unlike an a_LU/D certificate, literal zone inclusion cannot be invalidated by a later atom:
  // every concrete valuation and every successor of `s` is already present in `cover`.  Do not
  // retain a fake "provisional" record in that case.  The append-only provenance is needed only
  // for covers that rely on the relation currently being refined.
  if (cover >= 0 && stored_includes(cover, s.zone)) return cover;

  // Do not turn an ordinary provisional cover into a full stored state.  The expanded source
  // is immutable, and successor generation is deterministic, so (source, ordinal) is an exact
  // recipe for this candidate.  A later D refinement replays only recipes at the affected key;
  // until then the raw edge points at the current representative.  Literal inclusion has already
  // returned above, so every recipe retained here is genuinely revocable.
  if (cover >= 0 && source >= 0 && ordinal >= 0) {
    const int raw_slot = source < static_cast<int>(online_raw_slot_.size())
                             ? online_raw_slot_[source]
                             : -1;
    if (raw_slot < 0 || static_cast<std::uint64_t>(raw_slot) > UINT32_MAX ||
        static_cast<std::uint64_t>(ordinal) > UINT32_MAX) {
      online_failed_ = true;
      return cover;
    }
    const std::uint64_t handle = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(raw_slot)) << 32) |
                                 static_cast<std::uint32_t>(ordinal);
    if (online_deferred_index_enabled_) ki.deferred.push_back(handle);
    if (deferred) *deferred = 1;
    if (profile) ++online_perf.deferred_created;
    return cover;
  }

  // The source-less path is used for the root and for a deferred successor that has just lost
  // its cover.  Exact convex-union canonicalisation may still keep the active antichain small;
  // the union is set-theoretically exact and is materialised as a fresh immutable record.
  std::chrono::steady_clock::time_point join_start;
  if (profile) join_start = std::chrono::steady_clock::now();
  if (cover < 0) {
    Dbm old;
    // Iterated to a fixpoint, not stopped at the first success: each union enlarges the
    // candidate, and a larger zone can be convex with an active record the smaller one was not.
    // Still exact -- Dbm::try_union succeeds only when the union really is one zone -- so this
    // adds no point and removes none; it only leaves a tighter antichain behind, which is one
    // fewer record for ::online_compact_key to retire later.
    for (bool grew = true; grew;) {
      grew = false;
      for (int j = ki.active_head; j >= 0; j = online_active_next_[j]) {
        if (j == 0 || !online_active_[j]) continue;
        // A record the candidate already contains has nothing to add: their union IS the
        // candidate. Every record absorbed by an earlier pass is in that state, so without this
        // test each one is re-analysed in full on every later pass, which is what made the
        // fixpoint quadratic in the width of the antichain.
        if (includes_stored(s.zone, j)) continue;
        // Read the stored operand at its own entry type. ::dbm_try_union is templated on it, so
        // widening it into a scratch first is a ZoneStore::load bought for nothing.
        Dbm joined = s.zone;
        const bool merged = zs_.with(j, [&](const auto* pz) {
          return dbm_try_union(joined.data().data(), pz, zdim_);
        });
        if (!merged || joined == s.zone) continue;
        s.zone = std::move(joined);
        grew = true;
      }
    }
  }
  if (profile)
    online_perf.intern_join_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - join_start).count();

  std::chrono::steady_clock::time_point append_start;
  if (profile) append_start = std::chrono::steady_clock::now();
  const int id = online_append(std::move(s), key);
  if (profile)
    online_perf.intern_append_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - append_start).count();
  if (cover < 0) {
    online_activate(id);               // otherwise the materialised record remains inactive
    // Rebalance before this new representative reaches the work queue.  If it covers a queued
    // peer, the peer need not generate successors before becoming inactive; if a future atom
    // breaks the cover, that materialised record becomes active again.
    online_compact_key(id);
  } else {
    online_set_cover(id, cover);
  }
  return id;
}

/// @brief Regenerate exactly one provisionally covered successor from its immutable source.
bool LivenessEngine::online_replay(int source, int ordinal, int key, SymState& out) const {
  if (source < 0 || source >= static_cast<int>(nrec()) || ordinal < 0 || key < 0 ||
      key >= static_cast<int>(online_key_by_id_.size()))
    return false;

  const int source_key = online_key_of_[source];
  if (source_key < 0 || source_key >= static_cast<int>(online_key_by_id_.size())) return false;
  const OnlineKeyInfo& sk = *online_key_by_id_[source_key];
  SymState src;
  key_state(sk.id, src);
  zread(source, src.zone);

  std::vector<SymState> succ;
  std::vector<GfpTag> stags;
  tagged_successors(net, src, L, U, qmask, emode, succ, stags,
                  avoid_forward_ ? avoid_ : nullptr);
  if (ordinal >= static_cast<int>(succ.size())) return false;

  const OnlineKeyInfo& tk = *online_key_by_id_[key];
  SymState& candidate = succ[static_cast<std::size_t>(ordinal)];
  const int* tl = key_locs(tk.id);
  if (!std::equal(candidate.locs.begin(), candidate.locs.end(), tl, tl + key_loc_stride_))
    return false;
  const long* ti = key_ints(tk.id);
  if (!std::equal(candidate.ints.begin(), candidate.ints.end(), ti, ti + key_int_stride_))
    return false;
  out = std::move(candidate);
  return true;
}

/// @brief Test the discrete E[] restriction while online nodes still share key storage.
bool LivenessEngine::online_is_nq(int v) {
  if (v < 0 || v >= static_cast<int>(online_key_of_.size())) return false;
  grow();
  if (qc[v] == 2) {
    const OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
    SymState state;
    key_state(ki.id, state);
    qc[v] = holds(q, state) ? 1 : 0;
  }
  return qc[v] == 0;
}

/// @brief Certify an E[] witness that ends here, without running a partial backward solve.
///
/// Every materialised online record is an exact reachable zone reached through states that have
/// stayed inside psi.  A time-divergent valuation or a genuine deadlock of the unrestricted
/// network is therefore already a concrete maximal psi-run.  This is stronger than a graph-cycle
/// guess: no subsumption edge participates in the certificate.
/// @brief Does this record hold a deadlocked valuation?
///
/// The `A[] not deadlock` counterpart of ::LivenessEngine::online_eg_terminal_witness: the same
/// exact stored zone and the same per-key `may_deadlock` filter, but time divergence is not a
/// deadlock and takes no part in the test.
bool LivenessEngine::online_deadlock_at(int v, bool no_successor) {
  if (v < 0 || v >= static_cast<int>(online_key_of_.size())) return false;
  const OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
  if (!ki.may_deadlock && !no_successor) return false;
  SymState s;
  key_state(ki.id, s);
  zread(v, s.zone);
  return state_has_deadlock(net, s);
}

bool LivenessEngine::online_eg_terminal_witness(int v, bool no_restricted_successor) {
  if (!online_eg_prune_ || v < 0 || v >= static_cast<int>(online_key_of_.size())) return false;
  const OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
  SymState s;
  key_state(ki.id, s);
  zread(v, s.zone);
  if (time_diverges(s)) return true;

  // A universe seed that proved Dead(key) empty cannot contain a deadlocked sub-zone here.
  // Conversely, a key with a possible deadlock is tested exactly on this reached zone.  The
  // cheap no-successor case is also tested because psi may have removed every restricted edge.
  if (!ki.may_deadlock && !no_restricted_successor) return false;
  if (dead_net != &net || avoid_) {
    // If the original location lets time run for ever, an action-deadlocked valuation continues
    // by delay outside psi and is not the end of a maximal run.
    if (orig_time_open(s)) return false;
  }
  Federation dead;
  return state_has_deadlock(*dead_net, s, &dead);
}

/// @brief Certify a `p --> q` counterexample at a freshly expanded node, before the graph is done.
///
/// `p --> q` fails exactly when some reachable state satisfies `p /\ EG !q`. A valuation that
/// satisfies `p`, sits at a `!q` key, and can end a maximal run -- by deadlocking, or by letting
/// time diverge -- witnesses `EG !q` outright: the run stops (or runs on) there and never leaves
/// `!q`. So the forward search can stop the moment one is reached, with no backward pass and no
/// finished graph.
///
/// Every online record holds an exact reachable zone, so a valuation found here is genuinely
/// reachable and the witness is concrete: no subsumption edge takes part in it. That exactness is
/// also why this is confined to the a_LU lane. Where extrapolation is load-bearing (the diagonal
/// route) a stored zone over-approximates, and a stop valuation enters positively here, so a
/// spurious one would fabricate a violation.
///
/// @param v             the node just expanded
/// @param no_successor  whether ::tagged_successors produced nothing for it
/// @return @c true iff a `p`-satisfying valuation of @p v ends a maximal `!q` run
bool LivenessEngine::leadsto_witness(const SymState& s, bool no_successor, bool key_may_deadlock) {
  if (!leadsto_early_ || !use_alu) return false;
  // With a forbidden region in force, `q` is only partly discrete: ::LivenessEngine::holds deliberately
  // ignores a term that constrains a clock, because ::detail::DelayAvoidSet carries that part and
  // the backward pass cuts the zone with it. A stop valuation could then satisfy `q` through the
  // clock part while `holds` reports `!q`, and this witness would claim a violation that is not
  // one. Deciding it here would mean intersecting the stop set with the `!q` region; until that
  // is done, the region route keeps the ordinary construct-then-solve path.
  if (avoid_) return false;
  if (holds(q, s)) return false;                  // `q` holds here: not a witness position
  if (!p_discrete(s)) return false;               // `p`'s discrete guard cannot hold either

  // Where the run may stop. The two whole-zone cases are exact as such; only a partial deadlock
  // needs the per-valuation subtraction.
  Federation stop;
  if (no_successor || time_diverges(s)) {
    stop.push_back(s.zone);
  } else {
    if (!key_may_deadlock) return false;          // no zone at this key can deadlock
    if (!state_has_deadlock(*dead_net, s, &stop)) return false;
  }
  if (fed_is_empty(stop)) return false;

  // The witness must satisfy `p` as well as stop. With a purely discrete `p` the region is the
  // whole zone and this is a no-op.
  if (p.forbid.empty()) return true;
  return !fed_is_empty(fed_intersect(stop, p_region_of(s)));
}

/// @brief ::leadsto_witness at an online record, whose discrete half lives in its key.
bool LivenessEngine::online_leadsto_witness(int v, bool no_successor) {
  if (!leadsto_early_ || v < 0 || v >= static_cast<int>(online_key_of_.size())) return false;
  const OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
  SymState s;
  key_state(ki.id, s);
  if (holds(q, s) || !p_discrete(s)) return false;   // cheap discrete reject before any zone work
  zread(v, s.zone);
  return leadsto_witness(s, no_successor, ki.may_deadlock);
}

/// @brief Hash a discrete state without allocating its byte key.
std::uint64_t disc_hash(const LocVec& locs, const IntVec& ints) {
  std::uint64_t h = 1469598103934665603ULL;
  const auto mix = [&h](std::uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  };
  for (int l : locs) mix(std::uint64_t(std::uint32_t(l)));
  mix(0x5bf03635ULL);
  for (long v : ints) mix(std::uint64_t(std::int64_t(v)));
  return h;
}

/// @brief Insert @p s into the passed set, returning the node that now represents it.
///
/// Covering is a_LU subsumption when @ref use_alu is set and plain zone inclusion otherwise,
/// refined in both cases by the constraint map's cell tags when one is installed.
///
/// Three outcomes, in this order: an existing node already covers @p s (return it, nothing is
/// stored); @p s covers an existing node, which is then *enlarged in place*; or the exact
/// convex union of @p s with some stored node is itself a zone, in which case that node absorbs
/// @p s. Only if all three fail is a new node appended.
///
/// @param s  the candidate state; consumed
/// @return the index into @ref node representing @p s
///
/// @post @ref intern_grew is @c true iff a stored zone was rewritten (enlarged or merged). The
///       caller must then invalidate that node's adjacency: it was computed from the smaller
///       zone. ::LivenessEngine::expand does this.
/// @post Every stored zone remains atom-homogeneous only in the sense ::LivenessEngine::cover_rec needs,
///       i.e. not at all: straddling zones are stored whole and resolved at comparison time.
///
/// @note The two rewriting outcomes are what a purely append-only passed set would forgo. An
///       append-only rule makes every stored node immutable from birth, which is what "start
///       backward work only on vertices known not to change" wants, but it pays for that with
///       an antichain of zones the rewriting version collapses. Rewriting is kept, and the
///       invalidation above is what makes it safe.

int LivenessEngine::intern(SymState&& s) {
  intern_grew = false;
  // the cell tag. `tag` is the In-mask and `tag_out` the Out-mask of ::classify_atoms; a zone
  // set in neither straddles that atom, and every obligation a straddle leaves undecidable is
  // deferred to LivenessEngine::cover_rec.
  //
  // The obligation the tag enforces is the implication `Z |= phi => Z' |= phi`, in that
  // direction only, where Z covers Z'. Dead(q) = Inv(q) /\ AND_e !Esc_e with each Esc_e a
  // conjunction of atoms, so a stuck valuation needs, per outgoing edge e, some atom of Esc_e
  // that its simulator also falsifies. Equivalently `tag(Z) & ~tag(Z') == 0`, which is strictly
  // coarser than same-cell equality: equality needlessly refuses the pair (Z disjoint from phi,
  // Z' contained in phi).
  std::uint64_t tag = 0, tag_out = 0;
  bool tagged = false;
  const std::vector<AtomC>* atoms = nullptr;
  if (amap_) {
    auto it = amap_->diag.find(disc_key_bytes(s.locs, s.ints));
    if (it != amap_->diag.end() && !it->second.empty()) {
      const std::vector<AtomC>& as = it->second;
      tagged = true;
      atoms = &as;
      classify_atoms(s.zone, as, &tag, &tag_out);
    }
  }
  auto& b = bucket[disc_hash(s.locs, s.ints)];
  // Exact discrete equality, checked before any zone test: the bucket is a hash bucket, so it
  // may hold nodes of a different discrete state.
  auto same_disc = [&](int j) {
    const SymState& n = node[j];
    if (n.locs.size() != s.locs.size() || n.ints.size() != s.ints.size()) return false;
    for (std::size_t t = 0; t < s.locs.size(); ++t)
      if (n.locs[t] != s.locs[t]) return false;
    for (std::size_t t = 0; t < s.ints.size(); ++t)
      if (n.ints[t] != s.ints[t]) return false;
    return true;
  };
  if (use_alu) {
    if (net.lbound_loc.empty()) {
      bL = aL;
      bU = aU;
    } else {
      local_bounds(s.locs.data(), bL, bU);
    }
    // The per-location bounds must carry the non-diagonal atoms' constants too, for the same
    // reason the global arrays do in LivenessEngine::setup: raising L or U only makes a_LU finer, and the
    // simulation argument needs the raise to hold along every path.
    //
    // Raising one clock to `max(L, U)` is a_M ON that clock: the three clauses of the LU
    // preorder — nu'(x) = nu(x), or L(x) < nu'(x) < nu(x), or U(x) < nu(x) < nu'(x) — collapse
    // to "equal, or both above M(x)", which is symmetric, hence a bisimulation, hence
    // deadlock-preserving (Behrmann–Bouyer–Larsen–Pelánek, STTT 8(3), 2006, Lemma 1 against
    // Lemma 3). Raising every clock yields the global a_M relation; the production online path
    // refines only the clocks required by its discovered D atoms.
    if (amap_)
      for (int c = 1; c <= net.nclocks && c < (int)amap_->bump.size(); ++c)
        if (amap_->bump[c] != kNoBump) {
          bL[c] = std::max(bL[c], amap_->bump[c]);
          bU[c] = std::max(bU[c], amap_->bump[c]);
        }
  }
  const int dim2 = s.zone.dim();
  const bool tagcheck = tagged;
  const bool reccheck = tagcheck;
  Dbm zsc;                                                  // widening scratch, reused
  auto covered = [&](int j) {
    if (reccheck) {                                         // straddling operands: recurse
      return cover_rec(s.zone, tag, zread(j, zsc), ntag_out_[j], *atoms, 0);
    }
    if (tagcheck && (ntag_[j] & ~tag) != 0) return false;   // Z |= phi but Z' does not: refuse
    return use_alu ? alu_le_stored(s.zone, j) : stored_includes(j, s.zone);
  };
  // what the rewriting paths owe.
  //
  //   In-place enlargement replaces the stored zone by the newcomer, which under a_LU need not
  //   be a superset of it, so everything already routed to j must still be covered. That is
  //   transitivity of the covering relation, and the relation is transitive (<=_LU is a preorder
  //   and `nu' |= phi => nu |= phi` composes), so the ordinary three-valued test in the opposite
  //   direction is exactly and only what is owed.
  //
  //   The convex-union merge owes nothing at all: try_union succeeds only when the union really
  //   is a single zone, so the stored zone afterwards is a set superset of both operands.
  //   Growing the right-hand side of an exists-witness cannot invalidate a witness, and the
  //   newcomer is covered by the result via the identity witness.
  //
  // Both paths must refresh the stored masks, since the stored zone changed.
  auto covers = [&](int j) {
    if (reccheck) {
      return cover_rec(zread(j, zsc), ntag_[j], s.zone, tag_out, *atoms, 0);
    }
    if (tagcheck && ntag_[j] != tag) return false;          // rewriting: equality, see above
    return use_alu ? alu_le_new(j, s.zone) : includes_stored(s.zone, j);
  };
  for (int j : b)
    if (same_disc(j) && covered(j)) return j;
  for (std::size_t k = 0; k < b.size(); ++k)
      if (same_disc(b[k]) && covers(b[k])) {
        zstore(b[k], Dbm(s.zone));
        if (reccheck) { ntag_[b[k]] = tag; ntag_out_[b[k]] = tag_out; }
        intern_grew = true;
        return b[k];
      }
  // exact convex-union merge. Dbm::try_union succeeds only when the union of the two zones really
  // is a single zone, so the stored zone afterwards denotes exactly the set the two it replaced
  // denoted: no point added, none removed. That exactness is what makes the merge usable here at
  // all — unlike a convex hull it is not an over-approximation, so it cannot invent a certificate
  // in the under-approximating backward lane.
  {
    for (int j : b) {
      if (!same_disc(j)) continue;
      Dbm t = zread(j, zsc);              // copy: try_union rewrites its left operand
      if (t.dim() != dim2) continue;
      if (t.try_union(s.zone)) {          // union is exactly convex: absorb, no approximation
        if (reccheck) classify_atoms(t, *atoms, &ntag_[j], &ntag_out_[j]);
        zstore(j, std::move(t));
        intern_grew = true;
        return j;
      }
    }
  }
  const int id = static_cast<int>(nrec());
  Dbm znew = std::move(s.zone);           // taken out first: zstore decides where it lands
  s.zone = Dbm();
  node.push_back(std::move(s));
  if (ntag_.size() <= static_cast<std::size_t>(id)) ntag_.resize(id + 1, 0);
  if (ntag_out_.size() <= static_cast<std::size_t>(id)) ntag_out_.resize(id + 1, 0);
  ntag_[id] = tag;
  ntag_out_[id] = tag_out;
  zstore(id, std::move(znew));
  b.push_back(id);
  ++outer_nodes;
  grow();
  return id;
}

/// @brief Reference-path hook for atom splitting.
///
/// The production online path never calls this routine: it discharges atom obligations while
/// testing a cover rather than splitting materialised records.  The reference path uses the same
/// comparison-time strategy, so its output is the original zone as one piece.
void LivenessEngine::split_by_atoms(SymState&& s, std::vector<SymState>& out) {
  out.clear();
  out.push_back(std::move(s));
}

/// @brief Add @p t to the tag pool, reusing an equal tag if one is already there.
///
/// @param t  the tag produced for one transition
/// @return the index into ::LivenessEngine::tags_ that ::OtfEdge::tag must carry
///
/// @note Tags are immutable once pooled, and ::exact_predecessor only ever reads them, so sharing one
///       between edges is safe. Deduplication is by value, not by identity: two transitions
///       that fire the same participants with the same abstain constraints invert identically.
int LivenessEngine::intern_tag(const GfpTag& t) {
  const std::uint64_t h = tag_hash(t);
  auto& slots = tagidx_[h];
  for (int i : slots)
    if (tag_equal(tags_[i], t)) return i;
  const int id = static_cast<int>(tags_.size());
  tags_.push_back(t);
  slots.push_back(id);
  return id;
}

/// @brief Generate and intern every symbolic successor of node @p u.
///
/// A rewritten representative has successors computed for an older, smaller zone.  Its cached
/// adjacency is therefore cleared and the ordinary forward worklist expands it again.  This is
/// the non-online exact/inclusion path only; online records are immutable and use
/// ::LivenessEngine::online_expand instead.
void LivenessEngine::expand(int u) {
  grow();
  if (cancelled()) return;   // choke point: every forward step goes through here
  if (expanded[u]) return;
  expanded[u] = 1;
  ++expand_calls;
  ++outer_expands;
  std::vector<SymState> succ;
  std::vector<GfpTag> tags;
  std::vector<SymState> pieces;
  SymState usc;
  tagged_successors(net, sread(u, usc), L, U, qmask, emode, succ, tags,
                  avoid_forward_ ? avoid_ : nullptr);
  // Same concrete counterexample as the online route takes; `usc` still holds this node's exact
  // state. `may_deadlock` is not tracked per key here, so the exact test is always allowed to run.
  if (deadlock_scan_ && state_has_deadlock(net, usc)) {
    deadlock_hit_ = true;
    eg_early_hit_ = true;
    return;
  }
  if (leadsto_early_ && leadsto_witness(usc, succ.empty(), /*key_may_deadlock=*/true)) {
    eg_early_hit_ = true;
    return;
  }
  for (std::size_t k = 0; k < succ.size(); ++k) {
    pieces.clear();
    if (amap_) {
      split_by_atoms(std::move(succ[k]), pieces);
    } else {
      pieces.push_back(std::move(succ[k]));
    }
    // One tag serves every piece of this successor: the pieces partition the successor zone,
    // and the transition that produced them is the same one.
    const int tid = intern_tag(tags[k]);
    for (std::size_t pi = 0; pi < pieces.size(); ++pi) {
    const int v = intern(std::move(pieces[pi]));
    grow();
    adj[u].push_back(OtfEdge{v, tid});
    // A rewrite invalidates the rewritten node's adjacency; see the @warning above.
    if (intern_grew) {
      if (v != u) {
        expanded[v] = 0;
        adj[v].clear();
      } else {
        expanded[u] = 0;
      }
    }
    }
  }
}

/// @brief Can time grow without bound from @p s, with no transition ever taken?
///
/// @param s  a symbolic state
/// @return @c true iff every clock is unbounded in the delay closure of @p s's zone under the
///         location invariants — i.e. the whole zone admits an infinite time-divergent run.
///
/// @note A committed or urgent location admits no positive delay at all, so it can never
///       diverge. Delaying unconditionally would report divergence there and fabricate a
///       maximal-run witness out of nothing.
bool LivenessEngine::time_diverges(const SymState& s) const {
  if (net.nclocks == 0) return false;
  { bool fc, fu; frozen_flags(net, s.locs, s.ints, &fc, &fu); if (fc || fu) return false; }
  std::vector<Constraint> inv;
  all_invariants(net, s.locs, inv);
  // The unrouted answer, which is also a sound PRE-FILTER for the routed one below: every piece
  // of a routed delay is a subset of this delay closure, and cutting a set down can only tighten
  // its bounds, so a clock this zone already caps is capped in every piece. When it caps one,
  // no piece can be unbounded and the federation subtraction inside ::DelayAvoidSet::delay is
  // pure loss -- which is the common case, since an invariant that bounds a clock bounds it
  // whichever way the trajectory is routed.
  Dbm z = s.zone;
  z.delay();
  apply_all(z, inv);
  if (z.empty()) return false;
  for (int i = 1; i <= net.nclocks; ++i)
    if (z.get(i, 0) != kInf) return false;
  // With a forbidden region the delay is routed, and divergence must be witnessed by a piece of
  // the routed delay: a trajectory that would run into the region stops there rather than
  // diverging, and one that already lies past it may still run away.
  if (avoid_ && avoid_->applies(s.locs, s.ints)) {
    Federation parts;
    avoid_->delay(s.zone, s.locs, s.ints, parts);
    for (Dbm& p : parts) {
      if (!apply_all(p, inv) || p.empty()) continue;
      bool unbounded = true;
      for (int i = 1; i <= net.nclocks && unbounded; ++i)
        if (p.get(i, 0) != kInf) unbounded = false;
      if (unbounded) return true;
    }
    return false;
  }
  return true;
}

/// @brief The end-of-run term of the fixpoint operator: the valuations of node @p u from which
///        the run may legitimately stop.
///
/// @param u  node index; @p u must be expanded, or an empty adjacency is read as "stuck" when it
///          only means "not yet explored"
/// @return the federation of valuations of `node[u].zone` that either enable no transition ever
///         or admit a time-divergent run
///
/// @pre Stored zones are exact (::LivenessEngine::setup's invariant). A stop valuation occurs *positively*
///      in the fixpoint operator, so a spurious one fabricates a counterexample rather than
///      losing one.
/// @note Two of the three cases are whole-zone and exact as such: an empty adjacency means every
///       valuation is stuck, and ::time_diverges is a property of the delay-closed zone as a
///       whole. Only the third — partial deadlock — needs the per-valuation subtraction, and
///       hora::detail::state_has_deadlock supplies it, recomputing the delay closure first so
///       that the active-clock reduction cannot leave the stored zone unclosed.
/// @brief Can time run away for ever here in the unrestricted system?
///
/// @param s the state; only its location vector and integers are read
/// @return @c true iff the original invariant leaves every clock unbounded after a delay, so a
///         valuation with no enabled transition continues by delaying rather than stopping
///
/// @note Only meaningful when a `psi`-restriction is in force. It is what separates a genuine
///       timelock -- a state that truly cannot be extended, and therefore ends a maximal run --
///       from a state that merely cannot be extended *inside psi*.
bool LivenessEngine::orig_time_open(const SymState& s) const {
  if (net.nclocks == 0) return false;
  bool c, u;
  frozen_flags(*dead_net, s.locs, s.ints, &c, &u);
  if (c || u) return false;                       // frozen: no positive delay at all
  Dbm z = Dbm::universe(net.nclocks);
  z.delay();
  std::vector<Constraint> inv;
  all_invariants(*dead_net, s.locs, inv);
  if (!apply_all(z, inv)) return false;
  for (int i = 1; i <= net.nclocks; ++i)
    if (z.get(i, 0) != kInf) return false;        // some clock is capped: time cannot run away
  return true;
}

Federation LivenessEngine::stop_set(int u) {
  Federation f;
  SymState usc;
  const SymState& su = sread(u, usc);
  // A psi-restriction changes what "maximal" means, so the two cases are kept apart rather than
  // merged. Unrestricted -- every `p --> q` run -- keeps the original reading exactly: psi is
  // discrete there, a delay can never leave it, so a node with no discrete successor always ends
  // a maximal run.
  // A psi-restriction may arrive as a restricted network (psi's clock part convex, folded into
  // the invariants) or as a forbidden region (non-convex, routed around). Either one changes what
  // maximality means, so both take the second branch.
  if (dead_net == &net && !avoid_) {
    if (adj[u].empty() || time_diverges(su)) { f.push_back(su.zone); return f; }
    Federation d0;
    if (state_has_deadlock(net, su, &d0)) f.swap(d0);
    return f;
  }
  // Restricted: `adj[u].empty()` no longer implies maximality. The successors may all have been
  // dropped for leaving psi, and a run that can only continue by leaving psi is not maximal, it
  // is simply not a witness. Two things genuinely end a maximal psi-run:
  //
  //   (i)  time diverges inside psi -- judged on the restricted invariant, which is `net`;
  //   (ii) the state is timelocked in the original system -- no transition is ever enabled AND
  //        the original invariant caps the delay, so there is no divergent continuation either.
  //
  // Dropping (ii)'s second half is what made `E[] x <= 3` on af-sat.xml report satisfied: the
  // sink location is deadlocked, but the original bounds no clock there, so its maximal run is
  // the divergent one -- which carries x past 3 and out of psi.
  if (time_diverges(su)) { f.push_back(su.zone); return f; }
  if (orig_time_open(su)) return f;               // a deadlock here continues by delaying
  Federation d;
  // Deadlock is judged in `dead_net`, which is the unrestricted network when `E[] psi` supplied
  // one. Maximality is a property of the original system: a run that can only continue by leaving
  // psi is not maximal, it is simply not a witness.
  //
  // Divergence is the other way round and stays on `net`: the run must diverge inside psi, so the
  // restricted invariant is exactly the right one to consult.
  if (state_has_deadlock(*dead_net, su, &d)) f.swap(d);
  return f;
}

/// @brief Solve `W = W /\ (NoSucc \/ Pre(W))` by descending iteration confined to @p S.
///
/// @param S      the nodes to iterate over; every other node is pinned to the empty federation
/// @param[out] W indexed by node; sized to `nrec()`, empty outside @p S
///
/// @pre Every element of @p S is an expanded `!q` node (see ::LivenessEngine::stop_set).
/// @post On normal return @p W is the exact greatest fixpoint of the operator *restricted to*
///       @p S, hence a post-fixpoint of the operator on the finished graph, hence a sound
///       under-approximation of the true value.
/// @post If @ref capped or @ref cancelled fired, the iteration did **not** converge and @p W is
///       an arbitrary intermediate iterate. It proves nothing in either direction.
///
/// @warning @ref capped is sticky. A capped pass must be reported as unusable, never laundered
///          into a verdict; see ::LivenessEngine::certificate_in.

}  // namespace hora
