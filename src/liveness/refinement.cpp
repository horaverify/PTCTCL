/// @file src/liveness/refinement.cpp
/// @brief Discovering D atoms, pulling them backward, and sealing the graph.
///
/// A newly learned atom is pulled back through the reset-labelled key skeleton, which can revoke
/// a provisional cover. Once both worklists are stable the active records are compacted into the
/// final quotient by redirecting edges already stored.

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

// ---- Online D atom worklist ---------------------------------------------------------------

/// @brief Index provisional replay edges only if D actually changes.
///
/// While D is empty no key can be dirty, so per-key vectors containing every revocable edge are
/// dead weight.  The raw adjacency already contains the exact replay marker and target key.  At
/// the first real atom/bound raise, index the explored prefix once; later edges are appended to
/// their key directly.  This keeps the zero-atom case allocation-free without losing a recipe.
void LivenessEngine::online_enable_deferred_index() {
  if (online_deferred_index_enabled_) return;
  online_deferred_index_enabled_ = true;
  for (std::size_t raw_slot = 0; raw_slot < online_raw_adj_.size(); ++raw_slot) {
    if (raw_slot > UINT32_MAX) { online_failed_ = true; return; }
    std::vector<OtfEdge>& edges = online_raw_adj_[raw_slot];
    for (std::size_t ordinal = 0; ordinal < edges.size(); ++ordinal) {
      const OtfEdge& edge = edges[ordinal];
      if (edge.deferred < 0) continue;
      if (ordinal > UINT32_MAX || edge.tgt < 0 ||
          edge.tgt >= static_cast<int>(online_key_of_.size())) {
        online_failed_ = true;
        return;
      }
      const int key = online_key_of_[edge.tgt];
      const std::uint64_t handle =
          (static_cast<std::uint64_t>(static_cast<std::uint32_t>(raw_slot)) << 32) |
          static_cast<std::uint32_t>(ordinal);
      online_key_by_id_[key]->deferred.push_back(handle);
    }
  }
}

bool LivenessEngine::online_add_diag(int key, const AtomC& atom, bool closure) {
  if (key < 0 || key >= static_cast<int>(online_key_by_id_.size())) {
    online_failed_ = true;
    return false;
  }
  OnlineKeyInfo& ki = *online_key_by_id_[key];
  if (!ki.diag) ki.diag = std::make_unique<std::vector<AtomC>>();
  std::vector<AtomC>& as = *ki.diag;
  if (std::find(as.begin(), as.end(), atom) != as.end()) return false;
  online_enable_deferred_index();
  if (online_failed_) return false;
  if (!online_key_graph_enabled_) online_enable_key_graph();
  if (online_failed_) return false;
  as.push_back(atom);
  online_atom_work_.push_back(OnlineAtomWork{key, atom});
  if (!online_dirty_queued_[key]) {
    online_dirty_queued_[key] = 1;
    online_dirty_keys_.push_back(key);
  }
  if (closure) ++online_am_.closure_atoms;
  return true;
}

bool LivenessEngine::online_raise_bump(int clock, std::int32_t bound, bool closure) {
  if (clock <= 0 || clock >= (int)online_am_.bump.size()) return false;
  if (bound <= online_am_.bump[clock]) return false;
  online_enable_deferred_index();
  if (online_failed_) return false;
  online_am_.bump[clock] = bound;
  online_am_.live[clock] = 1;
  ++online_bound_epoch_;
  online_global_dirty_ = true;
  if (closure) ++online_am_.closure_atoms;
  return true;
}

/// @brief Seed the completion map at a newly discovered discrete key.
///
/// This deliberately uses the universe, not the zone that discovered the key: D is a property
/// of the completed automaton at the discrete key, and a later reached zone must not make the
/// local deadlock condition disappear.
void LivenessEngine::online_seed_key(int key_id) {
  if (online_failed_) return;
  const int nc = net.nclocks;
  OnlineKeyInfo& ki = *online_key_by_id_.at(key_id);
  SymState u;
  key_state(ki.id, u);
  bool fc, fu;
  frozen_flags(*dead_net, u.locs, u.ints, &fc, &fu);
  std::vector<Constraint> ci;
  all_invariants(*dead_net, u.locs, ci);

  // In a non-frozen state, time is unbounded whenever the invariant has no absolute upper
  // constraint x <= c.  Difference and lower constraints alone cannot anchor an upper bound to
  // clock 0.  Detect that syntactically before allocating and closing two dense DBMs; this is the
  // overwhelmingly common seed case in scalable networks.
  if (!fc && !fu) {
    bool has_upper = false;
    for (const Constraint& c : ci)
      if (c.i > 0 && c.j == 0 && c.bound < kInf) { has_upper = true; break; }
    if (!has_upper) return;
  }

  u.zone = Dbm::universe(nc);
  Dbm inv = Dbm::universe(nc);
  if (!fc && !fu) inv.delay();
  // Keep this seed computation local.  Most keys never participate in an atom pullback, and a
  // dense invariant DBM per discrete key is far larger than the key skeleton itself.  If a
  // later pullback actually reaches this key, online_pullback reconstructs and caches the same
  // invariant lazily.
  if (!apply_all(inv, ci)) return;

  // A non-frozen time-open key has an infinite delay from every valuation of its invariant.
  // Its deadlock completion is therefore empty, so constructing all enabled/escape federations
  // cannot reveal a seed atom.  This test used to sit after state_has_deadlock(), which made
  // every such key pay the full symbolic subtraction before reaching the same conclusion.
  bool open = true;
  for (int c = 1; c <= nc; ++c)
    if (inv.get(c, 0) != kInf) { open = false; break; }
  if (open && !fc && !fu) return;                 // divergence already covers every stop

  Federation stuck, esc;
  if (!state_has_deadlock(*dead_net, u, &stuck, &esc)) return;
  ki.may_deadlock = true;

  Federation kept;
  kept.reserve(esc.size());
  for (const Dbm& w : esc) {
    Dbm t = w;
    t.intersect(inv);
    if (t.empty()) { ++online_am_.dropped_vacuous; continue; }
    kept.push_back(w);
  }
  if (kept.empty()) return;                       // Dead(q) == Inv(q): no distinguishing atom
  esc.swap(kept);
  ++online_am_.seed_keys;

  const int dim = nc + 1;
  std::vector<std::uint32_t> mbits(mindbm_bits2words(std::uint32_t(dim) * std::uint32_t(dim)));
  std::vector<int> mfirst(dim), mnext(dim);
  auto extract = [&](const Federation& f, std::vector<AtomC>& diag,
                     std::vector<std::pair<int, std::int32_t>>& nond) {
    for (const Dbm& d : f) {
      if (d.empty()) continue;
      mindbm_analyze(d.data().data(), dim, mbits.data(), mfirst.data(), mnext.data());
      for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
          if (i == j) continue;
          const std::uint32_t idx = std::uint32_t(i) * std::uint32_t(dim) + std::uint32_t(j);
          if (!((mbits[idx >> 5] >> (idx & 31)) & 1u)) continue;
          const raw_t r = d.get(i, j);
          if (r >= kInf) continue;
          if (i == 0 && r >= kZeroLe) continue;
          if (inv.get(i, j) <= r) continue;
          if (raw_add(inv.get(j, i), r) < kZeroLe) {
            ++online_am_.dropped_disjoint;
            continue;
          }
          if (i > 0 && j > 0) {
            const AtomC a{i, j, r};
            if (std::find(diag.begin(), diag.end(), a) == diag.end()) diag.push_back(a);
          } else {
            const int c = i == 0 ? j : i;
            const std::int32_t b = i == 0 ? -raw_bound(r) : raw_bound(r);
            bool seen = false;
            for (auto& x : nond)
              if (x.first == c) { x.second = std::max(x.second, b); seen = true; break; }
            if (!seen) nond.push_back({c, b});
          }
        }
    }
  };

  std::vector<AtomC> residue_diag, escape_diag;
  std::vector<std::pair<int, std::int32_t>> residue_nond, escape_nond;
  extract(stuck, residue_diag, residue_nond);
  extract(esc, escape_diag, escape_nond);
  const bool take_escape = escape_diag.size() + 4 * escape_nond.size()
                         <= residue_diag.size() + 4 * residue_nond.size();
  const std::vector<AtomC>& diag = take_escape ? escape_diag : residue_diag;
  const auto& nond = take_escape ? escape_nond : residue_nond;
  online_am_.seed_atoms += diag.size() + nond.size();
  for (const AtomC& a : diag) online_add_diag(ki.id, a, /*closure=*/false);
  for (const auto& x : nond) {
    ++online_am_.nondiag_atoms;
    online_raise_bump(x.first, x.second, /*closure=*/false);
  }
}

/// @brief Pull one target diagonal atom backwards through one reset-labelled key edge.
void LivenessEngine::online_pullback(const OnlineKeyEdge& e, const AtomC& atom) {
  const std::vector<int>& resets = online_reset_sets_[e.reset_class];
  const auto reset = [&](int c) {
    return c > 0 && std::binary_search(resets.begin(), resets.end(), c);
  };
  const int i = reset(atom.i) ? 0 : atom.i;
  const int j = reset(atom.j) ? 0 : atom.j;
  if (i == j) return;
  if (i == 0 || j == 0) {
    const int c = i == 0 ? j : i;
    const std::int32_t b = i == 0 ? -raw_bound(atom.b) : raw_bound(atom.b);
    online_raise_bump(c, b, /*closure=*/true);
    return;
  }

  OnlineKeyInfo& si = *online_key_by_id_.at(e.src);
  if (!si.inv_ready) {
    SymState rep;
    key_state(si.id, rep);
    si.inv_ready = true;
    Dbm inv;
    if (key_invariant(*dead_net, rep, net.nclocks, inv))
      si.inv = std::make_unique<Dbm>(std::move(inv));
  }
  if (si.inv) {
    if (si.inv->get(i, j) <= atom.b) return;
    if (raw_add(si.inv->get(j, i), atom.b) < kZeroLe) {
      ++online_am_.dropped_disjoint;
      return;
    }
  }
  online_add_diag(si.id, AtomC{i, j, atom.b}, /*closure=*/true);
}

void LivenessEngine::online_drain_atoms() {
  while (!online_atom_work_.empty() && !online_failed_) {
    OnlineAtomWork w = std::move(online_atom_work_.front());
    online_atom_work_.pop_front();
    if (w.key < 0 || w.key >= static_cast<int>(online_key_by_id_.size())) continue;
    const OnlineKeyInfo& ki = *online_key_by_id_[w.key];
    // A pullback may append an atom to the source of a self-loop.  Copy each intrusive edge
    // record before the call so neither atom-map growth nor a future edge-arena reallocation can
    // invalidate the iteration state.
    for (int ei = ki.incoming; ei >= 0;) {
      const OnlineKeyEdge edge = online_key_edges_[ei];
      ei = edge.next_in;
      online_pullback(edge, w.atom);
    }
  }
}

int LivenessEngine::online_register_key(const LocVec& locs, const IntVec& ints,
                             std::uint64_t packed, bool packed_ready) {
  // Most generated successors revisit an existing discrete key.  The packed path is lossless;
  // wider keys use the exact vector bucket below.  D atoms are indexed by this numeric key, so
  // the online path never serialises a discrete state into a temporary string.
  std::uint64_t h = 0;
  std::vector<int>* slots = nullptr;
  if (online_packed_key_usable_) {
    if (!packed_ready) {
      packed = 0;
      for (int p = 0; p < static_cast<int>(locs.size()); ++p)
        packed |= static_cast<std::uint64_t>(static_cast<unsigned>(locs[p]))
                  << online_disc_shift_[p];
      for (std::size_t v = 0; v < ints.size(); ++v)
        packed |= static_cast<std::uint64_t>(ints[v] - net.intvars[v].lo)
                  << online_disc_shift_[net.nprocs() + static_cast<int>(v)];
    }
    std::size_t slot = static_cast<std::size_t>(mix_packed_key(packed)) & online_packed_key_mask_;
    while (online_packed_key_values_[slot] >= 0) {
      if (online_packed_key_keys_[slot] == packed) return online_packed_key_values_[slot];
      slot = (slot + 1) & online_packed_key_mask_;
    }
  } else {
    h = disc_hash(locs, ints);
    slots = &online_key_index_[h];
    for (int id : *slots) {
      const int* kl = key_locs(id);
      if (!std::equal(locs.begin(), locs.end(), kl, kl + key_loc_stride_)) continue;
      const long* kint = key_ints(id);
      if (!std::equal(ints.begin(), ints.end(), kint, kint + key_int_stride_)) continue;
      return id;
    }
  }

  online_keys_.emplace_back();
  OnlineKeyInfo& ki = online_keys_.back();
  ki.id = static_cast<int>(online_key_by_id_.size());
  ki.packed_key = packed;
  key_locs_.insert(key_locs_.end(), locs.begin(), locs.end());
  key_ints_.insert(key_ints_.end(), ints.begin(), ints.end());
  online_key_by_id_.push_back(&ki);
  online_dirty_queued_.push_back(0);
  if (online_packed_key_usable_) {
    if ((online_packed_key_count_ + 1) * 10 >= online_packed_key_values_.size() * 7) {
      std::vector<std::uint64_t> old_keys = std::move(online_packed_key_keys_);
      std::vector<int> old_values = std::move(online_packed_key_values_);
      const std::size_t capacity = old_values.size() * 2;
      online_packed_key_keys_.resize(capacity);
      online_packed_key_values_.assign(capacity, -1);
      online_packed_key_mask_ = capacity - 1;
      for (std::size_t old = 0; old < old_values.size(); ++old) {
        if (old_values[old] < 0) continue;
        std::size_t slot = static_cast<std::size_t>(mix_packed_key(old_keys[old])) &
                           online_packed_key_mask_;
        while (online_packed_key_values_[slot] >= 0)
          slot = (slot + 1) & online_packed_key_mask_;
        online_packed_key_keys_[slot] = old_keys[old];
        online_packed_key_values_[slot] = old_values[old];
      }
    }
    std::size_t slot = static_cast<std::size_t>(mix_packed_key(packed)) & online_packed_key_mask_;
    while (online_packed_key_values_[slot] >= 0)
      slot = (slot + 1) & online_packed_key_mask_;
    online_packed_key_keys_[slot] = packed;
    online_packed_key_values_[slot] = ki.id;
    ++online_packed_key_count_;
  } else {
    slots->push_back(ki.id);
  }
  online_seed_key(ki.id);
  return ki.id;
}

int LivenessEngine::online_reset_class(int tag) {
  if (tag < 0 || tag >= static_cast<int>(tags_.size())) {
    online_failed_ = true;
    return -1;
  }
  if (online_tag_reset_class_.size() <= static_cast<std::size_t>(tag))
    online_tag_reset_class_.resize(static_cast<std::size_t>(tag) + 1, -1);
  int& cached = online_tag_reset_class_[tag];
  if (cached >= 0) return cached;

  std::vector<int> resets;
  for (const auto& pe : tags_[tag].parts)
    for (int c : net.procs[pe.first].edges[pe.second].clock_resets)
      if (c >= 1 && c <= net.nclocks) resets.push_back(c);
  std::sort(resets.begin(), resets.end());
  resets.erase(std::unique(resets.begin(), resets.end()), resets.end());

  for (std::size_t i = 0; i < online_reset_sets_.size(); ++i)
    if (online_reset_sets_[i] == resets) return cached = static_cast<int>(i);
  online_reset_sets_.push_back(std::move(resets));
  return cached = static_cast<int>(online_reset_sets_.size() - 1);
}

/// @brief Build the reset-labelled key skeleton only when the first diagonal D atom exists.
///
/// Before that point there is nothing to pull backward.  Raw graph edges already retain their
/// transition tags and target representatives, so the exact key skeleton for the explored
/// prefix can be indexed once, locally, without regenerating a successor or rebuilding a graph.
void LivenessEngine::online_enable_key_graph() {
  if (online_key_graph_enabled_) return;
  online_key_graph_enabled_ = true;
  for (std::size_t raw_slot = 0; raw_slot < online_raw_adj_.size(); ++raw_slot) {
    if (raw_slot >= online_raw_owner_.size()) { online_failed_ = true; return; }
    const int owner = online_raw_owner_[raw_slot];
    if (owner < 0 || owner >= static_cast<int>(online_key_of_.size())) {
      online_failed_ = true;
      return;
    }
    const int src = online_key_of_[owner];
    for (const OtfEdge& edge : online_raw_adj_[raw_slot]) {
      if (edge.tgt < 0 || edge.tgt >= static_cast<int>(online_key_of_.size())) {
        online_failed_ = true;
        return;
      }
      online_register_edge(src, online_key_of_[edge.tgt], online_reset_class(edge.tag));
      if (online_failed_) return;
    }
  }
}

void LivenessEngine::online_register_edge(int src, int tgt, int reset_class) {
  if (reset_class < 0 || reset_class >= static_cast<int>(online_reset_sets_.size())) {
    online_failed_ = true;
    return;
  }
  OnlineKeyInfo& source = *online_key_by_id_.at(src);
  for (int ei = source.outgoing; ei >= 0; ei = online_key_edges_[ei].next_out) {
    const OnlineKeyEdge& old = online_key_edges_[ei];
    if (old.tgt == tgt && old.reset_class == reset_class) return;
  }
  const int id = static_cast<int>(online_key_edges_.size());
  OnlineKeyInfo& target = *online_key_by_id_.at(tgt);
  online_key_edges_.push_back(OnlineKeyEdge{src, tgt, reset_class, source.outgoing,
                                             target.incoming});
  source.outgoing = id;
  target.incoming = id;

  // An edge discovered after its target's atoms must observe all of them immediately; future
  // atoms arrive through online_atom_work_ and use the same incoming list.
  if (target.diag) {
    const std::vector<AtomC> atoms = *target.diag;
    for (const AtomC& a : atoms) online_pullback(online_key_edges_[id], a);
  }
}

void LivenessEngine::online_recheck_key(int key) {
  if (key < 0 || key >= static_cast<int>(online_key_by_id_.size())) return;
  OnlineKeyInfo& ki = *online_key_by_id_[key];
  const std::size_t count = ki.node_count;
  int v = ki.node_head;  // activation itself does not append ahead of this captured list
  if (online_perf.enabled) {
    ++online_perf.recheck_calls;
    online_perf.recheck_nodes += count + ki.deferred.size();
  }
  for (std::size_t n = 0; n < count && v >= 0; ++n) {
    const int current = v;
    v = online_next_in_key_[current];
    if (online_active_[current]) continue;
    const int c = online_cover_[current];
    if (c >= 0 && c < (int)nrec() && online_active_[c]) {
      if (online_covers(current, c, ki.id))
        continue;  // its recorded certificate survived the newly settled D map
    }
    const int cover = online_coverer(current, /*require_expanded=*/false);
    if (cover >= 0) {
      online_set_cover(current, cover);
    } else {
      online_activate(current);
    }
  }

  // Provisional successors that were never active have no node and no stored DBM.  Recreate
  // only those belonging to this D-dirty key, retest the exact candidate, and materialise it
  // only if the refined map leaves it uncovered.  The raw edge remains in the same graph and is
  // simply retargeted; this is local lazy reconstruction, not another reachability traversal.
  const std::size_t deferred_count = ki.deferred.size();
  for (std::size_t n = 0; n < deferred_count; ++n) {
    const std::uint64_t handle = ki.deferred[n];
    const int raw_slot = static_cast<int>(static_cast<std::uint32_t>(handle >> 32));
    const int ordinal = static_cast<int>(static_cast<std::uint32_t>(handle));
    if (raw_slot < 0 || raw_slot >= static_cast<int>(online_raw_adj_.size()) ||
        raw_slot >= static_cast<int>(online_raw_owner_.size()) || ordinal < 0 ||
        ordinal >= static_cast<int>(online_raw_adj_[raw_slot].size())) {
      online_failed_ = true;
      return;
    }
    OtfEdge& edge = online_raw_adj_[raw_slot][ordinal];
    if (edge.deferred < 0) continue;
    const int source = online_raw_owner_[raw_slot];

    SymState candidate;
    if (!online_replay(source, ordinal, ki.id, candidate)) {
      online_failed_ = true;
      return;
    }
    if (online_perf.enabled) ++online_perf.deferred_replayed;

    int cover = -1;
    if (edge.tgt >= 0 && edge.tgt < static_cast<int>(nrec()) && online_active_[edge.tgt] &&
        online_covers(candidate, edge.tgt, ki.id)) {
      cover = edge.tgt;
    } else {
      for (int j = ki.active_head; j >= 0; j = online_active_next_[j])
        if (online_active_[j] && online_covers(candidate, j, ki.id)) {
          cover = j;
          break;
        }
    }

    if (cover >= 0) {
      edge.tgt = cover;
      // These two relations are future-proof.  Once either holds, even another D refinement
      // cannot wake the candidate, so its replay recipe can be discarded permanently.
      const bool permanent = stored_includes(cover, candidate.zone);
      if (permanent) {
        edge.deferred = -1;
      }
      continue;
    }

    int nested_deferred = -1;
    const int v = online_intern(std::move(candidate), ki.id, -1, -1, &nested_deferred);
    if (nested_deferred >= 0) {
      online_failed_ = true;
      return;
    }
    edge.tgt = v;
    edge.deferred = -1;
    if (online_perf.enabled) ++online_perf.deferred_promoted;
  }
}

void LivenessEngine::online_recheck_dirty() {
  if (online_global_dirty_) {
    online_global_dirty_ = false;
    online_dirty_keys_.clear();
    std::fill(online_dirty_queued_.begin(), online_dirty_queued_.end(), 0);
    const std::size_t count = online_key_by_id_.size();
    for (std::size_t key = 0; key < count; ++key)
      online_recheck_key(static_cast<int>(key));
    return;
  }
  std::vector<int> keys = std::move(online_dirty_keys_);
  online_dirty_keys_.clear();
  for (int key : keys) {
    if (key >= 0 && key < static_cast<int>(online_dirty_queued_.size()))
      online_dirty_queued_[key] = 0;
    online_recheck_key(key);
  }
}

/// @brief Build the root through the common exact root pipeline, then retain it online.
bool LivenessEngine::online_build_all() {
  return build_all();
}

/// @brief Generate one raw state's semantic successors exactly once.
///
/// The two phases are intentional.  First register every target key and, once diagonal D exists,
/// its reset-labelled key edge.  A new atom may then invalidate an earlier certificate.  Only
/// after the atom worklist reaches its current fixed point are the successors covered or made
/// active.
void LivenessEngine::online_expand(int u) {
  grow();
  if (cancelled() || online_failed_ || u < 0 || u >= (int)nrec()) return;
  if (!online_active_[u] || expanded[u]) return;

  expanded[u] = 1;
  ++expand_calls;
  ++outer_expands;

  std::vector<SymState> succ;
  std::vector<GfpTag> stags;
  const int src = online_key_of_[u];
  const OnlineKeyInfo& src_info = *online_key_by_id_[src];
  const bool profile = online_perf.enabled;
  SymState usc;
  key_state(src_info.id, usc);
  zread(u, usc.zone);
  std::chrono::steady_clock::time_point successor_start;
  if (profile) successor_start = std::chrono::steady_clock::now();
  tagged_successors(net, usc, L, U, qmask, emode, succ, stags,
                  avoid_forward_ ? avoid_ : nullptr);
  std::chrono::steady_clock::time_point successor_done;
  if (profile) successor_done = std::chrono::steady_clock::now();

  // A concrete terminal/divergent witness is conclusive immediately.  Unlike the old repeated
  // gfp probe this performs no backward pass on a changing graph; if it misses, construction
  // continues to D-quiescence and the one normal backward pass decides the query.
  // `A[] not deadlock` is settled by one deadlocked valuation, so the search stops at the first
  // record holding one rather than finishing the graph and reading it off afterwards.
  if (deadlock_scan_ && online_deadlock_at(u, succ.empty())) {
    deadlock_hit_ = true;
    eg_early_hit_ = true;                  // the shared signal that halts the construction
    return;
  }
  if (eg_early_ && online_eg_terminal_witness(u, succ.empty())) {
    eg_early_hit_ = true;
    return;
  }
  if (leadsto_early_ && online_leadsto_witness(u, succ.empty())) {
    eg_early_hit_ = true;
    return;
  }

  std::vector<int> target_keys(succ.size());
  std::vector<int> tag_ids(succ.size());
  const auto packed_successor = [&](std::size_t k) {
    std::uint64_t packed = src_info.packed_key;
    for (const auto& part : stags[k].parts) {
      const int p = part.first;
      const std::uint64_t field_mask = online_disc_mask_[p] << online_disc_shift_[p];
      packed = (packed & ~field_mask) |
               ((static_cast<std::uint64_t>(static_cast<unsigned>(succ[k].locs[p])) &
                 online_disc_mask_[p]) << online_disc_shift_[p]);
    }
    // Generated and ordinary updates may change integer variables independently of the
    // participating process list.  Integer vectors are normally empty; when present, compare
    // them exactly and patch only changed fields.
    for (std::size_t v = 0; v < succ[k].ints.size(); ++v) {
      if (succ[k].ints[v] == key_ints(src_info.id)[v]) continue;
      const int field = net.nprocs() + static_cast<int>(v);
      const std::uint64_t field_mask = online_disc_mask_[field] << online_disc_shift_[field];
      const std::uint64_t value = static_cast<std::uint64_t>(succ[k].ints[v] - net.intvars[v].lo);
      packed = (packed & ~field_mask) |
               ((value & online_disc_mask_[field]) << online_disc_shift_[field]);
    }
    return packed;
  };
  for (std::size_t k = 0; k < succ.size(); ++k) {
    if (profile) {
      auto t = std::chrono::steady_clock::now();
      const std::uint64_t packed = online_packed_key_usable_ ? packed_successor(k) : 0;
      target_keys[k] = online_register_key(succ[k].locs, succ[k].ints, packed,
                                            online_packed_key_usable_);
      online_perf.register_key_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t).count();
      t = std::chrono::steady_clock::now();
      tag_ids[k] = intern_tag(stags[k]);
      online_perf.register_tag_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t).count();
    } else {
      const std::uint64_t packed = online_packed_key_usable_ ? packed_successor(k) : 0;
      target_keys[k] = online_register_key(succ[k].locs, succ[k].ints, packed,
                                            online_packed_key_usable_);
      tag_ids[k] = intern_tag(stags[k]);
    }
    if (online_failed_) return;
  }
  if (online_key_graph_enabled_) {
    for (std::size_t k = 0; k < succ.size(); ++k) {
      if (profile) {
        auto t = std::chrono::steady_clock::now();
        const int reset_class = online_reset_class(tag_ids[k]);
        online_perf.register_reset_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t).count();
        t = std::chrono::steady_clock::now();
        online_register_edge(src, target_keys[k], reset_class);
        online_perf.register_edge_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t).count();
      } else {
        online_register_edge(src, target_keys[k], online_reset_class(tag_ids[k]));
      }
      if (online_failed_) return;
    }
  }
  std::chrono::steady_clock::time_point registration_done;
  if (profile) registration_done = std::chrono::steady_clock::now();
  online_drain_atoms();
  online_recheck_dirty();
  if (cancelled() || online_failed_) return;

  // Only expanded representatives own an outgoing raw-edge vector.  Keep its slot separate
  // from the persistent record id.  Provisionally covered successors are not materialised, so
  // allocating an empty vector for every possible target would dominate the metadata footprint.
  if (online_raw_slot_[u] < 0) {
    online_raw_slot_[u] = static_cast<int>(online_raw_adj_.size());
    online_raw_adj_.emplace_back();
    online_raw_owner_.push_back(u);
  }
  const int raw_slot = online_raw_slot_[u];
  online_raw_adj_[raw_slot].reserve(online_raw_adj_[raw_slot].size() + succ.size());
  for (std::size_t k = 0; k < succ.size(); ++k) {
    int deferred = -1;
    const int v = online_intern(std::move(succ[k]), target_keys[k], u,
                                static_cast<int>(k), &deferred);
    online_raw_adj_[raw_slot].push_back(OtfEdge{v, tag_ids[k], deferred});
    if (cancelled() || online_failed_) return;
  }
  if (profile) {
    const auto intern_done = std::chrono::steady_clock::now();
    online_perf.successor_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        successor_done - successor_start).count();
    online_perf.register_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        registration_done - successor_done).count();
    online_perf.intern_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        intern_done - registration_done).count();
  }

  // Keep the active quotient lean.  A materialised representative that becomes covered turns
  // inactive; its already generated raw edges remain available if a later atom reactivates it.
  online_compact_key(u);
}

/// @brief Drain the coupled expansion / atom / cover-certificate worklists to quiescence.
void LivenessEngine::online_drain_forward_work() {
  for (;;) {
    online_drain_atoms();
    online_recheck_dirty();
    if (cancelled() || online_failed_ || eg_early_hit_ || online_work_.empty()) break;

    const int u = online_work_.front();
    online_work_.pop_front();
    if (u < 0 || u >= (int)nrec()) continue;
    online_inq_[u] = 0;
    if (!online_active_[u] || expanded[u]) continue;
    if (online_eg_prune_ && !online_is_nq(u)) {
      // E[] never returns to psi after leaving it.  Keep the target vertex for incoming edge
      // redirection and D seeding, but its successors cannot influence the initial-state gfp.
      expanded[u] = 1;
      if (online_raw_slot_[u] < 0) {
        online_raw_slot_[u] = static_cast<int>(online_raw_adj_.size());
        online_raw_adj_.emplace_back();
        online_raw_owner_.push_back(u);
      }
      continue;
    }
    online_expand(u);
  }
}

/// @brief Finish exact adjacencies only when the terminal-witness fast path did not decide E[].
///
/// During the first forward phase an unexpanded representative may leave the active antichain;
/// its exact zone is parked, not deleted.  This is what lets a concrete terminal witness be
/// reached with the small D quotient.  If no such witness exists, the backward solver needs the
/// exact incoming image of every materialised record, so those parked records are activated and
/// expanded here before the single gfp starts.  No state is rediscovered from the root and no
/// graph is rebuilt: this drains retained records through the same online-D worklist.
void LivenessEngine::online_complete_eg_history() {
  online_eg_repair_ = true;
  const std::size_t parked = nrec();
  std::size_t pending = 0;
  for (std::size_t v = 0; v < parked; ++v)
    if (!expanded[v]) { ++pending; online_activate(static_cast<int>(v)); }
  if (online_perf.enabled)
    std::fprintf(stderr, "ONLINE-EG-REPAIR before=%zu pending=%zu queued=%zu\n",
                 parked, pending, online_work_.size());
  online_drain_forward_work();
  if (online_perf.enabled) {
    std::size_t remaining = 0;
    for (std::size_t v = 0; v < nrec(); ++v) remaining += !expanded[v];
    std::fprintf(stderr, "ONLINE-EG-REPAIR after=%zu remaining=%zu\n",
                 nrec(), remaining);
  }
}

/// @brief Start the persistent D forward construction and settle all online refinement work.
void LivenessEngine::online_base_forward() {
  fwd_ok_ = online_build_all();
  if (!fwd_ok_) return;
  online_drain_forward_work();
}

/// @brief Seal the final D quotient by redirecting raw edges through final active coverers.
///
/// No successor is regenerated here.  The defensive wake-up loop only handles an edge whose
/// target lost its last certificate after the main worklist checked it; it activates that retained
/// raw record and lets the ordinary online worklist expand it for the first (and only) time.
void LivenessEngine::online_finalize_graph() {
  if (!fwd_ok_ || cancelled() || online_failed_) return;

  for (;;) {
    bool woke = false;
    for (std::size_t u = 0; u < nrec(); ++u) {
      if (!online_active_[u] || !expanded[u]) continue;
      const int raw_slot = online_raw_slot_[u];
      if (raw_slot < 0) { online_failed_ = true; return; }
      for (const OtfEdge& e : online_raw_adj_[raw_slot]) {
        if (online_resolve_cover(e.tgt, /*require_expanded=*/true) >= 0) continue;
        // A live replay recipe must still resolve through the certificate established by the
        // last key recheck.  Activating its old cover would not reconstruct the candidate and
        // would therefore be the wrong defensive recovery.
        if (e.deferred >= 0) { online_failed_ = true; return; }
        // This is normally an inactive materialised record whose last D cover was invalidated.
        // It is activated, never rediscovered from the root.  An active but
        // unexpanded target is also explicitly queued here, guarding against stale queue bits
        // left by representative compaction.
        if (!online_active_[e.tgt] || !expanded[e.tgt]) {
          online_activate(e.tgt);
          woke = true;
        } else {
          online_failed_ = true;
          return;
        }
      }
    }
    if (!woke) break;
    online_drain_forward_work();
    if (cancelled() || online_failed_) return;
  }

  online_am_.keys_tagged = 0;
  for (const OnlineKeyInfo* ki : online_key_by_id_)
    if (ki->diag && !ki->diag->empty()) ++online_am_.keys_tagged;
  online_am_.bumped = 0;
  for (int c = 1; c <= net.nclocks; ++c)
    if (online_am_.bump[c] != kNoBump) ++online_am_.bumped;
  online_am_.usable = !online_failed_;

  // Every materialised record remains a backward vertex, including an old representative that
  // later left the forward antichain.  Its incoming edge was generated for this exact zone; a
  // forward-sound redirection to a simulator is not an exact inverse for exact_predecessor.  Compaction
  // therefore removes records only from future cover scans, never from the finished graph.
  // With that invariant the existing numbering, zones, and raw adjacency can all be moved.
  bool already_dense = online_active_.size() == nrec();
  for (std::size_t v = 0; v < nrec() && already_dense; ++v)
    already_dense = expanded[v] && online_raw_slot_[v] >= 0;

  if (already_dense) {
    adj.clear();
    adj.resize(nrec());
    node_key_.assign(online_key_of_.begin(), online_key_of_.begin() + nrec());
    for (std::size_t v = 0; v < nrec(); ++v) {
      std::vector<OtfEdge>& raw = online_raw_adj_[online_raw_slot_[v]];
      for (OtfEdge& edge : raw) {
        // Materialised targets keep their own identity, even after leaving the active forward
        // antichain. A never-materialised provisional successor already names its final active
        // coverer here; dirty-key replay would have promoted it if that certificate had failed.
        if (edge.tgt < 0 || edge.tgt >= static_cast<int>(nrec()) ||
            !expanded[edge.tgt] || online_raw_slot_[edge.tgt] < 0) {
          const int representative = online_resolve_cover(edge.tgt, /*require_expanded=*/true);
          if (representative < 0) { online_failed_ = true; return; }
          edge.tgt = representative;
        }
        edge.deferred = -1;
      }
      adj[v] = std::move(raw);
    }
  } else {
    // Reindex only when refinement really left inactive materialised records. No successor is
    // recomputed and no D decision is revisited.
    //
    // IN place. Compaction only ever drops, so the new index of a survivor is at most its old one,
    // and rewriting ascending can therefore only overwrite slots already read.
    std::vector<int> remap(nrec(), -1);
    std::size_t kept = 0;
    // Count first, then size the key index exactly.
    for (std::size_t v = 0; v < nrec(); ++v)
      if (expanded[v] && online_raw_slot_[v] >= 0) ++kept;
    node_key_.assign(kept, -1);
    kept = 0;
    Dbm zsc;
    for (std::size_t old = 0; old < nrec(); ++old) {
      if (!expanded[old] || online_raw_slot_[old] < 0) continue;
      const int now = static_cast<int>(kept++);
      remap[old] = now;
      // Alias rather than store: a packed record must not be re-encoded (see ZoneStore::alias),
      // and even dense this is a memcpy instead of a load-and-re-store.
      zs_.alias(now, static_cast<int>(old));
      node_key_[now] = online_key_of_[old];
    }
    // The adjacency moves the same way, and each raw vector is moved out rather than copied.
    adj.clear();
    adj.resize(kept);
    for (std::size_t old = 0; old < nrec(); ++old) {
      const int now = remap[old];
      if (now < 0) continue;
      std::vector<OtfEdge>& raw = online_raw_adj_[online_raw_slot_[old]];
      for (OtfEdge& e : raw) {
        int representative = e.tgt;
        if (representative < 0 || representative >= static_cast<int>(nrec()) ||
            !expanded[representative] || online_raw_slot_[representative] < 0)
          representative = online_resolve_cover(e.tgt, /*require_expanded=*/true);
        if (representative < 0) { online_failed_ = true; return; }
        const int tgt = remap[representative];
        if (tgt < 0) { online_failed_ = true; return; }
        e.tgt = tgt;
        e.deferred = -1;
      }
      adj[now] = std::move(raw);
    }
    zs_.truncate(kept);
  }
  // From here on `node_key_` IS the graph: the compaction branch keeps fewer entries than
  // records were created, and every table below is sized from ::nrec.  `node` stays empty on
  // this route -- the discrete half is read out of the key arenas, which therefore outlive the
  // rest of the online state.
  online_nrec_ = node_key_.size();
  bucket.clear();
  {
    SymState dsc;
    for (std::size_t v = 0; v < nrec(); ++v) {
      nstate(static_cast<int>(v), dsc);
      bucket[disc_hash(dsc.locs, dsc.ints)].push_back(static_cast<int>(v));
    }
  }
  expanded.assign(nrec(), 1);
  pc.assign(nrec(), 2);
  qc.assign(nrec(), 2);
  ntag_.assign(nrec(), 0);
  ntag_out_.assign(nrec(), 0);
  online_active_.clear();
  online_active_next_.clear();
  online_active_prev_.clear();
  online_cover_.clear();
  online_raw_adj_.clear();
  online_raw_owner_.clear();
  online_raw_slot_.clear();
  online_inq_.clear();
  online_work_.clear();
  online_key_by_id_.clear();
  // key_locs_/key_ints_ and the strides deliberately survive: `node_key_` points into them for
  // the rest of the run.  ::online_reset releases them at the start of the next construction.
  online_key_of_.clear();
  online_next_in_key_.clear();
  online_keys_.clear();
  online_key_index_.clear();
  online_packed_key_keys_.clear();
  online_packed_key_values_.clear();
  online_packed_key_mask_ = 0;
  online_packed_key_count_ = 0;
  online_disc_shift_.clear();
  online_disc_mask_.clear();
  online_key_edges_.clear();
  online_reset_sets_.clear();
  online_tag_reset_class_.clear();
  online_atom_work_.clear();
  online_dirty_keys_.clear();
  online_dirty_queued_.clear();
  online_bound_key_ = -1;
  // Subsequent code is the ordinary backward solver over the compact final graph.
  online_d_ = false;
}

/// @brief Build the whole reachable zone graph.
///
/// @pre ::LivenessEngine::setup has run.
/// @post @ref fwd_ok_ is @c false iff the initial state is vacuous; otherwise every reachable
///       node is stored and expanded, subject to cancellation.
///
///       frontier. A frontier that pushes a successor whenever `!expanded[v]` is harmless while
///       ::LivenessEngine::intern only appends, since `expanded` then goes 0 -> 1 once per node; it becomes
///       quadratic as soon as intern rewrites, because every rewrite resets `expanded[v]` and
///       re-queues `v` once per in-edge, every level. Deduplicating changes only how often a
///       node is queued, never which nodes are explored, so stored-state counts are unaffected.

}  // namespace hora
