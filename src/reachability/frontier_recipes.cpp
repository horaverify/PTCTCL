/// @file src/reachability/frontier_recipes.cpp
/// @brief The host half of GPU expansion: a discrete state's transitions as flat recipes.
#include "hora/detail/frontier_gpu.hpp"

#include "hora/reachability.hpp"
#include "hora/detail/semantics.hpp"

namespace hora {
namespace detail {

namespace {

/// Emits one recipe; owns the scratch successor state so the enumeration allocates nothing per
/// transition once warm.
struct RecipeSink {
  const Network& net;
  const KeyLayout& kl;
  const NetTarget& target;
  const std::function<void(const LocVec&, IntVec&)>& canon;
  RecipeTable& out;
  LocVec locs;
  IntVec ints;

  void guards(const std::vector<Constraint>& cs) {
    for (const auto& c : cs) { out.g.push_back(c.i); out.g.push_back(c.j); out.g.push_back(c.bound); }
  }
  void resets(const std::vector<int>& rs) { out.r.insert(out.r.end(), rs.begin(), rs.end()); }

  // Called once the successor's discrete part (locs, ints) and its guards and resets are in place.
  void close() {
    std::uint32_t flags = 0;
    bool committed, urgent;
    frozen_flags(net, locs, ints, &committed, &urgent);
    if (committed || urgent) flags |= RecipeTable::kFrozen;
    bool hit = !target.never;
    for (const auto& lr : target.loc_req) hit = hit && locs[lr.first] == lr.second;
    if (hit && int_ok(target.int_cond, ints)) flags |= RecipeTable::kTargetDisc;
    if (canon) canon(locs, ints);
    out.key.push_back(kl.encode(locs, ints));
    out.flags.push_back(flags);
    out.g_off.push_back(static_cast<std::uint32_t>(out.g.size() / 3));
    out.r_off.push_back(static_cast<std::uint32_t>(out.r.size()));
  }
};

}  // namespace

void append_recipes(const Network& net, const KeyLayout& kl, const NetTarget& target,
                    const std::function<void(const LocVec&, IntVec&)>& canon,
                    const LocVec& s_locs, const IntVec& s_ints, RecipeTable& out) {
  // Plain locals, not thread_local: LocVec and IntVec draw on HDBM's per-thread pool, and a
  // thread_local container outlives that pool at thread exit and frees into it after it is gone.
  SymState s;   // build_broadcast_parts reads a SymState
  s.locs = s_locs;
  s.ints = s_ints;
  RecipeSink k{net, kl, target, canon, out, {}, {}};
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, s.ints, &src_committed, &src_urgent);
  const auto committed_at = [&](int p) { return net.procs[p].locations[s.locs[p]].committed; };

  // ---- location edges ----
  for (int p = 0; p < net.nprocs(); ++p) {
    if (src_committed && !committed_at(p)) continue;
    for (int ei : net.procs[p].edges_of(s.locs[p])) {
      const NEdge& e = net.procs[p].edges[ei];
      if (e.sync_chan >= 0) continue;
      if (!int_ok(e.int_guard, s.ints) || !gen_ok(net, e, s.ints.data(), s.ints.size())) continue;
      k.locs.assign(s.locs.begin(), s.locs.end());
      k.locs[p] = e.tgt;
      k.ints.assign(s.ints.begin(), s.ints.end());
      for (const auto& a : e.int_updates) k.ints[a.var] = a.val;
      if (!gen_apply(net, e, k.ints.data(), k.ints.size())) continue;
      k.guards(e.clock_guard);
      k.resets(e.clock_resets);
      k.close();
    }
  }

  // ---- binary channel synchronisation ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1 || net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      const bool i_comm = committed_at(i);
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan != se.sync_chan || re.sync_dir != -1) continue;
          if (src_committed && !(i_comm || committed_at(j))) continue;
          if (!int_ok(re.int_guard, s.ints) || !gen_ok(net, re, s.ints.data(), s.ints.size())) continue;
          k.locs.assign(s.locs.begin(), s.locs.end());
          k.locs[i] = se.tgt;
          k.locs[j] = re.tgt;
          k.ints.assign(s.ints.begin(), s.ints.end());
          for (const auto& a : se.int_updates) k.ints[a.var] = a.val;
          for (const auto& a : re.int_updates) k.ints[a.var] = a.val;
          if (!gen_apply(net, se, k.ints.data(), k.ints.size()) ||
              !gen_apply(net, re, k.ints.data(), k.ints.size())) continue;
          k.guards(se.clock_guard);
          k.guards(re.clock_guard);
          k.resets(se.clock_resets);
          k.resets(re.clock_resets);
          k.close();
        }
      }
    }
  }

  // ---- broadcast synchronisation: every receiver participates or, where legal, abstains ----
  // As in gen_succ_reuse, a broadcast applies the integer assignments but not gen_update.
  std::vector<std::pair<int, std::vector<RecvOpt>>> parts;
  std::vector<std::size_t> pick;
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1 || !net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      build_broadcast_parts(net, s, se, i, parts);
      pick.assign(parts.size(), 0);
      for (;;) {
        bool part_committed = committed_at(i);
        for (std::size_t t = 0; t < parts.size(); ++t)
          if (parts[t].second[pick[t]].edge >= 0 && committed_at(parts[t].first)) part_committed = true;
        if (!src_committed || part_committed) {
          k.locs.assign(s.locs.begin(), s.locs.end());
          k.ints.assign(s.ints.begin(), s.ints.end());
          k.locs[i] = se.tgt;
          for (const auto& a : se.int_updates) k.ints[a.var] = a.val;
          k.guards(se.clock_guard);
          k.resets(se.clock_resets);
          for (std::size_t t = 0; t < parts.size(); ++t) {
            const RecvOpt& o = parts[t].second[pick[t]];
            if (o.edge < 0) { k.guards(o.extra); continue; }   // abstains: only its condition
            const NEdge& re = net.procs[parts[t].first].edges[o.edge];
            k.guards(re.clock_guard);
            k.resets(re.clock_resets);
            k.locs[parts[t].first] = re.tgt;
            for (const auto& a : re.int_updates) k.ints[a.var] = a.val;
          }
          k.close();
        }
        if (parts.empty()) break;
        std::size_t t = 0;
        for (; t < parts.size(); ++t) { if (++pick[t] < parts[t].second.size()) break; pick[t] = 0; }
        if (t == parts.size()) break;
      }
    }
  }
}

}  // namespace detail
}  // namespace hora
