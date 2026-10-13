/// @file frontier_gpu.hpp
/// @brief The frontier-batched GPU reachability engine: its inputs, and the host half of
///        expansion (discrete transitions flattened into recipes).
///
/// The search is level-synchronous. Each level, every zone of the frontier is expanded along
/// every transition of its discrete state at once; the successors land in a candidate buffer,
/// are sorted by discrete key, and are reduced against the stored zones of the same key by an
/// all-pairs coverage test; the survivors join the passed set and form the next frontier.
///
/// The discrete half of a transition -- which edges are enabled, the integer update, the
/// synchronisation enumeration, committed and urgent semantics -- runs on the host, because the
/// modelling language's guards and updates are interpreted expressions. It depends only on the
/// discrete state, so it is computed once per distinct key of the frontier and shared by all of
/// that key's zones. The zone half runs on the GPU.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

#include "hora/network.hpp"
#include "hora/reach.hpp"
#include "hora/detail/sym_state.hpp"

namespace hora {
namespace detail {

/// @brief Packed discrete keys: one bit field per process, then per integer variable.
///
/// Must agree bit for bit with the reachability engine's ::DiscKeyCodecT, which owns the layout;
/// the engine fills this from its codec.
struct KeyLayout {
  std::vector<int> shift;            ///< bit offset of each field: processes first, then ints
  std::vector<std::uint64_t> mask;   ///< value mask of each field
  std::vector<long> lo;              ///< low bound of each integer variable
  int nprocs = 0;

  std::uint64_t encode(const LocVec& locs, const IntVec& ints) const {
    std::uint64_t k = 0;
    for (int p = 0; p < nprocs; ++p) k |= std::uint64_t(unsigned(locs[p])) << shift[p];
    for (std::size_t v = 0; v < ints.size(); ++v)
      k |= std::uint64_t(ints[v] - lo[v]) << shift[nprocs + v];
    return k;
  }
  void decode(std::uint64_t k, LocVec& locs, IntVec& ints) const {
    locs.resize(nprocs);
    ints.resize(lo.size());
    for (int p = 0; p < nprocs; ++p) locs[p] = int((k >> shift[p]) & mask[p]);
    for (std::size_t v = 0; v < lo.size(); ++v)
      ints[v] = long((k >> shift[nprocs + v]) & mask[nprocs + v]) + lo[v];
  }
};

/// @brief Transitions flattened for the GPU expand kernel: one *recipe* per transition.
///
/// A recipe is everything the zone half of a transition needs: the clock guards of every
/// participant (and a broadcast receiver's abstain condition), the clocks reset, the successor's
/// packed discrete key, and what the successor's discrete state alone decides -- whether time may
/// pass there, whether it meets the target's discrete part, its combined invariant, and which
/// clocks are dead there. Those last two depend only on the target locations, so the host derives
/// them once per transition instead of the GPU re-deriving them per process for every zone.
struct RecipeTable {
  static constexpr std::uint32_t kFrozen = 1;      ///< target committed or urgent: no delay
  static constexpr std::uint32_t kTargetDisc = 2;  ///< successor's discrete part meets the target

  std::vector<std::uint64_t> key;        ///< successor's packed key, integers already sliced
  std::vector<std::uint32_t> flags;      ///< kFrozen | kTargetDisc
  std::vector<std::uint32_t> g_off{0};   ///< recipe r's guards: triples g_off[r] .. g_off[r+1]
  std::vector<std::int32_t>  g;          ///< (i, j, raw bound) triples
  std::vector<std::uint32_t> r_off{0};   ///< recipe r's resets: r[r_off[r] .. r_off[r+1])
  std::vector<std::int32_t>  r;          ///< reset clocks
  std::vector<std::uint32_t> v_off{0};   ///< recipe r's target invariant: triples v_off[r] .. v_off[r+1]
  std::vector<std::int32_t>  v;          ///< (i, j, raw bound) triples, as ::all_invariants lists them
  std::vector<std::uint64_t> dead;       ///< clocks ::reset_inactive resets at the target (bit c)

  std::size_t size() const { return key.size(); }
  void clear() {
    key.clear(); flags.clear(); g.clear(); r.clear(); v.clear(); dead.clear();
    g_off.assign(1, 0); r_off.assign(1, 0); v_off.assign(1, 0);
  }
};

/// @brief Append a recipe for every transition of the discrete state (@p locs, @p ints).
///
/// Enumerates exactly what ::gen_succ_reuse does, in its order: local edges, binary handshakes,
/// broadcasts with every receiver's participate-or-abstain choice; the same committed filtering,
/// integer guards and updates. A transition whose integer update faults is dropped here.
///
/// @param canon  the engine's discrete slicing, applied to the successor's integers before the
///               key is packed and after the target and frozen tests read them, as on the CPU
/// @param query_clocks  clocks the query reads (bit c), which ::reset_inactive never resets
void append_recipes(const Network& net, const KeyLayout& kl, const NetTarget& target,
                    const std::function<void(const LocVec&, IntVec&)>& canon,
                    std::uint64_t query_clocks, const LocVec& locs, const IntVec& ints,
                    RecipeTable& out);

/// @brief Everything the frontier engine needs from the reachability engine, computed there.
struct FrontierProblem {
  const Network* net = nullptr;
  const NetTarget* target = nullptr;
  KeyLayout keys;
  int key_bits = 0;                       ///< significant bits of any packed key
  std::vector<char> qmask;                ///< clocks the query reads
  /// Per (process, location) a_LU bounds, and the floors every state's bounds are maxed with.
  /// Mirrors the engine's per-state bounds: L(q)[c] = max(Lq[c], max_p Lloc[p][q_p][c]), and
  /// L(q)[0] = 0.
  std::vector<std::vector<std::vector<std::int32_t>>> Lloc, Uloc;
  std::vector<std::int32_t> Lq, Uq;
  /// Global bounds for Extra_LU+, which the engine switches on -- as the CPU engine does -- once
  /// the passed set holds kAdaptRatio zones per discrete state.
  std::vector<std::int32_t> Lx, Ux;
  double adapt_ratio = 4.0;
  std::function<void(const LocVec&, IntVec&)> canon;
  SymState s0;                            ///< the initial state, sliced
  ReachStats* stats = nullptr;
  const std::atomic<bool>* cancel = nullptr;
};

/// @brief Run the search on the Metal GPU.
/// @return 1 when a target state is reachable, 0 when none is (or the search was cancelled),
///         -1 when the GPU path declines -- no device, or a model outside its scope -- and the
///         caller should run the CPU engine instead
int frontier_reach_metal(const FrontierProblem& p);

}  // namespace detail
}  // namespace hora
