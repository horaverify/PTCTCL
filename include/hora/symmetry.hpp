// symmetry.hpp — process-symmetry reduction for networks of timed automata.
//
// Many benchmark networks are built from N copies of the same template wired in a ring or a
// clique: the dining philosophers, Fischer, arbiter, CSMA. Permuting the copies maps the network
// onto itself, so the reachable state space is covered by exploring ONE representative per orbit.
// For an N-fold symmetry that is up to N times fewer states, N times less memory, and N times
// less work — a reduction no amount of parallelism can give you, and one that composes with it.
//
// This finds the symmetry itself: an `Automorphism` is a permutation of PROCESSES together with
// the induced permutations of CLOCKS and CHANNELS, and it is only accepted after being verified
// structurally against the network, so a wrong guess by the search cannot make the result wrong
// — it can only fail to find a symmetry and lose the reduction.
//
// UPPAAL offers the same reduction but only for models the author wrote with `scalar` types;
// here it is recovered from a plain flat network by structure alone.
#pragma once

#include <cstddef>
#include <vector>

#include "hora/network.hpp"

namespace hora {

/// @brief A network automorphism: process, clock, channel and integer-VALUE permutations that
///        agree.
///
/// The value permutations are what make Fischer-style models reachable. There the shared `id`
/// holds a PROCESS INDEX, so permuting processes has to permute `id`'s values in step -- a
/// network symmetric in every other respect is not an automorphism until the data moves too.
/// Values are only permuted, never reinterpreted: variables compared with `<`/`<=` are refused,
/// since an arbitrary permutation need not preserve an ordering.
struct Automorphism {
  std::vector<int> proc;   ///< proc[p] = image of process p
  std::vector<int> clk;    ///< clk[c]  = image of clock c; clk[0] == 0
  std::vector<int> chan;   ///< chan[a] = image of channel a
  /// ival[v][x - intvars[v].lo] = image of value x. Empty for a variable left pointwise fixed.
  std::vector<std::vector<long>> ival;
  bool identity = false;

  /// @brief Image of value @p x of integer variable @p v (identity when v is not permuted).
  long map_val(int v, long x, long lo) const {
    if (v >= static_cast<int>(ival.size()) || ival[v].empty()) return x;
    const std::size_t i = static_cast<std::size_t>(x - lo);
    return i < ival[v].size() ? ival[v][i] : x;
  }
};

/// @brief The symmetry group of @p net, as an explicit element list including the identity.
///
/// Returns a single (identity) element when no symmetry is found or the network is out of
/// scope. Out of scope for now: networks with integer variables, because a permutation of
/// processes generally has to permute index-VALUED data too (Fischer's `id`), which needs the
/// variable's meaning, not just the network's shape.
///
/// @param cap Maximum group size to materialise; the search stops early beyond it.
std::vector<Automorphism> symmetry_group(const Network& net, std::size_t cap = 256);

/// @brief Apply @p a to a location vector: out[a.proc[p]] = locs[p].
void permute_locs(const Automorphism& a, const int* locs, int np, int* out);

}  // namespace hora
