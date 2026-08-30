/// @file src/liveness/internal.hpp
/// @brief Declarations shared between the liveness engine's translation units.
///
/// The liveness engine is split by concern -- the symbolic operators, the quotient construction,
/// the D refinement, the backward fixpoint, and the query drivers each have their own file. This
/// header carries the few names more than one of them needs, and nothing else: it is private to
/// src/liveness and is not installed.
///
/// @see hora/detail/liveness_engine.hpp for the engine's own state and contracts.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hora/zone.hpp"
#include "hora/detail/liveness_engine.hpp"
#include "hora/detail/semantics.hpp"

namespace hora {

using detail::SymState;
using detail::LocVec;
using detail::IntVec;
using detail::ExNone;
using detail::ExLuPlus;
using detail::ExNoDiag;
using detail::apply_all;
using detail::int_ok;
using detail::all_invariants;
using detail::frozen_flags;
using detail::reset_inactive;
using detail::close_succ_zone;
using detail::disc_key_bytes;
using detail::RecvOpt;
using detail::build_broadcast_parts;
using detail::state_has_deadlock;

// Temporary profiling is intentionally opt-in: it is useful for separating successor
// generation from quotient maintenance without perturbing normal benchmark runs.
struct OnlinePerf {
  bool enabled = std::getenv("HORA_ONLINE_PROFILE") != nullptr;
  std::uint64_t cover_calls = 0;
  std::uint64_t cover_ns = 0;
  std::uint64_t intern_calls = 0;
  std::uint64_t intern_active_scan = 0;
  std::uint64_t intern_cover_probe_ns = 0;
  std::uint64_t intern_join_ns = 0;
  std::uint64_t intern_append_ns = 0;
  std::uint64_t deferred_created = 0;
  std::uint64_t deferred_replayed = 0;
  std::uint64_t deferred_promoted = 0;
  std::uint64_t coverer_calls = 0;
  std::uint64_t coverer_active_scan = 0;
  std::uint64_t compact_calls = 0;
  std::uint64_t compact_active_scan = 0;
  std::uint64_t compact_deact = 0;
  std::uint64_t compact_deact_unexp = 0;
  std::uint64_t compact_deact_permanent = 0;
  std::uint64_t recheck_calls = 0;
  std::uint64_t recheck_nodes = 0;
  std::uint64_t successor_ns = 0;
  std::uint64_t register_ns = 0;
  std::uint64_t register_key_ns = 0;
  std::uint64_t register_tag_ns = 0;
  std::uint64_t register_reset_ns = 0;
  std::uint64_t register_edge_ns = 0;
  std::uint64_t intern_ns = 0;
};

/// @brief The counters, shared by the construction and the drivers that report them.
extern OnlinePerf online_perf;

class OnlinePerfTimer {
 public:
  explicit OnlinePerfTimer(std::uint64_t* total) : total_(total), start_(std::chrono::steady_clock::now()) {}
  ~OnlinePerfTimer() {
    *total_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - start_).count();
  }

 private:
  std::uint64_t* total_;
  std::chrono::steady_clock::time_point start_;
};

/// @brief Mix a packed discrete key so it spreads across a power-of-two table.
std::uint64_t mix_packed_key(std::uint64_t x);

/// @brief Per-node data the backward predecessor needs that depends only on the discrete state.
struct PreNode {
  std::uint32_t inv_off = 0;     ///< into the caller's shared Constraint arena
  std::uint32_t inv_len = 0;
  std::uint64_t dead = 0;        ///< bit c set iff clock c is read by no process here
  bool frozen = false;           ///< committed or urgent: no positive delay
  bool ready = false;
};

/// @brief Generate the successors of @p s, each tagged with the edges that produced it.
void tagged_successors(const Network& net, const detail::SymState& s,
                       const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U,
                       const std::vector<char>& qmask, int emode,
                       std::vector<detail::SymState>& out, std::vector<GfpTag>& tags,
                       const detail::DelayAvoidSet* avoid);

/// @brief An empty zone of the network's dimension.
Dbm empty_zone(int nclocks);

/// @brief The exact concrete predecessor of one tagged transition.
Dbm exact_predecessor(const Network& net, const GfpTag& t, const PreNode& sn, const PreNode& tn,
                      const std::vector<Constraint>& invs,
                      const std::vector<unsigned char>& rmask, unsigned char* dead_scratch,
                      const Dbm& X);

/// @brief ::exact_predecessor for a run confined to a non-convex predicate.
void exact_predecessor_avoiding(const Network& net, const GfpTag& t, const detail::SymState& src,
                                const detail::SymState& tgt, const Dbm& X,
                                const detail::DelayAvoidSet& avoid, Federation& out);

/// @brief Does @p f denote the empty set?
bool fed_is_empty(const Federation& f);

/// @brief Is every point of @p a inside the union @p b?
bool fed_contains(const Federation& a, const Federation& b);

/// @brief Normalise a federation: drop empties and redundancies, merge convex pairs.
void fed_compact(Federation& f, std::size_t cap, bool* capped);

/// @brief Value equality for a transition tag, so the tag pool can deduplicate.
bool tag_equal(const GfpTag& a, const GfpTag& b);

/// @brief Hash of a transition tag's value, keyed on the same fields ::tag_equal compares.
std::uint64_t tag_hash(const GfpTag& t);

/// @brief Rebuild a discrete key's invariant zone from a representative state.
bool key_invariant(const Network& net, const detail::SymState& rep, int nc, Dbm& out);

/// @brief Hash a discrete state without allocating its byte key.
std::uint64_t disc_hash(const detail::LocVec& locs, const detail::IntVec& ints);

/// @brief Is the deadlock refinement provably empty for the whole network?
bool structurally_empty_d(const Network& net);

/// @brief Resident set in MiB for the HORA_RSS phase report; 0 when it is not asked for.
std::size_t rss_mb();

}  // namespace hora
