/// @file hora/config.hpp
/// @brief Every process-global switch the engine reads, in one place.
///
/// Each switch changes *which* symbolic states are stored — never a verdict — so they are
/// opt-in per run rather than always on. They are set once, from the command line, before any
/// query runs, and are not safe to change while a search is in flight.
///
/// @see hora::detail::EngineOptions for the storage behind them.
#pragma once

#include <string>
#include "hora/detail/engine_config.hpp"

#include <cstddef>

namespace hora {

/// @brief Explore one representative per orbit of the network's automorphism group.
///
/// @param on @c true to enable process-symmetry reduction
/// @warning Exact for the query, but it changes which states are stored, so it is opted into
///          per run rather than enabled by default.
/// @see symmetry.hpp for how the group is computed.
void set_symmetry(bool on);

/// @brief Report whether process-symmetry reduction is enabled.
/// @return the value last passed to ::set_symmetry (default @c false)
bool symmetry_on();

/// @brief Derive integer ranges from an interval analysis instead of from the declarations.
///
/// The discrete key spends `bits(hi - lo + 1)` per integer cell, taken from the declared range.
/// An unannotated `int` in the modelling language declares `[-32768, 32767]`, so a model with a
/// handful of unannotated array cells can overflow the 63-bit key on width alone, before state
/// count enters into it. With this on the range comes from an over-approximating interval
/// analysis of the values a cell can actually hold.
///
/// @param on @c true to narrow ranges before packing the discrete key
/// @see hora::narrow_int_ranges
void set_narrow_ranges(bool on);

/// @brief Report whether integer-range narrowing is enabled.
/// @return the value last passed to ::set_narrow_ranges (default @c false)
bool narrow_ranges_on();

/// @brief Canonicalise dead integer cells before computing the discrete key.
///
/// An integer cell that no process can read before overwriting, and that the query does not
/// mention, cannot influence any run from the current state, so two states differing only in it
/// have the same future. Such cells are set to their low value before the key is computed, which
/// merges those states. This is the discrete analogue of the active-clock reduction already
/// applied to zones.
///
/// @param on @c true to canonicalise dead cells
/// @warning Changes which states are stored; off by default.
/// @see hora::Network::iread for the liveness analysis this consumes.
void set_slice(bool on);

/// @brief Report whether the discrete liveness reduction is enabled.
/// @return the value last passed to ::set_slice (default @c false)
bool slice_on();

/// @brief Subsume with the diagonal-free G-simulation instead of a_LU.
///
/// The G-simulation is the a_LU test carried out on bounds that retain each guard's strictness
/// rather than only its constant. For a diagonal-free constraint map the relation collapses to
/// at most one upper and one lower bound per clock (Gastin–Mukherjee–Srivathsan, CAV 2019), so
/// the single strictness bit per bound is the entire difference from a_LU here.
///
/// @param on @c true to use the G-simulation
/// @warning Changes which states are stored; off by default.
/// @see hora::Network::lraw_loc, hora::Network::uraw_loc
void set_gsim(bool on);

/// @brief Select the recurrence criterion used by the forward liveness route.
/// @param mode "poly" (default), "guided" or "exact"
/// @return @c false if @p mode is not one of those
bool set_criterion(const std::string& mode);
/// @return the criterion last selected
detail::EngineOptions::Criterion criterion_mode();

/// @brief Store liveness zones as minimal graphs instead of dense matrices.
///
/// Far smaller resident, at one `mindbm_unpack` per read. Exact either way — the codec round
/// trips a canonical DBM bit for bit — so this only trades memory against time. Whether it pays
/// depends on the model's read rate; see hora::ZoneStore.
void set_packed_zones(bool on);

/// @brief Report whether G-simulation subsumption is enabled.
/// @return the value last passed to ::set_gsim (default @c false)
bool gsim_on();

/// @brief Insert each BFS level in a canonical order, making state counts thread-invariant.
///
/// The convex-union merge is order-sensitive, and by default successors are inserted as workers
/// produce them, so the interleaving decides which merges happen. On most families that is
/// invisible (arbiter, fischer, twolock and viking are bit-identical at any thread count), but
/// where every state shares one discrete key it dominates: `single7` explores 3,494 states at one
/// thread and ~39,800 at eight. With this on, successors are buffered with the
/// (frontier index, successor index) that produced them and inserted in that order, which is
/// exactly the sequential order -- so the count equals the one-thread count at any thread count.
/// @note Costs a level's worth of buffered successors and serialises insertion, so it is off by
///       default: it buys reproducibility, not speed.
/// @brief Explicitly select the `E[] psi` expansion order: @c true for depth-first, @c false for
///        breadth-first.
///
/// Without this call, the engine uses its automatic policy: it selects depth-first only when the
/// model has exactly one possible discrete key, and otherwise preserves breadth-first expansion.
/// Order only -- it changes cost, never the verdict.
/// @param on @c true to expand depth-first
void set_eg_dfs(bool on);

/// @return @c true iff depth-first was explicitly selected with ::set_eg_dfs
bool eg_dfs_on();

/// @return @c true iff no explicit `E[]` worklist order was selected
bool eg_dfs_auto_on();

void set_deterministic(bool on);
/// @brief Skip the shared-lock subsumption pre-filter before the exclusive insert.
///
/// The parallel insert probes a stripe under a SHARED lock first and only takes it exclusively
/// when the successor survives; the probe is a hint, since ::NetPassed::add re-checks under the
/// exclusive lock. This turns the probe off, which exists to make that design choice measurable
/// rather than assumed. Measured at 8 threads: the probe's hit rate is bimodal (2.7% on fischer11
/// and 3.4% on twolock11, against 77% on arbiter14 and 83% on phil12), and turning it off changes
/// wall time by -1%/-2% on the first pair but +29%/+20% on the second. It costs almost nothing
/// when it misses -- an uncontended shared lock over a chain that is short BECAUSE subsumption is
/// not hitting -- so it is on by default and there is no adaptive version worth building.
void set_no_probe(bool on);
/// @return the value last passed to ::set_no_probe (default @c false)
bool no_probe_on();
/// @return the value last passed to ::set_deterministic (default @c false)
bool deterministic_on();

/// @brief Is minimal-graph zone storage in force? @see set_packed_zones
bool packed_zones_on();

/// @brief Memoise the successor-zone pipeline.
///
/// A successor zone is a function of (source zone, edge, target-location data), and on models
/// with few distinct zones the same O(dim^3) computation recurs enormously often. The cache is
/// adaptive: a sliding window disables it when the hit rate is low, so it costs nothing where it
/// does not pay.
///
/// @param on @c true to enable the cache
/// @note A pure cache: it changes neither the stored states nor the verdict.
void set_memo(bool on);

/// @brief Report whether successor-zone memoisation is enabled.
/// @return the value last passed to ::set_memo (default @c false)
bool memo_on();

// ---- general modelling-language guard/update -------------------------------

}  // namespace hora
