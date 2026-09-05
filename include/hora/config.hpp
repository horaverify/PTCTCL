/// @file hora/config.hpp
/// @brief Every process-global switch the engine reads, in one place.
///
/// Each switch changes *which* symbolic states are stored — never a verdict — so they are
/// opt-in per run rather than always on. They are set once, from the command line, before any
/// query runs, and are not safe to change while a search is in flight.
///
/// @see hora::detail::EngineOptions for the storage behind them.
#pragma once

#include <cstddef>

namespace hora {

/// @brief Select how the GPU backends schedule the passed-list merge.
///
/// In the default *naive* mode the merge always runs as one GPU thread per discrete-state
/// bucket. In *adaptive* mode the flat driver calibrates GPU against multi-core CPU merge
/// throughput on the early BFS levels and routes each later level to the predicted-faster path,
/// which recovers models with few but very wide antichains, where one thread per bucket is the
/// wrong shape.
///
/// @param on @c true to enable adaptive routing, @c false for the naive schedule
/// @note No effect on the CPU-only backends (hora::Backend::Seq, hora::Backend::Omp).
/// @note Routing changes only the schedule, never the merge result, so the verdict is unaffected.
void set_adaptive_merge(bool on);

/// @brief Report whether adaptive GPU merge routing is enabled.
/// @return the value last passed to ::set_adaptive_merge (default @c false)
bool adaptive_merge();

/// @brief Deduplicate zones in the GPU passed set.
///
/// The GPU driver stores zones densely because its merge kernels consume dense matrices, which
/// costs one full matrix per (discrete state, zone). Interning collapses repeats to a single
/// stored matrix plus a 4-byte index per survivor, so the per-level rebuild copies indices
/// instead of matrices. This is a large win where few distinct zones are reachable and neutral
/// where nearly every zone is distinct, hence opt-in.
///
/// @param on @c true to intern GPU-side zones
/// @note Trades a hash lookup per zone for memory; it does not change which states are stored.
/// @see hora::detail::GpuZoneIntern
void set_gpu_intern(bool on);

/// @brief Expand each BFS level in bounded batches instead of in one shot.
///
/// With chunking off, a level is expanded whole, so the unified-memory pools must be sized to
/// the widest level and peak memory scales with level width. With it on, the pools are sized to
/// one batch and peak memory becomes independent of level width, raising the model size that
/// fits in RAM.
///
/// @param on     @c true to process the frontier in batches
/// @param states frontier states per batch; @c 0 sizes each batch from the memory budget set by
///               ::set_chunk_budget
/// @note Sound: the frontier is a worklist, so batching only reorders the work. It can shift
///       order-sensitive convex-union counts, never the verdict.
void set_chunking(bool on, int states = 0);

/// @brief Set the GPU pool budget used to size an automatic chunk.
/// @param bytes GPU pool bytes per batch; @c 0 selects the built-in default
/// @note Only consulted when chunking is on and ::set_chunking was given `states == 0`.
void set_chunk_budget(std::size_t bytes);

/// @brief Report whether frontier chunking is enabled.
/// @return the value last passed to ::set_chunking (default @c false)
bool chunking();


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
/// @brief Select the `E[] psi` expansion order: @c true for depth-first, @c false (default) for
///        breadth-first. Order only -- it changes cost, never the verdict.
/// @param on @c true to expand depth-first
void set_eg_dfs(bool on);

/// @return the value last passed to ::set_eg_dfs (default @c false)
bool eg_dfs_on();

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
