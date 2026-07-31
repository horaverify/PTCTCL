/// @file engine_config.hpp
/// @brief One place for every knob the reachability engine reads.
///
/// Two kinds of setting live here and they behave differently, so they are kept visibly apart:
///
///   * **Programmatic options** (EngineOptions) — set by the CLI before a run via the setters
///     declared in `netreach.hpp`. Mutable, and read on the hot path only through the accessor.
///   * **Environment knobs** (`PTCTLMC_*`) — read ONCE, on first use, into a function-local
///     static. They are experiment/ablation controls, not API, and are deliberately immutable
///     for the life of the process so a run cannot change policy halfway through and produce a
///     table nobody can reproduce.
///
/// These used to be a dozen file-scope `static` globals and `static` member functions scattered
/// through `netreach.cpp`, which meant the engine's policy could not be read, tested or
/// documented without reading the whole 4000-line file.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace ptctlmc {
namespace detail {

// ---------------------------------------------------------------------------- engine selection

/// @brief Which convex-union implementation (and mirror policy) a run uses.
///
/// `PTCTLMC_ENGINE=legacy` selects the engine of the 2026-07-29 evaluation: the pre-`10583cf`
/// convex union (::dbm_try_union_legacy) with that era's fixed `hotlen=64` mirror. Anything else
/// (including unset) selects the current engine with adaptive mirror promotion.
///
/// Both engines perform convex merging; only the implementation differs. `10583cf` added a
/// per-call `mindbm_analyze` that prunes facets already implied by another path — a win at large
/// `dim`, a loss at the dimensions the process-network models use.
///
/// @return true if the legacy engine was requested.
bool engine_legacy();

// ------------------------------------------------------------------------------ mirror policy

/// @brief How a chain earns a dense mirror.
enum class MirrorMode {
  Off,       ///< never mirror. ABLATION ONLY — this is not a historical mode, see engine_legacy()
  Fixed,     ///< promote at a fixed chain length (::hot_len_fixed)
  Adaptive   ///< promote on accumulated work, visits * length (::hot_work)
};

/// @brief Selected mirror policy (`PTCTLMC_MIRROR`, default Adaptive; legacy forces Fixed).
MirrorMode mirror_mode();

/// @brief Work threshold for adaptive promotion: promote once `visits * length` reaches this.
///
/// `PTCTLMC_HOTWORK`, default 16; 0 disables promotion entirely. Generalises the old length-only
/// rule — a length-16 chain promotes on its first visit, a length-8 chain on its second.
std::size_t hot_work();

/// @brief Minimum chain length below which a chain is never mirrored, however hot.
///
/// `PTCTLMC_HOTMINLEN`, default 8. A gate is required, not optional: mirroring a length-1 chain
/// stores a dense PRIVATE copy of a zone the interner was SHARING, so it defeats interning and
/// inflates the map every insert probes. Measured with the gate at 4, twolock12 went
/// 0.76 -> 1.21 GB and +3% time; 8 is the smallest value that leaves every process-network
/// family byte-identical to the un-mirrored run.
std::size_t hot_min_len();

/// @brief Fixed-length promotion threshold, or 0 when the adaptive rule is in force.
///
/// `PTCTLMC_HOTLEN` pins length-only promotion (the pre-adaptive rule, default 64).
std::size_t hot_len_fixed();

/// @brief Byte ceiling on all dense mirrors, process-wide.
///
/// `PTCTLMC_HOTBUDGET` in MB; default a quarter of physical RAM clamped to [1 GB, 32 GB].
/// Admission control is sound because a mirror is a PURE CACHE — the packed records remain the
/// storage of record, so refusing to promote can never change a result.
std::size_t hot_budget();

/// @brief Live mirror bytes, summed across every stripe. Reset at the start of each run.
std::atomic<std::size_t>& hot_bytes();

// -------------------------------------------------------------------------------- merge window

/// @brief Newest live zones a new zone attempts to convex-merge with.
///
/// `PTCTLMC_MERGESCAN`, default 32; 0 restores the unbounded whole-chain scan. This knob
/// CHANGES WHAT IS STORED, so it is a precision/time trade, not a pure optimisation. 32 is a
/// measured optimum in both directions: a window of 8 stores 1.65x more zones and runs 3.2x
/// slower (a narrow window misses the merges that collapse the antichain), while unbounded pays
/// 28x the try_union calls and, because merging is non-confluent, explores MORE states.
std::size_t merge_scan();

// --------------------------------------------------------------------- programmatic options

/// @brief Options the CLI sets before a run. Plain data; see `netreach.hpp` for the setters.
struct EngineOptions {
  bool adaptive_merge = false;   ///< widen when antichains fill up
  bool gpu_intern = false;       ///< intern zones on the device
  bool chunking = true;          ///< expand a BFS level in batches sized to a byte budget
  int chunk_states = 0;          ///< 0 = derive the batch size from the budget
  std::size_t chunk_budget = 0;  ///< GPU pool bytes per batch; 0 = auto
  bool symmetry = false;         ///< process-symmetry reduction (changes WHICH states are stored)
};

/// @brief The process-wide options block.
EngineOptions& options();

/// @brief GPU pool bytes allowed for one batch, resolving the `auto` default.
std::size_t chunk_budget_bytes();

}  // namespace detail
}  // namespace ptctlmc
