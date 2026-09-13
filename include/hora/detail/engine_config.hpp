/// @file engine_config.hpp
/// @brief One place for every knob the reachability engine reads.
///
/// Two kinds of setting live here and they are kept visibly apart:
///
///   * **Programmatic options** (EngineOptions) — set by the CLI before a run via the setters
///     declared in `netreach.hpp`. Mutable, and read on the hot path only through the accessor.
///   * **Tuning constants** — fixed policy, with the measurement that fixed them recorded on
///     each. They are not configurable: a run cannot change policy halfway through and produce
///     a table nobody can reproduce.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace hora {
namespace detail {

// ------------------------------------------------------------------------------ mirror policy

/// @brief Work threshold for a dense mirror: a chain is promoted once `visits * length` reaches
/// this. Generalises the old length-only rule -- a length-16 chain promotes on its first visit,
/// a length-8 chain on its second.
/// fixed length-16 rule and 13.34 s un-mirrored, with fischer/twolock memory unchanged.
constexpr std::size_t kHotWork = 16;

/// @brief Minimum chain length below which a chain is never mirrored, however hot.
///
/// A gate is required, not optional: mirroring a length-1 chain stores a dense private copy of
/// a zone the interner was sharing, so it defeats interning and inflates the map every insert
/// probes. A smaller gate costs resident memory for no time; 8 is the
/// smallest value that leaves every process-network family byte-identical to the un-mirrored run.
constexpr std::size_t kHotMinLen = 8;

/// @brief Byte ceiling on all dense mirrors, process-wide: a quarter of physical RAM clamped to
/// [1 GB, 32 GB]. Admission control is sound because a mirror is a pure cache -- the packed
/// records remain the storage of record, so refusing to promote can never change a result.
std::size_t hot_budget();

/// @brief Live mirror bytes, summed across every stripe. Reset at the start of each run.
std::atomic<std::size_t>& hot_bytes();

// -------------------------------------------------------------------------------- merge window

/// @brief Newest live zones a new zone attempts to convex-merge with.
///
/// This changes what IS stored, so it is a precision/time trade, not a pure optimisation. 32 is
/// a measured optimum in both directions: a window of 8 stores 1.65x more zones and runs 3.2x
/// slower (a narrow window misses the merges that collapse the antichain), while unbounded pays
/// 28x the try_union calls and, because merging is non-confluent, explores more states.
constexpr std::size_t kMergeScan = 32;

// --------------------------------------------------------------------- programmatic options

/// @brief Options the CLI sets before a run. Plain data; see `netreach.hpp` for the setters.
struct EngineOptions {
  bool adaptive_merge = false;   ///< widen when antichains fill up
  bool gpu_intern = false;       ///< intern zones on the device
  bool chunking = true;          ///< expand a BFS level in batches sized to a byte budget
  int chunk_states = 0;          ///< 0 = derive the batch size from the budget
  std::size_t chunk_budget = 0;  ///< GPU pool bytes per batch; 0 = auto
  bool symmetry = false;         ///< process-symmetry reduction (changes WHICH states are stored)
  bool narrow_ranges = false;    ///< pack dk from inferred integer ranges, not declared ones
  bool slice = false;            ///< canonicalise dead integer cells before the discrete key
  bool gsim = false;             ///< subsume with the G-simulation instead of a_LU
  bool packed_zones = false;     ///< store liveness zones as minimal graphs, not dense matrices
  bool memo = false;             ///< memoise the successor-zone pipeline (see gen_succ_reuse)
  bool deterministic = false;    ///< insert each level in canonical order; see reach_engine
  bool no_probe = false;         ///< skip the shared-lock subsumption pre-filter (ablation)
  /// @brief Which recurrence criterion the forward liveness route uses; see `--criterion`.
  ///
  /// A key-SCC owes refinement obligations only when it can host an infinite psi-run.  Deciding
  /// that exactly is exponential, so the default is a pair of polynomial tests that are
  /// sufficient but not complete; the other two settle the components those leave live.
  enum class Criterion {
    Poly,    ///< polynomial tests only: every cycle resets something, and resets imply guards
    Guided,  ///< ... then branch on a clock that actually conflicts in an SCC
    Exact,   ///< ... then enumerate the choice functions, pruning on acyclicity
  };
  Criterion criterion = Criterion::Poly;
  /// @brief Expand the `E[] psi` online construction depth-first rather than breadth-first.
  ///
  /// Only the order the worklist is popped in; the graph, the abstraction and the verdict are
  /// unchanged. It matters because the two constant-time witness tests in
  /// ::hora::LivenessEngine::online_eg_terminal_witness recognise a DIVERGENT or a TERMINAL state
  /// and nothing else, so a witness that is a CYCLE stops the construction only once the cycle is
  /// closed among expanded records. Breadth-first closes it last, after the whole breadth;
  /// depth-first closes it after one path. On `single7` `E[] x0 <= 6` that is 40197 records and
  /// 172 s against 122 records and 0.008 s, for the same SAT.
  bool eg_dfs = false;
};

/// @brief The process-wide options block.
EngineOptions& options();

/// @brief GPU pool bytes allowed for one batch, resolving the `auto` default.
std::size_t chunk_budget_bytes();

}  // namespace detail
}  // namespace hora
