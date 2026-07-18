// dbm_pool.hpp — per-thread free-list allocator for DBM zone storage.
//
// Every symbolic state owns a zone (dim*dim raw_t entries), and reachability
// churns through millions of short-lived zones (successor copies that get
// subsumed or merged away). With the default std::allocator each churned zone
// is a malloc/free pair, and malloc holds a PROCESS-GLOBAL lock — so under the
// OpenMP network backend all worker threads serialize on the allocator, which
// is exactly what caps parallel scaling at large N (the speedup collapses from
// ~3.6x to ~1.8x as the frontier grows).
//
// This allocator gives each thread its own free-list, so recycling a zone is a
// lock-free vector pop/push. All zones in a run share one byte size, so the hot
// path is a single free-list ("primary"); the rare odd size (e.g. the 1-entry
// placeholder Dbm) falls back to a small map. Buffers may be allocated on one
// thread and freed on another (successors are generated then merged on possibly
// different workers) — that is harmless: every buffer is a plain std::malloc
// block owned by exactly one free-list at a time, freed exactly once at thread
// exit. A per-bucket cap bounds pool memory so a huge level's discarded zones
// can't be retained forever.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>
#include <unordered_map>
#include <vector>

namespace ptctlmc {
namespace detail {

struct ThreadDbmPool {
  static constexpr std::size_t kMaxPerBucket = 8192;  // ~ one big level's worth
  static constexpr int kFast = 4;  // distinct hot sizes: zone, per-state locs, ints (+spare)
  std::size_t sizes[kFast] = {0, 0, 0, 0};
  std::vector<void*> fast[kFast];
  std::unordered_map<std::size_t, std::vector<void*>> other;

  ~ThreadDbmPool() {
    for (int i = 0; i < kFast; ++i)
      for (void* p : fast[i]) std::free(p);
    for (auto& kv : other)
      for (void* p : kv.second) std::free(p);
  }

  // A handful of byte sizes recur (all zones share one; locs/ints each another),
  // so a tiny linear-probed table beats hashing every alloc/free. Rare sizes spill
  // to the map.
  std::vector<void*>* bucket_for(std::size_t bytes) {
    for (int i = 0; i < kFast; ++i)
      if (sizes[i] == bytes) return &fast[i];
    for (int i = 0; i < kFast; ++i)
      if (sizes[i] == 0) {
        sizes[i] = bytes;
        return &fast[i];
      }
    return &other[bytes];
  }
};

inline ThreadDbmPool& tls_dbm_pool() {
  static thread_local ThreadDbmPool pool;
  return pool;
}

inline void* pool_alloc(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  auto* b = tls_dbm_pool().bucket_for(bytes);
  if (!b->empty()) {
    void* p = b->back();
    b->pop_back();
    return p;
  }
  void* p = std::malloc(bytes);
  if (!p) throw std::bad_alloc();
  return p;
}

inline void pool_free(void* p, std::size_t bytes) noexcept {
  if (!p) return;
  auto* b = tls_dbm_pool().bucket_for(bytes);
  if (b->size() < ThreadDbmPool::kMaxPerBucket)
    b->push_back(p);
  else
    std::free(p);
}

}  // namespace detail

// Stateless, per-thread pooled allocator. Stateless (all instances compare
// equal) so std::vector may freely move buffers between vectors — the pool is
// keyed by the running thread, not by the allocator instance.
template <class T>
struct PoolAllocator {
  using value_type = T;
  PoolAllocator() noexcept = default;
  template <class U>
  PoolAllocator(const PoolAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    return static_cast<T*>(detail::pool_alloc(n * sizeof(T)));
  }
  void deallocate(T* p, std::size_t n) noexcept {
    detail::pool_free(p, n * sizeof(T));
  }

  template <class U>
  bool operator==(const PoolAllocator<U>&) const noexcept {
    return true;
  }
  template <class U>
  bool operator!=(const PoolAllocator<U>&) const noexcept {
    return false;
  }
};

}  // namespace ptctlmc
