// dbm.hpp — Difference Bound Matrix (clock zone), a thin host wrapper over the
// shared __host__ __device__ zone operations in dbm_ops.hpp. The CPU unit tests
// exercise these ops, and the CUDA backend calls the exact same functions on
// device memory — so the GPU runs validated code.
#pragma once

#include <vector>

#include "ptctlmc/dbm_ops.hpp"
#include "ptctlmc/dbm_pool.hpp"

namespace ptctlmc {

class Dbm {
 public:
  // Zone storage draws from a per-thread free-list (see dbm_pool.hpp): zones are
  // churned by the millions and malloc's global lock otherwise throttles the
  // parallel network backend. std::vector still owns/copies/moves the buffer.
  using Storage = std::vector<raw_t, PoolAllocator<raw_t>>;

  Dbm() : n_(0), dim_(1), m_(1, kZeroLe) {}  // 0-clock placeholder (for containers)
  explicit Dbm(int nclocks)
      : n_(nclocks), dim_(nclocks + 1), m_(static_cast<std::size_t>(dim_) * dim_, kZeroLe) {}

  int clocks() const { return n_; }
  int dim() const { return dim_; }
  raw_t get(int i, int j) const { return m_[i * dim_ + j]; }
  void set(int i, int j, raw_t r) { m_[i * dim_ + j] = r; }
  const Storage& data() const { return m_; }
  Storage& data() { return m_; }

  void close() { dbm_close(m_.data(), dim_); }
  bool empty() const { return dbm_empty(m_.data(), dim_); }
  void delay() { dbm_delay(m_.data(), dim_); }
  void reset(int k) { dbm_reset(m_.data(), dim_, k); }
  bool constrain(int i, int j, raw_t bound) { return dbm_constrain(m_.data(), dim_, i, j, bound); }
  void intersect(const Dbm& o) { dbm_intersect(m_.data(), o.m_.data(), dim_); }
  bool includes(const Dbm& o) const { return dbm_includes(m_.data(), o.m_.data(), dim_); }
  // Is *this subsumed by o under the LU-preorder ( [*this] ⊆ a_{≼LU}([o]) )?
  bool alu_subsumed_by(const Dbm& o, const std::vector<std::int32_t>& L,
                       const std::vector<std::int32_t>& U) const {
    return dbm_is_alu_le(m_.data(), o.m_.data(), dim_, L.data(), U.data());
  }
  // If *this ∪ o is exactly convex, become their hull and return true; else unchanged.
  bool try_union(const Dbm& o) { return dbm_try_union(m_.data(), o.m_.data(), dim_); }
  void extrapolate(const std::vector<std::int32_t>& maxc) { dbm_extrapolate(m_.data(), dim_, maxc.data()); }
  void extrapolate_lu(const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U) {
    dbm_extrapolate_lu(m_.data(), dim_, L.data(), U.data());
  }
  void extrapolate_lu_plus(const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U) {
    dbm_extrapolate_lu_plus(m_.data(), dim_, L.data(), U.data());
  }
  void extrapolate_lu_plus_nodiag(const std::vector<std::int32_t>& L,
                                  const std::vector<std::int32_t>& U) {
    dbm_extrapolate_lu_plus_nodiag(m_.data(), dim_, L.data(), U.data());
  }

  bool operator==(const Dbm& o) const { return m_ == o.m_; }
  bool operator!=(const Dbm& o) const { return m_ != o.m_; }

 private:
  int n_, dim_;
  Storage m_;
};

}  // namespace ptctlmc
