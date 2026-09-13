/// @file engine_config.cpp
/// @brief Definitions for the engine's configuration surface (see engine_config.hpp).
#include "hora/detail/engine_config.hpp"

#include "hora/config.hpp"
#include <string>

#include <algorithm>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#ifdef __linux__
#include <unistd.h>
#endif

namespace hora {

void set_narrow_ranges(bool on) { detail::options().narrow_ranges = on; }
bool narrow_ranges_on() { return detail::options().narrow_ranges; }
void set_slice(bool on) { detail::options().slice = on; }
bool slice_on() { return detail::options().slice; }
void set_gsim(bool on) { detail::options().gsim = on; }
bool set_criterion(const std::string& mode) {
  using C = detail::EngineOptions::Criterion;
  if (mode == "poly") detail::options().criterion = C::Poly;
  else if (mode == "guided") detail::options().criterion = C::Guided;
  else if (mode == "exact") detail::options().criterion = C::Exact;
  else return false;
  return true;
}
detail::EngineOptions::Criterion criterion_mode() { return detail::options().criterion; }
bool gsim_on() { return detail::options().gsim; }
void set_deterministic(bool on) { detail::options().deterministic = on; }
bool deterministic_on() { return detail::options().deterministic; }
void set_no_probe(bool on) { detail::options().no_probe = on; }
bool no_probe_on() { return detail::options().no_probe; }
void set_eg_dfs(bool on) { detail::options().eg_dfs = on; }
bool eg_dfs_on() { return detail::options().eg_dfs; }
void set_packed_zones(bool on) { detail::options().packed_zones = on; }
bool packed_zones_on() { return detail::options().packed_zones; }

namespace detail {

std::size_t hot_budget() {
  static const std::size_t v = [] {
    std::size_t phys = 0;
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pg = sysconf(_SC_PHYS_PAGES), ps = sysconf(_SC_PAGESIZE);
    if (pg > 0 && ps > 0) phys = static_cast<std::size_t>(pg) * static_cast<std::size_t>(ps);
#endif
    const std::size_t b = phys ? phys / 4 : (std::size_t(4) << 30);
    const std::size_t lo = std::size_t(1) << 30, hi = std::size_t(32) << 30;
    return std::min(std::max(b, lo), hi);
  }();
  return v;
}

std::atomic<std::size_t>& hot_bytes() {
  static std::atomic<std::size_t> b{0};
  return b;
}

EngineOptions& options() {
  static EngineOptions o;
  return o;
}

std::size_t chunk_budget_bytes() {
  const EngineOptions& o = options();
  if (o.chunk_budget) return o.chunk_budget;
  std::size_t ram = 0;
#ifdef __APPLE__
  std::size_t len = sizeof(ram);
  if (sysctlbyname("hw.memsize", &ram, &len, nullptr, 0) != 0) ram = 0;
#endif
#ifdef __linux__
  const long pg = sysconf(_SC_PHYS_PAGES), ps = sysconf(_SC_PAGESIZE);
  if (pg > 0 && ps > 0) ram = static_cast<std::size_t>(pg) * static_cast<std::size_t>(ps);
#endif
  if (!ram) ram = std::size_t(16) << 30;
  return std::min<std::size_t>(ram / 16, std::size_t(1) << 30);
}

}  // namespace detail

// ------------------------------------------------------- public API declared in netreach.hpp
void set_adaptive_merge(bool on) { detail::options().adaptive_merge = on; }
bool adaptive_merge() { return detail::options().adaptive_merge; }
void set_gpu_intern(bool on) { detail::options().gpu_intern = on; }
void set_chunking(bool on, int states) {
  detail::options().chunking = on;
  detail::options().chunk_states = states > 0 ? states : 0;
}
void set_chunk_budget(std::size_t bytes) { detail::options().chunk_budget = bytes; }
void set_memo(bool on) { detail::options().memo = on; }
bool memo_on() { return detail::options().memo; }
bool chunking() { return detail::options().chunking; }
void set_symmetry(bool on) { detail::options().symmetry = on; }
bool symmetry_on() { return detail::options().symmetry; }

}  // namespace hora
