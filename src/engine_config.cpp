/// @file engine_config.cpp
/// @brief Definitions for the engine's configuration surface (see engine_config.hpp).
///
/// Every environment knob resolves through a function-local static, so it is read once on first
/// use and is thread-safe to initialise (C++11 magic statics). None of these appear on a hot
/// path more than as a load of an already-initialised value.
#include "ptctlmc/detail/engine_config.hpp"

#include "ptctlmc/netreach.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#ifdef __linux__
#include <unistd.h>
#endif

namespace ptctlmc {
namespace detail {
namespace {

/// @brief Read a positive integer knob, or @p dflt when unset/invalid.
/// @param name  environment variable
/// @param dflt  value when the variable is absent
/// @param zero  value to return when the variable is present and <= 0
std::size_t env_size(const char* name, std::size_t dflt, std::size_t zero) {
  const char* e = std::getenv(name);
  if (!e) return dflt;
  const long n = std::atol(e);
  return n <= 0 ? zero : static_cast<std::size_t>(n);
}

}  // namespace

bool engine_legacy() {
  static const bool v = [] {
    const char* e = std::getenv("PTCTLMC_ENGINE");
    return e && !std::strcmp(e, "legacy");
  }();
  return v;
}

}  // namespace detail

void set_narrow_ranges(bool on) { detail::options().narrow_ranges = on; }
bool narrow_ranges_on() { return detail::options().narrow_ranges; }
void set_slice(bool on) { detail::options().slice = on; }
bool slice_on() { return detail::options().slice; }
void set_gsim(bool on) { detail::options().gsim = on; }
bool gsim_on() { return detail::options().gsim; }

namespace detail {

MirrorMode mirror_mode() {
  static const MirrorMode m = [] {
    if (engine_legacy()) return MirrorMode::Fixed;   // hl=64, the policy of that era
    const char* e = std::getenv("PTCTLMC_MIRROR");
    if (!e) return MirrorMode::Adaptive;
    if (!std::strcmp(e, "off")) return MirrorMode::Off;
    if (!std::strcmp(e, "fixed")) return MirrorMode::Fixed;
    return MirrorMode::Adaptive;
  }();
  return m;
}

std::size_t hot_work() {
  // 16 with a min length of 8: a length-8 chain promotes on its 2nd visit, a length-16 on its
  // first. Measured on phil13 (t32 c4): 8.56 s against 9.12 s for a fixed hl=16 and 13.34 s for
  // the un-mirrored run, with fischer/twolock memory unchanged.
  static const std::size_t v = env_size("PTCTLMC_HOTWORK", 16, static_cast<std::size_t>(-1));
  return v;
}

std::size_t hot_min_len() {
  static const std::size_t v = env_size("PTCTLMC_HOTMINLEN", 8, 8);
  return v;
}

std::size_t hot_len_fixed() {
  static const std::size_t v = [] {
    if (const char* e = std::getenv("PTCTLMC_HOTLEN")) {
      const long n = std::atol(e);
      return n <= 0 ? static_cast<std::size_t>(-1) : static_cast<std::size_t>(n);
    }
    // Fixed mode with no explicit length means the pre-adaptive default.
    if (mirror_mode() == MirrorMode::Fixed) return std::size_t(64);
    return std::size_t(0);                       // 0 => not pinned, use the adaptive rule
  }();
  return v;
}

std::size_t hot_budget() {
  static const std::size_t v = [] {
    if (const char* e = std::getenv("PTCTLMC_HOTBUDGET")) {
      const long n = std::atol(e);
      if (n >= 0) return static_cast<std::size_t>(n) << 20;
    }
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

std::size_t merge_scan() {
  static const std::size_t v = env_size("PTCTLMC_MERGESCAN", 32, static_cast<std::size_t>(-1));
  return v;
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
bool chunking() { return detail::options().chunking; }
void set_symmetry(bool on) { detail::options().symmetry = on; }
bool symmetry_on() { return detail::options().symmetry; }

}  // namespace ptctlmc
