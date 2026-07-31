/// @file spin_rw.hpp
/// @brief The striped reader/writer lock the passed-set uses.
#pragma once

#include <atomic>
#include <thread>

#ifdef PTCTLMC_HAVE_OPENMP
#include <omp.h>
#endif

namespace ptctlmc {
namespace detail {

// A stripe lock that SPINS before it blocks.
//
// Contention on the stripe locks is only 2-3%, but with std::shared_mutex each collision cost
// 0.3-1.8 MILLISECONDS: the loser blocks, the kernel deschedules it, and it waits for a whole
// scheduling quantum to service a critical section that runs in microseconds. On phil13 that
// convoy consumed ~378 of 759 CPU-seconds -- literally half the machine sat blocked, which is
// why utilisation pinned at ~400% of 800%.
//
// Spinning turns a collision into a few hundred nanoseconds. The bounded spin then falls back to
// yield() so the design degrades gracefully when threads > cores, instead of burning a full
// quantum spinning against a descheduled holder. 8 bytes per stripe instead of ~56 also matters
// at 16k stripes.
//
// Two variants were measured and REMOVED, so don't reintroduce them: test-and-test-and-set with
// exponential back-off (no gain, slightly worse -- a waiter that probes less often misses the
// release window), and writer preference (it moved wait from writers to readers and total wait
// did not change, because the stripes are genuinely serialized rather than unfairly scheduled).
// The cycle-accurate spin accounting that established this is gone with them.
class SpinRW {
  std::atomic<int> st{0};                       // -1 = writer, >=0 = reader count
  static void pause_() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
  }
  // Pauses before falling back to yield(). yield() is a SYSCALL, and it is only the right move
  // when the lock holder has been descheduled -- i.e. when threads outnumber cores. At threads <=
  // cores the holder is running and will release in microseconds, so yielding costs more than the
  // wait it avoids: phil13 measured 14.5M yields, tens of CPU-seconds of pure syscall on a 239
  // CPU-second run. PTCTLMC_SPIN overrides the threshold.
  static int spin_limit() {
    static const int v = [] {
      if (const char* e = std::getenv("PTCTLMC_SPIN")) return std::max(1, std::atoi(e));
      const unsigned hw = std::thread::hardware_concurrency();
      int nt = 1;
#ifdef PTCTLMC_HAVE_OPENMP
      nt = omp_get_max_threads();
#endif
      return (hw && static_cast<unsigned>(nt) <= hw) ? 4096 : 64;
    }();
    return v;
  }
  template <class F> static void spin_until(F&& try_once) {
    if (try_once()) return;                       // fast path: one CAS, nothing else
    const int lim = spin_limit();
    for (int n = 0;; ++n) {
      if (try_once()) return;
      if (n < lim) pause_();
      else { std::this_thread::yield(); n = 0; }
    }
  }
 public:
  void lock_shared() {
    spin_until([this] {
      int v = st.load(std::memory_order_relaxed);
      return v >= 0 && st.compare_exchange_weak(v, v + 1, std::memory_order_acquire,
                                                std::memory_order_relaxed);
    });
  }
  void unlock_shared() { st.fetch_sub(1, std::memory_order_release); }
  void lock() {
    spin_until([this] {
      int v = 0;
      return st.compare_exchange_weak(v, -1, std::memory_order_acquire,
                                      std::memory_order_relaxed);
    });
  }
  void unlock() { st.store(0, std::memory_order_release); }
};

}  // namespace detail
}  // namespace ptctlmc
