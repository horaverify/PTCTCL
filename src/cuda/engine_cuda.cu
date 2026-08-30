// engine_cuda.cu — GPU zone-graph reachability.
//
// The GPU parallelizes the expensive part of zone exploration: computing the
// successor zone for every (frontier-state, edge) pair. Each thread runs the full
// DBM pipeline (guard ∩, reset, target-invariant ∩, delay, extrapolate) using the
// SAME __host__ __device__ ops the CPU is validated on (dbm_ops.hpp). The host does
// the serial passed-list subsumption merge (the hard-to-parallelize inclusion step),
// mirroring the OpenMP engine.
//
// Compiles for sm_90 (H100). Correctness rests on the shared DBM ops (CPU-tested)
// and the CPU engine's validated control flow.
#ifdef HORA_HAVE_CUDA

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "hora/zone.hpp"
#include "hora/cuda_reach.hpp"

namespace hora {
namespace {

constexpr int kMaxDim = 24;  // up to 23 clocks

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess)                                                      \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_)); \
  } while (0)

// Flattened, device-resident timed automaton (edge-indexed CSR-style constraint lists).
struct DeviceTA {
  int dim = 0;
  int nedges = 0;
  int* edge_tgt = nullptr;      // [nedges]
  int* edge_urgent = nullptr;   // [nedges] 1 if target urgent/committed
  int* guard_off = nullptr;     // [nedges+1]
  int *gi = nullptr, *gj = nullptr; raw_t* gb = nullptr;
  int* reset_off = nullptr;     // [nedges+1]
  int* rclk = nullptr;
  int* inv_off = nullptr;       // [nedges+1] target-invariant per edge
  int *ii = nullptr, *ij = nullptr; raw_t* ib = nullptr;
  std::int32_t* L = nullptr;    // [dim] Extra_LU+ lower bounds (matches CPU engine)
  std::int32_t* U = nullptr;    // [dim] Extra_LU+ upper bounds
};

template <class T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, (v.empty() ? 1 : v.size()) * sizeof(T)));
  if (!v.empty()) CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

// One thread per (frontier-state, edge) work item: emit the successor zone.
__global__ void gen_kernel(const raw_t* frontier, int dim, const int* work_state,
                           const int* work_edge, int nwork, DeviceTA ta, raw_t* out_zones,
                           int* out_loc, int* out_valid) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwork) return;
  const int si = work_state[w];
  const int ei = work_edge[w];
  const int d2 = dim * dim;

  raw_t z[kMaxDim * kMaxDim];
  for (int k = 0; k < d2; ++k) z[k] = frontier[si * d2 + k];

  bool ok = true;
  for (int g = ta.guard_off[ei]; ok && g < ta.guard_off[ei + 1]; ++g)
    if (!dbm_constrain(z, dim, ta.gi[g], ta.gj[g], ta.gb[g])) ok = false;
  if (ok) {
    for (int r = ta.reset_off[ei]; r < ta.reset_off[ei + 1]; ++r) dbm_reset(z, dim, ta.rclk[r]);
    for (int c = ta.inv_off[ei]; ok && c < ta.inv_off[ei + 1]; ++c)
      if (!dbm_constrain(z, dim, ta.ii[c], ta.ij[c], ta.ib[c])) ok = false;
  }
  if (ok) {
    if (!ta.edge_urgent[ei]) {
      dbm_delay(z, dim);
      for (int c = ta.inv_off[ei]; ok && c < ta.inv_off[ei + 1]; ++c)
        if (!dbm_constrain(z, dim, ta.ii[c], ta.ij[c], ta.ib[c])) ok = false;
    }
    if (ok) {
      dbm_extrapolate_lu_plus(z, dim, ta.L, ta.U);  // same abstraction as the CPU engine
      if (dbm_empty(z, dim)) ok = false;
    }
  }
  out_valid[w] = ok ? 1 : 0;
  out_loc[w] = ta.edge_tgt[ei];
  if (ok)
    for (int k = 0; k < d2; ++k) out_zones[w * d2 + k] = z[k];
}

// ---- host helpers (mirror reach.cpp) -----------------------------------------
bool apply_host(Dbm& z, const std::vector<Constraint>& cs) {
  for (const auto& c : cs)
    if (!z.constrain(c.i, c.j, c.bound)) return false;
  return true;
}
void delay_in_host(Dbm& z, const Location& l) {
  if (l.urgent || l.committed) return;
  z.delay();
  apply_host(z, l.invariant);
}

}  // namespace

bool reachable_cuda(const TimedAutomaton& ta, const std::vector<char>& accept,
                    const std::vector<Constraint>& tgt_clocks,
                    const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U,
                    ReachStats* stats) {
  const int dim = ta.nclocks + 1;
  if (dim > kMaxDim)
    throw std::runtime_error("reachable_cuda: too many clocks (raise kMaxDim)");
  const int d2 = dim * dim;
  // `ta` is diagonal-free (the caller applied diagonal splitting) and L/U are the
  // Extra_LU+ bounds the caller precomputed — so the GPU path handles diagonal models too.

  // ---- flatten edges to host arrays, then upload once ----
  const int E = static_cast<int>(ta.edges.size());
  std::vector<int> h_tgt(E), h_urg(E), goff(E + 1, 0), roff(E + 1, 0), ioff(E + 1, 0);
  std::vector<int> gi, gj, rclk, ii, ij;
  std::vector<raw_t> gb, ib;
  for (int e = 0; e < E; ++e) {
    const Edge& ed = ta.edges[e];
    h_tgt[e] = ed.tgt;
    const Location& tl = ta.locations[ed.tgt];
    h_urg[e] = (tl.urgent || tl.committed) ? 1 : 0;
    for (const auto& c : ed.guard) { gi.push_back(c.i); gj.push_back(c.j); gb.push_back(c.bound); }
    goff[e + 1] = static_cast<int>(gi.size());
    for (int r : ed.resets) rclk.push_back(r);
    roff[e + 1] = static_cast<int>(rclk.size());
    for (const auto& c : tl.invariant) { ii.push_back(c.i); ij.push_back(c.j); ib.push_back(c.bound); }
    ioff[e + 1] = static_cast<int>(ii.size());
  }
  DeviceTA dta;
  dta.dim = dim;
  dta.nedges = E;
  dta.edge_tgt = upload(h_tgt);
  dta.edge_urgent = upload(h_urg);
  dta.guard_off = upload(goff);
  dta.gi = upload(gi); dta.gj = upload(gj); dta.gb = upload(gb);
  dta.reset_off = upload(roff);
  dta.rclk = upload(rclk);
  dta.inv_off = upload(ioff);
  dta.ii = upload(ii); dta.ij = upload(ij); dta.ib = upload(ib);
  dta.L = upload(L); dta.U = upload(U);

  // ---- host reachability driver; GPU does per-level successor generation ----
  const int nloc = static_cast<int>(ta.locations.size());
  std::vector<std::vector<Dbm>> passed(nloc);
  auto is_target = [&](int loc, const Dbm& z) {
    if (!accept[loc]) return false;
    Dbm t = z;
    return apply_host(t, tgt_clocks);
  };
  auto subsumed = [&](int loc, const Dbm& z) {
    for (const auto& zp : passed[loc]) if (zp.includes(z)) return true;
    return false;
  };

  Dbm z0(ta.nclocks);
  if (!apply_host(z0, ta.locations[ta.initial].invariant)) return false;
  delay_in_host(z0, ta.locations[ta.initial]);
  z0.extrapolate_lu_plus(L, U);
  bool found = is_target(ta.initial, z0);

  std::vector<int> front_loc{ta.initial};
  std::vector<raw_t> front_z(z0.data().begin(), z0.data().end());
  passed[ta.initial].push_back(z0);
  if (stats) ++stats->explored;

  raw_t* d_front = nullptr; raw_t* d_out = nullptr;
  int *d_ws = nullptr, *d_we = nullptr, *d_ol = nullptr, *d_ov = nullptr;
  std::size_t cap_front = 0, cap_work = 0;

  while (!found && !front_loc.empty()) {
    // build work list (state,edge) pairs for this frontier
    std::vector<int> ws, we;
    for (int s = 0; s < static_cast<int>(front_loc.size()); ++s)
      for (int ei : ta.edges_from(front_loc[s])) { ws.push_back(s); we.push_back(ei); }
    const int nwork = static_cast<int>(ws.size());
    if (nwork == 0) break;

    if (front_loc.size() * d2 > cap_front) {
      if (d_front) cudaFree(d_front);
      cap_front = front_loc.size() * d2;
      CUDA_CHECK(cudaMalloc(&d_front, cap_front * sizeof(raw_t)));
    }
    if (static_cast<std::size_t>(nwork) > cap_work) {
      if (d_out) cudaFree(d_out);
      if (d_ws) { cudaFree(d_ws); cudaFree(d_we); cudaFree(d_ol); cudaFree(d_ov); }
      cap_work = nwork;
      CUDA_CHECK(cudaMalloc(&d_out, cap_work * d2 * sizeof(raw_t)));
      CUDA_CHECK(cudaMalloc(&d_ws, cap_work * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&d_we, cap_work * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&d_ol, cap_work * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&d_ov, cap_work * sizeof(int)));
    }
    CUDA_CHECK(cudaMemcpy(d_front, front_z.data(), front_loc.size() * d2 * sizeof(raw_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ws, ws.data(), nwork * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_we, we.data(), nwork * sizeof(int), cudaMemcpyHostToDevice));

    const int threads = 128;
    gen_kernel<<<(nwork + threads - 1) / threads, threads>>>(d_front, dim, d_ws, d_we, nwork, dta, d_out, d_ol, d_ov);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> ol(nwork), ov(nwork);
    std::vector<raw_t> oz(static_cast<std::size_t>(nwork) * d2);
    CUDA_CHECK(cudaMemcpy(ol.data(), d_ol, nwork * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ov.data(), d_ov, nwork * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(oz.data(), d_out, static_cast<std::size_t>(nwork) * d2 * sizeof(raw_t), cudaMemcpyDeviceToHost));

    // serial subsumption merge -> next frontier
    std::vector<int> next_loc;
    std::vector<raw_t> next_z;
    for (int w = 0; w < nwork && !found; ++w) {
      if (!ov[w]) continue;
      if (stats) ++stats->generated;
      Dbm z(ta.nclocks);
      for (int k = 0; k < d2; ++k) z.data()[k] = oz[static_cast<std::size_t>(w) * d2 + k];
      if (is_target(ol[w], z)) { found = true; break; }
      if (subsumed(ol[w], z)) continue;
      passed[ol[w]].push_back(z);
      if (stats) ++stats->explored;
      next_loc.push_back(ol[w]);
      next_z.insert(next_z.end(), z.data().begin(), z.data().end());
    }
    front_loc.swap(next_loc);
    front_z.swap(next_z);
  }

  // ---- free device memory ----
  for (void* p : {(void*)dta.edge_tgt, (void*)dta.edge_urgent, (void*)dta.guard_off, (void*)dta.gi,
                  (void*)dta.gj, (void*)dta.gb, (void*)dta.reset_off, (void*)dta.rclk, (void*)dta.inv_off,
                  (void*)dta.ii, (void*)dta.ij, (void*)dta.ib, (void*)dta.L, (void*)dta.U, (void*)d_front, (void*)d_out,
                  (void*)d_ws, (void*)d_we, (void*)d_ol, (void*)d_ov})
    if (p) cudaFree(p);
  return found;
}

}  // namespace hora

#endif  // HORA_HAVE_CUDA
