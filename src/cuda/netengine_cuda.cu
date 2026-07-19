// netengine_cuda.cu — CUDA GPU successor generation + subsumption merge for NETWORKS.
//
// Device-for-device peer of netengine_metal.mm: the same two-kernel design (net_expand over
// successor "recipes", net_merge over discrete-state buckets) and the same host flat driver in
// netreach.cpp drive it — only the device layer differs. The kernels call the SAME __host__
// __device__ DBM ops as the CPU (dbm_ops.hpp), so the GPU runs CPU-validated algebra. Buffers the
// host reads/writes each level use CUDA managed (unified) memory so netreach.cpp's flat driver —
// which writes merge inputs and reads outputs through raw pointers — works unchanged.
//
// Built for sm_90 (H100). Only compiled when PTCTLMC_HAVE_CUDA is defined.
#ifdef PTCTLMC_HAVE_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ptctlmc/dbm_ops.hpp"
#include "ptctlmc/net_gpu.hpp"
#include "ptctlmc/net_gpu_encode.hpp"
#include "ptctlmc/network.hpp"

namespace ptctlmc {
namespace {

constexpr int kMaxDim = 24;    // <= 23 clocks (matches metal / single-TA engine)
constexpr int kMaxProc = 64;
constexpr int kMaxInt = 32;

#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    cudaError_t e_ = (call);                                                             \
    if (e_ != cudaSuccess)                                                               \
      throw std::runtime_error(std::string("CUDA(net): ") + cudaGetErrorString(e_));    \
  } while (0)

using netgpu::NetEncoding;

// device-resident static network tables (read-only in the kernels)
struct DevNet {
  int dim, nprocs, nintvars, nclocks;
  const int *edge_tgt, *cg_off, *cg_data, *rs_off, *rs_clk, *iu_off, *iu_data;
  const int *locbase, *inv_off, *inv_data, *frozen, *active, *qmask;
};

// combined invariant of the target location vector (mirrors all_invariants + apply_all)
__device__ inline bool apply_inv(raw_t* z, int dim, int nprocs, const int* nlocs,
                                 const int* locbase, const int* inv_off, const int* inv_data) {
  for (int q = 0; q < nprocs; ++q) {
    int Lq = locbase[q] + nlocs[q];
    for (int c = inv_off[Lq]; c < inv_off[Lq + 1]; ++c)
      if (!dbm_constrain(z, dim, inv_data[3 * c], inv_data[3 * c + 1], inv_data[3 * c + 2])) return false;
  }
  return true;
}

// One thread per successor recipe (participant list). Int guards + committed filtering + sync
// enumeration are done on the host; the kernel is the pure zone pipeline over the participants.
__global__ void net_expand_k(const int* front_locs, const int* front_ints, const raw_t* front_zone,
                             const int* work_state, const int* part_off, const int* part_cnt,
                             const int* part_proc, const int* part_edge, DevNet net,
                             int* out_valid, int* out_locs, int* out_ints, raw_t* out_zone, int nwork) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwork) return;
  const int dim = net.dim, d2 = dim * dim, nprocs = net.nprocs, nintvars = net.nintvars, nclocks = net.nclocks;
  out_valid[w] = 0;
  const int si = work_state[w], po = part_off[w], pc = part_cnt[w];

  int nlocs[kMaxProc];
  for (int q = 0; q < nprocs; ++q) nlocs[q] = front_locs[si * nprocs + q];
  for (int t = 0; t < pc; ++t) nlocs[part_proc[po + t]] = net.edge_tgt[part_edge[po + t]];

  raw_t z[kMaxDim * kMaxDim];
  for (int k = 0; k < d2; ++k) z[k] = front_zone[si * d2 + k];
  for (int t = 0; t < pc; ++t) {
    int ge = part_edge[po + t];
    for (int c = net.cg_off[ge]; c < net.cg_off[ge + 1]; ++c)
      if (!dbm_constrain(z, dim, net.cg_data[3 * c], net.cg_data[3 * c + 1], net.cg_data[3 * c + 2])) return;
  }
  for (int t = 0; t < pc; ++t) {
    int ge = part_edge[po + t];
    for (int r = net.rs_off[ge]; r < net.rs_off[ge + 1]; ++r) dbm_reset(z, dim, net.rs_clk[r]);
  }

  if (!apply_inv(z, dim, nprocs, nlocs, net.locbase, net.inv_off, net.inv_data)) return;
  bool frozen_any = false;
  for (int q = 0; q < nprocs; ++q) if (net.frozen[net.locbase[q] + nlocs[q]] & 3) { frozen_any = true; break; }
  if (!frozen_any) {
    dbm_delay(z, dim);
    if (!apply_inv(z, dim, nprocs, nlocs, net.locbase, net.inv_off, net.inv_data)) return;
  }
  for (int cc = 1; cc <= nclocks; ++cc) {
    bool any = false;
    for (int q = 0; q < nprocs; ++q)
      if (net.active[(net.locbase[q] + nlocs[q]) * (nclocks + 1) + cc]) { any = true; break; }
    if (!any && !net.qmask[cc]) dbm_reset(z, dim, cc);
  }
  if (dbm_empty(z, dim)) return;

  int nints[kMaxInt];
  for (int v = 0; v < nintvars; ++v) nints[v] = front_ints[si * nintvars + v];
  for (int t = 0; t < pc; ++t) {
    int ge = part_edge[po + t];
    for (int u = net.iu_off[ge]; u < net.iu_off[ge + 1]; ++u) nints[net.iu_data[2 * u]] = net.iu_data[2 * u + 1];
  }

  out_valid[w] = 1;
  for (int q = 0; q < nprocs; ++q) out_locs[w * nprocs + q] = nlocs[q];
  for (int v = 0; v < nintvars; ++v) out_ints[w * nintvars + v] = nints[v];
  for (int k = 0; k < d2; ++k) out_zone[w * d2 + k] = z[k];
}

// Device convex-union: dbm_ops.hpp's dbm_try_union is host-only (and sized to kDbmMaxDim=64),
// so this is the __device__ twin with kMaxDim (24) scratch — a line-for-line port of it.
__device__ inline bool d_try_union(raw_t* a, const raw_t* b, int dim) {
  raw_t H[kMaxDim * kMaxDim], R[kMaxDim * kMaxDim];
  for (int k = 0; k < dim * dim; ++k) H[k] = a[k] > b[k] ? a[k] : b[k];
  for (int pass = 0; pass < 2; ++pass) {
    const raw_t* z = (pass == 0) ? a : b;
    const raw_t* other = (pass == 0) ? b : a;
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) {
        if (i == j || z[i * dim + j] >= H[i * dim + j]) continue;
        for (int k = 0; k < dim * dim; ++k) R[k] = H[k];
        const int v = raw_bound(z[i * dim + j]);
        const raw_t neg = raw_is_strict(z[i * dim + j]) ? le(-v) : lt(-v);
        if (neg < R[j * dim + i]) { R[j * dim + i] = neg; dbm_close(R, dim); }
        if (dbm_empty(R, dim)) continue;
        if (!dbm_includes(other, R, dim)) return false;
      }
  }
  for (int k = 0; k < dim * dim; ++k) a[k] = H[k];
  dbm_close(a, dim);
  return true;
}

// One thread per discrete-state bucket; replicates add_to (aLU-subsumption both ways + exact
// convex-union) over the work pool the host prefilled with existing zones (isnew = 0).
__global__ void net_merge_k(const int* work_off, const int* exist_cnt, const int* oz,
                           const int* new_idx, const int* new_off, const int* new_cnt,
                           const int* Lb, const int* Ub, raw_t* work, int* work_isnew,
                           int* out_count, int dim, int nb, int has_diag) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nb) return;
  const int d2 = dim * dim;
  raw_t* W = work + work_off[b] * d2;
  int* WI = work_isnew + work_off[b];
  const int* L = Lb + b * dim;
  const int* U = Ub + b * dim;
  int cur = exist_cnt[b];
  raw_t zn[kMaxDim * kMaxDim], zw[kMaxDim * kMaxDim];

  for (int t = 0; t < new_cnt[b]; ++t) {
    const int widx = new_idx[new_off[b] + t];   // read the new zone in place from the expand output
    for (int k = 0; k < d2; ++k) zn[k] = oz[widx * d2 + k];

    bool subsumed = false;
    for (int i = 0; i < cur; ++i) {
      for (int k = 0; k < d2; ++k) zw[k] = W[i * d2 + k];
      if (has_diag ? dbm_includes(zw, zn, dim) : dbm_is_alu_le(zn, zw, dim, L, U)) { subsumed = true; break; }
    }
    if (subsumed) continue;

    int w = 0;
    for (int i = 0; i < cur; ++i) {
      for (int k = 0; k < d2; ++k) zw[k] = W[i * d2 + k];
      bool killed = has_diag ? dbm_includes(zn, zw, dim) : dbm_is_alu_le(zw, zn, dim, L, U);
      if (!killed) { if (w != i) { for (int k = 0; k < d2; ++k) W[w * d2 + k] = W[i * d2 + k]; WI[w] = WI[i]; } ++w; }
    }
    cur = w;

    bool merged = true;
    while (merged) {
      merged = false;
      for (int kk = 0; kk < cur; ++kk) {
        for (int k = 0; k < d2; ++k) zw[k] = W[kk * d2 + k];
        if (d_try_union(zn, zw, dim)) {
          for (int i = kk; i < cur - 1; ++i) { for (int k = 0; k < d2; ++k) W[i * d2 + k] = W[(i + 1) * d2 + k]; WI[i] = WI[i + 1]; }
          --cur; merged = true; break;
        }
      }
    }
    for (int k = 0; k < d2; ++k) W[cur * d2 + k] = zn[k];
    WI[cur] = 1; ++cur;
  }
  out_count[b] = cur;
}

template <class T>
T* dev_upload(const std::vector<T>& v) {
  T* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, (v.empty() ? 1 : v.size()) * sizeof(T)));
  if (!v.empty()) CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

class NetGpuCuda : public NetGpuExpander {
 public:
  explicit NetGpuCuda(NetEncoding&& en) : en_(std::move(en)) {
    d_.dim = en_.dim; d_.nprocs = en_.nprocs; d_.nintvars = en_.nintvars; d_.nclocks = en_.nclocks;
    d_.edge_tgt = dev_upload(en_.edge_tgt); d_.cg_off = dev_upload(en_.cg_off); d_.cg_data = dev_upload(en_.cg_data);
    d_.rs_off = dev_upload(en_.rs_off); d_.rs_clk = dev_upload(en_.rs_clk);
    d_.iu_off = dev_upload(en_.iu_off); d_.iu_data = dev_upload(en_.iu_data);
    d_.locbase = dev_upload(en_.locbase); d_.inv_off = dev_upload(en_.inv_off); d_.inv_data = dev_upload(en_.inv_data);
    d_.frozen = dev_upload(en_.frozen); d_.active = dev_upload(en_.active); d_.qmask = dev_upload(en_.qmask);
  }
  ~NetGpuCuda() override {
    for (void* p : owned_) cudaFree(p);
  }

  std::size_t expand(int nstates, const std::vector<int>& locs, const std::vector<long>& ints,
                     const std::vector<raw_t>& zones, const int*& out_valid, const int*& out_locs,
                     const int*& out_ints, const raw_t*& out_zones) override {
    const int np = en_.nprocs, ni = en_.nintvars, d2 = en_.dim * en_.dim;
    out_valid = out_locs = out_ints = nullptr;
    out_zones = nullptr;

    std::vector<int> ints32(static_cast<std::size_t>(nstates) * ni);
    for (std::size_t i = 0; i < ints32.size(); ++i) ints32[i] = static_cast<int>(ints[i]);
    auto int_ok_ge = [&](int ge, const int* si) {
      for (int g = en_.ig_off[ge]; g < en_.ig_off[ge + 1]; ++g) {
        const int var = en_.ig_data[3 * g], op = en_.ig_data[3 * g + 1], val = en_.ig_data[3 * g + 2];
        const int x = si[var]; bool ok;
        switch (op) { case 0: ok = x == val; break; case 1: ok = x != val; break; case 2: ok = x < val; break;
                      case 3: ok = x <= val; break; case 4: ok = x > val; break; default: ok = x >= val; }
        if (!ok) return false;
      }
      return true;
    };

    // enumerate successor recipes (identical structure + order to netengine_metal.mm)
    std::vector<int> ws, part_off, part_cnt, part_proc, part_edge;
    std::vector<std::pair<int, int>> parts;
    auto emit = [&](int s) {
      ws.push_back(s);
      part_off.push_back(static_cast<int>(part_proc.size()));
      part_cnt.push_back(static_cast<int>(parts.size()));
      for (auto& pr : parts) { part_proc.push_back(pr.first); part_edge.push_back(pr.second); }
    };
    for (int s = 0; s < nstates; ++s) {
      const int* sl = &locs[static_cast<std::size_t>(s) * np];
      const int* si = &ints32[static_cast<std::size_t>(s) * ni];
      bool src_comm = false;
      for (int q = 0; q < np; ++q) if (en_.frozen[en_.locbase[q] + sl[q]] & 1) { src_comm = true; break; }
      for (int p = 0; p < np; ++p) {
        if (src_comm && !(en_.frozen[en_.locbase[p] + sl[p]] & 1)) continue;
        for (int ge : en_.loc_edges[p][sl[p]]) {
          if (en_.edge_chan[ge] >= 0 || !int_ok_ge(ge, si)) continue;
          parts.assign({{p, ge}}); emit(s);
        }
      }
      for (int i = 0; i < np; ++i) {
        for (int ges : en_.loc_edges[i][sl[i]]) {
          if (en_.edge_chan[ges] < 0 || en_.edge_dir[ges] != 1 || en_.chan_bcast[en_.edge_chan[ges]]) continue;
          if (!int_ok_ge(ges, si)) continue;
          const bool i_comm = en_.frozen[en_.locbase[i] + sl[i]] & 1;
          for (int j = 0; j < np; ++j) {
            if (j == i) continue;
            for (int ger : en_.loc_edges[j][sl[j]]) {
              if (en_.edge_chan[ger] != en_.edge_chan[ges] || en_.edge_dir[ger] != -1) continue;
              if (src_comm && !(i_comm || (en_.frozen[en_.locbase[j] + sl[j]] & 1))) continue;
              if (!int_ok_ge(ger, si)) continue;
              parts.assign({{i, ges}, {j, ger}}); emit(s);
            }
          }
        }
      }
      for (int i = 0; i < np; ++i) {
        for (int ges : en_.loc_edges[i][sl[i]]) {
          if (en_.edge_chan[ges] < 0 || en_.edge_dir[ges] != 1 || !en_.chan_bcast[en_.edge_chan[ges]]) continue;
          if (!int_ok_ge(ges, si)) continue;
          std::vector<std::pair<int, std::vector<int>>> pj;
          for (int j = 0; j < np; ++j) {
            if (j == i) continue;
            std::vector<int> es;
            for (int ger : en_.loc_edges[j][sl[j]])
              if (en_.edge_chan[ger] == en_.edge_chan[ges] && en_.edge_dir[ger] == -1 && int_ok_ge(ger, si))
                es.push_back(ger);
            if (!es.empty()) pj.push_back({j, std::move(es)});
          }
          std::vector<std::size_t> pick(pj.size(), 0);
          for (;;) {
            bool part_comm = en_.frozen[en_.locbase[i] + sl[i]] & 1;
            for (auto& pr : pj) if (en_.frozen[en_.locbase[pr.first] + sl[pr.first]] & 1) part_comm = true;
            if (!src_comm || part_comm) {
              parts.assign({{i, ges}});
              for (std::size_t t = 0; t < pj.size(); ++t) parts.push_back({pj[t].first, pj[t].second[pick[t]]});
              emit(s);
            }
            if (pj.empty()) break;
            std::size_t k = 0;
            for (; k < pj.size(); ++k) { if (++pick[k] < pj[k].second.size()) break; pick[k] = 0; }
            if (k == pj.size()) break;
          }
        }
      }
    }
    const int nwork = static_cast<int>(ws.size());
    if (nwork == 0) return 0;

    // frontier + work list -> device; outputs -> managed (host reads in place)
    int* d_fl = tmp(dev_upload(locs)); int* d_fi = tmp(dev_upload(ints32)); raw_t* d_fz = tmp(dev_upload(zones));
    int* d_ws = tmp(dev_upload(ws)); int* d_po = tmp(dev_upload(part_off)); int* d_pc = tmp(dev_upload(part_cnt));
    int* d_pp = tmp(dev_upload(part_proc)); int* d_pe = tmp(dev_upload(part_edge));
    int* ov = grow_i(mv_, static_cast<std::size_t>(nwork));
    int* olb = grow_i(mol_, static_cast<std::size_t>(nwork) * np);
    int* oib = grow_i(moi_, static_cast<std::size_t>(nwork) * (ni ? ni : 1));
    raw_t* ozb = grow_r(moz_, static_cast<std::size_t>(nwork) * d2);
    const int T = 128, B = (nwork + T - 1) / T;
    net_expand_k<<<B, T>>>(d_fl, d_fi, d_fz, d_ws, d_po, d_pc, d_pp, d_pe, d_, ov, olb, oib, ozb, nwork);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    free_tmp();
    out_valid = ov; out_locs = olb; out_ints = oib; out_zones = ozb;
    return static_cast<std::size_t>(nwork);
  }

  void stage_merge(std::size_t work_zones, int d2, raw_t*& work_buf) override {
    work_buf = grow_r(mwork_, (work_zones ? work_zones : 1) * d2);
    int* wi = grow_i(mwi_, work_zones ? work_zones : 1);
    std::memset(wi, 0, (work_zones ? work_zones : 1) * sizeof(int));  // existing isnew = 0
  }

  void merge(int nb, int dim, bool has_diag,
             const std::vector<int>& work_off, const std::vector<int>& exist_cnt,
             const std::vector<int>& new_idx, const std::vector<int>& new_off,
             const std::vector<int>& new_cnt,
             const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U,
             std::vector<int>& out_work_off, std::vector<int>& out_count,
             const raw_t*& out_zones, const int*& out_isnew) override {
    out_work_off.clear();
    out_count.assign(nb, 0);
    out_zones = nullptr;
    out_isnew = nullptr;
    if (nb == 0) return;
    int* d_wo = tmp(dev_upload(work_off)); int* d_ec = tmp(dev_upload(exist_cnt));
    int* d_ni = tmp(dev_upload(new_idx)); int* d_no = tmp(dev_upload(new_off)); int* d_nc = tmp(dev_upload(new_cnt));
    int* d_L = tmp(dev_upload(L)); int* d_U = tmp(dev_upload(U));
    int* oc = grow_i(moc_, static_cast<std::size_t>(nb));
    const int T = 64, B = (nb + T - 1) / T;
    net_merge_k<<<B, T>>>(d_wo, d_ec, moz_.p, d_ni, d_no, d_nc, d_L, d_U, mwork_.p, mwi_.p, oc, dim, nb, has_diag ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    free_tmp();
    out_work_off = work_off;
    for (int b = 0; b < nb; ++b) out_count[b] = oc[b];
    out_zones = mwork_.p;
    out_isnew = mwi_.p;
  }

 private:
  template <class T> struct Managed { T* p = nullptr; std::size_t cap = 0; };
  // persistent managed (unified) buffers, grown on demand
  Managed<int> mv_, mol_, moi_, moc_, mwi_;
  Managed<raw_t> moz_, mwork_;
  std::vector<void*> owned_;   // static device buffers to free
  std::vector<void*> tmps_;    // per-call device uploads to free after the launch
  NetEncoding en_;
  DevNet d_{};

  int* grow_i(Managed<int>& m, std::size_t n) { return grow_m(m, n); }
  raw_t* grow_r(Managed<raw_t>& m, std::size_t n) { return grow_m(m, n); }
  template <class T> T* grow_m(Managed<T>& m, std::size_t n) {
    if (m.cap < n) {
      if (m.p) cudaFree(m.p);
      CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(&m.p), (n ? n : 1) * sizeof(T)));
      m.cap = n;
    }
    return m.p;
  }
  template <class T> T* tmp(T* p) { tmps_.push_back(p); return p; }
  void free_tmp() { for (void* p : tmps_) cudaFree(p); tmps_.clear(); }
};

}  // namespace

std::unique_ptr<NetGpuExpander> make_net_gpu_cuda(const Network& net, const std::vector<char>& qmask) {
  if (netgpu::network_has_diagonal(net)) return nullptr;               // exact-zone diagonal path stays on CPU
  if (net.nclocks + 1 > kMaxDim) return nullptr;
  if (net.nprocs() > kMaxProc || static_cast<int>(net.intvars.size()) > kMaxInt) return nullptr;
  int devs = 0;
  if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) return nullptr;
  try {
    return std::make_unique<NetGpuCuda>(netgpu::encode(net, qmask));
  } catch (...) {
    return nullptr;
  }
}

}  // namespace ptctlmc

#endif  // PTCTLMC_HAVE_CUDA
