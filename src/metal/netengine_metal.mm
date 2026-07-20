// netengine_metal.mm — Metal GPU successor generation for NETWORKS (stage 1).
//
// Peer of engine_metal.mm (single TA), but for a network of timed automata with shared int
// variables. One GPU thread per (frontier-state, process, location-edge) runs the exact
// successors_net pipeline for the non-diagonal / channel-free path; the host does the
// subsumption merge. The MSL DBM ops are the same line-for-line port of dbm_ops.hpp used by
// engine_metal.mm, so the GPU runs validated algebra.
//
// Only compiled when PTCTLMC_HAVE_METAL is defined (see CMake).
#ifdef PTCTLMC_HAVE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
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

constexpr int kMaxDim = 24;  // <=23 clocks (matches CUDA / single-TA engine)

#define PTCTLMC_MTL_CHECK(cond, msg) \
  do { if (!(cond)) throw std::runtime_error(std::string("Metal(net): ") + (msg)); } while (0)

struct NetParams {
  std::int32_t dim, nprocs, nintvars, nclocks, nwork;
};

// ---- MSL: DBM ops (port of dbm_ops.hpp) + network successor kernel -------------------
static NSString* const kNetSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define MAXDIM 24
#define MAXPROC 64
#define MAXINT 32
constant int kInf = 1 << 30;
constant int kZeroLe = 1;

inline int r_le(int v)     { return (v << 1) | 1; }
inline int r_lt(int v)     { return (v << 1); }
inline int r_bound(int r)  { return r >> 1; }
inline bool r_strict(int r){ return (r & 1) == 0; }
inline int r_add(int a, int b) {
  if (a >= kInf || b >= kInf) return kInf;
  int v = r_bound(a) + r_bound(b);
  bool weak = !r_strict(a) && !r_strict(b);
  return weak ? r_le(v) : r_lt(v);
}
inline bool d_empty(thread const int* d, int dim) {
  for (int i = 0; i < dim; ++i) if (d[i*dim+i] < kZeroLe) return true;
  return false;
}
inline void d_delay(thread int* d, int dim) { for (int i = 1; i < dim; ++i) d[i*dim+0] = kInf; }
inline void d_reset(thread int* d, int dim, int k) {
  for (int j = 0; j < dim; ++j) { d[k*dim+j] = d[0*dim+j]; d[j*dim+k] = d[j*dim+0]; }
  d[k*dim+k] = kZeroLe;
}
inline bool d_constrain(thread int* d, int dim, int a, int b, int bound) {
  if (bound >= d[a*dim+b]) return !d_empty(d, dim);
  d[a*dim+b] = bound;
  for (int i = 0; i < dim; ++i) { int via = r_add(d[i*dim+a], bound); if (via < d[i*dim+b]) d[i*dim+b] = via; }
  for (int i = 0; i < dim; ++i) {
    int ib = d[i*dim+b]; if (ib >= kInf) continue;
    for (int k = 0; k < dim; ++k) { int via = r_add(ib, d[b*dim+k]); if (via < d[i*dim+k]) d[i*dim+k] = via; }
  }
  return !d_empty(d, dim);
}

// Combined invariant of the *target* location vector: every process's current-location
// invariant intersected in. Returns false on emptiness (mirrors all_invariants + apply_all).
inline bool apply_inv(thread int* z, int dim, int nprocs, thread const int* nlocs,
                      device const int* locbase, device const int* inv_off,
                      device const int* inv_data) {
  for (int q = 0; q < nprocs; ++q) {
    int Lq = locbase[q] + nlocs[q];
    for (int c = inv_off[Lq]; c < inv_off[Lq+1]; ++c)
      if (!d_constrain(z, dim, inv_data[3*c], inv_data[3*c+1], inv_data[3*c+2])) return false;
  }
  return true;
}

// ---- merge ops: port of dbm_close / dbm_includes / dbm_is_alu_le / dbm_try_union -------
constant int kAluNegInf = -(1 << 29);
inline void d_close(thread int* d, int dim) {
  for (int k = 0; k < dim; ++k)
    for (int i = 0; i < dim; ++i) {
      int ik = d[i*dim+k]; if (ik >= kInf) continue;
      for (int j = 0; j < dim; ++j) { int s = r_add(ik, d[k*dim+j]); if (s < d[i*dim+j]) d[i*dim+j] = s; }
    }
}
// does a include b?  (b[i] <= a[i] for all i)
inline bool d_includes(thread const int* a, thread const int* b, int dim) {
  for (int i = 0; i < dim*dim; ++i) if (b[i] > a[i]) return false;
  return true;
}
// is Z subsumed by Zp under the aLU preorder?  (port of dbm_is_alu_le)
inline bool d_alu_le(thread const int* Z, thread const int* Zp, int dim,
                     device const int* L, device const int* U) {
  for (int x = 0; x < dim; ++x) {
    if (U[x] == kAluNegInf) continue;
    if (Z[0*dim+x] < r_le(-U[x])) continue;
    for (int y = 0; y < dim; ++y) {
      if (x == y) continue;
      if (L[y] == kAluNegInf) continue;
      if (Zp[y*dim+x] < Z[y*dim+x] && r_add(Zp[y*dim+x], r_lt(-L[y])) < Z[0*dim+x]) return false;
    }
  }
  return true;
}
// if a ∪ b is exactly convex, set a := hull and return true; else leave a and return false.
inline bool d_try_union(thread int* a, thread const int* b, int dim) {
  int H[MAXDIM*MAXDIM], R[MAXDIM*MAXDIM];
  for (int k = 0; k < dim*dim; ++k) H[k] = a[k] > b[k] ? a[k] : b[k];
  for (int pass = 0; pass < 2; ++pass) {
    thread const int* z = (pass == 0) ? a : b;
    thread const int* other = (pass == 0) ? b : a;
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) {
        if (i == j || z[i*dim+j] >= H[i*dim+j]) continue;
        for (int k = 0; k < dim*dim; ++k) R[k] = H[k];
        int v = r_bound(z[i*dim+j]);
        int neg = r_strict(z[i*dim+j]) ? r_le(-v) : r_lt(-v);
        if (neg < R[j*dim+i]) { R[j*dim+i] = neg; d_close(R, dim); }
        if (d_empty(R, dim)) continue;
        if (!d_includes(other, R, dim)) return false;
      }
  }
  for (int k = 0; k < dim*dim; ++k) a[k] = H[k];
  d_close(a, dim);
  return true;
}

struct Params { int dim, nprocs, nintvars, nclocks, nwork; };
struct MergeParams { int dim, nb, has_diag; };

// One thread per successor "recipe": a participant list of (proc, global-edge) — 1 for a
// location step, 2 for a binary handshake, 1+K for a broadcast. Int guards, committed
// filtering, and the sync enumeration are done on the host (which knows the discrete state);
// the kernel is the pure zone pipeline over the participants (exactly successors_net's zone
// steps: all guards, all resets, combined invariant, delay-if-not-frozen, invariant,
// reset_inactive, emptiness) plus the discrete update (locs/ints).
kernel void net_expand(
    device const int* front_locs [[buffer(0)]],
    device const int* front_ints [[buffer(1)]],
    device const int* front_zone [[buffer(2)]],
    device const int* work_state [[buffer(3)]],
    device const int* part_off   [[buffer(4)]],
    device const int* part_cnt   [[buffer(5)]],
    device const int* part_proc  [[buffer(6)]],
    device const int* part_edge  [[buffer(7)]],
    device const int* edge_tgt   [[buffer(8)]],
    device const int* cg_off     [[buffer(9)]],
    device const int* cg_data    [[buffer(10)]],
    device const int* rs_off     [[buffer(11)]],
    device const int* rs_clk     [[buffer(12)]],
    device const int* iu_off     [[buffer(13)]],
    device const int* iu_data    [[buffer(14)]],
    device const int* locbase    [[buffer(15)]],
    device const int* inv_off    [[buffer(16)]],
    device const int* inv_data   [[buffer(17)]],
    device const int* frozen     [[buffer(18)]],
    device const int* active     [[buffer(19)]],
    device const int* qmask      [[buffer(20)]],
    device int*       out_valid  [[buffer(21)]],
    device int*       out_locs   [[buffer(22)]],
    device int*       out_ints   [[buffer(23)]],
    device int*       out_zone   [[buffer(24)]],
    constant Params&  P          [[buffer(25)]],
    uint w [[thread_position_in_grid]])
{
  if (w >= uint(P.nwork)) return;
  const int dim = P.dim, d2 = dim * dim, nprocs = P.nprocs, nintvars = P.nintvars, nclocks = P.nclocks;
  out_valid[w] = 0;
  const int si = work_state[w], po = part_off[w], pc = part_cnt[w];

  // target location vector: apply every participant's target
  int nlocs[MAXPROC];
  for (int q = 0; q < nprocs; ++q) nlocs[q] = front_locs[si*nprocs + q];
  for (int t = 0; t < pc; ++t) nlocs[part_proc[po+t]] = edge_tgt[part_edge[po+t]];

  // zone: all participants' clock guards, then all their resets
  int z[MAXDIM * MAXDIM];
  for (int k = 0; k < d2; ++k) z[k] = front_zone[si*d2 + k];
  for (int t = 0; t < pc; ++t) {
    int ge = part_edge[po+t];
    for (int c = cg_off[ge]; c < cg_off[ge+1]; ++c)
      if (!d_constrain(z, dim, cg_data[3*c], cg_data[3*c+1], cg_data[3*c+2])) return;
  }
  for (int t = 0; t < pc; ++t) {
    int ge = part_edge[po+t];
    for (int r = rs_off[ge]; r < rs_off[ge+1]; ++r) d_reset(z, dim, rs_clk[r]);
  }

  // combined invariant; delay if no process is committed/urgent; invariant again
  if (!apply_inv(z, dim, nprocs, nlocs, locbase, inv_off, inv_data)) return;
  bool frozen_any = false;
  for (int q = 0; q < nprocs; ++q) if (frozen[locbase[q] + nlocs[q]] & 3) { frozen_any = true; break; }
  if (!frozen_any) {
    d_delay(z, dim);
    if (!apply_inv(z, dim, nprocs, nlocs, locbase, inv_off, inv_data)) return;
  }

  // reset clocks inactive at every process's target location and not read by the query
  for (int cc = 1; cc <= nclocks; ++cc) {
    bool any = false;
    for (int q = 0; q < nprocs; ++q)
      if (active[(locbase[q] + nlocs[q]) * (nclocks + 1) + cc]) { any = true; break; }
    if (!any && !qmask[cc]) d_reset(z, dim, cc);
  }
  if (d_empty(z, dim)) return;

  // integer updates, in participant order (sender then receivers)
  int nints[MAXINT];
  for (int v = 0; v < nintvars; ++v) nints[v] = front_ints[si*nintvars + v];
  for (int t = 0; t < pc; ++t) {
    int ge = part_edge[po+t];
    for (int u = iu_off[ge]; u < iu_off[ge+1]; ++u) nints[iu_data[2*u]] = iu_data[2*u+1];
  }

  out_valid[w] = 1;
  for (int q = 0; q < nprocs; ++q) out_locs[w*nprocs + q] = nlocs[q];
  for (int v = 0; v < nintvars; ++v) out_ints[w*nintvars + v] = nints[v];
  for (int k = 0; k < d2; ++k) out_zone[w*d2 + k] = z[k];
}

// One thread per discrete-state bucket. Replicates add_to EXACTLY: for each new zone (in
// order) — drop it if aLU-subsumed by a stored zone; drop stored zones it subsumes; absorb
// any stored zone whose convex union with it is exact; then append it. Sequential within a
// bucket (small), parallel across the thousands of buckets. `work` is prefilled by the host
// with the bucket's existing antichain (isnew=0); the kernel appends survivors (isnew=1).
kernel void net_merge(
    device const int* work_off   [[buffer(0)]],  // zone-unit start of bucket b's work slice
    device const int* exist_cnt  [[buffer(1)]],
    device const int* new_off    [[buffer(2)]],  // start into new_idx
    device const int* new_cnt    [[buffer(3)]],
    device const int* oz         [[buffer(4)]],  // expand output zones (new successors, in place)
    device const int* new_idx    [[buffer(5)]],  // expand work-item index of each new zone
    device const int* Lb         [[buffer(6)]],  // [nb*dim]
    device const int* Ub         [[buffer(7)]],  // [nb*dim]
    device int*       work       [[buffer(8)]],  // in/out antichain zone pool
    device int*       work_isnew [[buffer(9)]],
    device int*       out_count  [[buffer(10)]],
    constant MergeParams& P      [[buffer(11)]],
    uint b [[thread_position_in_grid]])
{
  if (b >= uint(P.nb)) return;
  const int dim = P.dim, d2 = dim * dim, has_diag = P.has_diag;
  // ulong indexing: work_off[b]*d2 overflows 32 bits once the pool exceeds ~3.7M zones at d2=576
  // (kept in step with the same fix in netengine_cuda.cu, where a datacentre GPU reaches it).
  device int* W = work + (ulong)work_off[b] * d2;
  device int* WI = work_isnew + work_off[b];
  device const int* L = Lb + b * dim;
  device const int* U = Ub + b * dim;
  int cur = exist_cnt[b];
  int zn[MAXDIM * MAXDIM], zw[MAXDIM * MAXDIM];

  for (int t = 0; t < new_cnt[b]; ++t) {
    int widx = new_idx[new_off[b] + t];
    for (int k = 0; k < d2; ++k) zn[k] = oz[(ulong)widx * d2 + k];

    bool subsumed = false;                                   // zn subsumed by a stored zone?
    for (int i = 0; i < cur; ++i) {
      for (int k = 0; k < d2; ++k) zw[k] = W[i*d2 + k];
      if (has_diag ? d_includes(zw, zn, dim) : d_alu_le(zn, zw, dim, L, U)) { subsumed = true; break; }
    }
    if (subsumed) continue;

    int w = 0;                                               // drop stored zones subsumed by zn
    for (int i = 0; i < cur; ++i) {
      for (int k = 0; k < d2; ++k) zw[k] = W[i*d2 + k];
      bool killed = has_diag ? d_includes(zn, zw, dim) : d_alu_le(zw, zn, dim, L, U);
      if (!killed) { if (w != i) { for (int k = 0; k < d2; ++k) W[w*d2 + k] = W[i*d2 + k]; WI[w] = WI[i]; } ++w; }
    }
    cur = w;

    bool merged = true;                                      // convex-union absorb
    while (merged) {
      merged = false;
      for (int kk = 0; kk < cur; ++kk) {
        for (int k = 0; k < d2; ++k) zw[k] = W[kk*d2 + k];
        if (d_try_union(zn, zw, dim)) {                      // zn becomes the hull
          for (int i = kk; i < cur - 1; ++i) { for (int k = 0; k < d2; ++k) W[i*d2 + k] = W[(i+1)*d2 + k]; WI[i] = WI[i+1]; }
          --cur; merged = true; break;
        }
      }
    }

    for (int k = 0; k < d2; ++k) W[cur*d2 + k] = zn[k];      // append
    WI[cur] = 1; ++cur;
  }
  out_count[b] = cur;
}
)MSL";

using netgpu::NetEncoding;  // shared network encoding (net_gpu_encode.hpp)
using netgpu::encode;

class NetGpuMetal : public NetGpuExpander {
 public:
  explicit NetGpuMetal(NetEncoding&& en) : en_(std::move(en)) {
    device_ = MTLCreateSystemDefaultDevice();
    PTCTLMC_MTL_CHECK(device_ != nil, "no system default Metal device");
    queue_ = [device_ newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib = [device_ newLibraryWithSource:kNetSource options:nil error:&err];
    PTCTLMC_MTL_CHECK(lib != nil, std::string("shader compile failed: ") +
                                      (err ? err.localizedDescription.UTF8String : "?"));
    id<MTLFunction> fn = [lib newFunctionWithName:@"net_expand"];
    PTCTLMC_MTL_CHECK(fn != nil, "kernel 'net_expand' not found");
    pso_ = [device_ newComputePipelineStateWithFunction:fn error:&err];
    PTCTLMC_MTL_CHECK(pso_ != nil, "pipeline creation failed");
    id<MTLFunction> fm = [lib newFunctionWithName:@"net_merge"];
    PTCTLMC_MTL_CHECK(fm != nil, "kernel 'net_merge' not found");
    pso_merge_ = [device_ newComputePipelineStateWithFunction:fm error:&err];
    PTCTLMC_MTL_CHECK(pso_merge_ != nil, "merge pipeline creation failed");
    // static (per-run) buffers
    b_edge_tgt_ = ibuf(en_.edge_tgt); b_cg_off_ = ibuf(en_.cg_off); b_cg_data_ = ibuf(en_.cg_data);
    b_rs_off_ = ibuf(en_.rs_off); b_rs_clk_ = ibuf(en_.rs_clk);
    b_iu_off_ = ibuf(en_.iu_off); b_iu_data_ = ibuf(en_.iu_data);  // int guards are host-side (en_.ig_*)
    b_locbase_ = ibuf(en_.locbase); b_inv_off_ = ibuf(en_.inv_off); b_inv_data_ = ibuf(en_.inv_data);
    b_frozen_ = ibuf(en_.frozen); b_active_ = ibuf(en_.active); b_qmask_ = ibuf(en_.qmask);
  }

  std::size_t expand(int nstates, const std::vector<int>& locs, const std::vector<long>& ints,
                     const std::vector<raw_t>& zones, const int*& out_valid, const int*& out_locs,
                     const int*& out_ints, const raw_t*& out_zones) override {
    const int np = en_.nprocs, ni = en_.nintvars, d2 = en_.dim * en_.dim;
    out_valid = out_locs = out_ints = nullptr;
    out_zones = nullptr;

    // source ints as int32 for guard evaluation + the GPU
    std::vector<int> ints32(static_cast<std::size_t>(nstates) * ni);
    for (std::size_t i = 0; i < ints32.size(); ++i) ints32[i] = static_cast<int>(ints[i]);

    auto int_ok_ge = [&](int ge, const int* si) {
      for (int g = en_.ig_off[ge]; g < en_.ig_off[ge + 1]; ++g) {
        const int var = en_.ig_data[3*g], op = en_.ig_data[3*g+1], val = en_.ig_data[3*g+2];
        const int x = si[var];
        bool ok;
        switch (op) { case 0: ok = x == val; break; case 1: ok = x != val; break; case 2: ok = x < val; break;
                      case 3: ok = x <= val; break; case 4: ok = x > val; break; default: ok = x >= val; }
        if (!ok) return false;
      }
      return true;
    };

    // enumerate successor recipes (participant lists), mirroring successors_net EXACTLY:
    // per state — location edges, then binary handshakes, then broadcasts.
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
      for (int p = 0; p < np; ++p) {                                  // location edges
        if (src_comm && !(en_.frozen[en_.locbase[p] + sl[p]] & 1)) continue;
        for (int ge : en_.loc_edges[p][sl[p]]) {
          if (en_.edge_chan[ge] >= 0 || !int_ok_ge(ge, si)) continue;
          parts.assign({{p, ge}}); emit(s);
        }
      }
      for (int i = 0; i < np; ++i) {                                  // binary handshakes
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
      for (int i = 0; i < np; ++i) {                                  // broadcasts
        for (int ges : en_.loc_edges[i][sl[i]]) {
          if (en_.edge_chan[ges] < 0 || en_.edge_dir[ges] != 1 || !en_.chan_bcast[en_.edge_chan[ges]]) continue;
          if (!int_ok_ge(ges, si)) continue;
          std::vector<std::pair<int, std::vector<int>>> pj;  // (proc, its enabled receivers)
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

    @autoreleasepool {
      id<MTLBuffer> b_fz = grow(pb_fz_, (zones.size() ? zones.size() : 1) * sizeof(raw_t));
      if (!zones.empty()) std::memcpy(b_fz.contents, zones.data(), zones.size() * sizeof(raw_t));
      id<MTLBuffer> b_fl = ibuf(locs), b_fi = ibuf(ints32);
      id<MTLBuffer> b_ws = ibuf(ws), b_po = ibuf(part_off), b_pc = ibuf(part_cnt),
                    b_pp = ibuf(part_proc), b_pe = ibuf(part_edge);
      id<MTLBuffer> b_ov = grow(pb_ov_, static_cast<std::size_t>(nwork) * sizeof(int)),
                    b_ol = grow(pb_ol_, static_cast<std::size_t>(nwork) * np * sizeof(int)),
                    b_oi = grow(pb_oi_, static_cast<std::size_t>(nwork) * (ni ? ni : 1) * sizeof(int)),
                    b_oz = grow(pb_oz_, static_cast<std::size_t>(nwork) * d2 * sizeof(int));
      NetParams P{en_.dim, np, ni, en_.nclocks, nwork};

      id<MTLCommandBuffer> cb = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso_];
      id<MTLBuffer> bufs[25] = {b_fl, b_fi, b_fz, b_ws, b_po, b_pc, b_pp, b_pe, b_edge_tgt_, b_cg_off_,
                                b_cg_data_, b_rs_off_, b_rs_clk_, b_iu_off_, b_iu_data_, b_locbase_,
                                b_inv_off_, b_inv_data_, b_frozen_, b_active_, b_qmask_, b_ov, b_ol, b_oi, b_oz};
      for (int i = 0; i < 25; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
      [enc setBytes:&P length:sizeof(P) atIndex:25];
      NSUInteger tg = pso_.maxTotalThreadsPerThreadgroup;
      if (tg > static_cast<NSUInteger>(nwork)) tg = nwork;
      [enc dispatchThreads:MTLSizeMake(nwork, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      PTCTLMC_MTL_CHECK(cb.status == MTLCommandBufferStatusCompleted, "kernel failed");

      out_valid = static_cast<const int*>(b_ov.contents);   // read valid rows in place
      out_locs = static_cast<const int*>(b_ol.contents);
      out_ints = static_cast<const int*>(b_oi.contents);
      out_zones = static_cast<const raw_t*>(b_oz.contents);
      return static_cast<std::size_t>(nwork);
    }
  }

  void stage_merge(std::size_t work_zones, int d2, raw_t*& work_buf) override {
    id<MTLBuffer> bw = grow(pb_work_, (work_zones ? work_zones : 1) * d2 * sizeof(raw_t));
    id<MTLBuffer> bi = grow(pb_wi_, (work_zones ? work_zones : 1) * sizeof(int));
    std::memset(bi.contents, 0, (work_zones ? work_zones : 1) * sizeof(int));  // existing isnew = 0
    work_buf = static_cast<raw_t*>(bw.contents);   // caller fills existing directly (unified memory)
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
    @autoreleasepool {
      id<MTLBuffer> b_work = pb_work_, b_wi = pb_wi_;   // work filled in place; new read from pb_oz_
      id<MTLBuffer> b_woff = ibuf(work_off), b_ec = ibuf(exist_cnt), b_ni = ibuf(new_idx),
                    b_noff = ibuf(new_off), b_nc = ibuf(new_cnt), b_L = Lbuf(L), b_U = Lbuf(U), b_oc = newi(nb);
      MergeParamsGPU MP{dim, nb, has_diag ? 1 : 0};
      id<MTLCommandBuffer> cb = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso_merge_];
      id<MTLBuffer> bufs[11] = {b_woff, b_ec, b_noff, b_nc, pb_oz_, b_ni, b_L, b_U, b_work, b_wi, b_oc};
      for (int i = 0; i < 11; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
      [enc setBytes:&MP length:sizeof(MP) atIndex:11];
      NSUInteger tg = pso_merge_.maxTotalThreadsPerThreadgroup;
      if (tg > static_cast<NSUInteger>(nb)) tg = nb;
      [enc dispatchThreads:MTLSizeMake(nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      PTCTLMC_MTL_CHECK(cb.status == MTLCommandBufferStatusCompleted, "merge kernel failed");
      const int* oc = static_cast<const int*>(b_oc.contents);
      out_work_off = work_off;
      for (int b = 0; b < nb; ++b) out_count[b] = oc[b];
      out_zones = static_cast<const raw_t*>(b_work.contents);
      out_isnew = static_cast<const int*>(b_wi.contents);
    }
  }

 private:
  struct MergeParamsGPU { std::int32_t dim, nb, has_diag; };
  id<MTLBuffer> Lbuf(const std::vector<std::int32_t>& v) {
    const void* p = v.empty() ? (const void*)&kZeroPad_ : (const void*)v.data();
    std::size_t bytes = v.empty() ? sizeof(std::int32_t) : v.size() * sizeof(std::int32_t);
    return [device_ newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
  }
  id<MTLBuffer> ibuf(const std::vector<int>& v) {
    const void* p = v.empty() ? (const void*)&kZeroPad_ : (const void*)v.data();
    std::size_t bytes = v.empty() ? sizeof(int) : v.size() * sizeof(int);
    return [device_ newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
  }
  id<MTLBuffer> rbuf(const std::vector<raw_t>& v) {
    const void* p = v.empty() ? (const void*)&kZeroPad_ : (const void*)v.data();
    std::size_t bytes = v.empty() ? sizeof(raw_t) : v.size() * sizeof(raw_t);
    return [device_ newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
  }
  id<MTLBuffer> newi(std::size_t count) {
    return [device_ newBufferWithLength:(count ? count : 1) * sizeof(int) options:MTLResourceStorageModeShared];
  }
  // Persistent, grow-on-demand shared buffer: reused across levels so we never re-allocate
  // (and never free) the big per-level zone pools — the profiler showed that alloc + upload
  // of hundreds of MB per level was the metal bottleneck. Returns the buffer; caller writes
  // straight into `.contents` (unified memory), skipping the intermediate host vector + copy.
  id<MTLBuffer> grow(__strong id<MTLBuffer>& b, std::size_t bytes) {
    if (bytes == 0) bytes = 16;
    if (b == nil || static_cast<std::size_t>(b.length) < bytes) {
      // Release the OLD pool before allocating the new one. Otherwise both are live across the
      // allocation and the driver reclaims the old pages lazily, so several generations of a
      // growing pool stay resident — which measured as GBs of RSS above what the buffers report.
      // Grow geometrically (>=1.5x) so a steadily-widening BFS re-allocates a handful of times
      // instead of on nearly every level.
      // Grow GEOMETRICALLY (>=1.5x). Measured: with exact sizing a steadily-widening BFS re-allocates
      // the pool on nearly every level, and each dropped generation is reclaimed by the driver only
      // lazily, so ~1 GB of stale pools stays resident. Over-allocating ~0.5 GB to make growth rare is
      // a net ~0.5 GB win (5.00 -> 4.50 GB on Fischer N11) at no cost in run time.
      const std::size_t want = std::max(bytes, static_cast<std::size_t>(b ? b.length * 3 / 2 : 0));
      b = nil;
      b = [device_ newBufferWithLength:want options:MTLResourceStorageModeShared];
    }
    return b;
  }

  NetEncoding en_;
  int kZeroPad_ = 0;
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLComputePipelineState> pso_ = nil;
  id<MTLComputePipelineState> pso_merge_ = nil;
  id<MTLBuffer> b_edge_tgt_, b_cg_off_, b_cg_data_, b_rs_off_, b_rs_clk_,
      b_iu_off_, b_iu_data_, b_locbase_, b_inv_off_, b_inv_data_, b_frozen_, b_active_, b_qmask_;
  id<MTLBuffer> pb_fz_, pb_oz_, pb_ol_, pb_oi_, pb_ov_;   // persistent expand in/out zone+disc pools
  id<MTLBuffer> pb_work_, pb_wi_;                         // persistent merge work pool + isnew

 public:
  // Sum of every persistent unified-memory pool (diagnostics: PTCTLMC_MEMPROF).
  std::size_t device_bytes() const override {
    std::size_t t = 0;
    for (id<MTLBuffer> b : {pb_fz_, pb_oz_, pb_ol_, pb_oi_, pb_ov_, pb_work_, pb_wi_})
      if (b != nil) t += static_cast<std::size_t>(b.length);
    return t;
  }

 private:
};

}  // namespace

std::unique_ptr<NetGpuExpander> make_net_gpu_metal(const Network& net,
                                                   const std::vector<char>& qmask) {
  // scope: non-diagonal networks (channels + shared ints are handled). Diagonal models use
  // the CPU exact-zone fallback.
  if (netgpu::network_has_diagonal(net)) return nullptr;
  if (net.nclocks + 1 > kMaxDim) return nullptr;
  if (net.nprocs() > 64 || static_cast<int>(net.intvars.size()) > 32) return nullptr;
  if (MTLCreateSystemDefaultDevice() == nil) return nullptr;
  try {
    return std::make_unique<NetGpuMetal>(encode(net, qmask));
  } catch (...) {
    return nullptr;  // any Metal failure -> CPU fallback
  }
}

}  // namespace ptctlmc

#endif  // PTCTLMC_HAVE_METAL
