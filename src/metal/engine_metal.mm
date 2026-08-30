// engine_metal.mm — Metal GPU zone-graph reachability (macOS).
//
// The Apple-GPU peer of engine_cuda.cu: the GPU computes the successor zone for every
// (frontier-state, edge) pair — each thread runs the full DBM pipeline (guard ∩, reset,
// target-invariant ∩, delay, Extra_LU+ extrapolate, emptiness) — while the host does the
// serial passed-list subsumption merge. The MSL DBM ops below are a line-for-line port of
// the __host__ __device__ ops in dbm_ops.hpp (which the CPU unit tests validate), so the
// GPU runs the same algebra; identical verdicts to seq/omp/cuda by construction.
//
// Buffers use shared storage (unified memory on Apple silicon): host<->device transfer is
// a plain memcpy. Only compiled when HORA_HAVE_METAL is defined (see CMake).
#ifdef HORA_HAVE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "hora/zone.hpp"
#include "hora/metal_reach.hpp"

namespace hora {
namespace {

constexpr int kMaxDim = 24;  // up to 23 clocks (matches the CUDA backend)

#define HORA_MTL_CHECK(cond, msg) \
  do { if (!(cond)) throw std::runtime_error(std::string("Metal: ") + (msg)); } while (0)

struct ParamsGPU { std::int32_t dim; std::int32_t nwork; };

// ---- MSL: DBM ops (port of dbm_ops.hpp) + one-thread-per-(state,edge) kernel --------
static NSString* const kSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define MAXDIM 24
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
inline void d_close(thread int* d, int dim) {
  for (int k = 0; k < dim; ++k)
    for (int i = 0; i < dim; ++i) {
      int ik = d[i*dim+k]; if (ik >= kInf) continue;
      for (int j = 0; j < dim; ++j) { int s = r_add(ik, d[k*dim+j]); if (s < d[i*dim+j]) d[i*dim+j] = s; }
    }
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
// Extra_LU+ (dim <= MAXDIM here, so no fallback needed).
inline void d_extrap_lu_plus(thread int* d, int dim, device const int* L, device const int* U) {
  int row0[MAXDIM];
  for (int k = 0; k < dim; ++k) row0[k] = d[k];
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) {
      if (i == j) continue;
      int e = d[i*dim+j]; if (e >= kInf) continue;
      if (r_bound(e) > L[i]) { d[i*dim+j] = kInf; continue; }
      if (-r_bound(row0[i]) > L[i]) { d[i*dim+j] = kInf; continue; }
      if (i != 0 && -r_bound(row0[j]) > U[j]) { d[i*dim+j] = kInf; continue; }
      if (e < r_lt(-U[j])) d[i*dim+j] = r_lt(-U[j]);
    }
  d_close(d, dim);
}

struct Params { int dim; int nwork; };

kernel void gen_kernel(
    device const int* frontier   [[buffer(0)]],
    device const int* work_state [[buffer(1)]],
    device const int* work_edge  [[buffer(2)]],
    device const int* edge_tgt   [[buffer(3)]],
    device const int* edge_urgent[[buffer(4)]],
    device const int* guard_off  [[buffer(5)]],
    device const int* gi         [[buffer(6)]],
    device const int* gj         [[buffer(7)]],
    device const int* gb         [[buffer(8)]],
    device const int* reset_off  [[buffer(9)]],
    device const int* rclk       [[buffer(10)]],
    device const int* inv_off    [[buffer(11)]],
    device const int* ii         [[buffer(12)]],
    device const int* ij         [[buffer(13)]],
    device const int* ib         [[buffer(14)]],
    device const int* L          [[buffer(15)]],
    device const int* U          [[buffer(16)]],
    device int*       out_zones  [[buffer(17)]],
    device int*       out_loc    [[buffer(18)]],
    device int*       out_valid  [[buffer(19)]],
    constant Params&  P          [[buffer(20)]],
    uint w [[thread_position_in_grid]])
{
  if (w >= uint(P.nwork)) return;
  int dim = P.dim, d2 = dim * dim;
  int si = work_state[w], ei = work_edge[w];
  int z[MAXDIM * MAXDIM];
  for (int k = 0; k < d2; ++k) z[k] = frontier[si * d2 + k];

  bool ok = true;
  for (int g = guard_off[ei]; ok && g < guard_off[ei+1]; ++g)
    if (!d_constrain(z, dim, gi[g], gj[g], gb[g])) ok = false;
  if (ok) {
    for (int r = reset_off[ei]; r < reset_off[ei+1]; ++r) d_reset(z, dim, rclk[r]);
    for (int c = inv_off[ei]; ok && c < inv_off[ei+1]; ++c)
      if (!d_constrain(z, dim, ii[c], ij[c], ib[c])) ok = false;
  }
  if (ok) {
    if (edge_urgent[ei] == 0) {
      d_delay(z, dim);
      for (int c = inv_off[ei]; ok && c < inv_off[ei+1]; ++c)
        if (!d_constrain(z, dim, ii[c], ij[c], ib[c])) ok = false;
    }
    if (ok) { d_extrap_lu_plus(z, dim, L, U); if (d_empty(z, dim)) ok = false; }
  }
  out_valid[w] = ok ? 1 : 0;
  out_loc[w] = edge_tgt[ei];
  if (ok) for (int k = 0; k < d2; ++k) out_zones[w * d2 + k] = z[k];
}
)MSL";

// ---- host helpers (mirror engine_cuda.cu / reach.cpp) --------------------------------
bool apply_host(Dbm& z, const std::vector<Constraint>& cs) {
  for (const auto& c : cs) if (!z.constrain(c.i, c.j, c.bound)) return false;
  return true;
}
void delay_in_host(Dbm& z, const Location& l) {
  if (l.urgent || l.committed) return;
  z.delay();
  apply_host(z, l.invariant);
}

}  // namespace

bool reachable_metal(const TimedAutomaton& ta, const std::vector<char>& accept,
                     const std::vector<Constraint>& tgt_clocks,
                     const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U,
                     ReachStats* stats) {
  const int dim = ta.nclocks + 1;
  if (dim > kMaxDim) throw std::runtime_error("reachable_metal: too many clocks (raise kMaxDim)");
  const int d2 = dim * dim;
  // `ta` is diagonal-free (the caller applied diagonal splitting) and L/U are the
  // Extra_LU+ bounds the caller precomputed — so the GPU path handles diagonal models too.

  // flatten edges to host arrays.
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

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    HORA_MTL_CHECK(device != nil, "no system default Metal device");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:kSource options:nil error:&err];
    HORA_MTL_CHECK(lib != nil, std::string("shader compile failed: ") +
                                      (err ? err.localizedDescription.UTF8String : "?"));
    id<MTLFunction> fn = [lib newFunctionWithName:@"gen_kernel"];
    HORA_MTL_CHECK(fn != nil, "kernel 'gen_kernel' not found");
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    HORA_MTL_CHECK(pso != nil, "pipeline creation failed");

    auto buf = [&](const void* p, std::size_t bytes) {
      if (bytes == 0) { std::int32_t z = 0; return [device newBufferWithBytes:&z length:sizeof(z)
                                                                     options:MTLResourceStorageModeShared]; }
      return [device newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
    };
    auto ibuf = [&](const std::vector<int>& v) { return buf(v.data(), v.size() * sizeof(int)); };
    auto rbuf = [&](const std::vector<raw_t>& v) { return buf(v.data(), v.size() * sizeof(raw_t)); };
    auto Lbuf = [&](const std::vector<std::int32_t>& v) { return buf(v.data(), v.size() * sizeof(std::int32_t)); };

    id<MTLBuffer> b_tgt = ibuf(h_tgt), b_urg = ibuf(h_urg), b_goff = ibuf(goff), b_gi = ibuf(gi),
                  b_gj = ibuf(gj), b_gb = rbuf(gb), b_roff = ibuf(roff), b_rclk = ibuf(rclk),
                  b_ioff = ibuf(ioff), b_ii = ibuf(ii), b_ij = ibuf(ij), b_ib = rbuf(ib),
                  b_L = Lbuf(L), b_U = Lbuf(U);

    // host reachability driver; GPU does per-level successor generation.
    const int nloc = static_cast<int>(ta.locations.size());
    std::vector<std::vector<Dbm>> passed(nloc);
    auto is_target = [&](int loc, const Dbm& z) {
      if (!accept[loc]) return false;
      Dbm t = z; return apply_host(t, tgt_clocks);
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

    while (!found && !front_loc.empty()) {
      std::vector<int> ws, we;
      for (int s = 0; s < static_cast<int>(front_loc.size()); ++s)
        for (int ei : ta.edges_from(front_loc[s])) { ws.push_back(s); we.push_back(ei); }
      const int nwork = static_cast<int>(ws.size());
      if (nwork == 0) break;

      id<MTLBuffer> b_front = buf(front_z.data(), front_z.size() * sizeof(raw_t));
      id<MTLBuffer> b_ws = ibuf(ws), b_we = ibuf(we);
      id<MTLBuffer> b_oz = [device newBufferWithLength:static_cast<std::size_t>(nwork) * d2 * sizeof(raw_t)
                                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> b_ol = [device newBufferWithLength:nwork * sizeof(int) options:MTLResourceStorageModeShared];
      id<MTLBuffer> b_ov = [device newBufferWithLength:nwork * sizeof(int) options:MTLResourceStorageModeShared];
      ParamsGPU params{dim, nwork};

      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      id<MTLBuffer> bufs[20] = {b_front, b_ws, b_we, b_tgt, b_urg, b_goff, b_gi, b_gj, b_gb,
                                b_roff, b_rclk, b_ioff, b_ii, b_ij, b_ib, b_L, b_U, b_oz, b_ol, b_ov};
      for (int i = 0; i < 20; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
      [enc setBytes:&params length:sizeof(params) atIndex:20];
      NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
      if (tg > static_cast<NSUInteger>(nwork)) tg = nwork;
      [enc dispatchThreads:MTLSizeMake(nwork, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      HORA_MTL_CHECK(cb.status == MTLCommandBufferStatusCompleted, "kernel failed");

      const int* ol = static_cast<const int*>(b_ol.contents);
      const int* ov = static_cast<const int*>(b_ov.contents);
      const raw_t* oz = static_cast<const raw_t*>(b_oz.contents);

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
    return found;
  }
}

}  // namespace hora

#endif  // HORA_HAVE_METAL
