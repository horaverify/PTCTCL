// net_gpu_encode.hpp — static, per-run encoding of a Network into flat GPU-friendly arrays,
// shared by the Metal (netengine_metal.mm) and CUDA (netengine_cuda.cu) network backends so the
// two stay bit-for-bit in step. Pure host C++ (no GPU types).
#pragma once

#include <cstddef>
#include <vector>

#include "hora/network.hpp"

namespace hora {
namespace netgpu {

struct NetEncoding {
  int dim = 0, nprocs = 0, nintvars = 0, nclocks = 0, total_locs = 0;
  std::vector<int> locbase;            // [nprocs+1] cumulative location counts
  std::vector<int> edge_tgt;           // [E] target loc (within its process)
  std::vector<int> cg_off, cg_data;    // clock guard: triples (i,j,bound)
  std::vector<int> rs_off, rs_clk;     // resets
  std::vector<int> ig_off, ig_data;    // int guard: triples (var,op,val)
  std::vector<int> iu_off, iu_data;    // int update: pairs (var,val)
  std::vector<int> inv_off, inv_data;  // per-loc invariant: triples (i,j,bound)
  std::vector<int> frozen;             // per-loc: bit0 committed, bit1 urgent
  std::vector<int> active;             // [total_locs*(nclocks+1)]
  // Same information as `active`, one bit per clock (bit c = clock c active here). The kernel's
  // reset-inactive step needs the union over all processes' target locations; with the [loc][clk]
  // table that is nprocs*nclocks device loads per successor. With the mask it is nprocs loads and a
  // bitwise OR. kMaxDim is 24, so 23 clocks always fit in a uint32.
  std::vector<std::uint32_t> active_mask;   // [total_locs]
  std::uint32_t qmask_bits = 0;             // clocks the query reads (same bit layout)
  std::vector<int> qmask;              // [nclocks+1]
  std::vector<int> edge_proc;          // [E] owning process of each global edge
  std::vector<int> edge_chan;          // [E] sync channel (-1 none)
  std::vector<int> edge_dir;           // [E] +1 '!', -1 '?', 0 none
  std::vector<char> chan_bcast;        // [nchan] broadcast?
  // host-only: per (process, location) the global edge indices of all its edges
  // Back-pointer from a global edge index to the NEdge it was flattened from.
  // The tables above only capture the `var OP const` / `var := const` fast
  // paths; an edge may additionally carry gen_guard/gen_update (arithmetic,
  // arrays, calls), which cannot be flattened into fixed-width triples. The GPU
  // host loop evaluates those through gen_ok/gen_apply, so it needs the NEdge.
  // Valid for as long as the source Network outlives the encoding.
  std::vector<const NEdge*> edge_src;   // [E]
  bool has_general = false;             // any edge carries gen_guard/gen_update

  std::vector<std::vector<std::vector<int>>> loc_edges;
};

// Declared, not defined: netreach.cpp owns the one definition. This header used to carry an
// `inline` copy of the same body, which was an ODR violation the moment a GPU translation unit
// and netreach.o were linked together (the strong symbol silently won).
bool network_has_diagonal(const Network& net);

inline NetEncoding encode(const Network& net, const std::vector<char>& qmask) {
  NetEncoding en;
  en.nclocks = net.nclocks;
  en.dim = net.nclocks + 1;
  en.nprocs = net.nprocs();
  en.nintvars = static_cast<int>(net.intvars.size());

  en.locbase.assign(en.nprocs + 1, 0);
  for (int p = 0; p < en.nprocs; ++p)
    en.locbase[p + 1] = en.locbase[p] + static_cast<int>(net.procs[p].locations.size());
  en.total_locs = en.locbase[en.nprocs];

  en.inv_off.assign(en.total_locs + 1, 0);
  en.frozen.assign(en.total_locs, 0);
  en.active.assign(static_cast<std::size_t>(en.total_locs) * (en.nclocks + 1), 0);
  en.active_mask.assign(en.total_locs, 0u);
  const bool have_active = !net.active.empty();
  for (int p = 0; p < en.nprocs; ++p) {
    for (int l = 0; l < static_cast<int>(net.procs[p].locations.size()); ++l) {
      const int Li = en.locbase[p] + l;
      const Location& loc = net.procs[p].locations[l];
      for (const auto& c : loc.invariant) { en.inv_data.push_back(c.i); en.inv_data.push_back(c.j); en.inv_data.push_back(c.bound); }
      en.inv_off[Li + 1] = static_cast<int>(en.inv_data.size() / 3);
      en.frozen[Li] = (loc.committed ? 1 : 0) | (loc.urgent ? 2 : 0);
      if (have_active)
        for (int c = 1; c <= en.nclocks; ++c)
          en.active[static_cast<std::size_t>(Li) * (en.nclocks + 1) + c] = net.active[p][l][c] ? 1 : 0;
      else
        for (int c = 1; c <= en.nclocks; ++c)
          en.active[static_cast<std::size_t>(Li) * (en.nclocks + 1) + c] = 1;  // no reduction => all active
      std::uint32_t m = 0;
      for (int c = 1; c <= en.nclocks; ++c)
        if (en.active[static_cast<std::size_t>(Li) * (en.nclocks + 1) + c]) m |= (1u << c);
      en.active_mask[Li] = m;
    }
  }
  for (int i = 1; i <= en.total_locs; ++i)
    if (en.inv_off[i] < en.inv_off[i - 1]) en.inv_off[i] = en.inv_off[i - 1];

  en.cg_off.push_back(0); en.rs_off.push_back(0); en.ig_off.push_back(0); en.iu_off.push_back(0);
  en.loc_edges.assign(en.nprocs, {});
  for (int p = 0; p < en.nprocs; ++p) {
    en.loc_edges[p].assign(net.procs[p].locations.size(), {});
    for (int ei = 0; ei < static_cast<int>(net.procs[p].edges.size()); ++ei) {
      const NEdge& e = net.procs[p].edges[ei];
      const int ge = static_cast<int>(en.edge_tgt.size());
      en.edge_tgt.push_back(e.tgt);
      en.edge_proc.push_back(p);
      en.edge_chan.push_back(e.sync_chan);
      en.edge_dir.push_back(e.sync_dir);
      for (const auto& c : e.clock_guard) { en.cg_data.push_back(c.i); en.cg_data.push_back(c.j); en.cg_data.push_back(c.bound); }
      en.cg_off.push_back(static_cast<int>(en.cg_data.size() / 3));
      for (int r : e.clock_resets) en.rs_clk.push_back(r);
      en.rs_off.push_back(static_cast<int>(en.rs_clk.size()));
      for (const auto& a : e.int_guard) { en.ig_data.push_back(a.var); en.ig_data.push_back(static_cast<int>(a.op)); en.ig_data.push_back(static_cast<int>(a.val)); }
      en.ig_off.push_back(static_cast<int>(en.ig_data.size() / 3));
      for (const auto& a : e.int_updates) { en.iu_data.push_back(a.var); en.iu_data.push_back(static_cast<int>(a.val)); }
      en.iu_off.push_back(static_cast<int>(en.iu_data.size() / 2));
      en.edge_src.push_back(&e);
      if (e.gen_guard || e.gen_update) en.has_general = true;
      en.loc_edges[p][e.src].push_back(ge);  // ALL edges (sync enumeration happens in expand)
    }
  }
  en.chan_bcast.assign(net.channels.size(), 0);
  for (std::size_t c = 0; c < net.chan_broadcast.size(); ++c) en.chan_bcast[c] = net.chan_broadcast[c];

  en.qmask.assign(en.nclocks + 1, 0);
  for (int c = 0; c <= en.nclocks && c < static_cast<int>(qmask.size()); ++c) en.qmask[c] = qmask[c] ? 1 : 0;
  for (int c = 1; c <= en.nclocks; ++c) if (en.qmask[c]) en.qmask_bits |= (1u << c);
  return en;
}

}  // namespace netgpu
}  // namespace hora
