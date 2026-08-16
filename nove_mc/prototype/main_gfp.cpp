// gfp_proto — differential harness for candidate C1.
//
// usage: gfp_proto <model.xml> "<Proc.Loc>" "<Proc.Loc>"
//   runs   p --> q   with BOTH the shipping leads_to_net and the prototype
//   leads_to_gfp, prints both verdicts and the prototype's storage figures.
//
// Verdict disagreement is the refutation this experiment exists to look for.

#include "ptctlmc/netreach.hpp"
#include "ptctlmc/network.hpp"
#include "ptctlmc/uppaal.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

namespace ptctlmc {
struct GfpStats {
  std::size_t fwd_nodes = 0, disc_keys = 0, fwd_edges = 0, iters = 0, fed_max = 0;
  double fed_avg = 0;
  std::size_t nonempty = 0;
  bool capped = false;
  std::size_t expanded_nodes = 0, passes = 0, pre_calls = 0;
  bool early = false;
  bool inconclusive = false;
  std::size_t oracle_nodes = 0;
  std::size_t outer_nodes = 0, outer_expands = 0;
  const char* mode = "gfp";
};
bool leads_to_gfp(const Network& net, const DiscPred& p, const DiscPred& q, GfpStats* gs);
bool leads_to_otf(const Network& net, const DiscPred& p, const DiscPred& q, const char* mode,
                  GfpStats* gs);
void retire_causality(const Network& net);
void defer_probe(const Network& net, int D);
void gen_profile(const Network& net);
void memo_search(const Network& net, bool use_memo);
}
namespace ptctlmc {
}  // namespace ptctlmc

using namespace ptctlmc;

// "P1.cs" -> DiscPred{ proc index of P1, loc index of cs, equal }
static bool parse_pred(const Network& net, const std::string& s, DiscPred* out) {
  const auto dot = s.find('.');
  if (dot == std::string::npos) return false;
  const std::string pn = s.substr(0, dot), ln = s.substr(dot + 1);
  for (int p = 0; p < net.nprocs(); ++p) {
    if (net.procs[p].name != pn) continue;
    for (std::size_t l = 0; l < net.procs[p].locations.size(); ++l)
      if (net.procs[p].locations[l].name == ln) {
        out->locs.push_back(LocReq{p, static_cast<int>(l), true});
        return true;
      }
    std::fprintf(stderr, "no location '%s' in process '%s'\n", ln.c_str(), pn.c_str());
    return false;
  }
  std::fprintf(stderr, "no process '%s'\n", pn.c_str());
  return false;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <model.xml> <Proc.Loc> <Proc.Loc>\n", argv[0]);
    return 2;
  }
  Network net = load_uppaal_network(argv[1]);
  for (auto& p : net.procs) p.build_edge_index();

  if (std::string(argv[2]) == "--retire-causality") { retire_causality(net); return 0; }
  if (std::string(argv[2]) == "--defer") { defer_probe(net, std::atoi(argv[3])); return 0; }
  if (std::string(argv[2]) == "--genprofile") { gen_profile(net); return 0; }
  if (std::string(argv[2]) == "--memo") {
    // 0 = baseline only, 1 = memo only (separate processes so peak RSS is attributable),
    // anything else = both in one process (peak RSS then meaningless).
    const std::string which = argc > 3 ? argv[3] : "2";
    if (which == "0") { memo_search(net, false); return 0; }
    if (which == "1") { memo_search(net, true);  return 0; }
    memo_search(net, false); memo_search(net, true); return 0;
  }

  // E[] phi  (exists an infinite run all of whose states satisfy phi) is EXACTLY the greatest
  // fixpoint this algorithm already computes: take q = !phi, so the !q-restricted region is phi
  // and the accepting set p /\ !q is phi as well. So
  //     E[] phi   ==   NOT leads_to_gfp(p = phi, q = !phi).
  // Compared against the engine's own exists_invariant_run.
  if (std::string(argv[2]) == "--eg") {
    DiscPred phi, notphi;
    if (!parse_pred(net, argv[3], &phi)) return 2;
    notphi = phi; for (auto& r : notphi.locs) r.equal = false;   // negate
    GfpStats g2;
    setenv("PTCTLMC_GFPINITONLY", "1", 1);   // E[] is evaluated AT THE INITIAL STATE
    const bool gfp_eg = !leads_to_gfp(net, phi, notphi, &g2);
    const bool ref_eg = exists_invariant_run(net, phi, Backend::Seq, nullptr);
    std::printf("model=%s  query=E[] %s\n", argv[1], argv[3]);
    std::printf("  reference exists_invariant_run : %s\n", ref_eg ? "SATISFIED" : "NOT satisfied");
    std::printf("  prototype backward gfp         : %s\n", gfp_eg ? "SATISFIED" : "NOT satisfied");
    std::printf("  AGREE=%s   stored=%zu disc-keys=%zu zones/dk=%.2f fed avg=%.2f max=%zu\n",
                ref_eg == gfp_eg ? "YES" : "*** NO ***", g2.fwd_nodes, g2.disc_keys,
                g2.disc_keys ? double(g2.fwd_nodes) / double(g2.disc_keys) : 0.0,
                g2.fed_avg, g2.fed_max);
    return ref_eg == gfp_eg ? 0 : 1;
  }

  DiscPred p, q;
  if (!parse_pred(net, argv[2], &p) || !parse_pred(net, argv[3], &q)) return 2;

  // PTCTLMC_GFPMODE selects the algorithm: "gfp" (the published three-phase version) or
  // one of the on-the-fly variants base/d1/d2/d3/d4/d5 implemented on this branch.
  const char* modes = std::getenv("PTCTLMC_GFPMODE");
  const std::string mode = modes ? modes : "gfp";
  GfpStats gs;
  const bool gfp = (mode == "gfp") ? leads_to_gfp(net, p, q, &gs)
                                   : leads_to_otf(net, p, q, mode.c_str(), &gs);
  const bool noref = std::getenv("PROTO_NOREF") != nullptr;
  if (noref) {
    std::printf("model=%s  query=%s --> %s  mode=%s\n", argv[1], argv[2], argv[3], gs.mode);
    // A capped fixpoint has NOT converged, so its value is neither an over- nor an
  // under-approximation and the verdict must be discarded rather than printed.
  std::printf("  prototype leads_to_gfp : %s\n",
              gs.capped ? "UNUSABLE (federation cap hit)"
                        : gs.inconclusive ? "INCONCLUSIVE (one-sided mode)"
                                          : (gfp ? "SATISFIED" : "NOT satisfied"));
    std::printf("  gfp storage: nodes=%zu disc-keys=%zu zones/dk=%.2f edges=%zu\n",
                gs.fwd_nodes, gs.disc_keys,
                gs.disc_keys ? double(gs.fwd_nodes) / double(gs.disc_keys) : 0.0, gs.fwd_edges);
    std::printf("  gfp fixpoint: rounds=%zu nonempty-nodes=%zu fed-width avg=%.2f max=%zu capped=%s\n",
                gs.iters, gs.nonempty, gs.fed_avg, gs.fed_max, gs.capped ? "YES(UNUSABLE)" : "no");
    std::printf("  otf: expanded=%zu passes=%zu pre-calls=%zu early=%s\n",
                gs.expanded_nodes, gs.passes, gs.pre_calls, gs.early ? "YES" : "no");
    std::printf("  outer-only (like-for-like with verifyta): stored=%zu explored=%zu\n", gs.outer_nodes, gs.outer_expands);
    if (gs.oracle_nodes) std::printf("  oracle: coarse-nodes=%zu (included in nodes=)\n", gs.oracle_nodes);
    std::fflush(stdout);
    return 0;
  }
  // The shipping engine's own state count is the like-for-like on-the-fly baseline: it is a
  // lazy forward Tarjan, so `explored` is how many symbolic states IT had to expand before
  // settling the same query. That is the number the on-the-fly gfp has to beat.
  ReachStats rs;
  const bool ref = leads_to_net(net, p, q, Backend::Seq, &rs);

  std::printf("model=%s  query=%s --> %s  mode=%s\n", argv[1], argv[2], argv[3], gs.mode);
  std::printf("  reference leads_to_net : %s  (explored=%zu)\n",
              ref ? "SATISFIED" : "NOT satisfied", (std::size_t)rs.explored);
  // A capped fixpoint has NOT converged, so its value is neither an over- nor an
  // under-approximation and the verdict must be discarded rather than printed.
  std::printf("  prototype leads_to_gfp : %s\n",
              gs.capped ? "UNUSABLE (federation cap hit)"
                        : gs.inconclusive ? "INCONCLUSIVE (one-sided mode)"
                                          : (gfp ? "SATISFIED" : "NOT satisfied"));
  std::printf("  AGREE=%s\n", ref == gfp ? "YES" : "*** NO ***");
  std::printf("  gfp storage: nodes=%zu disc-keys=%zu zones/dk=%.2f edges=%zu\n",
              gs.fwd_nodes, gs.disc_keys,
              gs.disc_keys ? double(gs.fwd_nodes) / double(gs.disc_keys) : 0.0, gs.fwd_edges);
  std::printf("  gfp fixpoint: rounds=%zu nonempty-nodes=%zu fed-width avg=%.2f max=%zu capped=%s\n",
              gs.iters, gs.nonempty, gs.fed_avg, gs.fed_max, gs.capped ? "YES(UNUSABLE)" : "no");
  std::printf("  otf: expanded=%zu passes=%zu pre-calls=%zu early=%s\n",
              gs.expanded_nodes, gs.passes, gs.pre_calls, gs.early ? "YES" : "no");
  return ref == gfp ? 0 : 1;
}
