// main.cpp — HORA entry point (timed-automata / TCTL model checker).
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "hora/zone.hpp"
#include "hora/bisimulation.hpp"
#include "hora/config.hpp"
#include "hora/reach.hpp"
#include "hora/automaton.hpp"
#include "hora/model_io.hpp"
#include "hora/query.hpp"

/// @brief `--stats`: append successor-generation counts to each verdict line.
/// The explored-state count is always reported; this adds the generated-zone count, which
/// together give the subsumption ratio used in evaluation.
static bool g_stats = false;


namespace {
constexpr const char* kVersion = "HORA 0.1.0";

int self_test() {
  using namespace hora;
  int fail = 0;
  Dbm z(1);
  z.delay();
  z.constrain(1, 0, le(5));
  if (z.empty()) { std::cerr << "zone unexpectedly empty\n"; ++fail; }
  Dbm y(1);
  y.delay();
  y.constrain(1, 0, le(5));
  y.constrain(0, 1, le(-3));
  if (!z.includes(y)) { std::cerr << "inclusion [0,5]⊇[3,5] failed\n"; ++fail; }
  if (y.includes(z)) { std::cerr << "inclusion [3,5]⊉[0,5] failed\n"; ++fail; }
  if (fail == 0) std::cout << "self-test: OK\n";
  return fail == 0 ? 0 : 1;
}

int run(const char* model, const char* query, hora::Backend backend) {
  using namespace hora;
  try {
    const char* bn = backend == Backend::Seq ? "seq" : backend == Backend::Omp ? "omp"
                   : backend == Backend::Metal ? "metal" : "cuda";
    std::vector<Query> queries = load_queries(query);

    if (network_process_count(model) > 1) {  // network of timed automata
      Network net = load_network(model);
      std::cout << "*** HORA — backend: " << bn << " | network: " << net.nprocs()
                << " procs, " << net.nclocks << " clocks, " << net.intvars.size() << " int vars\n";
      int rc = 0;
      for (const auto& q : queries) {
        bool sat = false;
        ReachStats st;
        std::cout << "Verifying formula: " << q.raw << "\n";
        if (check_network_query(net, q, &sat, backend, &st))
          std::cout << (sat ? "-- Formula is satisfied." : "-- Formula is NOT satisfied.")
                    << "  [explored " << st.explored << " symbolic states"
                    << (g_stats ? ", generated " + std::to_string(st.generated) : "") << "]\n";
        else {
          std::cout << "-- Formula is UNSUPPORTED" << (q.note.empty() ? "" : " (" + q.note + ")") << "\n";
          rc = 2;
        }
      }
      return rc;
    }

    TimedAutomaton ta = load_model(model);
    std::cout << "*** HORA — backend: " << bn << " | " << ta.locations.size() << " locations, "
              << ta.nclocks << " clocks, " << ta.edges.size() << " edges\n";
    Network net_fb;  // lazily loaded as a 1-process network for liveness / A[] / leads-to
    bool net_loaded = false, net_ok = false;
    int rc = 0;
    for (const auto& q : queries) {
      bool sat = false;
      ReachStats st;
      std::cout << "Verifying formula: " << q.raw << "\n";
      // A single automaton is verified by the network engine (as a 1-process network), not by the
      // single-TA one. The two are not equally developed: reach.cpp's passed list only ever appends
      // -- it tests whether a new zone is already covered, but never retires a stored zone that a
      // more general one subsumes, and never runs the convex-union merge -- so on a clock-heavy
      // automaton it is not an antichain at all. The network engine has the interned packed zones,
      // aLU subsumption with per-location bounds, the merge and the sharded passed list.
      //
      // except with diagonal constraints, where the single-TA engine is the better one: it splits
      // locations on the truth of each diagonal atom and stays exact, while the network engine can
      // only keep diagonals exact and relax clock-vs-zero, behind a state cap. So diagonal models
      // keep the old route.
      if (!net_loaded) {
        net_loaded = true;
        try { net_fb = load_network(model); net_ok = true; } catch (...) { net_ok = false; }
      }
      // A model declaring an `urgent chan` must take the network route whatever else is true of
      // it. The single-TA engine in reach.cpp has no notion of channels at all -- it drops the
      // synchronisation label and treats the edge as local -- so it cannot see that a
      // synchronisation is enabled and would let time pass where the model forbids it. That is
      // exactly the silent wrong verdict urgent channels were added to remove, so the diagonal
      // preference below yields to it: weaker covering is a cost, a wrong verdict is not.
      const bool urgent_chan = net_ok && net_fb.has_urgent_chan;
      const bool prefer_net = net_ok && (!network_has_diagonal(net_fb) || urgent_chan);
      bool net_tried = false, done = false;
      if (prefer_net) {
        net_tried = true;
        done = check_network_query(net_fb, q, &sat, backend, &st);
      }
      if (!done && !urgent_chan && check_query(ta, q, &sat, backend, &st)) done = true;
      // E[] / A<> / A[] / leads-to on a single automaton land here when the route above declined.
      else if (!done && net_ok && !net_tried)
        done = check_network_query(net_fb, q, &sat, backend, &st);
      if (done) {
        std::cout << (sat ? "-- Formula is satisfied." : "-- Formula is NOT satisfied.")
                  << "  [explored " << st.explored << " symbolic states"
                  << (g_stats ? ", generated " + std::to_string(st.generated) : "") << "]\n";
      } else {
        std::cout << "-- Formula is UNSUPPORTED" << (q.note.empty() ? "" : " (" + q.note + ")") << "\n";
        rc = 2;
      }
    }
    return rc;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
}  // namespace

int dump(const char* model) {
  using namespace hora;
  TimedAutomaton ta = load_model(model);
  std::cout << "clocks:";
  for (int c = 1; c <= ta.nclocks; ++c) std::cout << " " << ta.clock_names[c] << "(#" << c << ")";
  std::cout << "\ninitial: " << ta.initial << "\n";
  for (int i = 0; i < (int)ta.locations.size(); ++i)
    std::cout << "loc #" << i << " name='" << ta.locations[i].name << "' inv=" << ta.locations[i].invariant.size()
              << (ta.locations[i].urgent ? " urgent" : "") << (ta.locations[i].committed ? " committed" : "") << "\n";
  auto pc = [](const std::vector<Constraint>& cs) {
    for (auto& c : cs) std::cout << " [x" << c.i << "-x" << c.j << (raw_is_strict(c.bound) ? "<" : "<=")
                                 << raw_bound(c.bound) << "]";
  };
  for (int i = 0; i < (int)ta.locations.size(); ++i) { std::cout << "loc#" << i << " inv:"; pc(ta.locations[i].invariant); std::cout << "\n"; }
  for (auto& e : ta.edges) { std::cout << "edge " << e.src << "->" << e.tgt << " guard:"; pc(e.guard); std::cout << " resets=" << e.resets.size() << "\n"; }
  std::cout << "max_const:"; for (int c = 0; c <= ta.nclocks; ++c) std::cout << " " << ta.max_const[c]; std::cout << "\n";
  for (int i = 0; i < (int)ta.locations.size(); ++i) {
    ReachStats st;
    bool r = reachable(ta, i, {}, &st);
    std::cout << "reachable(loc#" << i << ")=" << (r ? "YES" : "no") << " explored=" << st.explored << "\n";
  }
  return 0;
}

// --bisim <spec>: decide a timed (bi)simulation between two automata described in a small text
// spec (written by the front-end, which owns the UPPAAL->BAut translation and its diagnostics).
//
//   kind  strongbisim|strongsim|weakbisim|weaksim
//   obs   <act>*                      (observable labels; used by the weak kinds)
//   aut                               (exactly two blocks: A then B)
//   clocks <n> / locs <n> / init <n>
//   inv   <loc> ge|gt|le|lt <clock> <value>   (location invariant; repeat to conjoin)
//   edge  <src> <dst> <act>           (subsequent g/r lines attach to this edge)
//   g     ge|gt|le|lt <clock> <value>
//   r     <clock>
int bisim_cmd(const char* path) {
  using namespace hora;
  std::ifstream in(path);
  if (!in) { std::cerr << "cannot open spec: " << path << "\n"; return 1; }

  BisimKind kind = BisimKind::StrongBisim;
  std::set<int> obs;
  std::vector<BAut> auts;
  std::string line;
  int lineno = 0;
  auto die = [&](const std::string& msg) { std::cerr << "spec line " << lineno << ": " << msg << "\n"; };

  while (std::getline(in, line)) {
    ++lineno;
    std::istringstream ls(line);
    std::string tok;
    if (!(ls >> tok) || tok[0] == '#') continue;
    if (tok == "kind") {
      std::string k; ls >> k;
      if (k == "strongbisim") kind = BisimKind::StrongBisim;
      else if (k == "strongsim") kind = BisimKind::StrongSim;
      else if (k == "weakbisim") kind = BisimKind::WeakBisim;
      else if (k == "weaksim") kind = BisimKind::WeakSim;
      else { die("unknown kind '" + k + "'"); return 2; }
    } else if (tok == "obs") {
      int v; while (ls >> v) obs.insert(v);
    } else if (tok == "aut") {
      auts.emplace_back();
    } else if (auts.empty()) {
      die("'" + tok + "' before any 'aut'"); return 2;
    } else if (tok == "clocks") { ls >> auts.back().nclocks; }
    else if (tok == "locs")     { ls >> auts.back().nlocs; }
    else if (tok == "init")     { ls >> auts.back().initial; }
    else if (tok == "inv") {
      int loc, clk, val; std::string op;
      if (!(ls >> loc >> op >> clk >> val)) { die("inv needs <loc> <op> <clock> <value>"); return 2; }
      if (loc < 0) { die("negative location"); return 2; }
      auto& iv = auts.back().invariants;
      if (static_cast<int>(iv.size()) <= loc) iv.resize(loc + 1);
      if (op == "ge") iv[loc].push_back(c_ge(clk, val));
      else if (op == "gt") iv[loc].push_back(c_gt(clk, val));
      else if (op == "le") iv[loc].push_back(c_le(clk, val));
      else if (op == "lt") iv[loc].push_back(c_lt(clk, val));
      else { die("unknown invariant op '" + op + "'"); return 2; }
    }
    else if (tok == "edge") {
      BEdge e; ls >> e.src >> e.dst >> e.act;
      auts.back().edges.push_back(e);
    } else if (tok == "g" || tok == "r") {
      if (auts.back().edges.empty()) { die("'" + tok + "' before any 'edge'"); return 2; }
      if (tok == "r") { int c; ls >> c; auts.back().edges.back().resets.push_back(c); }
      else {
        std::string op; int clk, val; ls >> op >> clk >> val;
        auto& g = auts.back().edges.back().guard;
        if (op == "ge") g.push_back(c_ge(clk, val));
        else if (op == "gt") g.push_back(c_gt(clk, val));
        else if (op == "le") g.push_back(c_le(clk, val));
        else if (op == "lt") g.push_back(c_lt(clk, val));
        else { die("unknown guard op '" + op + "'"); return 2; }
      }
    } else { die("unknown directive '" + tok + "'"); return 2; }
  }
  if (auts.size() != 2) {
    std::cerr << "expected exactly 2 'aut' blocks, got " << auts.size() << "\n";
    return 2;
  }
  bool holds = decide_bisim(auts[0], auts[1], kind, obs);
  std::cout << "RESULT: " << (holds ? "yes" : "no") << "\n";
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && (!std::strcmp(argv[1], "--version") || !std::strcmp(argv[1], "-V"))) {
    std::cout << kVersion << "\n";
    return 0;
  }
  if (argc > 1 && !std::strcmp(argv[1], "--selftest")) return self_test();
  if (argc >= 3 && !std::strcmp(argv[1], "--dump")) return dump(argv[2]);
  if (argc >= 3 && !std::strcmp(argv[1], "--bisim")) return bisim_cmd(argv[2]);

  hora::Backend backend = hora::Backend::Omp;
  const char* files[2] = {nullptr, nullptr};
  int nf = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--backend" && i + 1 < argc) {
      std::string b = argv[++i];
      backend = (b == "seq") ? hora::Backend::Seq
              : (b == "cuda") ? hora::Backend::Cuda
              : (b == "metal") ? hora::Backend::Metal : hora::Backend::Omp;
    } else if (a == "--adaptive") {   // GPU merge routing: calibrate CPU vs GPU per level
      hora::set_adaptive_merge(true);
    } else if (a == "--naive") {      // GPU merge routing: always one thread per bucket (default)
      hora::set_adaptive_merge(false);
    } else if (a == "--gpu-intern") {   // GPU passed set: deduplicate zones (memory over speed)
      hora::set_gpu_intern(true);
    } else if (a == "--zones" && i + 1 < argc) {
      const std::string mode = argv[++i];
      if (mode == "packed") {
        hora::set_packed_zones(true);
      } else if (mode != "dense") {
        std::cerr << "unknown --zones mode '" << mode << "' (expected dense or packed)\n";
        return 2;
      }
    } else if (a == "--stats") {       // report generated successors alongside explored states
      g_stats = true;
    } else if (a == "--memo") {        // memoise the successor-zone pipeline (adaptive; off by default)
      hora::set_memo(true);
    } else if (a == "--symmetry") {     // explore one representative per automorphism orbit
      hora::set_symmetry(true);
    } else if (a == "--narrow") {       // pack dk from inferred integer ranges, not declared ones
      hora::set_narrow_ranges(true);
    } else if (a == "--slice") {        // canonicalise dead integer cells before the discrete key
      hora::set_slice(true);
    } else if (a == "--sim" && i + 1 < argc) {   // alu (default) | g
      hora::set_gsim(std::string(argv[++i]) == "g");
    } else if (a == "--omp-chunk") {  // OpenMP dynamic-schedule chunk size (default 8)
      if (i + 1 < argc) {
        const int v = std::atoi(argv[++i]);
        hora::set_omp_chunk(v);
      }
    } else if (a == "--mem" && i + 1 < argc) {
      // Peak GPU-pool budget per batch, in GB. A BFS level is expanded in batches sized so the
      // expand + merge pools stay inside this; smaller means less memory and more batches.
      const double gb = std::atof(argv[++i]);
      if (gb > 0) hora::set_chunk_budget(static_cast<std::size_t>(gb * (1024.0*1024.0*1024.0)));
    } else if (a == "--chunking") {   // expand the frontier in bounded batches (caps peak memory)
      int states = 65536;             // optional size: --chunking 32768
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const int v = std::atoi(argv[i + 1]);
        if (v > 0) { states = v; ++i; }
      }
      hora::set_chunking(true, states);
    } else if (a[0] != '-' && nf < 2) {
      files[nf++] = argv[i];
    }
  }
  if (nf == 2) return run(files[0], files[1], backend);

  std::cout << kVersion << " — Parallel Timed-CTL Model Checker (UPPAAL-style)\n"
            << "usage: hora [--backend seq|omp|metal|cuda] [--adaptive] [--chunking [N]] [--stats]\n              [--zones dense|packed] <model.xml> <query.q>\n"
            << "       --stats: also report the number of successor zones generated\n"
            << "       --zones dense|packed: how liveness stores its zones (default dense).\n"
            << "                   `packed` keeps each zone as a minimal graph -- an order of\n"
            << "                   magnitude smaller resident -- and unpacks it on every read.\n"
            << "                   Same verdicts either way; it trades memory for time, and\n"
            << "                   which way that goes depends on the model's probe rate.\n"
            << "       --gpu-intern: GPU passed set deduplicates zones (large win on models with\n"
            << "                     few distinct zones, e.g. viking; neutral elsewhere)\n"
            << "       --adaptive: GPU merge auto-routes CPU/GPU per level (default --naive)\n"
            << "       --chunking: expand the frontier in batches of N states (default 65536),\n"
            << "                   capping peak GPU memory independently of BFS level width\n"
            << "       hora --bisim <spec>   decide timed (bi)simulation between two automata\n"
            << "       hora [--version] [--selftest] [--dump <model.xml>]\n";
  return 0;
}
