// main.cpp — PTCTLMC entry point (timed-automata / TCTL model checker).
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "ptctlmc/bisim.hpp"
#include "ptctlmc/dbm.hpp"
#include "ptctlmc/netreach.hpp"
#include "ptctlmc/reach.hpp"
#include "ptctlmc/ta.hpp"
#include "ptctlmc/uppaal.hpp"

namespace {
constexpr const char* kVersion = "PTCTLMC 0.1.0";

int self_test() {
  using namespace ptctlmc;
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

int run(const char* model, const char* query, ptctlmc::Backend backend) {
  using namespace ptctlmc;
  try {
    const char* bn = backend == Backend::Seq ? "seq" : backend == Backend::Omp ? "omp"
                   : backend == Backend::Metal ? "metal" : "cuda";
    std::vector<Query> queries = load_queries(query);

    if (uppaal_process_count(model) > 1) {  // network of timed automata
      Network net = load_uppaal_network(model);
      std::cout << "*** PTCTLMC — backend: " << bn << " | network: " << net.nprocs()
                << " procs, " << net.nclocks << " clocks, " << net.intvars.size() << " int vars\n";
      int rc = 0;
      for (const auto& q : queries) {
        bool sat = false;
        ReachStats st;
        std::cout << "Verifying formula: " << q.raw << "\n";
        if (check_net_query(net, q, &sat, backend, &st))
          std::cout << (sat ? "-- Formula is satisfied." : "-- Formula is NOT satisfied.")
                    << "  [explored " << st.explored << " symbolic states]\n";
        else {
          std::cout << "-- Formula is UNSUPPORTED" << (q.note.empty() ? "" : " (" + q.note + ")") << "\n";
          rc = 2;
        }
      }
      return rc;
    }

    TimedAutomaton ta = load_uppaal_xml(model);
    std::cout << "*** PTCTLMC — backend: " << bn << " | " << ta.locations.size() << " locations, "
              << ta.nclocks << " clocks, " << ta.edges.size() << " edges\n";
    Network net_fb;  // lazily loaded as a 1-process network for liveness / A[] / leads-to
    bool net_loaded = false, net_ok = false;
    int rc = 0;
    for (const auto& q : queries) {
      bool sat = false;
      ReachStats st;
      std::cout << "Verifying formula: " << q.raw << "\n";
      if (check_query(ta, q, &sat, backend, &st)) {
        std::cout << (sat ? "-- Formula is satisfied." : "-- Formula is NOT satisfied.")
                  << "  [explored " << st.explored << " symbolic states]\n";
        continue;
      }
      // Fall back to the network engine (E[] / A<> / A[] / leads-to on a single automaton).
      if (!net_loaded) {
        net_loaded = true;
        try { net_fb = load_uppaal_network(model); net_ok = true; } catch (...) { net_ok = false; }
      }
      if (net_ok && check_net_query(net_fb, q, &sat, backend, &st)) {
        std::cout << (sat ? "-- Formula is satisfied." : "-- Formula is NOT satisfied.")
                  << "  [explored " << st.explored << " symbolic states]\n";
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
  using namespace ptctlmc;
  TimedAutomaton ta = load_uppaal_xml(model);
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
//   edge  <src> <dst> <act>           (subsequent g/r lines attach to this edge)
//   g     ge|gt|le|lt <clock> <value>
//   r     <clock>
int bisim_cmd(const char* path) {
  using namespace ptctlmc;
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

  ptctlmc::Backend backend = ptctlmc::Backend::Omp;
  const char* files[2] = {nullptr, nullptr};
  int nf = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--backend" && i + 1 < argc) {
      std::string b = argv[++i];
      backend = (b == "seq") ? ptctlmc::Backend::Seq
              : (b == "cuda") ? ptctlmc::Backend::Cuda
              : (b == "metal") ? ptctlmc::Backend::Metal : ptctlmc::Backend::Omp;
    } else if (a == "--adaptive") {   // GPU merge routing: calibrate CPU vs GPU per level
      ptctlmc::set_adaptive_merge(true);
    } else if (a == "--naive") {      // GPU merge routing: always one thread per bucket (default)
      ptctlmc::set_adaptive_merge(false);
    } else if (a == "--omp-chunk") {  // OpenMP dynamic-schedule chunk size (default 8)
      if (i + 1 < argc) {
        const int v = std::atoi(argv[++i]);
        ptctlmc::set_omp_chunk(v);
      }
    } else if (a == "--chunking") {   // expand the frontier in bounded batches (caps peak memory)
      int states = 65536;             // optional size: --chunking 32768
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const int v = std::atoi(argv[i + 1]);
        if (v > 0) { states = v; ++i; }
      }
      ptctlmc::set_chunking(true, states);
    } else if (a[0] != '-' && nf < 2) {
      files[nf++] = argv[i];
    }
  }
  if (nf == 2) return run(files[0], files[1], backend);

  std::cout << kVersion << " — Parallel Timed-CTL Model Checker (UPPAAL-style)\n"
            << "usage: ptctlmc [--backend seq|omp|metal|cuda] [--adaptive] [--chunking [N]] <model.xml> <query.q>\n"
            << "       --adaptive: GPU merge auto-routes CPU/GPU per level (default --naive)\n"
            << "       --chunking: expand the frontier in batches of N states (default 65536),\n"
            << "                   capping peak GPU memory independently of BFS level width\n"
            << "       ptctlmc --bisim <spec>   decide timed (bi)simulation between two automata\n"
            << "       ptctlmc [--version] [--selftest] [--dump <model.xml>]\n";
  return 0;
}
