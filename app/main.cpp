// main.cpp — PTCTLMC entry point (timed-automata / TCTL model checker).
#include <cstring>
#include <iostream>

#include "ptctlmc/dbm.hpp"
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
    const char* bn = backend == Backend::Seq ? "seq" : backend == Backend::Omp ? "omp" : "cuda";
    std::vector<Query> queries = load_queries(query);

    if (uppaal_template_count(model) > 1) {  // network of timed automata
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
    int rc = 0;
    for (const auto& q : queries) {
      bool sat = false;
      ReachStats st;
      std::cout << "Verifying formula: " << q.raw << "\n";
      if (check_query(ta, q, &sat, backend, &st)) {
        std::cout << (sat ? "-- Formula is satisfied." : "-- Formula is NOT satisfied.")
                  << "  [explored " << st.explored << " symbolic states]\n";
      } else {
        std::cout << "-- Formula is UNSUPPORTED"
                  << (q.note.empty() ? "" : " (" + q.note + ")") << "\n";
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

int main(int argc, char** argv) {
  if (argc > 1 && (!std::strcmp(argv[1], "--version") || !std::strcmp(argv[1], "-V"))) {
    std::cout << kVersion << "\n";
    return 0;
  }
  if (argc > 1 && !std::strcmp(argv[1], "--selftest")) return self_test();
  if (argc >= 3 && !std::strcmp(argv[1], "--dump")) return dump(argv[2]);

  ptctlmc::Backend backend = ptctlmc::Backend::Omp;
  const char* files[2] = {nullptr, nullptr};
  int nf = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--backend" && i + 1 < argc) {
      std::string b = argv[++i];
      backend = (b == "seq") ? ptctlmc::Backend::Seq
              : (b == "cuda") ? ptctlmc::Backend::Cuda : ptctlmc::Backend::Omp;
    } else if (a[0] != '-' && nf < 2) {
      files[nf++] = argv[i];
    }
  }
  if (nf == 2) return run(files[0], files[1], backend);

  std::cout << kVersion << " — Parallel Timed-CTL Model Checker (UPPAAL-style)\n"
            << "usage: ptctlmc [--backend seq|omp|cuda] <model.xml> <query.q>\n"
            << "       ptctlmc [--version] [--selftest] [--dump <model.xml>]\n";
  return 0;
}
