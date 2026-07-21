// uppaal.hpp — load UPPAAL .xml timed automata and .q queries (clock fragment).
//
// A compact, standalone loader for the common single-template, clock-only subset:
// clocks, locations (+ invariants, urgent/committed), edges (guards, clock resets).
// Networks/channels and integer variables are future work. Queries: `E<>` and
// `A[]` over a location and/or clock constraints.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "ptctlmc/netreach.hpp"
#include "ptctlmc/network.hpp"
#include "ptctlmc/reach.hpp"
#include "ptctlmc/ta.hpp"

namespace ptctlmc {

struct UppaalError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

TimedAutomaton load_uppaal_xml(const std::string& path);

// Number of <template> elements (>1 => a network).
// Number of processes the <system> declaration puts in the network (one
// template may be instantiated several times); falls back to the template
// count for models with no system list.
int uppaal_process_count(const std::string& path);
// Load a multi-template network (clocks + integer variables + int guards/updates).
Network load_uppaal_network(const std::string& path);

struct Query {
  enum Kind { EF, AG, EG, AF, LEADSTO } kind = EF;  // E<>, A[], E[], A<>, p-->q
  std::string raw;
  std::string loc_name;             // target location name ("" = any)
  std::vector<Constraint> clock_cond;
  std::string lhs;                  // LEADSTO: left predicate text (rhs goes in `note`)
  bool supported = true;
  std::string note;
};

std::vector<Query> load_queries(const std::string& path);

// Evaluate `q` on `ta`. Returns true if the query is supported (sets *satisfied);
// false if unsupported (see q.note). `backend` selects the reachability engine.
bool check_query(const TimedAutomaton& ta, const Query& q, bool* satisfied,
                 Backend backend = Backend::Omp, ReachStats* stats = nullptr);

// Evaluate a query on a network (E<> over process locations / int / clock atoms).
bool check_net_query(const Network& net, const Query& q, bool* satisfied,
                     Backend backend = Backend::Omp, ReachStats* stats = nullptr);

}  // namespace ptctlmc
