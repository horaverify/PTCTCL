/// @file src/frontend/query_reader.cpp
/// @brief Reads a query file into a list of hora::Query.
///
/// One query per non-blank, non-comment line, kept verbatim. Lowering a query into engine
/// predicates happens in src/query/dispatch.cpp, not here.

#include "hora/model_io.hpp"
#include "hora/query.hpp"
#include "hora/config.hpp"
#include "hora/liveness.hpp"
#include "hora/reachability.hpp"
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <unordered_map>
#include "hora/reach.hpp"
#include "src/frontend/parse_util.hpp"

namespace hora {

using frontend::ClockMap;
using frontend::ident_char;
using frontend::parse_atom;
using frontend::parse_net_atom;
using frontend::split_conjuncts;
using frontend::trim;

std::vector<Query> load_queries(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw ModelError("cannot open query file: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  std::string all = ss.str();
  // strip /* */ comments
  for (std::size_t p; (p = all.find("/*")) != std::string::npos;) {
    std::size_t e = all.find("*/", p);
    all.erase(p, (e == std::string::npos ? all.size() : e + 2) - p);
  }
  std::vector<Query> out;
  std::stringstream ls(all);
  std::string line;
  while (std::getline(ls, line)) {
    std::size_t c = line.find("//");
    if (c != std::string::npos) line = line.substr(0, c);
    line = trim(line);
    if (line.empty()) continue;
    Query q;
    q.raw = line;
    std::string cond;
    std::size_t arrow = line.find("-->");
    if (arrow != std::string::npos) {  // leads-to:  p --> q
      q.kind = Query::LEADSTO;
      q.lhs = trim(line.substr(0, arrow));
      cond = trim(line.substr(arrow + 3));
    } else if (line.compare(0, 3, "E<>") == 0) { q.kind = Query::EF; cond = line.substr(3); }
    else if (line.compare(0, 3, "A[]") == 0) { q.kind = Query::AG; cond = line.substr(3); }
    else if (line.compare(0, 3, "E[]") == 0) { q.kind = Query::EG; cond = line.substr(3); }
    else if (line.compare(0, 3, "A<>") == 0) { q.kind = Query::AF; cond = line.substr(3); }
    else { q.supported = false; q.note = "unsupported query operator"; out.push_back(q); continue; }
    out.push_back(q);
    out.back().raw = line;
    out.back().note = cond;  // condition text stored raw; resolved later (needs clocks)
  }
  return out;
}

/// @brief Resolve and decide one query against a single timed automaton.
/// @param ta   the automaton
/// @param q_in the query, with its condition text unresolved
/// @param[out] satisfied receives the verdict; written only on @c true
/// @param backend which expansion backend to run
/// @param[out] stats optional statistics; unchanged when null
/// @return @c false if the query is unsupported or its condition does not resolve against @p ta
/// @note A @c false return is reported to the user as unsupported. It never means "not
///       satisfied", which is why @p satisfied is left untouched.

}  // namespace hora
