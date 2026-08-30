/// @file hora/query.hpp
/// @brief A parsed query, and the entry point that answers one.
///
/// A hora::Query is the query file's text plus the little structure the front end can determine
/// without a model. Lowering it into engine predicates, and deciding which engine can answer it,
/// happens in ::check_query and ::check_network_query -- neither of which is specific to the
/// model's file format.
#pragma once

#include <string>
#include <vector>

#include "hora/automaton.hpp"
#include "hora/network.hpp"
#include "hora/reach.hpp"
#include "hora/zone.hpp"

namespace hora {

/// @brief One query, as read from a `.q` file.
struct Query {
  /// @brief Which temporal operator the query uses.
  enum Kind { EF, AG, EG, AF, LEADSTO };   ///< `E<>`, `A[]`, `E[]`, `A<>`, `p --> q`

  Kind kind = EF;
  std::string raw;                    ///< the source text, verbatim
  std::string loc_name;               ///< target location, empty for "any location"
  std::vector<Constraint> clock_cond; ///< the clock part, where it is convex
  std::string lhs;                    ///< `p` of `p --> q`; `q` is carried in @ref note
  bool supported = true;              ///< false when @ref note says why it cannot be decided
  std::string note;
};

/// @brief Read every query in a `.q` file, one per non-blank, non-comment line.
/// @throws ModelError if the file cannot be read.
std::vector<Query> load_queries(const std::string& path);

/// @brief Answer @p q on a single timed automaton.
/// @param[out] satisfied the verdict; written only when this returns @c true
/// @return @c false if the query is outside the supported fragment, with the reason in
///         @ref Query::note
bool check_query(const TimedAutomaton& ta, const Query& q, bool* satisfied,
                 Backend backend = Backend::Omp, ReachStats* stats = nullptr);

/// @brief Answer @p q on a network.
/// @copydetails check_query
bool check_network_query(const Network& net, const Query& q, bool* satisfied,
                         Backend backend = Backend::Omp, ReachStats* stats = nullptr);

}  // namespace hora
