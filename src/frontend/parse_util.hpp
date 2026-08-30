/// @file src/frontend/parse_util.hpp
/// @brief Lexical helpers shared by the model reader, the query reader and the query dispatcher.
///
/// These are the only pieces of the front end that more than one translation unit needs, so they
/// are the only ones with external linkage; everything else the front end uses stays file-local.
/// Nothing outside src/frontend and src/query has any use for them, which is why this header is
/// not under include/.
///
/// The definitions live in src/frontend/model_reader.cpp, beside the private types they parse into.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "hora/network.hpp"
#include "hora/zone.hpp"

namespace hora {
namespace frontend {

/// @brief A clock's name mapped to its index in a zone. Index 0 is the reference clock.
using ClockMap = std::unordered_map<std::string, int>;

/// @brief May @p c appear inside an identifier?
///
/// A textual operator such as `and` counts as an operator only when it stands as a whole token.
/// Both sides must be checked: an identifier merely ending in the keyword would otherwise be split
/// mid-name, silently dropping a declaration with no diagnostic.
bool ident_char(unsigned char c);

/// @brief @p s without leading or trailing whitespace.
std::string trim(const std::string& s);

/// @brief Split a textual conjunction into its conjuncts, on `&&` and on `and`.
std::vector<std::string> split_conjuncts(const std::string& s);

/// @brief Parse one comparison of a single-automaton predicate into clock constraints.
/// @return @c false when the atom is not a clock comparison this front end understands.
bool parse_atom(const std::string& atom, const ClockMap& cm, std::vector<Constraint>& out);

/// @brief Parse one network-level atom into either a clock constraint or an integer test.
void parse_net_atom(const std::string& atom, const ClockMap& cm,
                    const std::unordered_map<std::string, int>& im,
                    std::vector<Constraint>& cc, std::vector<IntAtom>& ic);

}  // namespace frontend
}  // namespace hora
