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

#include "hora/expression.hpp"
#include "hora/network.hpp"
#include "hora/zone.hpp"

namespace hora {
namespace frontend {

/// @brief A clock's name mapped to its index in a zone. Index 0 is the reference clock.
using ClockMap = std::unordered_map<std::string, int>;

/// @brief Where a guard, invariant or assignment resolves a symbolic bound.
///
/// A bound that is neither a clock nor an integer variable is a declared constant -- `x <= SA`
/// with `const int SA = 20;`. Resolving it needs the scope's symbol table, which only the model
/// reader has; the query side passes none and keeps the older, lenient reading, where an
/// unresolvable token is simply not a clock atom and the caller retries it as a location name.
///
/// @warning Without a context an unresolvable bound silently became `0`, which turned `x <= SA`
///          into `x <= 0` and quietly replaced the model with a degenerate one. When a context
///          IS supplied the same case now raises ::ModelError instead.
struct BoundScope {
  const SymTab* st = nullptr;
  const std::vector<Func>* funcs = nullptr;
};

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
                    std::vector<Constraint>& cc, std::vector<IntAtom>& ic,
                    const BoundScope& ev = {});

}  // namespace frontend
}  // namespace hora
