/// @file src/query/dispatch.cpp
/// @brief Answers a parsed query by lowering it and selecting an engine.
///
/// Where a hora::Query becomes a call into reachability or liveness: the predicate is lowered
/// to the engine's form, the shape of the formula decides which engine can answer it, and the
/// verdict comes back. Nothing here is specific to the model's file format.

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

/// @brief Constant-fold the boolean literals of a conjunction.
/// @return ::kConstFalse when some conjunct is `false` (so the conjunction is constantly false),
///         ::kConstTrue when every conjunct is `true` or empty, ::kNotConst otherwise.
///
/// `true` was already dropped as a no-op conjunct wherever a conjunction is walked, but `false`
/// had no case at all: it fell through to the location lookup, matched nothing, and was reported
/// UNSUPPORTED. That is a parse gap rather than a real limitation -- both literals are part of
/// the query language -- so they are folded here, once, for every caller that walks conjuncts.
///
/// @note Folding picks the verdict, NOT the cost. Each literal still costs what its dual
///       reachability query costs, so the two are not symmetric: `A[] false` (= `!E<> true`) is
///       refuted by the initial state and stops there, while `A[] true` (= `!E<> false`) has a
///       target no state matches and can only answer by exhausting the reachable set. That makes
///       `A[] true` the honest "measure the whole state space" query -- the one to use for a
///       family whose natural property is violated and would otherwise stop at a counterexample.
enum : int { kConstFalse = 0, kConstTrue = 1, kNotConst = -1 };
int const_fold_conj(const std::string& text) {
  bool all_true = true;
  for (auto atom : split_conjuncts(text)) {
    atom = trim(atom);
    if (atom.empty() || atom == "true") continue;
    if (atom == "false") return kConstFalse;
    all_true = false;
  }
  return all_true ? kConstTrue : kNotConst;
}

bool check_query(const TimedAutomaton& ta, const Query& q_in, bool* satisfied, Backend backend,
                 ReachStats* stats) {
  if (!q_in.supported) return false;
  // rebuild clock map from the automaton
  ClockMap cm;
  for (int c = 1; c <= ta.nclocks; ++c) cm[ta.clock_names[c]] = c;

  // parse the condition (stored raw in note by load_queries)
  Query q = q_in;
  std::string cond = q.note;
  q.note.clear();
  // Verdicts for the constant conditions. The single-automaton path answers them without
  // exploring -- unlike the network path, which runs `A[] true` to exhaustion for its state
  // count. The verdicts agree; only the reported cost differs, and every benchmark model is a
  // network, so the measuring path is the one that explores.
  if (q.kind == Query::EF || q.kind == Query::AG) {
    const int cf = const_fold_conj(cond);
    if (cf != kNotConst) { *satisfied = (cf == kConstTrue); return true; }
  }
  std::string loc_name;
  std::vector<Constraint> clocks;
  for (auto atom : split_conjuncts(cond)) {
    atom = trim(atom);
    if (atom.empty() || atom == "true") continue;
    std::vector<Constraint> cs;
    if (parse_atom(atom, cm, cs)) { clocks.insert(clocks.end(), cs.begin(), cs.end()); continue; }
    if (atom == "deadlock") { *satisfied = false; return false; }
    // location atom "Proc.Loc" or "Loc"
    std::size_t dot = atom.rfind('.');
    loc_name = (dot == std::string::npos) ? atom : atom.substr(dot + 1);
  }
  int target = -1;
  if (!loc_name.empty()) {
    for (int i = 0; i < (int)ta.locations.size(); ++i)
      if (ta.locations[i].name == loc_name) { target = i; break; }
    if (target < 0) return false;  // unknown location
  }

  if (q.kind == Query::EF) {
    *satisfied = reachable(ta, target, clocks, stats, backend);
    return true;
  }
  // A[] cond  (clock safety):  satisfied iff no reachable state violates cond.
  if (target >= 0) return false;  // A[] with a location target not supported in v1
  for (const auto& c : clocks)
    if (reachable(ta, -1, {negate(c)}, stats, backend)) { *satisfied = false; return true; }
  *satisfied = true;
  return true;
}

bool check_network_query(const Network& net, const Query& q, bool* satisfied, Backend backend,
                     ReachStats* stats) {
  if (!q.supported) return false;
  ClockMap cm;
  for (int c = 1; c <= net.nclocks; ++c) cm[net.clock_names[c]] = c;
  std::unordered_map<std::string, int> im;
  for (int i = 0; i < (int)net.intvars.size(); ++i) im[net.intvars[i].name] = i;

  // Conjunction -> NetTarget (E<> path; clock atoms allowed).
  auto build_target = [&](const std::string& text, NetTarget& tgt) -> bool {
    for (auto atom : split_conjuncts(text)) {
      atom = trim(atom);
      if (atom.empty() || atom == "true") continue;
      if (atom == "deadlock") return false;
      std::vector<Constraint> cc; std::vector<IntAtom> ic;
      parse_net_atom(atom, cm, im, cc, ic);
      if (!cc.empty() || !ic.empty()) {
        tgt.clock_cond.insert(tgt.clock_cond.end(), cc.begin(), cc.end());
        tgt.int_cond.insert(tgt.int_cond.end(), ic.begin(), ic.end());
        continue;
      }
      std::size_t dot = atom.rfind('.');
      std::string proc = dot == std::string::npos ? "" : atom.substr(0, dot);
      std::string loc = dot == std::string::npos ? atom : atom.substr(dot + 1);
      int pi, li;
      if (!net.resolve_location(proc, loc, &pi, &li)) return false;
      tgt.loc_req.push_back({pi, li});
    }
    return true;
  };
  // Conjunction of location/int atoms -> DiscPred (liveness; clock atoms rejected).
  // Parse a list of atoms into `out`, optionally negating each one as it is pushed.
  //
  // Negating atom by atom is what lets a disjunction be accepted: `a \/ b` is `!(!a /\ !b)`, and
  // the engine represents a conjunction and the negation of a conjunction, so a disjunction of
  // single atoms lands in the second class. It does not represent anything wider, which is why an
  // atom whose own negation is not a single atom -- a clock equality, since `x == 5` parses to
  // two constraints and negates to a disjunction -- is refused here rather than approximated.
  auto build_atoms = [&](const std::vector<std::string>& atoms, DiscPred& out, bool neg) -> bool {
    for (auto atom : atoms) {
      atom = trim(atom);
      if (atom.empty() || atom == "true") continue;
      std::vector<Constraint> cc; std::vector<IntAtom> ic;
      parse_net_atom(atom, cm, im, cc, ic);
      // A clock atom is kept, not refused: ::exists_invariant_run adjoins the conjunction to
      // every location invariant, which is the psi-restricted semantics and is exact because a
      // conjunction of constraints is convex.
      if (!cc.empty()) {
        if (neg) {
          if (cc.size() != 1) return false;   // `x == 5` negates to a disjunction
          out.clocks.push_back(negate(cc[0]));
        } else {
          out.clocks.insert(out.clocks.end(), cc.begin(), cc.end());
        }
        continue;
      }
      if (!ic.empty()) {
        for (const auto& a : ic) {
          if (!neg) { out.ints.push_back(a); continue; }
          IntOp n = a.op;
          switch (a.op) {
            case IntOp::Eq: n = IntOp::Ne; break; case IntOp::Ne: n = IntOp::Eq; break;
            case IntOp::Lt: n = IntOp::Ge; break; case IntOp::Ge: n = IntOp::Lt; break;
            case IntOp::Le: n = IntOp::Gt; break; case IntOp::Gt: n = IntOp::Le; break;
          }
          out.ints.push_back({a.var, n, a.val});
        }
        continue;
      }
      bool eq = true;
      std::string a = atom;
      if (!a.empty() && a[0] == '!') { eq = false; a = trim(a.substr(1)); }
      else if (a.compare(0, 4, "not ") == 0) { eq = false; a = trim(a.substr(4)); }
      std::size_t dot = a.rfind('.');
      std::string proc = dot == std::string::npos ? "" : a.substr(0, dot);
      std::string loc = dot == std::string::npos ? a : a.substr(dot + 1);
      int pi, li;
      if (!net.resolve_location(proc, loc, &pi, &li)) return false;
      out.locs.push_back({pi, li, neg ? !eq : eq});
    }
    return true;
  };
  auto build_pred = [&](const std::string& text, DiscPred& out) -> bool {
    return build_atoms(split_conjuncts(text), out, false);
  };
  // ---- general boolean predicates -------------------------------------------------------
  //
  // Anything that is not a plain conjunction is handled by putting the negation of the predicate
  // into disjunctive normal form. Each resulting term is a conjunction, hence a discrete guard
  // plus a convex clock region, which is the only shape the engine ever has to represent; the
  // non-convexity lives in the union of terms and is absorbed by intersecting the per-term delay
  // results. See hora::detail::DelayAvoidSet.
  //
  // Grammar: or := and ('||' and)* ; and := unary ('&&' unary)* ; unary := '!' unary | '(' or ')'
  //          | atom. `and`/`or`/`not` spell the operators too, and ',' reads as '&&'.
  struct BExpr {
    enum Kind { Atom, Not, And, Or } kind = Atom;
    std::string atom;
    std::vector<BExpr> kids;
  };
  // A hand-rolled parser rather than reuse of parse_expression: that one yields an arithmetic
  // ExprP over integer state, while what is needed here is the boolean skeleton, with the leaves
  // left as text for parse_net_atom to interpret as a location, integer or clock atom.
  struct BParse {
    const std::string& t; std::size_t i = 0; bool ok = true;
    explicit BParse(const std::string& s) : t(s) {}
    void ws() { while (i < t.size() && std::isspace((unsigned char)t[i])) ++i; }
    bool word(const char* w) {
      ws(); const std::size_t n = std::strlen(w);
      if (t.compare(i, n, w) != 0) return false;
      const bool alpha = std::isalpha((unsigned char)w[0]);
      if (alpha && i + n < t.size() && ident_char((unsigned char)t[i + n])) return false;
      if (alpha && i > 0 && ident_char((unsigned char)t[i - 1])) return false;
      i += n; return true;
    }
    BExpr parse_or() {
      BExpr a = parse_and();
      while (true) { const std::size_t save = i; if (word("||") || word("or")) {
          BExpr b = parse_and(); BExpr n; n.kind = BExpr::Or; n.kids = {a, b}; a = n;
        } else { i = save; break; } }
      return a;
    }
    BExpr parse_and() {
      BExpr a = parse_unary();
      while (true) { const std::size_t save = i; if (word("&&") || word("and") || word(",")) {
          BExpr b = parse_unary(); BExpr n; n.kind = BExpr::And; n.kids = {a, b}; a = n;
        } else { i = save; break; } }
      return a;
    }
    BExpr parse_unary() {
      ws();
      if (word("!") || word("not")) { BExpr k = parse_unary(); BExpr n; n.kind = BExpr::Not; n.kids = {k}; return n; }
      if (word("(")) { BExpr k = parse_or(); if (!word(")")) ok = false; return k; }
      // An atom runs to the next top-level boolean operator or close-paren.
      const std::size_t start = i; int depth = 0;
      while (i < t.size()) {
        if (t[i] == '(') ++depth;
        else if (t[i] == ')') { if (depth == 0) break; --depth; }
        else if (depth == 0) {
          if (t.compare(i, 2, "||") == 0 || t.compare(i, 2, "&&") == 0 || t[i] == ',') break;
          if ((t.compare(i, 2, "or") == 0 || t.compare(i, 3, "and") == 0) &&
              (i == 0 || !ident_char((unsigned char)t[i - 1]))) {
            const std::size_t n = t.compare(i, 2, "or") == 0 ? 2 : 3;
            if (i + n >= t.size() || !ident_char((unsigned char)t[i + n])) break;
          }
        }
        ++i;
      }
      BExpr a; a.kind = BExpr::Atom; a.atom = trim(t.substr(start, i - start));
      if (a.atom.empty()) ok = false;
      return a;
    }
  };
  // Push negations to the leaves, then distribute. Returns the DNF of the expression as a list of
  // terms, each a list of (atom text, negated) literals.
  using Lit = std::pair<std::string, bool>;
  std::function<std::vector<std::vector<Lit>>(const BExpr&, bool)> to_dnf =
      [&](const BExpr& e, bool neg) -> std::vector<std::vector<Lit>> {
    switch (e.kind) {
      case BExpr::Atom: return {{{e.atom, neg}}};
      case BExpr::Not:  return to_dnf(e.kids[0], !neg);
      case BExpr::And: case BExpr::Or: {
        // De Morgan: negating swaps the connective.
        const bool as_or = (e.kind == BExpr::Or) != neg;
        auto a = to_dnf(e.kids[0], neg), b = to_dnf(e.kids[1], neg);
        if (as_or) { for (auto& t : b) a.push_back(std::move(t)); return a; }
        std::vector<std::vector<Lit>> out;
        for (const auto& x : a)
          for (const auto& y : b) { auto t = x; t.insert(t.end(), y.begin(), y.end()); out.push_back(std::move(t)); }
        return out;
      }
    }
    return {};
  };

  // Build the predicate a liveness query must maintain, from the query text.
  //
  // `want_negated` says whether the caller wants `!text` (as `A<>` does) rather than `text`.
  // Either way the result is a conjunction plus a flag, because those are the two shapes the
  // engine represents; a disjunction is turned into the negation of a conjunction of negated
  // atoms, which flips the flag. Mixing `&&` and `||` in one predicate is refused: the result
  // would be neither shape.
  // Split `psi` into an arbitrary boolean over discrete atoms and a conjunction of clock atoms.
  //
  // This is the fragment the backward engine decides. The clock conjunction is convex, so it
  // becomes an invariant of a restricted network and the run stays inside it by construction; the
  // discrete part is constant across a vertex, so any boolean structure in it is a vertex filter
  // and needs no convexity at all. What is refused is a clock atom under a disjunction or a
  // negation, because that is a non-convex region the run must stay inside, and the backward
  // operator for it -- Pre through a delay routed around an obstacle -- does not exist yet.
  //
  // @param[out] cc  the clock conjunction
  // @param[out] ndnf  DNF of the negated discrete part, i.e. where V_psi does NOT hold
  std::function<bool(const BExpr&, std::vector<const BExpr*>&)> flatten_and =
      [&](const BExpr& e, std::vector<const BExpr*>& out) -> bool {
    if (e.kind == BExpr::And) return flatten_and(e.kids[0], out) && flatten_and(e.kids[1], out);
    out.push_back(&e);
    return true;
  };
  std::function<bool(const BExpr&)> has_clock = [&](const BExpr& e) -> bool {
    if (e.kind == BExpr::Atom) {
      std::vector<Constraint> cc; std::vector<IntAtom> ic;
      parse_net_atom(e.atom, cm, im, cc, ic);
      return !cc.empty();
    }
    for (const auto& k : e.kids) if (has_clock(k)) return true;
    return false;
  };
  auto split_psi = [&](const std::string& text, std::vector<Constraint>& cc,
                       std::vector<DiscTerm>& ndnf) -> bool {
    BParse bp(text);
    BExpr e = bp.parse_or();
    bp.ws();
    if (!bp.ok || bp.i != text.size()) return false;
    std::vector<const BExpr*> conj;
    flatten_and(e, conj);
    std::vector<const BExpr*> disc, nonconvex;
    for (const BExpr* k : conj) {
      if (!has_clock(*k)) { disc.push_back(k); continue; }
      if (k->kind == BExpr::Atom) {                  // a bare clock atom: convex, an invariant
        std::vector<Constraint> a; std::vector<IntAtom> ic;
        parse_net_atom(k->atom, cm, im, a, ic);
        cc.insert(cc.end(), a.begin(), a.end());
        continue;
      }
      nonconvex.push_back(k);                        // clock under a disjunction or a negation
    }
    // A conjunct putting a clock under a disjunction or a negation is not an invariant: the run
    // must stay inside a non-convex region. Its negation goes into DNF, giving convex terms with
    // discrete guards that detail::DelayAvoidSet routes each delay around -- forward through
    // delay_avoiding, backward through pre_avoiding.
    for (const BExpr* k : nonconvex)
      for (const auto& term : to_dnf(*k, /*neg=*/true)) {
        // A negated literal can itself be a disjunction -- `x == 3` parses to two constraints, so
        // `!(x == 3)` is `x < 3 \/ x > 3` -- and a forbidden term must stay a conjunction, so the
        // term splits into one copy per alternative, cross-producted over every such literal.
        DiscTerm base;
        std::vector<std::vector<Constraint>> alts;
        for (const auto& lit : term) {
          std::vector<std::string> one{lit.first};
          DiscPred acc;
          if (!lit.second) {
            if (!build_atoms(one, acc, false)) return false;
          } else {
            std::vector<Constraint> cc2; std::vector<IntAtom> ic2;
            parse_net_atom(lit.first, cm, im, cc2, ic2);
            if (cc2.size() > 1) {
              std::vector<Constraint> a;
              for (const auto& c : cc2) a.push_back(negate(c));
              alts.push_back(std::move(a));
              continue;
            }
            if (!build_atoms(one, acc, true)) return false;
          }
          base.locs.insert(base.locs.end(), acc.locs.begin(), acc.locs.end());
          base.ints.insert(base.ints.end(), acc.ints.begin(), acc.ints.end());
          base.clocks.insert(base.clocks.end(), acc.clocks.begin(), acc.clocks.end());
        }
        std::vector<DiscTerm> expanded{base};
        for (const auto& a : alts) {
          std::vector<DiscTerm> next;
          for (const auto& tt : expanded)
            for (const auto& c : a) { DiscTerm cp = tt; cp.clocks.push_back(c); next.push_back(std::move(cp)); }
          expanded.swap(next);
        }
        for (auto& tt : expanded) ndnf.push_back(std::move(tt));
      }
    // The discrete conjuncts, negated and put in DNF: that is exactly `!psi_discrete`.
    for (const BExpr* k : disc)
      for (const auto& term : to_dnf(*k, /*neg=*/true)) {
        DiscTerm t;
        DiscPred acc;
        std::vector<std::string> pos, negs;
        for (const auto& lit : term) (lit.second ? negs : pos).push_back(lit.first);
        if (!build_atoms(pos, acc, false) || !build_atoms(negs, acc, true)) return false;
        if (!acc.clocks.empty()) return false;       // should not happen: `disc` has no clocks
        t.locs = std::move(acc.locs); t.ints = std::move(acc.ints);
        ndnf.push_back(std::move(t));
      }
    return true;
  };

  if (q.kind == Query::EF) {
    if (trim(q.note) == "deadlock") {  // E<> deadlock
      *satisfied = deadlock_reachable(net, backend, stats);
      return true;
    }
    {
      const int cf = const_fold_conj(q.note);
      // `E<> true` is met by the initial state, so it stops there. `E<> false` matches nothing,
      // so it can only answer by exhausting the reachable set -- and that is what makes its dual
      // `A[] true` the honest full-state-space probe.
      if (cf == kConstTrue) { *satisfied = true; return true; }
      if (cf == kConstFalse) {
        NetTarget tgt; tgt.never = true;
        *satisfied = reachable_net(net, tgt, backend, stats);   // false, after a full search
        return true;
      }
    }
    NetTarget tgt;
    if (!build_target(q.note, tgt)) return false;
    *satisfied = reachable_net(net, tgt, backend, stats);
    return true;
  }
  if (q.kind == Query::EG) {  // E[] phi
    // The boolean literals are decided here, before any lowering, because the general path gets
    // one of them WRONG rather than merely slow. `E[] true` lowers `!true` to an EMPTY
    // conjunction, and an empty conjunction is vacuously TRUE, so every state lands in the
    // forbidden set and the answer comes back NOT -- where a maximal run always exists and the
    // property holds. `E[] false` is refused instead: `false` matches no location, integer or
    // clock atom, so ::split_psi declines it and the query reports UNSUPPORTED.
    //
    // @note Neither literal costs a search, and that is NOT the asymmetry `E<>`/`A[]` have. There
    //       `A[] true` (= `!E<> false`) has a target no state matches and must exhaust the
    //       reachable set to say so, which is what makes it the honest full-state-space probe.
    //       Under `E[]` the polarity flips: `false` empties the psi-restricted network, so there
    //       is nothing to explore, and `true` is witnessed by any maximal run. verifyta agrees on
    //       both verdicts and reports 0 states explored for each.
    // @note `E[] true` assumes the initial state exists. verifyta makes the same assumption --
    //       it answers satisfied without exploring -- so a vacuous initial state is out of scope
    //       for both tools rather than a disagreement between them.
    {
      const int cf = const_fold_conj(q.note);
      if (cf == kConstFalse) { *satisfied = false; return true; }
      if (cf == kConstTrue) { *satisfied = true; return true; }
    }
    std::vector<Constraint> cc;
    DiscPred nq;
    nq.is_dnf = true;   // empty means psi's discrete part never fails, not that it always does
    if (!split_psi(q.note, cc, nq.forbid)) return false;
    *satisfied = exists_invariant_run(net, cc, nq, stats, &nq.forbid);
    return true;
  }
  if (q.kind == Query::AF) {  // A<> phi = !E[] !phi
    // The literals, for the same reason ::Query::EG folds them: `!phi` for a literal `phi` lowers
    // to a conjunction the general path cannot represent, and `A<> false` is reported UNSUPPORTED
    // rather than answered. `A<> true` holds at the initial state; `A<> false` asks every run to
    // reach a state satisfying `false`, which none can. verifyta agrees on both.
    {
      const int cf = const_fold_conj(q.note);
      if (cf == kConstTrue) { *satisfied = true; return true; }
      if (cf == kConstFalse) { *satisfied = false; return true; }
    }
    // `!phi` is what the run must maintain. Its clock part is a conjunction only when `phi`'s is
    // a single atom, so the split below refuses the rest -- a regression in reach against the
    // forward path, taken deliberately: the forward path decided them by cycle detection on a
    // subsumed graph, which is unsound.
    {
      std::vector<Constraint> cc;
      DiscPred nq;
      nq.is_dnf = true;
      // psi = !phi, so psi's negation is phi itself: the DNF of `phi` is what nq wants, and the
      // clock conjunction of `psi` is the negation of phi's single clock atom.
      std::string neg = "!(" + q.note + ")";
      if (!split_psi(neg, cc, nq.forbid)) return false;
      *satisfied = !exists_invariant_run(net, cc, nq, stats, &nq.forbid);
      return true;
    }
  }
  auto neg_int = [](IntOp o) {
    switch (o) {
      case IntOp::Eq: return IntOp::Ne; case IntOp::Ne: return IntOp::Eq;
      case IntOp::Lt: return IntOp::Ge; case IntOp::Ge: return IntOp::Lt;
      case IntOp::Le: return IntOp::Gt; default: return IntOp::Le;
    }
  };
  // Is a state satisfying !atom reachable?  atom = conjunction of constraints; !atom is the
  // disjunction of the negated constraints, so any one being reachable violates the atom.
  auto violated = [&](const std::string& atom) -> int {  // 1 reachable, 0 not, -1 unsupported
    std::vector<Constraint> cc; std::vector<IntAtom> ic;
    parse_net_atom(atom, cm, im, cc, ic);
    if (!cc.empty()) {
      for (const auto& c : cc) { NetTarget nt; nt.clock_cond.push_back(negate(c));
        if (reachable_net(net, nt, backend, stats)) return 1; }
      return 0;
    }
    if (!ic.empty()) {
      for (const auto& a : ic) { NetTarget nt; nt.int_cond.push_back({a.var, neg_int(a.op), a.val});
        if (reachable_net(net, nt, backend, stats)) return 1; }
      return 0;
    }
    return -1;  // location atom: negation not expressible as a positive target
  };

  if (q.kind == Query::AG) {  // A[] phi  =  ! E<> !phi
    std::string body = trim(q.note);
    // Safety form  A[] not(X)  /  A[] !(X)  =  ! E<>(X)
    std::string inner;
    if (!body.empty() && body[0] == '!') inner = trim(body.substr(1));
    else if (body.compare(0, 4, "not(") == 0 || body.compare(0, 4, "not ") == 0) inner = trim(body.substr(3));
    if (!inner.empty()) {
      if (inner.size() >= 2 && inner.front() == '(' && inner.back() == ')') inner = trim(inner.substr(1, inner.size() - 2));
      if (inner == "deadlock") {  // A[] not deadlock = ! E<> deadlock
        *satisfied = !deadlock_reachable(net, backend, stats);
        return true;
      }
      {
        const int cf = const_fold_conj(inner);
        if (cf == kConstTrue) { *satisfied = false; return true; }   // A[] not true: initial state
        if (cf == kConstFalse) {                                     // A[] not false == A[] true
          NetTarget tgt; tgt.never = true;
          *satisfied = !reachable_net(net, tgt, backend, stats);
          return true;
        }
      }
      NetTarget tgt;
      if (!build_target(inner, tgt)) return false;
      *satisfied = !reachable_net(net, tgt, backend, stats);
      return true;
    }
    // General:  A[] (a1 & a2 & ...) = AND_i !E<>(!ai)
    {
      const int cf = const_fold_conj(body);
      if (cf == kConstFalse) { *satisfied = false; return true; }  // refuted by the initial state
      if (cf == kConstTrue) {   // nothing can violate it, but the COST is the whole state space
        NetTarget tgt; tgt.never = true;
        *satisfied = !reachable_net(net, tgt, backend, stats);
        return true;
      }
    }
    for (auto atom : split_conjuncts(body)) {
      atom = trim(atom);
      if (atom.empty() || atom == "true") continue;
      const int v = violated(atom);
      if (v < 0) return false;             // unsupported atom (positive location)
      if (v == 1) { *satisfied = false; return true; }
    }
    *satisfied = true;
    return true;
  }
  if (q.kind == Query::LEADSTO) {  // p --> q
    DiscPred p, qq;
    // `p --> q` is violated by EF(p /\ EG !q), so the two operands are handled differently.
    //
    // `q` supplies psi = !q, exactly as A<> does: split_psi on the negated text yields the clock
    // conjunction of psi and the DNF of !psi, which is q itself -- the region the run must avoid
    // wherever q's discrete part holds.
    //
    // `p` is evaluated at one valuation under EF, with no obligation along a delay, so it needs
    // no convexity treatment at all: its DNF is carried positively and intersected with the
    // fixpoint at the end.
    {
      std::vector<Constraint> qcc;
      DiscPred nqp;
      nqp.is_dnf = true;
      if (!split_psi("!(" + q.note + ")", qcc, nqp.forbid)) return false;
      // psi's own clock conjunction must be empty here: psi = !q, and a conjunction in psi would
      // mean q contributed a disjunction, which split_psi routes through `forbid` instead.
      if (!qcc.empty()) return false;
      DiscPred pp;
      pp.is_dnf = true;
      {
        BParse bp(q.lhs);
        BExpr pe = bp.parse_or();
        bp.ws();
        if (!bp.ok || bp.i != q.lhs.size()) return false;
        for (const auto& term : to_dnf(pe, /*neg=*/false)) {
          DiscTerm t;
          DiscPred acc;
          std::vector<std::string> pos, negs;
          for (const auto& lit : term) (lit.second ? negs : pos).push_back(lit.first);
          if (!build_atoms(pos, acc, false) || !build_atoms(negs, acc, true)) return false;
          t.locs = std::move(acc.locs); t.ints = std::move(acc.ints); t.clocks = std::move(acc.clocks);
          pp.forbid.push_back(std::move(t));
        }
      }
      const bool has_clocks = [&] {
        for (const auto& t : pp.forbid) if (!t.clocks.empty()) return true;
        for (const auto& t : nqp.forbid) if (!t.clocks.empty()) return true;
        return false;
      }();
      if (has_clocks || pp.forbid.size() > 1 || nqp.forbid.size() > 1) {
        GfpStats gs2;
        *satisfied = leads_to_clocks(net, pp, nqp, &gs2);
        if (stats) stats->explored = gs2.outer_nodes;
        return true;
      }
      // Purely discrete and single-term on both sides: the long-standing path, untouched.
    }
    if (!build_pred(q.lhs, p) || !build_pred(q.note, qq)) return false;
    // the backward greatest fixpoint IS the engine. Forward SCC search closes cycles that
    // subsumption invented; the backward fixpoint only ever reads a target's W as an upper
    // bound, so subsumption is harmless to it.
    GfpStats gs;
    *satisfied = leads_to_otf(net, p, qq, &gs);
    if (stats) {
      // `explored` is the like-for-like count: successor computations actually performed.
      // On a violated query the on-the-fly engine settles early, so this is far below
      // fwd_nodes -- which is the entire point of the engine and must not be papered over.
      stats->explored = gs.expanded_nodes ? gs.expanded_nodes : gs.fwd_nodes;
      stats->generated = gs.fwd_edges;
    }
    return true;
  }
  return false;
}

}  // namespace hora
