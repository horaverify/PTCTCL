/// @file uppaal.cpp
/// @brief Front end: parses model files and query files into the engine's own structures.
///
/// Everything here is translation, never analysis. It reads the modelling language's XML format
/// and produces either a ptctlmc::TimedAutomaton (single automaton) or a ptctlmc::Network
/// (parallel composition), plus the ptctlmc::Query list a run checks. It also owns the query
/// routers ::check_query and ::check_net_query, which choose an engine per query kind.
///
/// @par What the parser refuses
/// A construct that cannot be represented faithfully is rejected, so a query over it reports
/// UNSUPPORTED rather than being answered against different semantics. The one place this rule is
/// broken today is `urgent chan`, which is accepted and downgraded silently — see
/// KNOWN_LIMITATION.md §1.
///
/// @warning Declarations are matched textually. Any keyword test must therefore check both sides
///          of the match for identifier characters; testing only the prefix would split an
///          identifier that merely begins or ends with the keyword and drop the declaration with
///          no diagnostic.

#include "ptctlmc/uppaal.hpp"

#include "ptctlmc/netreach.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <unordered_map>

#include "ptctlmc/reach.hpp"

namespace ptctlmc {
namespace {

// ---- a minimal XML tree, sufficient for the flat model format ---------------

/// @brief One XML element: its tag, attributes, text and children.
struct Xml {
  std::string tag;
  std::vector<std::pair<std::string, std::string>> attrs;
  std::string text;
  std::vector<Xml> kids;
  const Xml* first(const std::string& t) const {
    for (const auto& k : kids) if (k.tag == t) return &k;
    return nullptr;
  }
  std::string attr(const std::string& a) const {
    for (const auto& p : attrs) if (p.first == a) return p.second;
    return "";
  }
};

/// @brief Expand the XML entity references that appear in model files.
/// @param s raw attribute or text content
/// @return @p s with the standard entities replaced
std::string unescape(const std::string& s) {
  std::string o;
  o.reserve(s.size());
  for (std::size_t i = 0; i < s.size();) {
    if (s[i] == '&') {
      if (s.compare(i, 4, "&lt;") == 0) { o += '<'; i += 4; continue; }
      if (s.compare(i, 4, "&gt;") == 0) { o += '>'; i += 4; continue; }
      if (s.compare(i, 5, "&amp;") == 0) { o += '&'; i += 5; continue; }
      if (s.compare(i, 6, "&quot;") == 0) { o += '"'; i += 6; continue; }
      if (s.compare(i, 6, "&apos;") == 0) { o += '\''; i += 6; continue; }
    }
    o += s[i++];
  }
  return o;
}

/// @brief Recursive-descent XML reader producing an ::Xml tree.
/// @note Handles only the subset the model format uses; it is not a general XML parser.
struct XmlParser {
  const std::string& s;
  std::size_t i = 0;
  explicit XmlParser(const std::string& src) : s(src) {}

  void ws() { while (i < s.size() && std::isspace((unsigned char)s[i])) ++i; }

  void skip_misc() {
    for (;;) {
      ws();
      if (s.compare(i, 4, "<!--") == 0) { auto e = s.find("-->", i); i = (e == std::string::npos) ? s.size() : e + 3; }
      else if (s.compare(i, 2, "<?") == 0) { auto e = s.find("?>", i); i = (e == std::string::npos) ? s.size() : e + 2; }
      else if (s.compare(i, 2, "<!") == 0) { auto e = s.find('>', i); i = (e == std::string::npos) ? s.size() : e + 1; }
      else break;
    }
  }

  std::string name() {
    std::size_t b = i;
    while (i < s.size() && (std::isalnum((unsigned char)s[i]) || s[i] == '_' || s[i] == '-' || s[i] == ':')) ++i;
    return s.substr(b, i - b);
  }

  Xml element() {
    Xml n;
    ++i;  // consume '<'
    n.tag = name();
    // attributes
    for (;;) {
      ws();
      if (i >= s.size() || s[i] == '>' || s[i] == '/') break;
      std::string an = name();
      ws();
      if (i < s.size() && s[i] == '=') { ++i; ws(); }
      char q = s[i++];
      std::size_t b = i;
      while (i < s.size() && s[i] != q) ++i;
      n.attrs.emplace_back(an, unescape(s.substr(b, i - b)));
      ++i;  // closing quote
    }
    ws();
    if (s.compare(i, 2, "/>") == 0) { i += 2; return n; }
    ++i;  // consume '>'
    // content
    std::string text;
    for (;;) {
      if (i >= s.size()) break;
      if (s[i] == '<') {
        if (s.compare(i, 4, "<!--") == 0) { auto e = s.find("-->", i); i = (e == std::string::npos) ? s.size() : e + 3; continue; }
        if (s.compare(i, 2, "</") == 0) { i += 2; name(); ws(); if (i < s.size()) ++i; break; }
        n.kids.push_back(element());
      } else {
        text += s[i++];
      }
    }
    n.text = unescape(text);
    return n;
  }

  Xml parse() { skip_misc(); return element(); }
};

// ---- expression tokenizing / parsing (clock constraints & assignments) -------
/// @brief Strip leading and trailing whitespace.
std::string trim(const std::string& s) {
  std::size_t a = 0, b = s.size();
  while (a < b && std::isspace((unsigned char)s[a])) ++a;
  while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
  return s.substr(a, b - a);
}

/// @brief May @p c appear inside an identifier?
///
/// @warning Used to require that a textual operator such as `and` is a whole token. Both sides
///          must be checked: an identifier merely *ending* in the keyword would otherwise be split
///          mid-name, silently dropping the declaration and corrupting the model with no
///          diagnostic.
static inline bool ident_char(unsigned char c) { return std::isalnum(c) != 0 || c == '_'; }

/// @brief Split a textual conjunction into its conjuncts.
/// @param s the expression text
/// @return the conjuncts, trimmed; a single-element result when there is no top-level operator
/// @note Splits on both the symbolic and the word form of conjunction, the latter only when it
///       stands as a whole token; see ::ident_char.
std::vector<std::string> split_conjuncts(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (std::size_t i = 0; i < s.size();) {
    if (s.compare(i, 2, "&&") == 0) { out.push_back(trim(cur)); cur.clear(); i += 2; }
    else if (s.compare(i, 3, "and") == 0
             && (i == 0 || !ident_char((unsigned char)s[i - 1]))
             && (i + 3 >= s.size() || !ident_char((unsigned char)s[i + 3]))) {
      out.push_back(trim(cur)); cur.clear(); i += 3;
    }
    else if (s[i] == ',') { out.push_back(trim(cur)); cur.clear(); ++i; }
    else cur += s[i++];
  }
  if (!trim(cur).empty()) out.push_back(trim(cur));
  return out;
}

/// @brief Split on top-level `||` / `or`, the disjunctive counterpart of ::split_conjuncts.
/// @note Used only by the liveness predicates. A disjunction is admissible there precisely when
///       its negation is a conjunction, which is the shape both engine mechanisms can represent;
///       see ::build_pred_parts.
std::vector<std::string> split_disjuncts(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (std::size_t i = 0; i < s.size();) {
    if (s.compare(i, 2, "||") == 0) { out.push_back(trim(cur)); cur.clear(); i += 2; }
    else if (s.compare(i, 2, "or") == 0
             && (i == 0 || !ident_char((unsigned char)s[i - 1]))
             && (i + 2 >= s.size() || !ident_char((unsigned char)s[i + 2]))) {
      out.push_back(trim(cur)); cur.clear(); i += 2;
    }
    else cur += s[i++];
  }
  if (!trim(cur).empty()) out.push_back(trim(cur));
  return out;
}

/// @brief Does @p s contain a top-level `||` or `or`?
bool has_disjunction(const std::string& s) { return split_disjuncts(s).size() > 1; }

using ClockMap = std::unordered_map<std::string, int>;

enum class Cmp { LE, LT, GE, GT, EQ };

/// @brief One side of a comparison: a constant, a single clock, or a clock difference.
struct Side {
  bool is_const = false;  ///< the side is a literal constant
  long val = 0;           ///< its value, when constant
  int a = 0;              ///< the clock, or the minuend of a difference
  int b = -1;             ///< the subtrahend of a difference, or -1 for a single clock
};

/// @brief Parse one side of a clock comparison.
/// @param raw the text
/// @param cm  clock name to index
/// @return the parsed side
/// @throws UppaalError if the text is neither a constant nor a known clock or clock difference
Side parse_side(const std::string& raw, const ClockMap& cm) {
  std::string t = trim(raw);
  Side s;
  // clock difference "x - y"
  std::size_t minus = std::string::npos;
  int depth_ident = 0;
  for (std::size_t i = 1; i < t.size(); ++i)  // skip leading '-' (unary)
    if (t[i] == '-') { minus = i; break; }
  (void)depth_ident;
  if (minus != std::string::npos) {
    std::string l = trim(t.substr(0, minus)), r = trim(t.substr(minus + 1));
    auto il = cm.find(l), ir = cm.find(r);
    if (il != cm.end() && ir != cm.end()) { s.a = il->second; s.b = ir->second; return s; }
  }
  auto it = cm.find(t);
  if (it != cm.end()) { s.a = it->second; s.b = -1; return s; }
  // constant (maybe negative)
  s.is_const = true;
  s.val = std::stol(t);
  return s;
}

/// @brief Emit the DBM constraints encoding one clock comparison.
/// @param a  the clock, or the minuend of a difference
/// @param b  the subtrahend, or -1 for a single clock
/// @param op the comparison operator
/// @param k  the constant compared against
/// @param[out] out constraints are appended
/// @note An equality emits *two* constraints, an upper and a lower bound, since a DBM entry can
///       express only one direction.
void emit_clock_cmp(int a, int b, Cmp op, long k, std::vector<Constraint>& out) {
  switch (op) {
    case Cmp::LE: out.push_back(b < 0 ? Constraint{a, 0, le((int)k)} : Constraint{a, b, le((int)k)}); break;
    case Cmp::LT: out.push_back(b < 0 ? Constraint{a, 0, lt((int)k)} : Constraint{a, b, lt((int)k)}); break;
    case Cmp::GE: out.push_back(b < 0 ? Constraint{0, a, le((int)-k)} : Constraint{b, a, le((int)-k)}); break;
    case Cmp::GT: out.push_back(b < 0 ? Constraint{0, a, lt((int)-k)} : Constraint{b, a, lt((int)-k)}); break;
    case Cmp::EQ: emit_clock_cmp(a, b, Cmp::LE, k, out); emit_clock_cmp(a, b, Cmp::GE, k, out); break;
  }
}

/// @brief Reverse a comparison operator, for rewriting `const op x` as `x op' const`.
Cmp flip(Cmp c) {
  switch (c) { case Cmp::LE: return Cmp::GE; case Cmp::LT: return Cmp::GT;
               case Cmp::GE: return Cmp::LE; case Cmp::GT: return Cmp::LT; default: return Cmp::EQ; }
}

/// @brief Parse a single comparison into clock constraints.
/// @param atom the atom text
/// @param cm   clock name to index
/// @param[out] out constraints are appended on success
/// @return @c false if @p atom is not a clock comparison, leaving @p out untouched
bool parse_atom(const std::string& atom, const ClockMap& cm, std::vector<Constraint>& out) {
  static const std::pair<const char*, Cmp> ops[] = {
      {"<=", Cmp::LE}, {">=", Cmp::GE}, {"==", Cmp::EQ}, {"<", Cmp::LT}, {">", Cmp::GT}};
  std::size_t pos = std::string::npos;
  Cmp op = Cmp::EQ;
  int oplen = 0;
  for (auto& o : ops) {
    std::size_t p = atom.find(o.first);
    if (p != std::string::npos && (pos == std::string::npos || p < pos)) { pos = p; op = o.second; oplen = (int)std::string(o.first).size(); }
  }
  if (pos == std::string::npos) return false;
  Side l = parse_side(atom.substr(0, pos), cm);
  Side r = parse_side(atom.substr(pos + oplen), cm);
  if (l.is_const && !r.is_const) { std::swap(l, r); op = flip(op); }
  if (l.is_const || !r.is_const) return false;  // need clock op const
  emit_clock_cmp(l.a, l.b, op, r.val, out);
  return true;
}

/// @brief Parse a conjunction of clock comparisons.
/// @param expr the expression text
/// @param cm   clock name to index
/// @return the conjoined constraints
std::vector<Constraint> parse_constraints(const std::string& expr, const ClockMap& cm) {
  std::vector<Constraint> out;
  for (const auto& atom : split_conjuncts(expr)) {
    if (atom.empty()) continue;
    parse_atom(atom, cm, out);  // non-clock atoms (int guards) silently ignored in v1
  }
  return out;
}

/// @brief Parse a reset list, i.e. the clock assignments of an edge's update.
/// @param expr the update text
/// @param cm   clock name to index
/// @return the indices of the clocks reset
/// @note Only assignments to zero are resets; the engine's zone operations support no other form.
std::vector<int> parse_resets(const std::string& expr, const ClockMap& cm) {
  std::vector<int> out;
  for (auto part : split_conjuncts(expr)) {
    // "x = 0" or "x := 0"
    std::size_t p = part.find(":=");
    std::size_t adv = 2;
    if (p == std::string::npos) { p = part.find('='); adv = 1; }
    if (p == std::string::npos) continue;
    std::string lhs = trim(part.substr(0, p)), rhs = trim(part.substr(p + adv));
    auto it = cm.find(lhs);
    if (it != cm.end() && rhs == "0") out.push_back(it->second);
  }
  return out;
}

/// @brief Register every clock a declaration block declares.
/// @param decl the declaration text
/// @param[in,out] cm    clock name to index, extended
/// @param[in,out] names index to clock name, extended in step with @p cm
/// @post Clock indices start at 1; index 0 is the reference clock.
void collect_clocks(const std::string& decl, ClockMap& cm, std::vector<std::string>& names) {
  // find "clock a, b, c;" statements
  std::size_t p = 0;
  while ((p = decl.find("clock", p)) != std::string::npos) {
    bool word = (p == 0 || !std::isalnum((unsigned char)decl[p - 1])) &&
                (!std::isalnum((unsigned char)decl[p + 5]));
    if (!word) { p += 5; continue; }
    std::size_t e = decl.find(';', p);
    std::string list = decl.substr(p + 5, (e == std::string::npos ? decl.size() : e) - (p + 5));
    for (auto id : split_conjuncts(list)) {  // comma-separated
      id = trim(id);
      if (!id.empty() && (std::isalpha((unsigned char)id[0]) || id[0] == '_')) {
        int idx = (int)names.size();  // names already holds the "0" reference at index 0
        cm[id] = idx;
        names.push_back(id);
      }
    }
    p = (e == std::string::npos) ? decl.size() : e + 1;
  }
}

// ---- the system block -------------------------------------------------------

/// @brief The system block: template instantiations, and which of them form the network.
///
/// The block does two separate things, and ignoring either silently under-models the system: it
/// instantiates templates under new names, and it lists which processes actually compose. A
/// template that is never listed is not part of the system at all.
struct SystemDecl {
  bool has_list = false;
  std::vector<std::string> order;                     // names in `system ...;`
  std::unordered_map<std::string, std::string> inst;  // instance name -> template
};

/// @brief Remove line and block comments from declaration text.
/// @note Runs before any keyword matching, so a keyword inside a comment cannot be mistaken for a
///       declaration.
std::string strip_comments(const std::string& s) {
  std::string o;
  o.reserve(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '/' && i + 1 < s.size() && s[i + 1] == '/') {
      while (i < s.size() && s[i] != '\n') ++i;
      o.push_back('\n');
    } else if (s[i] == '/' && i + 1 < s.size() && s[i + 1] == '*') {
      i += 2;
      while (i + 1 < s.size() && !(s[i] == '*' && s[i + 1] == '/')) ++i;
      ++i;
      o.push_back(' ');
    } else {
      o.push_back(s[i]);
    }
  }
  return o;
}

bool ident_char(char c) { return std::isalnum((unsigned char)c) || c == '_'; }

/// @brief Parse the system block.
/// @param raw the block text
/// @return the instantiations and the composed process list
/// @post When no explicit system line is present, every declared template is taken to compose, so
///       a model without one still yields a network.
SystemDecl parse_system(const std::string& raw) {
  SystemDecl sd;
  const std::string s = strip_comments(raw);
  std::size_t p = 0;
  while (p < s.size()) {
    std::size_t e = s.find(';', p);
    std::string stmt = trim(s.substr(p, (e == std::string::npos ? s.size() : e) - p));
    p = (e == std::string::npos) ? s.size() : e + 1;
    if (stmt.empty()) continue;

    // `system A, B, C;` — UPPAAL also allows '<' between priority groups.
    if (stmt.size() >= 6 && stmt.compare(0, 6, "system") == 0 &&
        (stmt.size() == 6 || !ident_char(stmt[6]))) {
      sd.has_list = true;
      std::string list = stmt.substr(6);
      for (char& c : list) if (c == '<') c = ',';
      for (auto n : split_conjuncts(list)) {
        n = trim(n);
        if (!n.empty()) sd.order.push_back(n);
      }
      continue;
    }

    // `Name = Template(args);`
    std::size_t eq = stmt.find('=');
    if (eq == std::string::npos) continue;
    std::string lhs = trim(stmt.substr(0, eq));
    std::string rhs = trim(stmt.substr(eq + 1));
    if (rhs.empty() || rhs.back() != ')') continue;  // not an instantiation
    std::size_t op = rhs.find('(');
    if (op == std::string::npos) continue;
    std::string tname = trim(rhs.substr(0, op));
    std::string args = trim(rhs.substr(op + 1, rhs.size() - op - 2));
    if (lhs.empty() || tname.empty()) continue;
    if (!args.empty())
      throw UppaalError("parameterised template instantiation is not supported: " + stmt);
    sd.inst[lhs] = tname;
  }
  return sd;
}

}  // namespace

/// @brief Load a model file as a single timed automaton.
/// @param path path to the model file
/// @return the automaton, with its max-constant arrays already computed
/// @throws UppaalError if the file cannot be opened or the model cannot be represented
/// @note For a model composing several processes, use ::load_uppaal_network instead.
TimedAutomaton load_uppaal_xml(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw UppaalError("cannot open model: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  Xml nta = XmlParser(ss.str()).parse();
  if (nta.tag != "nta") throw UppaalError("not an <nta> UPPAAL model");

  TimedAutomaton ta;
  ClockMap cm;
  ta.clock_names.push_back("0");
  if (const Xml* d = nta.first("declaration")) collect_clocks(d->text, cm, ta.clock_names);
  const Xml* tmpl = nta.first("template");
  if (!tmpl) throw UppaalError("no <template> found");
  if (const Xml* d = tmpl->first("declaration")) collect_clocks(d->text, cm, ta.clock_names);
  ta.nclocks = (int)ta.clock_names.size() - 1;

  std::unordered_map<std::string, int> loc_id;  // xml id -> index
  for (const auto& k : tmpl->kids) {
    if (k.tag != "location") continue;
    Location loc;
    loc.name = (k.first("name") ? k.first("name")->text : k.attr("id"));
    loc.committed = k.first("committed") != nullptr;
    loc.urgent = k.first("urgent") != nullptr;
    for (const auto& lb : k.kids)
      if (lb.tag == "label" && lb.attr("kind") == "invariant")
        loc.invariant = parse_constraints(lb.text, cm);
    loc_id[k.attr("id")] = (int)ta.locations.size();
    ta.locations.push_back(std::move(loc));
  }
  if (const Xml* in0 = tmpl->first("init")) {
    auto it = loc_id.find(in0->attr("ref"));
    ta.initial = (it != loc_id.end()) ? it->second : 0;
  }
  for (const auto& k : tmpl->kids) {
    if (k.tag != "transition") continue;
    Edge e;
    e.src = loc_id[k.first("source")->attr("ref")];
    e.tgt = loc_id[k.first("target")->attr("ref")];
    for (const auto& lb : k.kids) {
      if (lb.tag != "label") continue;
      const std::string kind = lb.attr("kind");
      if (kind == "guard") e.guard = parse_constraints(lb.text, cm);
      else if (kind == "assignment") e.resets = parse_resets(lb.text, cm);
    }
    ta.edges.push_back(std::move(e));
  }
  ta.compute_max_constants();
  return ta;
}

/// @brief Load a query file.
/// @param path path to the query file
/// @return one ptctlmc::Query per query line
/// @throws UppaalError if the file cannot be opened
/// @post A query whose operator is not supported is returned with `supported == false` and a note
///       explaining why, rather than being dropped — so every input line is accounted for in the
///       output.
/// @note Each query's condition text is stored raw and resolved later, since resolving it needs
///       the clock and variable maps of the model it will be checked against.
std::vector<Query> load_queries(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw UppaalError("cannot open query file: " + path);
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
/// @note A @c false return is reported to the user as UNSUPPORTED. It never means "not
///       satisfied", which is why @p satisfied is left untouched.
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

// ============================ networks ======================================
namespace {


enum class SideKind { Const, Clock, ClockDiff, Int };
/// @brief One side of a network-level comparison, which may be a clock, a variable or a constant.
struct Side2 {
  SideKind kind;  ///< which of the three forms this side takes
  int a;          ///< clock or variable index, by @c kind
  int b;          ///< subtrahend clock for a difference, else unused
  long val;       ///< the value, when constant
};

/// @brief Parse one side of a network-level comparison.
/// @param raw the text
/// @param cm  clock name to index
/// @return the parsed side, classified as clock, variable or constant
Side2 parse_side2(const std::string& raw, const ClockMap& cm,
                  const std::unordered_map<std::string, int>& im) {
  std::string t = trim(raw);
  std::size_t minus = std::string::npos;
  for (std::size_t i = 1; i < t.size(); ++i) if (t[i] == '-') { minus = i; break; }
  if (minus != std::string::npos) {
    auto il = cm.find(trim(t.substr(0, minus)));
    auto ir = cm.find(trim(t.substr(minus + 1)));
    if (il != cm.end() && ir != cm.end()) return {SideKind::ClockDiff, il->second, ir->second, 0};
  }
  auto ic = cm.find(t); if (ic != cm.end()) return {SideKind::Clock, ic->second, -1, 0};
  auto ii = im.find(t); if (ii != im.end()) return {SideKind::Int, ii->second, -1, 0};
  long v = 0; try { v = std::stol(t); } catch (...) {}
  return {SideKind::Const, 0, -1, v};
}

/// @brief Parse one network-level atom into either a clock constraint or an integer atom.
/// @param atom the atom text
/// @param cm   clock name to index
/// @param[out] cc clock constraints are appended here
/// @param[out] ic integer atoms are appended here
/// @post At most one of @p cc and @p ic grows, which is how callers tell the two kinds apart —
///       ::build_pred relies on this to refuse a clock constraint in a liveness predicate.
void parse_net_atom(const std::string& atom, const ClockMap& cm,
                    const std::unordered_map<std::string, int>& im,
                    std::vector<Constraint>& cc, std::vector<IntAtom>& ic) {
  static const std::pair<const char*, int> ops[] = {{"<=", 0}, {">=", 1}, {"==", 2}, {"!=", 3}, {"<", 4}, {">", 5}};
  std::size_t pos = std::string::npos; int opi = -1, oplen = 0;
  for (auto& o : ops) {
    std::size_t q = atom.find(o.first);
    if (q != std::string::npos && (pos == std::string::npos || q < pos)) { pos = q; opi = o.second; oplen = (int)std::string(o.first).size(); }
  }
  if (pos == std::string::npos) return;
  Side2 l = parse_side2(atom.substr(0, pos), cm, im);
  Side2 r = parse_side2(atom.substr(pos + oplen), cm, im);
  auto flip = [](int o) { switch (o) { case 0: return 1; case 1: return 0; case 4: return 5; case 5: return 4; default: return o; } };
  if (l.kind == SideKind::Const && r.kind != SideKind::Const) { std::swap(l, r); opi = flip(opi); }
  if (r.kind != SideKind::Const) return;
  if (l.kind == SideKind::Clock || l.kind == SideKind::ClockDiff) {
    if (opi == 3) return;  // != is not a clock constraint
    Cmp cmp = opi == 0 ? Cmp::LE : opi == 1 ? Cmp::GE : opi == 2 ? Cmp::EQ : opi == 4 ? Cmp::LT : Cmp::GT;
    emit_clock_cmp(l.a, l.b, cmp, r.val, cc);
  } else if (l.kind == SideKind::Int) {
    IntOp io = opi == 0 ? IntOp::Le : opi == 1 ? IntOp::Ge : opi == 2 ? IntOp::Eq : opi == 3 ? IntOp::Ne : opi == 4 ? IntOp::Lt : IntOp::Gt;
    ic.push_back({l.a, io, r.val});
  }
}

/// @brief Parse an edge guard into its clock and integer parts.
/// @param text the guard text
/// @param cm   clock name to index
/// @param[out] cc clock constraints
/// @param[out] ic integer atoms
void parse_net_guard(const std::string& text, const ClockMap& cm,
                     const std::unordered_map<std::string, int>& im,
                     std::vector<Constraint>& cc, std::vector<IntAtom>& ic) {
  for (auto atom : split_conjuncts(text)) { atom = trim(atom); if (!atom.empty()) parse_net_atom(atom, cm, im, cc, ic); }
}

/// @brief Parse an edge update into clock resets and integer assignments.
/// @param text the update text
/// @param cm   clock name to index
/// @param[out] resets clock indices reset by this update
/// @param[out] ia     integer assignments
void parse_net_assign(const std::string& text, const ClockMap& cm,
                      const std::unordered_map<std::string, int>& im,
                      std::vector<int>& resets, std::vector<IntAsgn>& updates) {
  for (auto part : split_conjuncts(text)) {
    std::size_t p = part.find(":="); std::size_t adv = 2;
    if (p == std::string::npos) { p = part.find('='); adv = 1; }
    if (p == std::string::npos) continue;
    std::string lhs = trim(part.substr(0, p)), rhs = trim(part.substr(p + adv));
    auto cit = cm.find(lhs);
    if (cit != cm.end()) { if (rhs == "0") resets.push_back(cit->second); continue; }
    auto iit = im.find(lhs);
    if (iit != im.end()) { try { updates.push_back({iit->second, std::stol(rhs)}); } catch (...) {} }
  }
}

}  // namespace

/// @brief How many processes a model file composes, without building the network.
/// @param path path to the model file
/// @return the number of composed processes, or 1 when the model declares no system block
/// @note Used to choose between the single-automaton and network engines before paying for a full
///       parse.
int uppaal_process_count(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw UppaalError("cannot open model: " + path);
  std::stringstream ss; ss << in.rdbuf();
  Xml nta = XmlParser(ss.str()).parse();
  // The system list decides how many processes there are: one template may be
  // instantiated several times, and a template may not be in the system at all.
  if (const Xml* sys = nta.first("system")) {
    SystemDecl sd = parse_system(sys->text);
    if (sd.has_list) return (int)sd.order.size();
  }
  int n = 0;
  for (const auto& k : nta.kids) if (k.tag == "template") ++n;
  return n;
}

// ---- full modelling-language declarations ----------------------------------
//
// Everything the language can declare is parsed here: constants, typedefs, arrays of variables
// and of channels, user functions, and the plain scalar forms. An array becomes consecutive
// IntVar cells and a channel array becomes consecutive channels, so nothing downstream needs to
// know arrays exist.

/// @brief Symbol tables threaded through declaration parsing.
struct DeclCtx {
  SymTab st;                                           // constants, cells, arrays, functions
  std::unordered_map<std::string, std::pair<long, long>> tdefs;  // typedef -> [lo,hi]
  // channel-array name -> (first channel id, count), needed to expand a
  // state-dependent index such as `stop[tail()]!`.
  std::unordered_map<std::string, std::pair<int, int>> chan_arr;
};

/// @brief Split a declaration block into top-level statements.
/// @param src the block text, comments already stripped
/// @return one string per declaration
/// @note A function definition ends at its closing brace; every other declaration ends at a
///       semicolon. Nesting is tracked so a semicolon inside a function body does not split it.
std::vector<std::string> split_decls(const std::string& src) {
  std::vector<std::string> out;
  std::string cur;
  int depth = 0;
  for (std::size_t i = 0; i < src.size(); ++i) {
    const char c = src[i];
    if (c == '/' && i + 1 < src.size() && src[i + 1] == '/') {
      while (i < src.size() && src[i] != '\n') ++i;
      continue;
    }
    if (c == '/' && i + 1 < src.size() && src[i + 1] == '*') {
      i += 2;
      while (i + 1 < src.size() && !(src[i] == '*' && src[i + 1] == '/')) ++i;
      ++i;
      continue;
    }
    cur.push_back(c);
    if (c == '{') ++depth;
    else if (c == '}') {
      if (--depth == 0) { out.push_back(cur); cur.clear(); }
    } else if (c == ';' && depth == 0) {
      out.push_back(cur);
      cur.clear();
    }
  }
  if (!trim(cur).empty()) out.push_back(cur);
  return out;
}

/// @brief Does @p s begin with the whole word @p w?
/// @return @c true only when the match is followed by a non-identifier character, so that a
///         longer identifier beginning with @p w does not match
bool starts_word(const std::string& s, const char* w) {
  const std::size_t n = std::strlen(w);
  if (s.compare(0, n, w) != 0) return false;
  return s.size() == n || !(std::isalnum((unsigned char)s[n]) || s[n] == '_');
}

/// @brief Resolve a type token to the value range of one element.
/// @param tok the declaration text, starting at the type
/// @param[in,out] pos advanced past the type on success
/// @param dc  the symbol tables, for typedef names and constants
/// @return @c false if @p tok does not name a known type
/// @note An unannotated integer type takes the language's default range, which is wide. That
///       width is charged to the packed discrete key, so models relying on it can exceed the key
///       and be rejected; ptctlmc::set_narrow_ranges exists for exactly that case.
bool type_range(const std::string& tok, std::size_t* pos, const DeclCtx& dc,
                const std::vector<Func>& funcs, long* lo, long* hi) {
  std::string s = trim(tok);
  if (starts_word(s, "int") || starts_word(s, "bool")) {
    const bool isbool = starts_word(s, "bool");
    std::size_t p = isbool ? 4 : 3;
    while (p < s.size() && std::isspace((unsigned char)s[p])) ++p;
    if (!isbool && p < s.size() && s[p] == '[') {
      const std::size_t rb = s.find(']', p);
      if (rb == std::string::npos) return false;
      const std::string range = s.substr(p + 1, rb - p - 1);
      const std::size_t comma = range.find(',');
      if (comma == std::string::npos) return false;
      SymTab tmp = dc.st;
      *lo = const_eval(parse_expression(trim(range.substr(0, comma)), tmp, funcs), funcs);
      *hi = const_eval(parse_expression(trim(range.substr(comma + 1)), tmp, funcs), funcs);
      *pos = rb + 1;
      return true;
    }
    // UPPAAL's default int range; bool is [0,1].
    *lo = isbool ? 0 : -32768;
    *hi = isbool ? 1 : 32767;
    *pos = p;
    return true;
  }
  // typedef'd name
  std::size_t p = 0;
  while (p < s.size() && (std::isalnum((unsigned char)s[p]) || s[p] == '_')) ++p;
  auto it = dc.tdefs.find(s.substr(0, p));
  if (it == dc.tdefs.end()) return false;
  *lo = it->second.first;
  *hi = it->second.second;
  *pos = p;
  return true;
}

/// @brief Parse one declaration block, global or template-local.
/// @param text   the block text
/// @param prefix qualifies per-instance local state, so that two instances of a template each
///               declaring the same clock get distinct qualified names
/// @param[in,out] net the network being built, extended with the declared clocks, variables,
///                    channels and functions
void parse_decl_block(const std::string& text, const std::string& prefix, Network& net,
                      ClockMap& cm, std::unordered_map<std::string, int>& im,
                      std::unordered_map<std::string, int>& chm, DeclCtx& dc) {
  const std::vector<std::string> stmts = split_decls(text);

  // Pass 1: register function NAMES so bodies and guards can call them in any order.
  std::vector<std::pair<std::size_t, std::string>> fn_bodies;  // (func index, raw text)
  for (const std::string& raw : stmts) {
    const std::string s = trim(raw);
    if (s.empty() || s.back() != '}') continue;
    const std::size_t par = s.find('(');
    if (par == std::string::npos) continue;
    // "<rettype> <name> ( params ) { body }"
    std::string head = trim(s.substr(0, par));
    const std::size_t sp = head.find_last_of(" \t\n*&");
    if (sp == std::string::npos) continue;
    const std::string name = trim(head.substr(sp + 1));
    if (name.empty()) continue;
    SymTab::Sym sym;
    sym.kind = SymTab::Sym::Function;
    sym.idx = static_cast<int>(net.funcs.size());
    dc.st.put(name, sym);
    Func f;
    f.name = name;
    net.funcs.push_back(std::move(f));
    fn_bodies.emplace_back(net.funcs.size() - 1, s);
  }

  // Pass 2: everything else, in order.
  for (const std::string& raw : stmts) {
    std::string s = trim(raw);
    if (!s.empty() && s.back() == ';') s.pop_back();
    s = trim(s);
    if (s.empty()) continue;
    if (s.back() == '}') continue;  // function, handled below

    if (starts_word(s, "typedef")) {
      std::string rest = trim(s.substr(7));
      std::size_t p = 0;
      long lo = 0, hi = 0;
      if (!type_range(rest, &p, dc, net.funcs, &lo, &hi))
        throw UppaalError("unsupported typedef: " + s);
      const std::string name = trim(rest.substr(p));
      if (name.empty()) throw UppaalError("typedef without a name: " + s);
      dc.tdefs[name] = {lo, hi};
      continue;
    }

    bool is_const = false;
    if (starts_word(s, "const")) { is_const = true; s = trim(s.substr(5)); }

    if (starts_word(s, "clock")) {
      for (auto part : split_conjuncts(s.substr(5))) {
        part = trim(part);
        if (part.empty()) continue;
        const std::string nm = prefix + part;
        if (!cm.count(nm)) { cm[nm] = static_cast<int>(net.clock_names.size()); net.clock_names.push_back(nm); }
        if (!prefix.empty() && !cm.count(part)) cm[part] = cm[nm];  // unqualified inside the template
      }
      continue;
    }

    // `urgent` and `broadcast` are independent prefixes: `urgent chan`, `broadcast chan` and
    // `urgent broadcast chan` are all legal, and each keyword is recorded rather than peeled
    // and dropped.
    const bool urgent = starts_word(s, "urgent");
    std::string cs = urgent ? trim(s.substr(6)) : s;
    const bool bcast = starts_word(cs, "broadcast");
    if (bcast) cs = trim(cs.substr(9));
    if (starts_word(cs, "chan")) {
      for (auto part : split_conjuncts(cs.substr(4))) {
        part = trim(part);
        if (part.empty()) continue;
        long count = 1;
        std::string nm = part;
        const std::size_t br = part.find('[');
        if (br != std::string::npos) {
          const std::size_t rb = part.rfind(']');
          nm = trim(part.substr(0, br));
          SymTab tmp = dc.st;
          count = const_eval(parse_expression(trim(part.substr(br + 1, rb - br - 1)), tmp, net.funcs),
                             net.funcs);
          if (count <= 0) throw UppaalError("channel array with non-positive size: " + part);
        }
        const int first = static_cast<int>(net.channels.size());
        for (long i = 0; i < count; ++i) {
          const std::string full =
              (br == std::string::npos) ? (prefix + nm) : (prefix + nm + "[" + std::to_string(i) + "]");
          if (chm.count(full)) continue;
          chm[full] = static_cast<int>(net.channels.size());
          net.channels.push_back(full);
          net.chan_broadcast.push_back(bcast ? 1 : 0);
          net.chan_urgent.push_back(urgent ? 1 : 0);
          if (urgent) net.has_urgent_chan = true;
        }
        if (br != std::string::npos) dc.chan_arr[nm] = {first, static_cast<int>(count)};
      }
      continue;
    }

    // Variables: `<type> name[=init]` or `<type> name[size]`, comma-separated.
    std::size_t tp = 0;
    long lo = 0, hi = 0;
    if (!type_range(s, &tp, dc, net.funcs, &lo, &hi)) continue;  // not a declaration we handle
    for (auto part : split_conjuncts(s.substr(tp))) {
      part = trim(part);
      if (part.empty()) continue;
      std::string name = part, init_txt;
      const std::size_t eq = part.find('=');
      if (eq != std::string::npos) {
        name = trim(part.substr(0, eq));
        init_txt = trim(part.substr(eq + 1));
      }
      long count = 0;  // 0 => scalar
      const std::size_t br = name.find('[');
      if (br != std::string::npos) {
        const std::size_t rb = name.rfind(']');
        SymTab tmp = dc.st;
        count = const_eval(parse_expression(trim(name.substr(br + 1, rb - br - 1)), tmp, net.funcs),
                           net.funcs);
        name = trim(name.substr(0, br));
        if (count <= 0) throw UppaalError("array with non-positive size: " + part);
      }
      if (name.empty()) continue;

      if (is_const && count == 0) {  // `const int N = 8;`
        SymTab tmp = dc.st;
        SymTab::Sym sym;
        sym.kind = SymTab::Sym::Const;
        sym.val = init_txt.empty() ? 0 : const_eval(parse_expression(init_txt, tmp, net.funcs), net.funcs);
        dc.st.put(name, sym);
        continue;
      }

      long init = 0;
      if (!init_txt.empty() && init_txt[0] != '{') {
        SymTab tmp = dc.st;
        try { init = const_eval(parse_expression(init_txt, tmp, net.funcs), net.funcs); } catch (...) {}
      }
      const std::string full = prefix + name;
      if (count == 0) {
        if (im.count(full)) continue;
        SymTab::Sym sym;
        sym.kind = SymTab::Sym::Cell;
        sym.idx = static_cast<int>(net.intvars.size());
        dc.st.put(name, sym);
        im[full] = sym.idx;
        net.intvars.push_back({full, lo, hi, init});
      } else {
        SymTab::Sym sym;
        sym.kind = SymTab::Sym::Array;
        sym.idx = static_cast<int>(net.intvars.size());
        sym.len = static_cast<int>(count);
        dc.st.put(name, sym);
        for (long i = 0; i < count; ++i) {
          const std::string cell = full + "[" + std::to_string(i) + "]";
          im[cell] = static_cast<int>(net.intvars.size());
          net.intvars.push_back({cell, lo, hi, init});
        }
      }
    }
  }

  // Pass 3: function bodies, now that every name (including other functions) resolves.
  for (const auto& fb : fn_bodies) {
    const std::string& s = fb.second;
    const std::size_t par = s.find('(');
    const std::size_t rpar = s.find(')', par);
    const std::size_t lb = s.find('{', rpar);
    const std::size_t rb = s.rfind('}');
    Func& f = net.funcs[fb.first];
    const std::size_t mark = dc.st.mark();
    int nloc = 0;
    for (auto p : split_conjuncts(s.substr(par + 1, rpar - par - 1))) {
      p = trim(p);
      if (p.empty()) continue;
      if (starts_word(p, "const")) p = trim(p.substr(5));
      std::size_t tp = 0;
      long plo = 0, phi = 0;
      if (!type_range(p, &tp, dc, net.funcs, &plo, &phi))
        throw UppaalError("unsupported parameter in function '" + f.name + "': " + p);
      const std::string pname = trim(p.substr(tp));
      if (pname.find('&') != std::string::npos)
        throw UppaalError("reference parameters are not supported: " + p);
      SymTab::Sym sym;
      sym.kind = SymTab::Sym::Local;
      sym.idx = nloc++;
      dc.st.put(pname, sym);
    }
    f.nparams = nloc;
    f.body = parse_statements(s.substr(lb + 1, rb - lb - 1), dc.st, net.funcs, &nloc);
    f.nlocals = nloc;
    dc.st.release(mark);
  }
}

/// @brief Split a guard or update into its top-level parts.
/// @param s      the text
/// @param andand split on conjunction (a guard) rather than on commas (an update)
/// @return the parts, in order
/// @note Separators inside brackets or parentheses are ignored, so an indexed access or a call
///       with several arguments is not split mid-expression.
std::vector<std::string> split_parts(const std::string& s, bool andand) {
  std::vector<std::string> out;
  std::string cur;
  int depth = 0;
  for (std::size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (c == '(' || c == '[') ++depth;
    else if (c == ')' || c == ']') --depth;
    if (depth == 0) {
      if (andand && c == '&' && i + 1 < s.size() && s[i + 1] == '&') {
        out.push_back(cur); cur.clear(); ++i; continue;
      }
      if (!andand && c == ',') { out.push_back(cur); cur.clear(); continue; }
    }
    cur.push_back(c);
  }
  if (!trim(cur).empty()) out.push_back(cur);
  return out;
}

/// @brief Does @p part mention a clock, i.e. belong on the zone side rather than the integer side?
/// @param part the expression text
/// @param cm   clock name to index
bool mentions_clock(const std::string& part, const ClockMap& cm) {
  for (const auto& kv : cm) {
    const std::string& n = kv.first;
    std::size_t p = 0;
    while ((p = part.find(n, p)) != std::string::npos) {
      const bool lok = p == 0 || !(std::isalnum((unsigned char)part[p - 1]) || part[p - 1] == '_');
      const std::size_t e = p + n.size();
      const bool rok = e >= part.size() || !(std::isalnum((unsigned char)part[e]) || part[e] == '_');
      if (lok && rok) return true;
      p += n.size();
    }
  }
  return false;
}

/// @brief Can this part be represented as a simple integer atom or assignment?
///
/// Simple means `var OP const` or `var := const`. Anything involving an index, a call or
/// arithmetic is not, and falls back to the general expression path.
///
/// @param part the expression text
/// @param im   variable name to index
/// @return @c true iff the simple form suffices
/// @note The distinction is a representation choice, not a semantic one: the general path
///       evaluates the same thing, but carries an expression tree and so cannot be compared as
///       cheaply. Keeping the common case simple is what lets most edges avoid it entirely.
bool part_is_simple(const std::string& part, const std::unordered_map<std::string, int>& im) {
  if (part.find('[') != std::string::npos || part.find('(') != std::string::npos) return false;
  if (part.find('+') != std::string::npos || part.find('-') != std::string::npos ||
      part.find('*') != std::string::npos || part.find('/') != std::string::npos ||
      part.find('%') != std::string::npos)
    return false;
  static const char* ops[] = {"==", "!=", "<=", ">=", "<", ">", ":=", "=", nullptr};
  for (int i = 0; ops[i]; ++i) {
    const std::size_t p = part.find(ops[i]);
    if (p == std::string::npos) continue;
    const std::string lhs = trim(part.substr(0, p));
    std::string rhs = trim(part.substr(p + std::strlen(ops[i])));
    if (!im.count(lhs)) return false;
    if (rhs.empty()) return false;
    if (rhs[0] == '-') rhs = rhs.substr(1);
    for (char c : rhs)
      if (!std::isdigit((unsigned char)c)) return false;
    return true;
  }
  return false;
}

/// @brief Conjoin two expressions, treating a null operand as absent.
/// @return @p b when @p a is null and vice versa, otherwise their conjunction
ExprP and_expr(ExprP a, ExprP b) {
  if (!a) return b;
  if (!b) return a;
  auto e = std::make_shared<Expr>();
  e->op = EOp::And;
  e->kids = {std::move(a), std::move(b)};
  return e;
}

/// @brief Reject clock guards on edges that synchronise over an urgent channel.
///
/// An urgent channel stops time exactly while a synchronisation on it is enabled. Enabledness is
/// therefore consulted at every state, and the whole design rests on it being decidable from the
/// discrete state alone: locations plus the integer valuation. A clock guard on such an edge
/// would make enabledness a property of the ZONE, and worse, one that can switch on partway
/// through a delay — the delay would then have to be clipped at that instant, a zone subtraction
/// whose result is in general non-convex.
///
/// UPPAAL imposes exactly this restriction on its modelling language, so no model that UPPAAL
/// itself accepts is refused here. Rejecting is the point: the alternative is the silent
/// downgrade this replaces, which verified the model against semantics it did not declare.
///
/// @param net the finalised network
/// @throws UppaalError naming the process, edge and channel of the first offending edge
static void check_urgent_chan_guards(const Network& net) {
  if (!net.has_urgent_chan) return;
  for (const auto& proc : net.procs)
    for (const auto& e : proc.edges) {
      if (e.sync_chan < 0 || e.sync_chan >= (int)net.chan_urgent.size()) continue;
      if (!net.chan_urgent[e.sync_chan] || e.clock_guard.empty()) continue;
      throw UppaalError(
          "clock guard on an edge synchronising over the urgent channel '" +
          net.channels[e.sync_chan] + "' (process " + proc.name + ", " +
          proc.locations[e.src].name + " -> " + proc.locations[e.tgt].name +
          "). Urgent-channel synchronisation may not be clock-guarded; see KNOWN_LIMITATION.md.");
    }
}

/// @brief Load a model file as a network of timed automata.
/// @param path path to the model file
/// @return the network, fully finalised
/// @throws UppaalError if the file cannot be opened or the model cannot be represented
/// @post Every finalisation analysis has run — bounds, per-location bounds, active clocks and
///       integer liveness — so the network is ready to explore.
Network load_uppaal_network(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw UppaalError("cannot open model: " + path);
  std::stringstream ss; ss << in.rdbuf();
  Xml nta = XmlParser(ss.str()).parse();
  if (nta.tag != "nta") throw UppaalError("not an <nta> model");

  Network net;
  ClockMap cm;
  std::unordered_map<std::string, int> im, chm;
  DeclCtx dc;
  net.clock_names.push_back("0");
  if (const Xml* d = nta.first("declaration")) parse_decl_block(d->text, "", net, cm, im, chm, dc);

  // Index templates by name, keeping document order.
  std::vector<const Xml*> tmpl_order;
  std::unordered_map<std::string, const Xml*> tmpl_by_name;
  for (const auto& k : nta.kids) {
    if (k.tag != "template") continue;
    tmpl_order.push_back(&k);
    tmpl_by_name[k.first("name") ? trim(k.first("name")->text) : "P"] = &k;
  }

  SystemDecl sd;
  if (const Xml* sys = nta.first("system")) sd = parse_system(sys->text);

  // Resolve the process list: (instance name, defining template). Without a
  // `system ...;` line, fall back to one process per template in document order.
  // A process entry is (instance name, template, parameter binding). A template
  // with a parameter over a bounded type -- `Train(const id_t id)` -- is
  // instantiated ONCE PER VALUE, which is what `system Train, Gate;` means in
  // UPPAAL. Treating it as a single process is exactly the silent
  // under-modelling that made viking12..20 masquerade as viking10.
  struct PEntry {
    std::string name;
    const Xml* tmpl;
    bool has_param = false;
    std::string param;
    long value = 0;
  };
  std::vector<PEntry> plist;

  auto template_param = [&](const Xml* t, std::string* pname, long* plo, long* phi) -> bool {
    const Xml* p = t->first("parameter");
    if (!p || trim(p->text).empty()) return false;
    std::string s = trim(p->text);
    if (s.find(',') != std::string::npos)
      throw UppaalError("templates with more than one parameter are not supported: " + s);
    if (s.find('&') != std::string::npos)
      throw UppaalError("reference parameters are not supported: " + s);
    if (starts_word(s, "const")) s = trim(s.substr(5));
    std::size_t tp = 0;
    if (!type_range(s, &tp, dc, net.funcs, plo, phi))
      throw UppaalError("unsupported template parameter type: " + s);
    *pname = trim(s.substr(tp));
    return true;
  };

  auto add_instances = [&](const std::string& base, const Xml* t) {
    std::string pname;
    long plo = 0, phi = 0;
    if (!template_param(t, &pname, &plo, &phi)) {
      plist.push_back({base, t, false, "", 0});
      return;
    }
    for (long v = plo; v <= phi; ++v)
      plist.push_back({base + std::to_string(v), t, true, pname, v});
  };

  if (sd.has_list) {
    for (const auto& nm : sd.order) {
      auto ii = sd.inst.find(nm);
      const std::string tn = (ii != sd.inst.end()) ? ii->second : nm;
      auto ti = tmpl_by_name.find(tn);
      if (ti == tmpl_by_name.end())
        throw UppaalError("system declaration names unknown process or template: " + nm);
      add_instances(nm, ti->second);
    }
  } else {
    for (const Xml* t : tmpl_order)
      add_instances(t->first("name") ? trim(t->first("name")->text) : "P", t);
  }

  for (const auto& pr : plist) {
    const Xml& tmpl = *pr.tmpl;
    // Per-instance scope. Template-local declarations are parsed PER INSTANCE
    // with the instance name as a prefix, so N instances of `Train` (each
    // declaring `clock x`) get N distinct clocks Train0.x .. TrainN-1.x --- v1
    // refused this case outright. The template parameter is a CONSTANT in this
    // scope, which is what makes `appr[id]` fold to a concrete channel.
    DeclCtx ldc = dc;
    if (pr.has_param) {
      SymTab::Sym sym;
      sym.kind = SymTab::Sym::Const;
      sym.val = pr.value;
      ldc.st.put(pr.param, sym);
    }
    ClockMap lcm = cm;
    if (const Xml* d = tmpl.first("declaration"))
      parse_decl_block(d->text, pr.name + ".", net, lcm, im, chm, ldc);

    NProc proc;
    proc.name = pr.name;
    std::unordered_map<std::string, int> lid;
    for (const auto& k : tmpl.kids) {
      if (k.tag != "location") continue;
      Location loc;
      loc.name = k.first("name") ? k.first("name")->text : k.attr("id");
      loc.committed = k.first("committed") != nullptr;
      loc.urgent = k.first("urgent") != nullptr;
      for (const auto& lb : k.kids)
        if (lb.tag == "label" && lb.attr("kind") == "invariant") {
          std::vector<IntAtom> ic; parse_net_guard(lb.text, lcm, im, loc.invariant, ic);
        }
      lid[k.attr("id")] = (int)proc.locations.size();
      proc.locations.push_back(std::move(loc));
    }
    if (const Xml* in0 = tmpl.first("init")) { auto it = lid.find(in0->attr("ref")); proc.initial = it != lid.end() ? it->second : 0; }
    for (const auto& k : tmpl.kids) {
      if (k.tag != "transition") continue;

      // `select e : id_t` binds e non-deterministically; the edge exists once per
      // value. Skipping it would silently drop transitions.
      std::string sel_name;
      long sel_lo = 0, sel_hi = -1;
      for (const auto& lb : k.kids) {
        if (lb.tag != "label" || lb.attr("kind") != "select") continue;
        const std::string t = trim(lb.text);
        const std::size_t colon = t.find(':');
        if (colon == std::string::npos) throw UppaalError("malformed select: " + t);
        if (t.find(',') != std::string::npos)
          throw UppaalError("multiple select bindings are not supported: " + t);
        sel_name = trim(t.substr(0, colon));
        std::string ty = trim(t.substr(colon + 1));
        std::size_t tp = 0;
        if (!type_range(ty, &tp, ldc, net.funcs, &sel_lo, &sel_hi))
          throw UppaalError("unsupported select type: " + ty);
      }

      for (long sv = sel_lo; sv <= (sel_name.empty() ? sel_lo : sel_hi); ++sv) {
        DeclCtx edc = ldc;
        if (!sel_name.empty()) {
          SymTab::Sym sym;
          sym.kind = SymTab::Sym::Const;
          sym.val = sv;
          edc.st.put(sel_name, sym);
        }

        NEdge base_edge;
        base_edge.src = lid[k.first("source")->attr("ref")];
        base_edge.tgt = lid[k.first("target")->attr("ref")];
        // (channel id, extra guard) alternatives; one entry unless the channel
        // index is state-dependent.
        std::vector<std::pair<int, ExprP>> sync_opts;
        int sync_dir = 0;

        for (const auto& lb : k.kids) {
          if (lb.tag != "label") continue;
          const std::string kind = lb.attr("kind");

          if (kind == "guard" || kind == "assignment") {
            const bool is_guard = kind == "guard";
            std::vector<std::string> plain, complex_;
            for (const auto& part : split_parts(lb.text, is_guard)) {
              if (trim(part).empty()) continue;
              if (mentions_clock(part, lcm)) {  // zone side, unchanged
                if (is_guard) {
                  std::vector<IntAtom> ic;
                  parse_net_guard(part, lcm, im, base_edge.clock_guard, ic);
                } else {
                  std::vector<IntAsgn> up;
                  parse_net_assign(part, lcm, im, base_edge.clock_resets, up);
                }
                continue;
              }
              (part_is_simple(part, im) ? plain : complex_).push_back(part);
            }
            // If every non-clock part is expressible the v1 way, keep using it so
            // the static analyses see the same atoms they always have.
            if (complex_.empty()) {
              for (const auto& part : plain) {
                if (is_guard) {
                  std::vector<Constraint> cc;
                  parse_net_guard(part, lcm, im, cc, base_edge.int_guard);
                } else {
                  std::vector<int> rs;
                  parse_net_assign(part, lcm, im, rs, base_edge.int_updates);
                }
              }
            } else {
              std::vector<std::string> all = plain;
              all.insert(all.end(), complex_.begin(), complex_.end());
              SymTab tmp = edc.st;
              if (is_guard) {
                for (const auto& part : all)
                  base_edge.gen_guard =
                      and_expr(base_edge.gen_guard, parse_expression(part, tmp, net.funcs));
              } else {
                std::string joined;
                for (const auto& part : all) joined += part + ";";
                int nloc = 0;
                base_edge.gen_update = parse_statements(joined, tmp, net.funcs, &nloc);
                base_edge.gen_locals = nloc;
              }
            }
          } else if (kind == "synchronisation") {
            const std::string t = trim(lb.text);
            if (t.empty() || (t.back() != '!' && t.back() != '?')) continue;
            sync_dir = (t.back() == '!') ? 1 : -1;
            const std::string raw = trim(t.substr(0, t.size() - 1));
            const std::size_t br = raw.find('[');
            if (br == std::string::npos) {
              auto it = chm.find(raw);
              if (it == chm.end()) throw UppaalError("unknown channel: " + raw);
              sync_opts.emplace_back(it->second, nullptr);
              continue;
            }
            const std::string cbase = trim(raw.substr(0, br));
            const std::string itxt = trim(raw.substr(br + 1, raw.rfind(']') - br - 1));
            SymTab tmp = edc.st;
            ExprP idx = parse_expression(itxt, tmp, net.funcs);
            auto ca = edc.chan_arr.find(cbase);
            if (ca == edc.chan_arr.end()) throw UppaalError("unknown channel array: " + cbase);
            long fixed = 0;
            bool is_const = true;
            try { fixed = const_eval(idx, net.funcs); } catch (const ParseError&) { is_const = false; }
            if (is_const) {
              if (fixed < 0 || fixed >= ca->second.second)
                throw UppaalError("channel index out of range: " + raw);
              sync_opts.emplace_back(ca->second.first + static_cast<int>(fixed), nullptr);
            } else {
              // State-dependent index (`stop[tail()]!`, `go[front()]!`): the
              // channel is not known statically. Expand into one edge per index
              // guarded by `idx == v` -- sound, and it keeps the synchronisation
              // machinery free of runtime channel resolution.
              for (int v = 0; v < ca->second.second; ++v) {
                auto eq = std::make_shared<Expr>();
                eq->op = EOp::Eq;
                auto konst = std::make_shared<Expr>();
                konst->op = EOp::Const;
                konst->val = v;
                eq->kids = {idx, konst};
                sync_opts.emplace_back(ca->second.first + v, eq);
              }
            }
          }
        }

        if (sync_opts.empty()) sync_opts.emplace_back(-1, nullptr);
        for (const auto& opt : sync_opts) {
          NEdge e = base_edge;
          e.sync_chan = opt.first;
          e.sync_dir = (opt.first >= 0) ? sync_dir : 0;
          if (opt.second) e.gen_guard = and_expr(e.gen_guard, opt.second);
          proc.edges.push_back(std::move(e));
        }
        if (sel_name.empty()) break;
      }
    }
    net.procs.push_back(std::move(proc));
  }
  net.nclocks = (int)net.clock_names.size() - 1;
  // Narrow declared integer ranges to the values actually reachable BEFORE the discrete key is
  // laid out, since the key spends bits(hi-lo+1) per cell. Opt-in (--narrow) while it is new:
  // it is an over-approximation and therefore sound, but it changes the packing of every model
  // with integer state, so the parity gate decides when it becomes the default.
  check_urgent_chan_guards(net);
  if (narrow_ranges_on()) narrow_int_ranges(net);
  net.compute_bounds();
  net.compute_loc_bounds();
  net.compute_active();
  net.compute_int_live();
  return net;
}

/// @brief Resolve and decide one query against a network, routing it to the right engine.
///
/// Reachability and safety go to ptctlmc::reachable_net, deadlock to
/// ptctlmc::deadlock_reachable, and the liveness forms to the greatest-fixpoint engine.
///
/// @param net the network
/// @param q   the query, with its condition text unresolved
/// @param[out] satisfied receives the verdict; written only on @c true
/// @param backend which expansion backend to run
/// @param[out] stats optional statistics; unchanged when null
/// @return @c false if the query is unsupported or does not resolve against @p net
///
/// @note A @c false return is reported as UNSUPPORTED and never means "not satisfied", which is
///       why @p satisfied is left untouched. This is the path a liveness predicate mentioning a
///       clock takes; see KNOWN_LIMITATION.md §3.
bool check_net_query(const Network& net, const Query& q, bool* satisfied, Backend backend,
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
  // Conjunction of LOCATION/INT atoms -> DiscPred (liveness; clock atoms rejected).
  // Parse a list of atoms into `out`, optionally negating each one as it is pushed.
  //
  // Negating atom by atom is what lets a DISJUNCTION be accepted: `a \/ b` is `!(!a /\ !b)`, and
  // the engine represents a conjunction and the negation of a conjunction, so a disjunction of
  // single atoms lands in the second class. It does not represent anything wider, which is why an
  // atom whose own negation is not a single atom -- a clock equality, since `x == 5` parses to
  // TWO constraints and negates to a disjunction -- is refused here rather than approximated.
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
  // Anything that is not a plain conjunction is handled by putting the NEGATION of the predicate
  // into disjunctive normal form. Each resulting term is a conjunction, hence a discrete guard
  // plus a convex clock region, which is the only shape the engine ever has to represent; the
  // non-convexity lives in the union of terms and is absorbed by intersecting the per-term delay
  // results. See ptctlmc::detail::DelayAvoidSet.
  //
  // Grammar: or := and ('||' and)* ; and := unary ('&&' unary)* ; unary := '!' unary | '(' or ')'
  //          | atom. `and`/`or`/`not` spell the operators too, and ',' reads as '&&'.
  struct BExpr {
    enum Kind { Atom, Not, And, Or } kind = Atom;
    std::string atom;
    std::vector<BExpr> kids;
  };
  // A hand-rolled parser rather than reuse of parse_expression: that one yields an arithmetic
  // ExprP over integer state, while what is needed here is the boolean SKELETON, with the leaves
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

  // Build the predicate a liveness query must MAINTAIN, from the query text.
  //
  // `want_negated` says whether the caller wants `!text` (as `A<>` does) rather than `text`.
  // Either way the result is a conjunction plus a flag, because those are the two shapes the
  // engine represents; a disjunction is turned into the negation of a conjunction of negated
  // atoms, which flips the flag. Mixing `&&` and `||` in one predicate is refused: the result
  // would be neither shape.
  // Split `psi` into an arbitrary boolean over DISCRETE atoms and a CONJUNCTION of clock atoms.
  //
  // This is the fragment the backward engine decides. The clock conjunction is convex, so it
  // becomes an invariant of a restricted network and the run stays inside it by construction; the
  // discrete part is constant across a vertex, so any boolean structure in it is a vertex filter
  // and needs no convexity at all. What is refused is a clock atom under a disjunction or a
  // negation, because that is a non-convex region the run must stay inside, and the backward
  // operator for it -- Pre through a delay routed around an obstacle -- does not exist yet.
  //
  // @param[out] cc  the clock conjunction
  // @param[out] ndnf  DNF of the NEGATED discrete part, i.e. where V_psi does NOT hold
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
    // must stay inside a non-convex region. Its NEGATION goes into DNF, giving convex terms with
    // discrete guards that detail::DelayAvoidSet routes each delay around -- forward through
    // delay_avoiding, backward through pre_avoiding.
    for (const BExpr* k : nonconvex)
      for (const auto& term : to_dnf(*k, /*neg=*/true)) {
        // A negated literal can itself be a disjunction -- `x == 3` parses to TWO constraints, so
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
    NetTarget tgt;
    if (!build_target(q.note, tgt)) return false;
    *satisfied = reachable_net(net, tgt, backend, stats);
    return true;
  }
  if (q.kind == Query::EG) {  // E[] phi
    std::vector<Constraint> cc;
    DiscPred nq;
    nq.is_dnf = true;   // empty means psi's discrete part never fails, not that it always does
    if (!split_psi(q.note, cc, nq.forbid)) return false;
    *satisfied = exists_invariant_run(net, cc, nq, stats, &nq.forbid);
    return true;
  }
  if (q.kind == Query::AF) {  // A<> phi = !E[] !phi
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
      NetTarget tgt;
      if (!build_target(inner, tgt)) return false;
      *satisfied = !reachable_net(net, tgt, backend, stats);
      return true;
    }
    // General:  A[] (a1 & a2 & ...) = AND_i !E<>(!ai)
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
    // conjunction of psi and the DNF of !psi, which is q itself -- the region the run must AVOID
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
    // THE BACKWARD GREATEST FIXPOINT IS THE ENGINE. Forward SCC search closes cycles that
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

}  // namespace ptctlmc
