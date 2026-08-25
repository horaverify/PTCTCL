#include "ptctlmc/uppaal.hpp"

#include "ptctlmc/netreach.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "ptctlmc/reach.hpp"

namespace ptctlmc {
namespace {

// ---- a minimal XML tree (enough for UPPAAL's flat-1_6 format) ----------------
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
std::string trim(const std::string& s) {
  std::size_t a = 0, b = s.size();
  while (a < b && std::isspace((unsigned char)s[a])) ++a;
  while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
  return s.substr(a, b - a);
}

// True if c may appear inside an identifier. Used to require that a textual "and" operator is a whole
// token: without checking BOTH sides, an identifier merely *ending* in "and" (land, command, rand) was
// split mid-name, silently dropping the declaration and corrupting the model with no diagnostic.
static inline bool ident_char(unsigned char c) { return std::isalnum(c) != 0 || c == '_'; }

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

using ClockMap = std::unordered_map<std::string, int>;

enum class Cmp { LE, LT, GE, GT, EQ };

// One side of a comparison: a constant, a single clock, or a clock difference.
struct Side { bool is_const = false; long val = 0; int a = 0, b = -1; };

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

void emit_clock_cmp(int a, int b /*-1 single*/, Cmp op, long k, std::vector<Constraint>& out) {
  switch (op) {
    case Cmp::LE: out.push_back(b < 0 ? Constraint{a, 0, le((int)k)} : Constraint{a, b, le((int)k)}); break;
    case Cmp::LT: out.push_back(b < 0 ? Constraint{a, 0, lt((int)k)} : Constraint{a, b, lt((int)k)}); break;
    case Cmp::GE: out.push_back(b < 0 ? Constraint{0, a, le((int)-k)} : Constraint{b, a, le((int)-k)}); break;
    case Cmp::GT: out.push_back(b < 0 ? Constraint{0, a, lt((int)-k)} : Constraint{b, a, lt((int)-k)}); break;
    case Cmp::EQ: emit_clock_cmp(a, b, Cmp::LE, k, out); emit_clock_cmp(a, b, Cmp::GE, k, out); break;
  }
}

Cmp flip(Cmp c) {
  switch (c) { case Cmp::LE: return Cmp::GE; case Cmp::LT: return Cmp::GT;
               case Cmp::GE: return Cmp::LE; case Cmp::GT: return Cmp::LT; default: return Cmp::EQ; }
}

// Parse "A op B" into constraint(s); returns false if not a clock atom.
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

std::vector<Constraint> parse_constraints(const std::string& expr, const ClockMap& cm) {
  std::vector<Constraint> out;
  for (const auto& atom : split_conjuncts(expr)) {
    if (atom.empty()) continue;
    parse_atom(atom, cm, out);  // non-clock atoms (int guards) silently ignored in v1
  }
  return out;
}

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

// ---- the <system> block -----------------------------------------------------
// It does two things, and ignoring either silently under-models the system:
//   Viking11 = Viking10();            instantiate a template under a new name
//   system Viking1, ..., Viking11;    which processes actually form the network
// A template that is never listed is not part of the system at all.
struct SystemDecl {
  bool has_list = false;
  std::vector<std::string> order;                     // names in `system ...;`
  std::unordered_map<std::string, std::string> inst;  // instance name -> template
};

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
struct Side2 { SideKind kind; int a; int b; long val; };

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

void parse_net_guard(const std::string& text, const ClockMap& cm,
                     const std::unordered_map<std::string, int>& im,
                     std::vector<Constraint>& cc, std::vector<IntAtom>& ic) {
  for (auto atom : split_conjuncts(text)) { atom = trim(atom); if (!atom.empty()) parse_net_atom(atom, cm, im, cc, ic); }
}

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
// Everything the modelling language can declare is parsed here: `const int N = 8;`,
// `typedef int[0,N-1] id_t;`, arrays (`int list[N+1];`, `chan appr[N];`) and user
// functions, as well as the plain `int[lo,hi] x;` / `chan a, b;` forms. Arrays become
// consecutive IntVar cells and channel arrays become consecutive channels, so nothing
// downstream has to know they exist. (Earlier standalone collect_ints/collect_channels
// scanners handled only the plain forms and were subsumed by this; they are gone.)

struct DeclCtx {
  SymTab st;                                           // constants, cells, arrays, functions
  std::unordered_map<std::string, std::pair<long, long>> tdefs;  // typedef -> [lo,hi]
  // channel-array name -> (first channel id, count), needed to expand a
  // state-dependent index such as `stop[tail()]!`.
  std::unordered_map<std::string, std::pair<int, int>> chan_arr;
};

// Split a declaration block into top-level statements. A function definition
// ends at its closing brace; everything else at ';'.
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

bool starts_word(const std::string& s, const char* w) {
  const std::size_t n = std::strlen(w);
  if (s.compare(0, n, w) != 0) return false;
  return s.size() == n || !(std::isalnum((unsigned char)s[n]) || s[n] == '_');
}

// `int[a,b]` / a typedef name / plain `int`|`bool` -> element range.
// Returns false if `tok` is not a known type.
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

// Parse one declaration block (global or template-local). `prefix` disambiguates
// per-instance local state: two Train instances each declaring `clock x` become
// `Train0.x` and `Train1.x`.
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

// Split on top-level `&&` (guards) or `,` (assignments), ignoring separators
// inside brackets or parentheses so `list[i+1]` and `f(a,b)` stay intact.
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

// True if `part` mentions a clock, i.e. belongs on the zone side.
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

// True if the v1 fast path can represent this part: `intvar OP const` /
// `intvar := const`. Anything with an index, a call, or arithmetic is not.
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

ExprP and_expr(ExprP a, ExprP b) {
  if (!a) return b;
  if (!b) return a;
  auto e = std::make_shared<Expr>();
  e->op = EOp::And;
  e->kids = {std::move(a), std::move(b)};
  return e;
}

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
  if (narrow_ranges_on()) narrow_int_ranges(net);
  net.compute_bounds();
  net.compute_loc_bounds();
  net.compute_active();
  net.compute_int_live();
  return net;
}

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
  auto build_pred = [&](const std::string& text, DiscPred& out) -> bool {
    for (auto atom : split_conjuncts(text)) {
      atom = trim(atom);
      if (atom.empty() || atom == "true") continue;
      std::vector<Constraint> cc; std::vector<IntAtom> ic;
      parse_net_atom(atom, cm, im, cc, ic);
      if (!cc.empty()) return false;  // clock constraint in a liveness predicate: unsupported
      if (!ic.empty()) { out.ints.insert(out.ints.end(), ic.begin(), ic.end()); continue; }
      bool eq = true;
      std::string a = atom;
      if (!a.empty() && a[0] == '!') { eq = false; a = trim(a.substr(1)); }
      else if (a.compare(0, 4, "not ") == 0) { eq = false; a = trim(a.substr(4)); }
      std::size_t dot = a.rfind('.');
      std::string proc = dot == std::string::npos ? "" : a.substr(0, dot);
      std::string loc = dot == std::string::npos ? a : a.substr(dot + 1);
      int pi, li;
      if (!net.resolve_location(proc, loc, &pi, &li)) return false;
      out.locs.push_back({pi, li, eq});
    }
    return true;
  };
  // Negate a SINGLE-atom predicate (A<> phi = !E[] !phi).
  auto negate_single = [&](const DiscPred& phi, DiscPred& out) -> bool {
    if (phi.locs.size() + phi.ints.size() != 1) return false;
    if (phi.locs.size() == 1) {
      out.locs.push_back({phi.locs[0].proc, phi.locs[0].loc, !phi.locs[0].equal});
      return true;
    }
    IntAtom a = phi.ints[0]; IntOp n = a.op;
    switch (a.op) {
      case IntOp::Eq: n = IntOp::Ne; break; case IntOp::Ne: n = IntOp::Eq; break;
      case IntOp::Lt: n = IntOp::Ge; break; case IntOp::Ge: n = IntOp::Lt; break;
      case IntOp::Le: n = IntOp::Gt; break; case IntOp::Gt: n = IntOp::Le; break;
    }
    out.ints.push_back({a.var, n, a.val});
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
    DiscPred phi;
    if (!build_pred(q.note, phi)) return false;
    *satisfied = exists_invariant_run(net, phi, backend, stats);
    return true;
  }
  if (q.kind == Query::AF) {  // A<> phi = !E[] !phi   (single-atom phi)
    DiscPred phi, nphi;
    if (!build_pred(q.note, phi) || !negate_single(phi, nphi)) return false;
    *satisfied = !exists_invariant_run(net, nphi, backend, stats);
    return true;
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
    if (!build_pred(q.lhs, p) || !build_pred(q.note, qq)) return false;
    // THE BACKWARD GREATEST FIXPOINT IS THE SHIPPING ENGINE. leads_to_net's forward SCC
    // search closes cycles that subsumption invented; the backward fixpoint only ever reads a
    // target's W as an upper bound, so subsumption is harmless to it. PTCTLMC_LIVENESS picks
    // an arm for A/B: `otf` (default), `gfp` (the whole-graph form), `net` (the old search).
    const char* arm = std::getenv("PTCTLMC_LIVENESS");
    const std::string a = arm ? arm : "otf";
    if (a == "net") { *satisfied = leads_to_net(net, p, qq, backend, stats); return true; }
    GfpStats gs;
    *satisfied = (a == "gfp") ? leads_to_gfp(net, p, qq, &gs)
                              : leads_to_otf(net, p, qq, std::getenv("PTCTLMC_GFPMODE"), &gs);
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
