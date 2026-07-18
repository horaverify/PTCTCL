#include "ptctlmc/uppaal.hpp"

#include <cctype>
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

std::vector<std::string> split_conjuncts(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (std::size_t i = 0; i < s.size();) {
    if (s.compare(i, 2, "&&") == 0) { out.push_back(trim(cur)); cur.clear(); i += 2; }
    else if (s.compare(i, 3, "and") == 0 && (i + 3 >= s.size() || !std::isalnum((unsigned char)s[i + 3]))) { out.push_back(trim(cur)); cur.clear(); i += 3; }
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

void collect_ints(const std::string& decl, std::unordered_map<std::string, int>& im,
                  std::vector<IntVar>& vars) {
  std::size_t p = 0;
  while ((p = decl.find("int", p)) != std::string::npos) {
    const bool wb = (p == 0 || !(std::isalnum((unsigned char)decl[p - 1]) || decl[p - 1] == '_')) &&
                    (p + 3 >= decl.size() || !(std::isalnum((unsigned char)decl[p + 3]) || decl[p + 3] == '_'));
    if (!wb) { p += 3; continue; }
    std::size_t q = p + 3;
    while (q < decl.size() && std::isspace((unsigned char)decl[q])) ++q;
    if (q < decl.size() && decl[q] == '[') { q = decl.find(']', q); if (q == std::string::npos) break; ++q; }
    std::size_t e = decl.find(';', q);
    std::string list = decl.substr(q, (e == std::string::npos ? decl.size() : e) - q);
    for (auto part : split_conjuncts(list)) {
      part = trim(part);
      long init = 0;
      std::string name = part;
      std::size_t eq = part.find('=');
      if (eq != std::string::npos) {
        name = trim(part.substr(0, eq));
        try { init = std::stol(trim(part.substr(eq + 1))); } catch (...) {}
      }
      name = trim(name);
      if (!name.empty() && (std::isalpha((unsigned char)name[0]) || name[0] == '_') && !im.count(name)) {
        im[name] = (int)vars.size();
        vars.push_back({name, 0, 0, init});
      }
    }
    p = (e == std::string::npos) ? decl.size() : e + 1;
  }
}

void collect_channels(const std::string& decl, std::unordered_map<std::string, int>& chm,
                      std::vector<std::string>& names) {
  std::size_t p = 0;
  while ((p = decl.find("chan", p)) != std::string::npos) {
    const bool wb = (p == 0 || !(std::isalnum((unsigned char)decl[p - 1]) || decl[p - 1] == '_')) &&
                    (p + 4 >= decl.size() || !(std::isalnum((unsigned char)decl[p + 4]) || decl[p + 4] == '_'));
    if (!wb) { p += 4; continue; }
    std::size_t e = decl.find(';', p);
    std::string list = decl.substr(p + 4, (e == std::string::npos ? decl.size() : e) - (p + 4));
    for (auto part : split_conjuncts(list)) {
      part = trim(part);
      if (!part.empty() && (std::isalpha((unsigned char)part[0]) || part[0] == '_') && !chm.count(part)) {
        chm[part] = (int)names.size();
        names.push_back(part);
      }
    }
    p = (e == std::string::npos) ? decl.size() : e + 1;
  }
}

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

int uppaal_template_count(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw UppaalError("cannot open model: " + path);
  std::stringstream ss; ss << in.rdbuf();
  Xml nta = XmlParser(ss.str()).parse();
  int n = 0;
  for (const auto& k : nta.kids) if (k.tag == "template") ++n;
  return n;
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
  net.clock_names.push_back("0");
  if (const Xml* d = nta.first("declaration")) { collect_clocks(d->text, cm, net.clock_names); collect_ints(d->text, im, net.intvars); collect_channels(d->text, chm, net.channels); }

  for (const auto& tmpl : nta.kids) {
    if (tmpl.tag != "template") continue;
    NProc proc;
    proc.name = tmpl.first("name") ? tmpl.first("name")->text : "P";
    if (const Xml* d = tmpl.first("declaration")) { collect_clocks(d->text, cm, net.clock_names); collect_ints(d->text, im, net.intvars); collect_channels(d->text, chm, net.channels); }
    std::unordered_map<std::string, int> lid;
    for (const auto& k : tmpl.kids) {
      if (k.tag != "location") continue;
      Location loc;
      loc.name = k.first("name") ? k.first("name")->text : k.attr("id");
      loc.committed = k.first("committed") != nullptr;
      loc.urgent = k.first("urgent") != nullptr;
      for (const auto& lb : k.kids)
        if (lb.tag == "label" && lb.attr("kind") == "invariant") {
          std::vector<IntAtom> ic; parse_net_guard(lb.text, cm, im, loc.invariant, ic);
        }
      lid[k.attr("id")] = (int)proc.locations.size();
      proc.locations.push_back(std::move(loc));
    }
    if (const Xml* in0 = tmpl.first("init")) { auto it = lid.find(in0->attr("ref")); proc.initial = it != lid.end() ? it->second : 0; }
    for (const auto& k : tmpl.kids) {
      if (k.tag != "transition") continue;
      NEdge e;
      e.src = lid[k.first("source")->attr("ref")];
      e.tgt = lid[k.first("target")->attr("ref")];
      for (const auto& lb : k.kids) {
        if (lb.tag != "label") continue;
        const std::string kind = lb.attr("kind");
        if (kind == "guard") parse_net_guard(lb.text, cm, im, e.clock_guard, e.int_guard);
        else if (kind == "assignment") parse_net_assign(lb.text, cm, im, e.clock_resets, e.int_updates);
        else if (kind == "synchronisation") {
          std::string t = trim(lb.text);
          if (!t.empty() && (t.back() == '!' || t.back() == '?')) {
            auto it = chm.find(trim(t.substr(0, t.size() - 1)));
            if (it != chm.end()) { e.sync_chan = it->second; e.sync_dir = (t.back() == '!') ? 1 : -1; }
          }
        }
      }
      proc.edges.push_back(std::move(e));
    }
    net.procs.push_back(std::move(proc));
  }
  net.nclocks = (int)net.clock_names.size() - 1;
  net.compute_bounds();
  net.compute_loc_bounds();
  net.compute_active();
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
  return false;  // leads-to over networks not supported yet
}

}  // namespace ptctlmc
