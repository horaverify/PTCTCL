// uexpr.cpp — tokenizer and recursive-descent parser for the UPPAAL modelling
// subset (see expr.hpp). Written from the language's documented syntax and
// semantics; it deliberately shares no code with UTAP, which is LGPL, while
// HORA is MIT and links neither UTAP nor UDBM.
//
// Supported: integer expressions with the usual C precedence, arrays (flattened
// to consecutive cells), user functions with locals/if/while/for/return,
// `const`, `typedef int[a,b]`, and the assignment sugar (`=`, `+=`, `-=`, `++`,
// `--`) that real models use.
//
// Deliberately NOT supported (rejected loudly rather than mis-modelled): structs,
// pointers/references, clocks inside functions, and recursion. A model using
// them raises ParseError instead of being silently under-verified -- the failure
// mode that let viking12..20 masquerade as viking10 before the <system> fix.
#include "hora/expression.hpp"

#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace hora {

const SymTab::Sym* SymTab::find(const std::string& name) const {
  for (auto it = syms.rbegin(); it != syms.rend(); ++it)  // innermost wins
    if (it->first == name) return &it->second;
  return nullptr;
}

namespace {

struct Tok {
  enum Kind { End, Num, Ident, Punct } kind = End;
  std::string s;
  long num = 0;
};

class Lexer {
 public:
  explicit Lexer(const std::string& src) : s_(src) { next(); }
  const Tok& peek() const { return t_; }
  Tok take() { Tok t = t_; next(); return t; }
  bool is(const char* p) const { return t_.kind == Tok::Punct && t_.s == p; }
  bool is_ident(const char* p) const { return t_.kind == Tok::Ident && t_.s == p; }
  bool accept(const char* p) { if (is(p)) { next(); return true; } return false; }
  bool accept_ident(const char* p) { if (is_ident(p)) { next(); return true; } return false; }
  void expect(const char* p) {
    if (!accept(p)) throw ParseError{std::string("expected '") + p + "' near '" + t_.s + "'"};
  }
  bool done() const { return t_.kind == Tok::End; }

 private:
  void next() {
    while (i_ < s_.size() && (std::isspace((unsigned char)s_[i_]) || s_[i_] == '\n')) ++i_;
    // comments
    if (i_ + 1 < s_.size() && s_[i_] == '/' && s_[i_ + 1] == '/') {
      while (i_ < s_.size() && s_[i_] != '\n') ++i_;
      return next();
    }
    if (i_ + 1 < s_.size() && s_[i_] == '/' && s_[i_ + 1] == '*') {
      i_ += 2;
      while (i_ + 1 < s_.size() && !(s_[i_] == '*' && s_[i_ + 1] == '/')) ++i_;
      i_ = std::min(i_ + 2, s_.size());
      return next();
    }
    if (i_ >= s_.size()) { t_ = Tok{}; return; }

    const char c = s_[i_];
    if (std::isdigit((unsigned char)c)) {
      std::size_t j = i_;
      while (j < s_.size() && std::isdigit((unsigned char)s_[j])) ++j;
      t_ = Tok{Tok::Num, s_.substr(i_, j - i_), std::strtol(s_.c_str() + i_, nullptr, 10)};
      i_ = j;
      return;
    }
    if (std::isalpha((unsigned char)c) || c == '_') {
      std::size_t j = i_;
      while (j < s_.size() && (std::isalnum((unsigned char)s_[j]) || s_[j] == '_')) ++j;
      t_ = Tok{Tok::Ident, s_.substr(i_, j - i_), 0};
      i_ = j;
      return;
    }
    // two-character operators first
    static const char* kTwo[] = {"==", "!=", "<=", ">=", "&&", "||", "++", "--",
                                 "+=", "-=", "*=", "/=", nullptr};
    for (int k = 0; kTwo[k]; ++k)
      if (s_.compare(i_, 2, kTwo[k]) == 0) {
        t_ = Tok{Tok::Punct, kTwo[k], 0};
        i_ += 2;
        return;
      }
    t_ = Tok{Tok::Punct, std::string(1, c), 0};
    ++i_;
  }

  const std::string& s_;
  std::size_t i_ = 0;
  Tok t_;
};

ExprP mk(EOp op) { auto e = std::make_shared<Expr>(); e->op = op; return e; }
ExprP mk_const(long v) { auto e = mk(EOp::Const); e->val = v; return e; }
ExprP mk_bin(EOp op, ExprP a, ExprP b) {
  auto e = mk(op);
  e->kids = {std::move(a), std::move(b)};
  return e;
}

class Parser {
 public:
  Parser(Lexer& lx, SymTab& st, const std::vector<Func>& funcs, int* nlocals)
      : lx_(lx), st_(st), funcs_(funcs), nlocals_(nlocals) {}

  // ---- expressions (precedence climbing) ----
  ExprP expr() { return ternary(); }

  ExprP ternary() {
    ExprP c = or_();
    if (lx_.accept("?")) {
      ExprP a = expr();
      lx_.expect(":");
      ExprP b = expr();
      auto e = mk(EOp::Cond);
      e->kids = {std::move(c), std::move(a), std::move(b)};
      return e;
    }
    return c;
  }
  ExprP or_() {
    ExprP a = and_();
    while (lx_.accept("||")) a = mk_bin(EOp::Or, a, and_());
    return a;
  }
  ExprP and_() {
    ExprP a = equality();
    while (lx_.accept("&&")) a = mk_bin(EOp::And, a, equality());
    return a;
  }
  ExprP equality() {
    ExprP a = relational();
    for (;;) {
      if (lx_.accept("==")) a = mk_bin(EOp::Eq, a, relational());
      else if (lx_.accept("!=")) a = mk_bin(EOp::Ne, a, relational());
      else return a;
    }
  }
  ExprP relational() {
    ExprP a = additive();
    for (;;) {
      if (lx_.accept("<=")) a = mk_bin(EOp::Le, a, additive());
      else if (lx_.accept(">=")) a = mk_bin(EOp::Ge, a, additive());
      else if (lx_.accept("<")) a = mk_bin(EOp::Lt, a, additive());
      else if (lx_.accept(">")) a = mk_bin(EOp::Gt, a, additive());
      else return a;
    }
  }
  ExprP additive() {
    ExprP a = multiplicative();
    for (;;) {
      if (lx_.accept("+")) a = mk_bin(EOp::Add, a, multiplicative());
      else if (lx_.accept("-")) a = mk_bin(EOp::Sub, a, multiplicative());
      else return a;
    }
  }
  ExprP multiplicative() {
    ExprP a = unary();
    for (;;) {
      if (lx_.accept("*")) a = mk_bin(EOp::Mul, a, unary());
      else if (lx_.accept("/")) a = mk_bin(EOp::Div, a, unary());
      else if (lx_.accept("%")) a = mk_bin(EOp::Mod, a, unary());
      else return a;
    }
  }
  ExprP unary() {
    if (lx_.accept("-")) { auto e = mk(EOp::Neg); e->kids = {unary()}; return e; }
    if (lx_.accept("!")) { auto e = mk(EOp::Not); e->kids = {unary()}; return e; }
    if (lx_.accept("+")) return unary();
    return primary();
  }

  // `x++` / `x--` appearing inside an expression -- the idiom that matters is
  // `list[len++] = element` in train-gate's enqueue(). The expression yields the
  // current value; the increment is queued and runs after the enclosing
  // statement, so `list[len] = element` is followed by `len = len + 1`.
  ExprP postfix(ExprP value, const SymTab::Sym& sym) {
    const bool inc = lx_.is("++");
    if (!inc && !lx_.is("--")) return value;
    lx_.take();
    auto s = std::make_shared<Stmt>();
    s->op = (sym.kind == SymTab::Sym::Local) ? SOp::AssignLocal : SOp::Assign;
    s->var = sym.idx;
    s->rhs = mk_bin(inc ? EOp::Add : EOp::Sub, value, mk_const(1));
    pending_.push_back(s);
    return value;
  }

  ExprP primary() {
    if (lx_.peek().kind == Tok::Num) return mk_const(lx_.take().num);
    if (lx_.accept("(")) { ExprP e = expr(); lx_.expect(")"); return e; }
    if (lx_.peek().kind != Tok::Ident)
      throw ParseError{"unexpected token '" + lx_.peek().s + "'"};

    const std::string name = lx_.take().s;
    if (name == "true") return mk_const(1);
    if (name == "false") return mk_const(0);

    const SymTab::Sym* sym = st_.find(name);
    if (!sym) throw ParseError{"unknown identifier '" + name + "'"};

    if (lx_.is("(")) {  // function call
      if (sym->kind != SymTab::Sym::Function)
        throw ParseError{"'" + name + "' is not a function"};
      lx_.expect("(");
      auto e = mk(EOp::Call);
      e->fn = sym->idx;
      if (!lx_.accept(")")) {
        for (;;) {
          e->kids.push_back(expr());
          if (lx_.accept(")")) break;
          lx_.expect(",");
        }
      }
      return e;
    }
    if (lx_.is("[")) {  // array element
      if (sym->kind != SymTab::Sym::Array)
        throw ParseError{"'" + name + "' is not an array"};
      lx_.expect("[");
      ExprP i = expr();
      lx_.expect("]");
      auto e = mk(EOp::Index);
      e->var = sym->idx;
      e->arr_len = sym->len;
      e->kids = {std::move(i)};
      return e;
    }
    switch (sym->kind) {
      case SymTab::Sym::Const: return mk_const(sym->val);
      case SymTab::Sym::Cell: { auto e = mk(EOp::Var); e->var = sym->idx; return postfix(e, *sym); }
      case SymTab::Sym::Local: { auto e = mk(EOp::Local); e->var = sym->idx; return postfix(e, *sym); }
      case SymTab::Sym::Array:
        throw ParseError{"array '" + name + "' used without an index"};
      case SymTab::Sym::Function:
        throw ParseError{"function '" + name + "' used without a call"};
    }
    throw ParseError{"unresolved '" + name + "'"};
  }

  // ---- statements ----
  StmtP block_or_stmt() {
    if (lx_.is("{")) return block();
    auto b = std::make_shared<Stmt>();
    b->op = SOp::Block;
    b->kids.push_back(statement());
    return b;
  }

  StmtP block() {
    lx_.expect("{");
    const std::size_t m = st_.mark();
    auto b = std::make_shared<Stmt>();
    b->op = SOp::Block;
    while (!lx_.accept("}")) {
      if (lx_.done()) throw ParseError{"unterminated block"};
      b->kids.push_back(statement());
    }
    st_.release(m);
    return b;
  }

  // Wraps statement_() so that any post-increments queued while parsing it are
  // appended immediately after, in source order.
  StmtP statement() {
    const std::size_t m = pending_.size();
    StmtP s = statement_();
    if (pending_.size() == m) return s;
    auto b = std::make_shared<Stmt>();
    b->op = SOp::Block;
    b->kids.push_back(std::move(s));
    for (std::size_t i = m; i < pending_.size(); ++i) b->kids.push_back(pending_[i]);
    pending_.resize(m);
    return b;
  }

  StmtP statement_() {
    if (lx_.is("{")) return block();
    if (lx_.accept(";")) { auto s = std::make_shared<Stmt>(); s->op = SOp::Nop; return s; }

    if (lx_.accept_ident("if")) {
      auto s = std::make_shared<Stmt>();
      s->op = SOp::If;
      lx_.expect("(");
      s->cond = expr();
      lx_.expect(")");
      s->kids.push_back(block_or_stmt());
      if (lx_.accept_ident("else")) s->kids.push_back(block_or_stmt());
      return s;
    }
    if (lx_.accept_ident("while")) {
      auto s = std::make_shared<Stmt>();
      s->op = SOp::While;
      lx_.expect("(");
      s->cond = expr();
      lx_.expect(")");
      s->kids.push_back(block_or_stmt());
      return s;
    }
    if (lx_.accept_ident("for")) {
      // Desugar `for (init; cond; step) body` into `{ init; while (cond) { body; step; } }`
      lx_.expect("(");
      const std::size_t m = st_.mark();
      auto outer = std::make_shared<Stmt>();
      outer->op = SOp::Block;
      if (!lx_.is(";")) outer->kids.push_back(simple_statement());
      lx_.expect(";");
      auto w = std::make_shared<Stmt>();
      w->op = SOp::While;
      w->cond = lx_.is(";") ? mk_const(1) : expr();
      lx_.expect(";");
      StmtP step;
      if (!lx_.is(")")) step = simple_statement();
      lx_.expect(")");
      auto body = std::make_shared<Stmt>();
      body->op = SOp::Block;
      body->kids.push_back(block_or_stmt());
      if (step) body->kids.push_back(step);
      w->kids.push_back(body);
      outer->kids.push_back(w);
      st_.release(m);
      return outer;
    }
    if (lx_.accept_ident("return")) {
      auto s = std::make_shared<Stmt>();
      s->op = SOp::Return;
      if (!lx_.is(";")) s->rhs = expr();
      lx_.accept(";");
      return s;
    }
    StmtP s = simple_statement();
    lx_.accept(";");
    return s;
  }

  // local declaration, or an assignment / increment expression-statement
  StmtP simple_statement() {
    if (lx_.is_ident("int") || lx_.is_ident("bool")) {
      lx_.take();
      // A local declaration. Range syntax `int[0,N] x` is accepted and the
      // bounds ignored: locals are scratch, only stored cells need a range.
      if (lx_.accept("[")) { expr(); lx_.expect(","); expr(); lx_.expect("]"); }
      const std::string name = lx_.take().s;
      const int slot = (*nlocals_)++;
      SymTab::Sym sym;
      sym.kind = SymTab::Sym::Local;
      sym.idx = slot;
      st_.put(name, sym);
      auto s = std::make_shared<Stmt>();
      s->op = SOp::AssignLocal;
      s->var = slot;
      s->rhs = lx_.accept("=") ? expr() : mk_const(0);
      return s;
    }
    return assignment();
  }

  StmtP assignment() {
    if (lx_.peek().kind != Tok::Ident)
      throw ParseError{"expected an assignment, got '" + lx_.peek().s + "'"};
    const std::string name = lx_.take().s;
    const SymTab::Sym* sym = st_.find(name);
    if (!sym) throw ParseError{"unknown identifier '" + name + "'"};

    // A bare call used as a statement: `enqueue(e)`, `dequeue()`.
    if (lx_.is("(") && sym->kind == SymTab::Sym::Function) {
      lx_.expect("(");
      auto call = mk(EOp::Call);
      call->fn = sym->idx;
      if (!lx_.accept(")")) {
        for (;;) {
          call->kids.push_back(expr());
          if (lx_.accept(")")) break;
          lx_.expect(",");
        }
      }
      // Evaluate for its side effects: assign into a throwaway local slot.
      auto s = std::make_shared<Stmt>();
      s->op = SOp::AssignLocal;
      s->var = (*nlocals_)++;
      s->rhs = call;
      return s;
    }

    ExprP index;
    if (lx_.accept("[")) {
      if (sym->kind != SymTab::Sym::Array) throw ParseError{"'" + name + "' is not an array"};
      index = expr();
      lx_.expect("]");
    }

    const bool is_local = sym->kind == SymTab::Sym::Local;
    if (!is_local && sym->kind != SymTab::Sym::Cell && sym->kind != SymTab::Sym::Array)
      throw ParseError{"cannot assign to '" + name + "'"};

    auto target = [&]() -> ExprP {
      if (index) {
        auto e = mk(EOp::Index);
        e->var = sym->idx;
        e->arr_len = sym->len;
        e->kids = {index};
        return e;
      }
      if (is_local) { auto e = mk(EOp::Local); e->var = sym->idx; return e; }
      auto e = mk(EOp::Var);
      e->var = sym->idx;
      return e;
    };

    auto make = [&](ExprP rhs) {
      auto s = std::make_shared<Stmt>();
      s->op = is_local ? SOp::AssignLocal : SOp::Assign;
      s->var = sym->idx;
      s->arr_len = index ? sym->len : 0;
      s->idx = index;
      s->rhs = std::move(rhs);
      return s;
    };

    if (lx_.accept("=")) return make(expr());
    if (lx_.accept("+=")) return make(mk_bin(EOp::Add, target(), expr()));
    if (lx_.accept("-=")) return make(mk_bin(EOp::Sub, target(), expr()));
    if (lx_.accept("*=")) return make(mk_bin(EOp::Mul, target(), expr()));
    if (lx_.accept("/=")) return make(mk_bin(EOp::Div, target(), expr()));
    if (lx_.accept("++")) return make(mk_bin(EOp::Add, target(), mk_const(1)));
    if (lx_.accept("--")) return make(mk_bin(EOp::Sub, target(), mk_const(1)));
    throw ParseError{"expected an assignment operator after '" + name + "'"};
  }

 private:
  Lexer& lx_;
  SymTab& st_;
  const std::vector<Func>& funcs_;
  int* nlocals_;
  std::vector<StmtP> pending_;  // post-increments queued by postfix()
};

}  // namespace

ExprP parse_expression(const std::string& text, SymTab& st,
                       const std::vector<Func>& funcs) {
  Lexer lx(text);
  int dummy = 0;
  Parser p(lx, st, funcs, &dummy);
  ExprP e = p.expr();
  if (!lx.done()) throw ParseError{"trailing input near '" + lx.peek().s + "'"};
  return e;
}

StmtP parse_statements(const std::string& text, SymTab& st,
                       const std::vector<Func>& funcs, int* nlocals) {
  Lexer lx(text);
  Parser p(lx, st, funcs, nlocals);
  auto b = std::make_shared<Stmt>();
  b->op = SOp::Block;
  const std::size_t m = st.mark();
  while (!lx.done()) {
    // UPPAAL separates several updates on one edge with ','
    if (lx.accept(",") || lx.accept(";")) continue;
    b->kids.push_back(p.statement());
  }
  st.release(m);
  return b;
}

long const_eval(const ExprP& e, const std::vector<Func>& funcs) {
  if (!e) return 0;
  EvalCtx cx;
  cx.funcs = &funcs;
  // No cells: any reference to state makes this a non-constant expression.
  static long none = 0;
  cx.cells = &none;
  // A Var/Index node would read `none`; detect that structurally instead.
  struct V {
    static bool state_dep(const Expr& x) {
      if (x.op == EOp::Var || x.op == EOp::Index || x.op == EOp::Call) return true;
      for (const auto& k : x.kids)
        if (k && state_dep(*k)) return true;
      return false;
    }
  };
  if (V::state_dep(*e)) throw ParseError{"expected a constant expression"};
  const long v = eval(*e, cx);
  if (cx.fault) throw ParseError{"bad constant expression"};
  return v;
}

}  // namespace hora
