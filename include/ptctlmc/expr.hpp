// expr.hpp — the UPPAAL modelling-language subset that guards, updates and
// declarations are written in: integer expressions, arrays, and user functions.
//
// Why this exists. v1 of the network engine only understood `var OP const`
// guards and `var := const` updates (network.hpp's IntAtom/IntAsgn). That is
// enough for Fischer and the viking suite, but not for idiomatic UPPAAL models:
// the classic train-gate keeps a FIFO of waiting trains in `int list[N+1]` and
// manipulates it with `enqueue()`/`dequeue()`, guards edges with
// `e == front()`, and synchronises on `stop[tail()]`. None of that is
// expressible as var-op-const, so such models were rejected outright.
//
// Design notes:
//  * Arrays are FLATTENED at parse time into consecutive IntVar cells
//    (`list[0] .. list[N]`), so the packed discrete state stays exactly what the
//    engine already indexes -- an array element is just another int cell. An
//    Index node therefore carries the base cell plus the number of cells, and
//    resolves to `base + eval(index)` with a bounds check.
//  * Functions are interpreted, not inlined: `dequeue()` contains a loop whose
//    trip count depends on `len`, so it has to run against the concrete state.
//  * Locals (function parameters and function-body declarations) live in a
//    separate frame indexed at parse time, so evaluation needs no name lookup.
//  * Everything is side-effect-free EXCEPT via Stmt, which is what an edge
//    update executes. Guards evaluate Expr only, so a guard can call a function
//    (`front()`) but a function that writes globals would be a modelling error;
//    we simply evaluate it against a scratch copy (see eval_guard).
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ptctlmc {

enum class EOp {
  Const,   // literal
  Var,     // scalar int cell (global state)
  Local,   // local/parameter slot
  Index,   // array element: cell `var + kids[0]`, bounds-checked against arr_len
  Add, Sub, Mul, Div, Mod,
  Eq, Ne, Lt, Le, Gt, Ge,
  And, Or, Not, Neg,
  Cond,    // kids[0] ? kids[1] : kids[2]
  Call,    // user function `fn` applied to kids
};

struct Expr;
using ExprP = std::shared_ptr<Expr>;

struct Expr {
  EOp op = EOp::Const;
  long val = 0;      // Const
  int var = -1;      // Var: cell index | Local: slot | Index: base cell
  int arr_len = 0;   // Index: cell count (bounds check)
  int fn = -1;       // Call: function index
  std::vector<ExprP> kids;
};

enum class SOp {
  Nop,
  Assign,   // cell `var` (scalar) or `var + idx` (array) := rhs
  AssignLocal,
  If,       // cond ? kids[0] : kids[1]
  While,    // while (cond) kids[0]
  Block,    // kids in order
  Return,   // rhs (may be null)
};

struct Stmt;
using StmtP = std::shared_ptr<Stmt>;

struct Stmt {
  SOp op = SOp::Nop;
  int var = -1;      // Assign target cell / AssignLocal slot
  int arr_len = 0;   // >0 when the target is an array element
  ExprP idx;         // array index (null for scalars)
  ExprP rhs;
  ExprP cond;
  std::vector<StmtP> kids;
};

// A user-declared function. Parameters occupy the first `nparams` local slots.
struct Func {
  std::string name;
  int nparams = 0;
  int nlocals = 0;   // total frame size (params + body declarations)
  StmtP body;
};

// Evaluation state: the model's flat int cells plus one call frame.
struct EvalCtx {
  long* cells = nullptr;              // global int state (may be read-only for guards)
  std::vector<long> locals;
  const std::vector<Func>* funcs = nullptr;
  bool fault = false;                 // set on out-of-range index / div by zero
  int depth = 0;                      // recursion guard
};

long eval(const Expr& e, EvalCtx& cx);
// Executes `s`; sets *returned/*retval when a `return` fires.
void exec(const Stmt& s, EvalCtx& cx, bool* returned, long* retval);

// Convenience: evaluate a boolean guard against a state without mutating it.
bool eval_guard(const ExprP& e, const long* cells, int ncells,
                const std::vector<Func>& funcs);

// ---- parsing ---------------------------------------------------------------
// Names visible while parsing a declaration block, guard, update or function
// body. `Const` covers both `const int N = 8;` and things fixed at instantiation
// time -- a template parameter (`Train(const id_t id)`) and a `select e : id_t`
// binding are constants for the edge copy being generated, which is what lets
// `appr[id]` and `stop[tail()]` resolve to a concrete channel.
struct SymTab {
  struct Sym {
    enum Kind { Const, Cell, Array, Local, Function } kind = Const;
    long val = 0;  // Const: value
    int idx = 0;   // Cell: cell | Array: base cell | Local: slot | Function: index
    int len = 0;   // Array: number of cells
  };
  std::vector<std::pair<std::string, Sym>> syms;  // innermost scope last
  const Sym* find(const std::string& name) const;
  void put(const std::string& name, Sym s) { syms.emplace_back(name, s); }
  std::size_t mark() const { return syms.size(); }
  void release(std::size_t m) { syms.resize(m); }
};

struct ParseError {
  std::string msg;
};

// Integer-valued/boolean expression. Throws ParseError.
ExprP parse_expression(const std::string& text, SymTab& st,
                       const std::vector<Func>& funcs);

// A statement sequence (an edge update, or a function body). `nlocals` receives
// the frame size needed. Throws ParseError.
StmtP parse_statements(const std::string& text, SymTab& st,
                       const std::vector<Func>& funcs, int* nlocals);

// Constant-fold an expression that must not depend on state. Throws ParseError
// if it does. Used for array sizes, typedef ranges and `const` initialisers.
long const_eval(const ExprP& e, const std::vector<Func>& funcs);

}  // namespace ptctlmc
