/// @file expr.cpp
/// @brief Evaluator for the modelling-language expression and statement subset.
///
/// Guards evaluate an hora::Expr; edge updates execute a hora::Stmt. Both may call user
/// functions, which are interpreted against the live integer cells rather than constant-folded,
/// since a function may loop over an array and depend on the current state.
///
/// @par Fault handling
/// Faults — an index out of range, division by zero, a runaway loop or exceeded call depth — do
/// **not** throw. They set hora::EvalCtx::fault and yield 0, leaving the decision to the
/// caller. A faulting guard is treated as false and a faulting update abandons the edge, so a
/// malformed model loses transitions rather than silently producing a wrong verdict.
///
/// @warning Every recursive entry point must check @c fault on return and propagate it. A fault
///          that is observed but not propagated turns an abandoned computation into a committed
///          value of 0.
#include "hora/expression.hpp"

#include <algorithm>

namespace hora {

namespace {
/// Maximum user-function call depth. The language's functions are not recursive, so this is a
/// backstop against a malformed model rather than a semantic limit.
constexpr int kMaxDepth = 64;
}

/// @brief Evaluate an expression.
/// @param e  the expression
/// @param[in,out] cx the evaluation context: integer cells, locals, function table and fault flag
/// @return the value, or 0 if a fault occurred
/// @post On a fault, `cx.fault` is set and the result is meaningless; callers must test it.
/// @note A called function shares the caller's integer cells, so a function with side effects can
///       write them. Callers that must not be perturbed evaluate against a copy; see
///       ::eval_guard.
long eval(const Expr& e, EvalCtx& cx) {
  switch (e.op) {
    case EOp::Const: return e.val;
    case EOp::Var:   return cx.cells[e.var];
    case EOp::Local:
      return (e.var >= 0 && e.var < static_cast<int>(cx.locals.size())) ? cx.locals[e.var] : 0;
    case EOp::Index: {
      const long i = eval(*e.kids[0], cx);
      if (i < 0 || i >= e.arr_len) { cx.fault = true; return 0; }
      return cx.cells[e.var + i];
    }
    case EOp::Neg: return -eval(*e.kids[0], cx);
    case EOp::Not: return eval(*e.kids[0], cx) ? 0 : 1;
    // Conjunction and disjunction short-circuit, which is load-bearing rather than an
    // optimisation: a guard may test a bound before indexing with it, and evaluating the right
    // side regardless would fault on exactly the states the left side exists to exclude.
    case EOp::And: return (eval(*e.kids[0], cx) && eval(*e.kids[1], cx)) ? 1 : 0;
    case EOp::Or:  return (eval(*e.kids[0], cx) || eval(*e.kids[1], cx)) ? 1 : 0;
    case EOp::Cond: return eval(*e.kids[0], cx) ? eval(*e.kids[1], cx) : eval(*e.kids[2], cx);
    case EOp::Call: {
      if (e.fn < 0 || !cx.funcs || e.fn >= static_cast<int>(cx.funcs->size())) {
        cx.fault = true; return 0;
      }
      const Func& f = (*cx.funcs)[e.fn];
      if (cx.depth + 1 > kMaxDepth) { cx.fault = true; return 0; }
      EvalCtx sub;
      sub.cells = cx.cells;             // functions see and may write the same cells
      sub.funcs = cx.funcs;
      sub.depth = cx.depth + 1;
      sub.locals.assign(static_cast<std::size_t>(std::max(f.nlocals, f.nparams)), 0);
      for (std::size_t i = 0; i < e.kids.size() && i < static_cast<std::size_t>(f.nparams); ++i)
        sub.locals[i] = eval(*e.kids[i], cx);
      bool returned = false;
      long ret = 0;
      if (f.body) exec(*f.body, sub, &returned, &ret);
      if (sub.fault) cx.fault = true;
      return ret;
    }
    default: break;
  }

  const long a = eval(*e.kids[0], cx);
  const long b = eval(*e.kids[1], cx);
  switch (e.op) {
    case EOp::Add: return a + b;
    case EOp::Sub: return a - b;
    case EOp::Mul: return a * b;
    case EOp::Div: if (b == 0) { cx.fault = true; return 0; } return a / b;
    case EOp::Mod: if (b == 0) { cx.fault = true; return 0; } return a % b;
    case EOp::Eq:  return a == b;
    case EOp::Ne:  return a != b;
    case EOp::Lt:  return a <  b;
    case EOp::Le:  return a <= b;
    case EOp::Gt:  return a >  b;
    case EOp::Ge:  return a >= b;
    default: cx.fault = true; return 0;
  }
}

/// @brief Execute a statement.
/// @param s  the statement
/// @param[in,out] cx the evaluation context
/// @param[in,out] returned set once a return statement executes; execution then unwinds
/// @param[out] retval the returned value, written only when @p returned becomes true
/// @post Execution stops at the first fault, leaving the cells partially updated — which is why
///       callers run updates against a copy and commit only on success.
void exec(const Stmt& s, EvalCtx& cx, bool* returned, long* retval) {
  if (*returned || cx.fault) return;
  switch (s.op) {
    case SOp::Nop: return;
    case SOp::Block:
      for (const auto& k : s.kids) {
        exec(*k, cx, returned, retval);
        if (*returned || cx.fault) return;
      }
      return;
    case SOp::Assign: {
      const long v = s.rhs ? eval(*s.rhs, cx) : 0;
      if (cx.fault) return;
      int cell = s.var;
      if (s.arr_len > 0) {
        const long i = s.idx ? eval(*s.idx, cx) : 0;
        if (cx.fault) return;
        if (i < 0 || i >= s.arr_len) { cx.fault = true; return; }
        cell = s.var + static_cast<int>(i);
      }
      cx.cells[cell] = v;
      return;
    }
    case SOp::AssignLocal: {
      const long v = s.rhs ? eval(*s.rhs, cx) : 0;
      if (cx.fault) return;
      if (s.var >= 0 && s.var < static_cast<int>(cx.locals.size())) cx.locals[s.var] = v;
      return;
    }
    case SOp::If: {
      const long c = s.cond ? eval(*s.cond, cx) : 0;
      if (cx.fault) return;
      if (c) { if (!s.kids.empty() && s.kids[0]) exec(*s.kids[0], cx, returned, retval); }
      else   { if (s.kids.size() > 1 && s.kids[1]) exec(*s.kids[1], cx, returned, retval); }
      return;
    }
    case SOp::While: {
      // Bounded so that a modelling error, or a loop condition that never falsifies, cannot hang
      // the checker. Exceeding the cap raises a fault rather than truncating the loop silently.
      // The arrays this language iterates are small, so the cap is unreachable for a well-formed
      // model.
      for (long iter = 0; iter < 1'000'000; ++iter) {
        const long c = s.cond ? eval(*s.cond, cx) : 0;
        if (cx.fault || !c) return;
        if (!s.kids.empty() && s.kids[0]) exec(*s.kids[0], cx, returned, retval);
        if (*returned || cx.fault) return;
      }
      cx.fault = true;
      return;
    }
    case SOp::Return:
      *retval = s.rhs ? eval(*s.rhs, cx) : 0;
      *returned = true;
      return;
  }
}

/// @brief Evaluate a guard expression against a state, without modifying it.
/// @param e      the guard; a null guard is vacuously satisfied
/// @param cells  the integer cells of the state
/// @param ncells number of cells
/// @param funcs  the function table
/// @return @c true iff the guard evaluates non-zero without faulting
/// @post The caller's cells are unchanged.
/// @note The copy is not incidental. A guard must not mutate the state it filters, but it may
///       call a function, and a function is free to write; evaluating against a scratch copy
///       makes that impossible rather than merely discouraged.
bool eval_guard(const ExprP& e, const long* cells, int ncells,
                const std::vector<Func>& funcs) {
  if (!e) return true;
  std::vector<long> scratch(cells, cells + ncells);
  EvalCtx cx;
  cx.cells = scratch.data();
  cx.funcs = &funcs;
  const long v = eval(*e, cx);
  return !cx.fault && v != 0;
}

}  // namespace hora
