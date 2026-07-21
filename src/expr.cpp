// expr.cpp — evaluator for the modelling-language subset declared in expr.hpp.
//
// Guards evaluate an Expr; edge updates execute a Stmt. Both may call user
// functions, which are interpreted against the live int cells (train-gate's
// `dequeue()` shifts a FIFO in a loop, so it cannot be constant-folded).
//
// Faults (index out of range, division by zero, runaway recursion) do not throw:
// they set EvalCtx::fault and yield 0. The caller decides. A faulting GUARD is
// treated as false and a faulting UPDATE abandons the edge, which keeps a
// malformed model from silently producing a wrong verdict.
#include "ptctlmc/expr.hpp"

#include <algorithm>

namespace ptctlmc {

namespace {
constexpr int kMaxDepth = 64;  // user functions here are not recursive; this is a backstop
}

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
    // && and || short-circuit: `len > 0 && list[len-1] == e` must not evaluate
    // the right side when len == 0, or it would index out of range.
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
      // Bounded: a modelling error (or a guard that never falsifies) must not hang
      // the model checker. Array sizes here are tiny, so this cap is unreachable
      // for well-formed models.
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

bool eval_guard(const ExprP& e, const long* cells, int ncells,
                const std::vector<Func>& funcs) {
  if (!e) return true;
  // Guards must not mutate the state, but they may call functions, and an
  // ill-behaved function could write. Evaluate against a scratch copy so a guard
  // can never perturb the state it is filtering.
  std::vector<long> scratch(cells, cells + ncells);
  EvalCtx cx;
  cx.cells = scratch.data();
  cx.funcs = &funcs;
  const long v = eval(*e, cx);
  return !cx.fault && v != 0;
}

}  // namespace ptctlmc
