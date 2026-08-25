// intrange.cpp — interval analysis over the integer cells, to pack `dk` from the values a
// variable can actually take rather than from the range its declaration happens to allow.
//
// WHY. The discrete key is a lossless bit-packing of the location vector plus every integer
// cell, and a cell costs bits(hi - lo + 1) of the 63 available. UPPAAL's default range for an
// undeclared `int` is [-32768, 32767], i.e. 16 bits, so a model that writes `int list[N+1];
// int len;` spends 80 bits on five cells whose real contents are train ids and a queue length
// -- three bits each. train0 is rejected outright today ("discrete state needs 91 bits but the
// packed key holds 63") while verifyta verifies it in 55 stored states.
//
// WHAT. A flow-INSENSITIVE, whole-network abstract interpretation over intervals: one interval
// per cell, joined over every assignment anywhere in the network, iterated to a fixpoint.
// Guards are ignored, every branch of every `if` is taken, and array writes through a
// non-constant index hit every cell the index range can reach. All four make the result an
// OVER-approximation of the reachable values, which is the only direction that is sound here:
// the engine discards a transition whose write leaves the variable's range (gen_apply), so
// narrowing below a reachable value would silently delete behaviour. Over-approximating cannot,
// because no reachable write lands outside an over-approximation.
//
// The lattice is finite (every interval is contained in the declared range) so the fixpoint
// terminates on its own, but `len++` inside a loop would crawl one unit per round, so both the
// outer fixpoint and every `while` widen to the declared range once they exceed a round budget.
// Widening only ever loses precision, never soundness.
#include "ptctlmc/network.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ptctlmc {
namespace {

// An interval over one cell. `bot` marks "no value seen yet", which is the identity for join
// and is NOT the same as [0,0] -- a cell that is never assigned keeps only its initial value.
struct Iv {
  long lo = 0, hi = 0;
  bool bot = true;
  static Iv point(long v) { return Iv{v, v, false}; }
  static Iv range(long a, long b) { return Iv{a, b, false}; }
  void join(const Iv& o) {
    if (o.bot) return;
    if (bot) { *this = o; return; }
    lo = std::min(lo, o.lo);
    hi = std::max(hi, o.hi);
  }
  bool operator==(const Iv& o) const {
    return bot == o.bot && (bot || (lo == o.lo && hi == o.hi));
  }
};

// Interval arithmetic. Everything saturates into the declared bounds later, so plain long
// arithmetic is safe here for the constant magnitudes UPPAAL models use; the multiply takes the
// extremal products because either operand may straddle zero.
Iv iv_add(const Iv& a, const Iv& b) {
  if (a.bot || b.bot) return Iv{};
  return Iv::range(a.lo + b.lo, a.hi + b.hi);
}
Iv iv_sub(const Iv& a, const Iv& b) {
  if (a.bot || b.bot) return Iv{};
  return Iv::range(a.lo - b.hi, a.hi - b.lo);
}
Iv iv_mul(const Iv& a, const Iv& b) {
  if (a.bot || b.bot) return Iv{};
  const long p[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
  return Iv::range(*std::min_element(p, p + 4), *std::max_element(p, p + 4));
}

struct Analysis {
  const Network& net;
  std::vector<Iv> cell;              // one interval per integer cell
  std::vector<Iv> loc;               // current call frame
  const std::vector<Func>* funcs;
  int depth = 0;                     // recursion guard for Call
  bool loop_widened = false;

  explicit Analysis(const Network& n) : net(n), funcs(&n.funcs) {
    cell.resize(n.intvars.size());
    for (std::size_t v = 0; v < n.intvars.size(); ++v) cell[v] = Iv::point(n.intvars[v].init);
  }

  Iv declared(int c) const {
    return Iv::range(net.intvars[c].lo, net.intvars[c].hi);
  }
  // Clamp to the declaration: a value outside it is one the engine would have discarded, so
  // keeping it would only widen the packing for behaviour that cannot happen.
  void write(int c, Iv v) {
    if (c < 0 || c >= static_cast<int>(cell.size()) || v.bot) return;
    const Iv d = declared(c);
    v.lo = std::max(v.lo, d.lo);
    v.hi = std::min(v.hi, d.hi);
    if (v.lo > v.hi) return;         // entirely outside the declaration: transition discarded
    cell[c].join(v);
  }

  // The cells an Index node may address: base + [idx.lo, idx.hi], clipped to the array.
  void index_cells(const Expr& e, int* first, int* last) {
    const Iv i = ev(*e.kids[0]);
    const long a = i.bot ? 0 : i.lo, b = i.bot ? e.arr_len - 1 : i.hi;
    *first = e.var + static_cast<int>(std::max<long>(a, 0));
    *last = e.var + static_cast<int>(std::min<long>(b, e.arr_len - 1));
  }

  Iv ev(const Expr& e) {
    switch (e.op) {
      case EOp::Const: return Iv::point(e.val);
      case EOp::Var:   return (e.var >= 0 && e.var < (int)cell.size()) ? cell[e.var] : Iv{};
      case EOp::Local: return (e.var >= 0 && e.var < (int)loc.size()) ? loc[e.var] : Iv{};
      case EOp::Index: {
        int f, l;
        index_cells(e, &f, &l);
        Iv r;
        for (int c = f; c <= l; ++c)
          if (c >= 0 && c < (int)cell.size()) r.join(cell[c]);
        return r;
      }
      case EOp::Add: return iv_add(ev(*e.kids[0]), ev(*e.kids[1]));
      case EOp::Sub: return iv_sub(ev(*e.kids[0]), ev(*e.kids[1]));
      case EOp::Mul: return iv_mul(ev(*e.kids[0]), ev(*e.kids[1]));
      // Division and modulo: bound by the numerator's magnitude rather than reasoning about the
      // divisor's sign, which is imprecise but always contains the true result.
      case EOp::Div: case EOp::Mod: {
        const Iv a = ev(*e.kids[0]);
        if (a.bot) return Iv{};
        const long m = std::max(std::abs(a.lo), std::abs(a.hi));
        return Iv::range(-m, m);
      }
      case EOp::Neg: { const Iv a = ev(*e.kids[0]); return a.bot ? Iv{} : Iv::range(-a.hi, -a.lo); }
      case EOp::Eq: case EOp::Ne: case EOp::Lt: case EOp::Le: case EOp::Gt: case EOp::Ge:
      case EOp::And: case EOp::Or: case EOp::Not:
        for (const auto& k : e.kids) if (k) ev(*k);   // visit for effect-free sub-reads
        return Iv::range(0, 1);
      case EOp::Cond: {
        ev(*e.kids[0]);
        Iv r = ev(*e.kids[1]);
        r.join(ev(*e.kids[2]));
        return r;
      }
      case EOp::Call: return call(e);
    }
    return Iv{};
  }

  Iv call(const Expr& e) {
    if (e.fn < 0 || e.fn >= static_cast<int>(funcs->size()) || depth > 8) return Iv{};
    const Func& f = (*funcs)[e.fn];
    std::vector<Iv> args;
    args.reserve(e.kids.size());
    for (const auto& k : e.kids) args.push_back(ev(*k));
    std::vector<Iv> saved;
    saved.swap(loc);
    loc.assign(std::max(f.nlocals, f.nparams), Iv{});
    for (std::size_t i = 0; i < args.size() && i < loc.size(); ++i) loc[i] = args[i];
    Iv ret;
    ++depth;
    if (f.body) ret = run(*f.body);
    --depth;
    loc.swap(saved);
    return ret;
  }

  // Executes `s` abstractly, joining every write into `cell`. Returns the joined interval of
  // every `return` inside it (bottom when there is none).
  Iv run(const Stmt& s) {
    Iv ret;
    switch (s.op) {
      case SOp::Nop: break;
      case SOp::Assign: {
        const Iv v = s.rhs ? ev(*s.rhs) : Iv{};
        if (s.arr_len > 0 && s.idx) {
          // Non-constant index: the write may land on any cell the index range reaches, so it
          // must be joined into all of them.
          const Iv i = ev(*s.idx);
          const long a = i.bot ? 0 : i.lo, b = i.bot ? s.arr_len - 1 : i.hi;
          const int f = s.var + static_cast<int>(std::max<long>(a, 0));
          const int l = s.var + static_cast<int>(std::min<long>(b, s.arr_len - 1));
          for (int c = f; c <= l; ++c) write(c, v);
        } else {
          write(s.var, v);
        }
        break;
      }
      case SOp::AssignLocal:
        if (s.var >= 0 && s.var < static_cast<int>(loc.size()) && s.rhs) loc[s.var].join(ev(*s.rhs));
        break;
      case SOp::If:
        if (s.cond) ev(*s.cond);
        for (const auto& k : s.kids) if (k) ret.join(run(*k));   // both branches, cond ignored
        break;
      case SOp::While: {
        // Re-run the body until the cell state stops moving. The budget is what stops `len++`
        // from taking 65k rounds; on exhaustion every cell the loop writes goes to its declared
        // range, which is exactly today's behaviour for those cells.
        constexpr int kBudget = 24;
        if (s.cond) ev(*s.cond);
        std::vector<Iv> before;
        for (int it = 0; it < kBudget; ++it) {
          before = cell;
          for (const auto& k : s.kids) if (k) ret.join(run(*k));
          if (s.cond) ev(*s.cond);
          if (cell == before) return ret;                        // loop reached its fixpoint
        }
        widen_written_by(s);
        loop_widened = true;
        break;
      }
      case SOp::Block:
        for (const auto& k : s.kids) if (k) ret.join(run(*k));
        break;
      case SOp::Return:
        if (s.rhs) ret.join(ev(*s.rhs));
        break;
    }
    return ret;
  }

  // Widening: every cell this statement can write goes to its full declared range.
  void widen_written_by(const Stmt& s) {
    if (s.op == SOp::Assign && s.var >= 0) {
      const int n = s.arr_len > 0 ? s.arr_len : 1;
      for (int c = s.var; c < s.var + n && c < static_cast<int>(cell.size()); ++c)
        cell[c] = declared(c);
    }
    for (const auto& k : s.kids) if (k) widen_written_by(*k);
  }
};

}  // namespace

namespace {

// Widening thresholds: every integer constant the model itself mentions, plus the array
// lengths. A cell that keeps growing is jumped to the nearest enclosing threshold rather than
// straight to the declared range, which is what keeps `len` at the queue capacity instead of at
// 32767. Widening to a threshold is sound in the standard way: iteration continues afterwards,
// so the result is only accepted once it is stable, i.e. a post-fixpoint of the transfer
// function -- and a post-fixpoint contains every reachable value.
void collect_expr(const ExprP& e, std::vector<long>& t) {
  if (!e) return;
  if (e->op == EOp::Const) t.push_back(e->val);
  if (e->op == EOp::Index) t.push_back(e->arr_len);
  for (const auto& k : e->kids) collect_expr(k, t);
}
void collect_stmt(const StmtP& s, std::vector<long>& t) {
  if (!s) return;
  if (s->arr_len > 0) t.push_back(s->arr_len);
  collect_expr(s->idx, t);
  collect_expr(s->rhs, t);
  collect_expr(s->cond, t);
  for (const auto& k : s->kids) collect_stmt(k, t);
}

std::vector<long> thresholds(const Network& net) {
  std::vector<long> t{0, 1};
  for (const auto& v : net.intvars) { t.push_back(v.lo); t.push_back(v.hi); t.push_back(v.init); }
  for (const auto& f : net.funcs) collect_stmt(f.body, t);
  for (const auto& p : net.procs)
    for (const auto& e : p.edges) {
      for (const auto& g : e.int_guard) t.push_back(g.val);
      for (const auto& u : e.int_updates) t.push_back(u.val);
      collect_expr(e.gen_guard, t);
      collect_stmt(e.gen_update, t);
    }
  // A constant `c` is usually an exclusive bound (`i < N`), so c+1 must be reachable as a
  // threshold too or a counter that legitimately reaches N would widen past it to the
  // declaration.
  const std::size_t n = t.size();
  for (std::size_t i = 0; i < n; ++i) t.push_back(t[i] + 1);
  std::sort(t.begin(), t.end());
  t.erase(std::unique(t.begin(), t.end()), t.end());
  return t;
}

}  // namespace

// Replace each integer variable's declared range by an over-approximation of the values it can
// actually hold. Returns true when at least one variable narrowed. Idempotent, and a no-op on a
// network with no integer variables.
bool narrow_int_ranges(Network& net) {
  if (net.intvars.empty()) return false;
  Analysis a(net);

  // Flow-insensitive outer fixpoint: sweep every edge of every process, joining its writes,
  // until nothing moves. Sweeping in a fixed order makes the result deterministic.
  const std::vector<long> th = thresholds(net);
  // Jump to a threshold once a cell has moved this many rounds without settling. Three is enough
  // to distinguish "assigned a couple of constants" from "counting".
  constexpr int kWidenAfter = 3;
  constexpr int kRounds = 64;
  int round = 0;
  for (; round < kRounds; ++round) {
    const std::vector<Iv> before = a.cell;
    for (const auto& p : net.procs)
      for (const auto& e : p.edges) {
        for (const auto& u : e.int_updates) a.write(u.var, Iv::point(u.val));
        if (e.gen_guard) a.ev(*e.gen_guard);        // a guard may call a function that reads cells
        if (e.gen_update) {
          a.loc.assign(std::max(e.gen_locals, 0), Iv{});
          a.run(*e.gen_update);
        }
      }
    if (a.cell == before) break;
    if (round >= kWidenAfter) {
      for (std::size_t c = 0; c < a.cell.size(); ++c) {
        if (a.cell[c].bot || before[c].bot || a.cell[c] == before[c]) continue;
        const Iv d = a.declared(static_cast<int>(c));
        if (a.cell[c].lo < before[c].lo) {   // still falling: down to the next threshold below
          auto it = std::lower_bound(th.begin(), th.end(), a.cell[c].lo);
          a.cell[c].lo = (it == th.begin()) ? d.lo : std::max(d.lo, *(it - 1));
        }
        if (a.cell[c].hi > before[c].hi) {   // still climbing: up to the next threshold above
          auto it = std::upper_bound(th.begin(), th.end(), a.cell[c].hi);
          a.cell[c].hi = (it == th.end()) ? d.hi : std::min(d.hi, *it);
        }
      }
    }
  }
  // Did not settle inside the budget: fall back to the declarations wholesale rather than ship a
  // half-converged (and therefore possibly unsound) result.
  if (round == kRounds) return false;

  bool narrowed = false;
  for (std::size_t v = 0; v < net.intvars.size(); ++v) {
    IntVar& iv = net.intvars[v];
    const Iv r = a.cell[v];
    if (r.bot) continue;
    const long lo = std::max(iv.lo, std::min(r.lo, iv.init));
    const long hi = std::min(iv.hi, std::max(r.hi, iv.init));
    if (lo > hi || (lo == iv.lo && hi == iv.hi)) continue;
    iv.lo = lo;
    iv.hi = hi;
    narrowed = true;
  }
  return narrowed;
}

}  // namespace ptctlmc
