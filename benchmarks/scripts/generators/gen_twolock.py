#!/usr/bin/env python3
"""Two-lock mutual exclusion: a Fischer-style protocol where each process must
acquire TWO shared locks (a then b) before entering its critical section. Two
shared int variables exercise the packed-uint64 discrete key across >1 int var
(the case a single-int Fischer never hits). High-concurrency, scalable.

  gen_twolock.py N [k]  > twolock.xml

Query idea:  A[] not (P0.cs and P1.cs)
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
clocks = ",".join(f"x{i}" for i in range(N))
print('<?xml version="1.0" encoding="utf-8"?>')
print("<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.6//EN' "
      "'http://www.it.uu.se/research/group/darts/uppaal/flat-1_6.dtd'>")
print("<nta>")
print(f"<declaration>clock {clocks}; int[0,{N}] a; int[0,{N}] b;</declaration>")
for i in range(N):
    pid = i + 1
    x = f"x{i}"
    print(f"<template><name>P{i}</name>")
    for loc in ("idle", "reqA", "waitA", "reqB", "waitB", "cs"):
        inv = f'<label kind="invariant">{x}&lt;={k}</label>' if loc in ("reqA", "reqB") else ""
        print(f'<location id="p{i}_{loc}"><name>{loc}</name>{inv}</location>')
    print(f'<init ref="p{i}_idle"/>')
    # ---- acquire lock a (Fischer on a) ----
    print(f'<transition><source ref="p{i}_idle"/><target ref="p{i}_reqA"/>'
          f'<label kind="guard">a==0</label><label kind="assignment">{x}=0</label></transition>')
    print(f'<transition><source ref="p{i}_reqA"/><target ref="p{i}_waitA"/>'
          f'<label kind="assignment">a={pid}, {x}=0</label></transition>')
    print(f'<transition><source ref="p{i}_waitA"/><target ref="p{i}_reqB"/>'
          f'<label kind="guard">{x}&gt;{k} &amp;&amp; a=={pid} &amp;&amp; b==0</label>'
          f'<label kind="assignment">{x}=0</label></transition>')
    print(f'<transition><source ref="p{i}_waitA"/><target ref="p{i}_idle"/>'
          f'<label kind="guard">{x}&gt;{k} &amp;&amp; a!={pid}</label></transition>')
    # ---- acquire lock b (Fischer on b), holding a ----
    print(f'<transition><source ref="p{i}_reqB"/><target ref="p{i}_waitB"/>'
          f'<label kind="assignment">b={pid}, {x}=0</label></transition>')
    print(f'<transition><source ref="p{i}_waitB"/><target ref="p{i}_cs"/>'
          f'<label kind="guard">{x}&gt;{k} &amp;&amp; b=={pid}</label></transition>')
    print(f'<transition><source ref="p{i}_waitB"/><target ref="p{i}_idle"/>'
          f'<label kind="guard">{x}&gt;{k} &amp;&amp; b!={pid}</label>'
          f'<label kind="assignment">a=0</label></transition>')
    # ---- release both ----
    print(f'<transition><source ref="p{i}_cs"/><target ref="p{i}_idle"/>'
          f'<label kind="assignment">a=0, b=0</label></transition>')
    print("</template>")
print("<system>system " + ",".join(f"P{i}" for i in range(N)) + ";</system>")
print("</nta>")
