#!/usr/bin/env python3
"""Dining philosophers (timed), N philosophers + N forks over 2N binary channels.

The parallel *showcase*: every philosopher can act independently each level, so the
BFS frontier is the widest of any benchmark here — the case where multicore/GPU
expansion has the most to chew on. Diagonal-free and channel-based, so it exercises
the GPU binary-sync path. Philosopher i uses fork i (left) and fork (i+1)%N (right).

  gen_phil.py N [EAT]  > phil.xml

Queries:  E<> P0.Eat                          (reachable — early exit)
          A[] not (P0.Eat and P1.Eat)         (safety, explores everything: adjacent
                                                philosophers share a fork, never eat together)
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EAT = int(sys.argv[2]) if len(sys.argv) > 2 else 2
clocks = ",".join(f"c{i}" for i in range(N))
chans = ",".join([f"take{i}" for i in range(N)] + [f"put{i}" for i in range(N)])
print('<?xml version="1.0" encoding="utf-8"?>')
print("<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.6//EN' "
      "'http://www.it.uu.se/research/group/darts/uppaal/flat-1_6.dtd'>")
print("<nta>")
print(f"<declaration>clock {clocks}; chan {chans};</declaration>")

# ---- forks (no clocks): Free <--take_i--> Used <--put_i--> Free ----
for i in range(N):
    print(f"<template><name>F{i}</name>")
    print(f'<location id="f{i}_free"><name>Free</name></location>')
    print(f'<location id="f{i}_used"><name>Used</name></location>')
    print(f'<init ref="f{i}_free"/>')
    print(f'<transition><source ref="f{i}_free"/><target ref="f{i}_used"/>'
          f'<label kind="synchronisation">take{i}?</label></transition>')
    print(f'<transition><source ref="f{i}_used"/><target ref="f{i}_free"/>'
          f'<label kind="synchronisation">put{i}?</label></transition>')
    print("</template>")

# ---- philosophers: grab left, grab right, eat, drop left, drop right ----
for i in range(N):
    r = (i + 1) % N          # right fork
    c = f"c{i}"
    print(f"<template><name>P{i}</name>")
    print(f'<location id="p{i}_think"><name>Think</name></location>')
    print(f'<location id="p{i}_holdl"><name>HoldL</name></location>')
    print(f'<location id="p{i}_eat"><name>Eat</name><label kind="invariant">{c}&lt;={EAT}</label></location>')
    print(f'<location id="p{i}_drop"><name>Dropped</name></location>')
    print(f'<init ref="p{i}_think"/>')
    print(f'<transition><source ref="p{i}_think"/><target ref="p{i}_holdl"/>'
          f'<label kind="synchronisation">take{i}!</label><label kind="assignment">{c}=0</label></transition>')
    print(f'<transition><source ref="p{i}_holdl"/><target ref="p{i}_eat"/>'
          f'<label kind="synchronisation">take{r}!</label><label kind="assignment">{c}=0</label></transition>')
    print(f'<transition><source ref="p{i}_eat"/><target ref="p{i}_drop"/>'
          f'<label kind="guard">{c}&gt;=1</label><label kind="synchronisation">put{i}!</label></transition>')
    print(f'<transition><source ref="p{i}_drop"/><target ref="p{i}_think"/>'
          f'<label kind="synchronisation">put{r}!</label></transition>')
    print("</template>")

procs = ",".join([f"F{i}" for i in range(N)] + [f"P{i}" for i in range(N)])
print(f"<system>system {procs};</system>")
print("</nta>")
