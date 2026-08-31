#!/usr/bin/env python3
"""Scalable single-automaton stress model: one location, n clocks, each with a
self-loop [x_i>=1]{x_i:=0}, invariant x_i<=D. The reachable zone graph grows
combinatorially with n and D — a stress test for zone exploration.

  gen_single.py n D  > model.xml     (writes model.q alongside via --q)
"""
import sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
D = int(sys.argv[2]) if len(sys.argv) > 2 else 6
decl = "clock " + ",".join(f"x{i}" for i in range(n)) + ";"
inv = " &amp;&amp; ".join(f"x{i}&lt;={D}" for i in range(n))
print('<?xml version="1.0" encoding="utf-8"?>')
print("<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.6//EN' "
      "'http://www.it.uu.se/research/group/darts/uppaal/flat-1_6.dtd'>")
print("<nta>")
print(f"<declaration>{decl}</declaration>")
print("<template><name>P</name>")
print(f'<location id="A"><name>A</name><label kind="invariant">{inv}</label></location>')
print('<init ref="A"/>')
for i in range(n):
    print(f'<transition><source ref="A"/><target ref="A"/>'
          f'<label kind="guard">x{i}&gt;=1</label>'
          f'<label kind="assignment">x{i}=0</label></transition>')
print("</template>")
print("<system>system P;</system>")
print("</nta>")
