#!/usr/bin/env python3
"""Generalised Railroad Crossing (train--gate), N trains + gate + controller.

Heterogeneous components over binary channels. The controller must remember how many
trains are approaching/inside; since the parser forbids integer increments, the count
is encoded in the *controller's location index* (low_k / dn_k) — the standard way to
keep a queue protocol in scope. Diagonal-free -> GPU-eligible.

  gen_grc.py N  > grc.xml

Queries:  A[] not (T0.in and G.up)   (safety headline: a train is never in the region
                                       while the gate is fully up; explores everything)
          E<> T0.in                  (reachability sanity)
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CD, GLT, GRT, TP, TDU, TDL = 1, 2, 2, 4, 15, 1   # TP >= CD+GLT keeps the gate down in time
print('<?xml version="1.0" encoding="utf-8"?>')
print("<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.6//EN' "
      "'http://www.it.uu.se/research/group/darts/uppaal/flat-1_6.dtd'>")
print("<nta>")
xs = ",".join(f"x{i}" for i in range(N))
print(f"<declaration>clock {xs}, y, z; chan approach, leave, lower, raise;</declaration>")

# ---- trains ----
for i in range(N):
    x = f"x{i}"
    print(f"<template><name>T{i}</name>")
    print(f'<location id="t{i}_far"><name>far</name></location>')
    print(f'<location id="t{i}_near"><name>near</name><label kind="invariant">{x}&lt;={TP}</label></location>')
    print(f'<location id="t{i}_in"><name>in</name><label kind="invariant">{x}&lt;={TDU}</label></location>')
    print(f'<init ref="t{i}_far"/>')
    print(f'<transition><source ref="t{i}_far"/><target ref="t{i}_near"/>'
          f'<label kind="synchronisation">approach!</label><label kind="assignment">{x}=0</label></transition>')
    print(f'<transition><source ref="t{i}_near"/><target ref="t{i}_in"/>'
          f'<label kind="guard">{x}&gt;={TP}</label><label kind="assignment">{x}=0</label></transition>')
    print(f'<transition><source ref="t{i}_in"/><target ref="t{i}_far"/>'
          f'<label kind="guard">{x}&gt;={TDL}</label><label kind="synchronisation">leave!</label></transition>')
    print("</template>")

# ---- gate ----
print("<template><name>G</name>")
print('<location id="g_up"><name>up</name></location>')
print(f'<location id="g_low"><name>low</name><label kind="invariant">y&lt;={GLT}</label></location>')
print('<location id="g_down"><name>down</name></location>')
print(f'<location id="g_rai"><name>rai</name><label kind="invariant">y&lt;={GRT}</label></location>')
print('<init ref="g_up"/>')
print('<transition><source ref="g_up"/><target ref="g_low"/>'
      '<label kind="synchronisation">lower?</label><label kind="assignment">y=0</label></transition>')
print(f'<transition><source ref="g_low"/><target ref="g_down"/><label kind="guard">y&gt;={GLT}</label></transition>')
print('<transition><source ref="g_down"/><target ref="g_rai"/>'
      '<label kind="synchronisation">raise?</label><label kind="assignment">y=0</label></transition>')
print(f'<transition><source ref="g_rai"/><target ref="g_up"/><label kind="guard">y&gt;={GRT}</label></transition>')
print("</template>")

# ---- controller: location index = number of approaching/in trains ----
print("<template><name>Ctl</name>")
print('<location id="c_up"><name>up</name></location>')
for k in range(1, N + 1):
    print(f'<location id="c_low{k}"><name>low_{k}</name><label kind="invariant">z&lt;={CD}</label></location>')
for k in range(1, N + 1):
    print(f'<location id="c_dn{k}"><name>dn_{k}</name></location>')
print(f'<location id="c_rai"><name>rai</name><label kind="invariant">z&lt;={CD}</label></location>')
print('<init ref="c_up"/>')
# up --approach?--> low_1
print('<transition><source ref="c_up"/><target ref="c_low1"/>'
      '<label kind="synchronisation">approach?</label><label kind="assignment">z=0</label></transition>')
for k in range(1, N + 1):
    if k < N:  # low_k --approach?--> low_{k+1}
        print(f'<transition><source ref="c_low{k}"/><target ref="c_low{k+1}"/>'
              f'<label kind="synchronisation">approach?</label></transition>')
    # low_k --lower!--> dn_k
    print(f'<transition><source ref="c_low{k}"/><target ref="c_dn{k}"/>'
          f'<label kind="synchronisation">lower!</label></transition>')
# low_1 --leave?--> up  (the single pending train left before the gate lowered)
print('<transition><source ref="c_low1"/><target ref="c_up"/>'
      '<label kind="synchronisation">leave?</label><label kind="assignment">z=0</label></transition>')
for k in range(1, N + 1):
    if k < N:  # dn_k --approach?--> dn_{k+1}
        print(f'<transition><source ref="c_dn{k}"/><target ref="c_dn{k+1}"/>'
              f'<label kind="synchronisation">approach?</label></transition>')
    if k > 1:  # dn_k --leave?--> dn_{k-1}
        print(f'<transition><source ref="c_dn{k}"/><target ref="c_dn{k-1}"/>'
              f'<label kind="synchronisation">leave?</label></transition>')
# dn_1 --leave?--> rai
print('<transition><source ref="c_dn1"/><target ref="c_rai"/>'
      '<label kind="synchronisation">leave?</label><label kind="assignment">z=0</label></transition>')
# rai --raise!--> up
print('<transition><source ref="c_rai"/><target ref="c_up"/>'
      '<label kind="synchronisation">raise!</label></transition>')
print("</template>")

procs = ",".join([f"T{i}" for i in range(N)] + ["G", "Ctl"])
print(f"<system>system {procs};</system>")
print("</nta>")
