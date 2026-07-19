#!/usr/bin/env python3
"""Arbiter mutex: N timed clients contend for a critical section mediated by a
central arbiter over BINARY handshake channels (acq!/acq?, rel!/rel?). The
arbiter serialises access (Free <-> Busy), so mutual exclusion holds by
construction. Exercises binary channel synchronisation; scalable in N clients.

  gen_arbiter.py N [P D]   > arbiter.xml   (P = think period, D = cs deadline)

Query idea:  A[] not (C0.cs and C1.cs)
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
P = int(sys.argv[2]) if len(sys.argv) > 2 else 2   # min think time before retry
D = int(sys.argv[3]) if len(sys.argv) > 3 else 4   # cs deadline
clocks = ",".join(f"x{i}" for i in range(N))
print('<?xml version="1.0" encoding="utf-8"?>')
print("<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.6//EN' "
      "'http://www.it.uu.se/research/group/darts/uppaal/flat-1_6.dtd'>")
print("<nta>")
print(f"<declaration>clock {clocks}; chan acq, rel;</declaration>")

# ---- arbiter (no clock): Free --acq?--> Busy --rel?--> Free ----
print("<template><name>Arb</name>")
print('<location id="arb_free"><name>Free</name></location>')
print('<location id="arb_busy"><name>Busy</name></location>')
print('<init ref="arb_free"/>')
print('<transition><source ref="arb_free"/><target ref="arb_busy"/>'
      '<label kind="synchronisation">acq?</label></transition>')
print('<transition><source ref="arb_busy"/><target ref="arb_free"/>'
      '<label kind="synchronisation">rel?</label></transition>')
print("</template>")

# ---- clients ----
for i in range(N):
    x = f"x{i}"
    print(f"<template><name>C{i}</name>")
    print(f'<location id="c{i}_idle"><name>idle</name></location>')
    print(f'<location id="c{i}_want"><name>want</name></location>')
    print(f'<location id="c{i}_cs"><name>cs</name><label kind="invariant">{x}&lt;={D}</label></location>')
    print(f'<init ref="c{i}_idle"/>')
    # idle -> want after thinking P
    print(f'<transition><source ref="c{i}_idle"/><target ref="c{i}_want"/>'
          f'<label kind="guard">{x}&gt;={P}</label></transition>')
    # want -> cs : acquire the lock (acq!) and reset the cs clock
    print(f'<transition><source ref="c{i}_want"/><target ref="c{i}_cs"/>'
          f'<label kind="synchronisation">acq!</label><label kind="assignment">{x}=0</label></transition>')
    # cs -> idle : release (rel!) and reset the think clock
    print(f'<transition><source ref="c{i}_cs"/><target ref="c{i}_idle"/>'
          f'<label kind="synchronisation">rel!</label><label kind="assignment">{x}=0</label></transition>')
    print("</template>")

print("<system>system Arb," + ",".join(f"C{i}" for i in range(N)) + ";</system>")
print("</nta>")
