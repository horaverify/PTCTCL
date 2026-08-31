#!/usr/bin/env python3
"""CSMA/CD (carrier-sense multiple-access, collision-detect), the classic timed
broadcast benchmark. N stations share a bus. A station begins sending (begin!);
if a second station begins within the propagation window sigma, the bus detects a
collision and BROADCASTS cd! — received simultaneously by every sending station,
which back off. Exercises broadcast synchronisation (1 sender -> all receivers);
scalable in N stations.

  gen_csma.py N [sigma lambda]  > csma.xml   (sigma = propagation, lambda = frame)

Query idea:  E<> Bus.Collision      (a collision is reachable)
             A[] not (S0.Send and S1.Send and Bus.Transmitting)  (no undetected overlap)
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
S = int(sys.argv[2]) if len(sys.argv) > 2 else 1    # propagation delay sigma
L = int(sys.argv[3]) if len(sys.argv) > 3 else 4    # frame length lambda (> sigma)
clocks = ",".join(f"y{i}" for i in range(N))
print('<?xml version="1.0" encoding="utf-8"?>')
print("<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.6//EN' "
      "'http://www.it.uu.se/research/group/darts/uppaal/flat-1_6.dtd'>")
print("<nta>")
print(f"<declaration>clock {clocks}, c; broadcast chan cd; chan begin, end;</declaration>")

# ---- bus: Idle --begin?--> Active; a 2nd begin? within sigma => Collided --cd!--> Idle;
#      no 2nd begin within sigma => Transmitting --end?--> Idle
print("<template><name>Bus</name>")
print('<location id="bus_idle"><name>Idle</name></location>')
print(f'<location id="bus_active"><name>Active</name><label kind="invariant">c&lt;={S}</label></location>')
print('<location id="bus_tx"><name>Transmitting</name></location>')
print('<location id="bus_coll"><name>Collided</name></location>')
print('<init ref="bus_idle"/>')
# a station starts sending
print('<transition><source ref="bus_idle"/><target ref="bus_active"/>'
      '<label kind="synchronisation">begin?</label><label kind="assignment">c=0</label></transition>')
# a SECOND station begins within sigma => real collision (both are now sending)
print('<transition><source ref="bus_active"/><target ref="bus_coll"/>'
      f'<label kind="guard">c&lt;={S}</label>'
      '<label kind="synchronisation">begin?</label></transition>')
# window survived with a single sender => clean transmission
print('<transition><source ref="bus_active"/><target ref="bus_tx"/>'
      f'<label kind="guard">c&gt;={S}</label></transition>')
print('<transition><source ref="bus_tx"/><target ref="bus_idle"/>'
      '<label kind="synchronisation">end?</label></transition>')
# broadcast the collision to every sending station, then bus recovers
print('<transition><source ref="bus_coll"/><target ref="bus_idle"/>'
      '<label kind="synchronisation">cd!</label></transition>')
print("</template>")

# ---- stations ----
for i in range(N):
    y = f"y{i}"
    print(f"<template><name>S{i}</name>")
    print(f'<location id="s{i}_wait"><name>Wait</name></location>')
    print(f'<location id="s{i}_send"><name>Send</name><label kind="invariant">{y}&lt;={L}</label></location>')
    print(f'<location id="s{i}_retry"><name>Retry</name></location>')
    print(f'<init ref="s{i}_wait"/>')
    # start sending
    print(f'<transition><source ref="s{i}_wait"/><target ref="s{i}_send"/>'
          f'<label kind="synchronisation">begin!</label><label kind="assignment">{y}=0</label></transition>')
    # collision heard (broadcast) -> back off
    print(f'<transition><source ref="s{i}_send"/><target ref="s{i}_retry"/>'
          f'<label kind="synchronisation">cd?</label><label kind="assignment">{y}=0</label></transition>')
    # frame done -> release bus
    print(f'<transition><source ref="s{i}_send"/><target ref="s{i}_wait"/>'
          f'<label kind="guard">{y}&gt;={L}</label>'
          f'<label kind="synchronisation">end!</label></transition>')
    # backoff done -> try again
    print(f'<transition><source ref="s{i}_retry"/><target ref="s{i}_wait"/>'
          f'<label kind="guard">{y}&gt;={S}</label></transition>')
    print("</template>")

print("<system>system Bus," + ",".join(f"S{i}" for i in range(N)) + ";</system>")
print("</nta>")
