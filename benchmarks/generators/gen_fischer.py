#!/usr/bin/env python3
"""Fischer's mutual-exclusion protocol, flattened to N concrete processes (the
classic scalable timed benchmark). Global int `id` (0 = free), one clock per
process. Emits a UPPAAL .xml valid for verifyta AND parseable by PTCTLMC.

  gen_fischer.py N [k]  > fischer.xml
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
clocks = ",".join(f"x{i}" for i in range(N))
print('<?xml version="1.0" encoding="utf-8"?>')
print("<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.6//EN' "
      "'http://www.it.uu.se/research/group/darts/uppaal/flat-1_6.dtd'>")
print("<nta>")
print(f"<declaration>clock {clocks}; int[0,{N}] id;</declaration>")
for i in range(N):
    pid = i + 1
    x = f"x{i}"
    print(f"<template><name>P{i}</name>")
    print(f'<location id="p{i}_idle"><name>idle</name></location>')
    print(f'<location id="p{i}_req"><name>req</name><label kind="invariant">{x}&lt;={k}</label></location>')
    print(f'<location id="p{i}_wait"><name>wait</name></location>')
    print(f'<location id="p{i}_cs"><name>cs</name></location>')
    print(f'<init ref="p{i}_idle"/>')
    # idle -> req : id==0, x:=0
    print(f'<transition><source ref="p{i}_idle"/><target ref="p{i}_req"/>'
          f'<label kind="guard">id==0</label><label kind="assignment">{x}=0</label></transition>')
    # req -> wait : id:=pid, x:=0
    print(f'<transition><source ref="p{i}_req"/><target ref="p{i}_wait"/>'
          f'<label kind="assignment">id={pid}, {x}=0</label></transition>')
    # wait -> cs : x>k && id==pid
    print(f'<transition><source ref="p{i}_wait"/><target ref="p{i}_cs"/>'
          f'<label kind="guard">{x}&gt;{k} &amp;&amp; id=={pid}</label></transition>')
    # wait -> idle : x>k && id!=pid  (retry)
    print(f'<transition><source ref="p{i}_wait"/><target ref="p{i}_idle"/>'
          f'<label kind="guard">{x}&gt;{k} &amp;&amp; id!={pid}</label></transition>')
    # cs -> idle : id:=0
    print(f'<transition><source ref="p{i}_cs"/><target ref="p{i}_idle"/>'
          f'<label kind="assignment">id=0</label></transition>')
    print("</template>")
print("<system>system " + ",".join(f"P{i}" for i in range(N)) + ";</system>")
print("</nta>")
