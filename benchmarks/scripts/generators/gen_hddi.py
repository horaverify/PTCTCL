#!/usr/bin/env python3
"""Generator that copies hddi_input_<N>.xml asset. Silently skips if missing."""
import os, sys
if len(sys.argv) < 2:
    sys.exit(1)
n = int(sys.argv[1])
HERE = os.path.dirname(os.path.abspath(__file__))
hddi_xml = os.path.join(HERE, "assets", "hddi", f"hddi_input_{n:02d}.xml")
if not os.path.exists(hddi_xml):
    sys.exit(1)
sys.stdout.write(open(hddi_xml).read())
