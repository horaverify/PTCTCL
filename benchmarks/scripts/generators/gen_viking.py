#!/usr/bin/env python3
"""Generator that copies viking<N>.xml asset. Silently skips if missing."""
import os, sys
if len(sys.argv) < 2:
    sys.exit(1)
n = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
viking_xml = os.path.join(HERE, "assets", "viking", f"viking{n}.xml")
if not os.path.exists(viking_xml):
    sys.exit(1)
sys.stdout.write(open(viking_xml).read())
