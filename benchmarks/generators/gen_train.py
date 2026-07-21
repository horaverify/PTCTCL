#!/usr/bin/env python3
"""Generator that copies train-gate.xml asset. Silently skips if missing."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
train_xml = os.path.join(HERE, "assets", "train", "train0.xml")
if not os.path.exists(train_xml):
    sys.exit(1)
sys.stdout.write(open(train_xml).read())
