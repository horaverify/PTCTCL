#!/bin/sh
# verifyta forced to depth-first search (-o 1). The recorded reach table was measured with the
# default order; this exists to measure whether that default is what costs UPPAAL its reach
# numbers, or whether the gap is the search itself.
exec /usr/local/bin/verifyta -o 1 "$@"
