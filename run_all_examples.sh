#!/bin/bash
# Regenerate optimized_structure.pdf for the example cases, headless (--no-gui).
# Usage:
#   ./run_all_examples.sh                       # every Examples/<case> with geometry+parameters json
#   ./run_all_examples.sh Examples/bridge_1 ... # only the given folders
# Continues on error; exit status is non-zero if any case failed.

set -u
cd "$(dirname "$0")"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"

if [ "$#" -gt 0 ]; then
    dirs=("$@")
else
    dirs=(Examples/*/)
fi

fail=0
for d in "${dirs[@]}"; do
    d="${d%/}"
    if [ ! -f "$d/geometry.json" ] || [ ! -f "$d/parameters.json" ]; then
        echo "--- skip $d (missing geometry.json or parameters.json)"
        continue
    fi
    echo "=== $d ==="
    if ! "$PY" src/main.py --example "$d" --no-gui; then
        echo "!!! FAILED: $d"
        fail=1
    fi
done

exit $fail
