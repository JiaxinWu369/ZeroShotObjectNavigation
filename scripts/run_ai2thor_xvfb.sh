#!/usr/bin/env bash
set -euo pipefail

xvfb-run -a -s "-screen 0 1024x768x24" python scripts/test_ai2thor.py
