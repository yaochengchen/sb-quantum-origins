#!/usr/bin/env bash
# Kept for convenience; the real driver is make_all_figs.py (works on Windows and Linux). Run from msbqa_phase1/.
cd "$(dirname "$0")/.." && python paper1_figs/make_all_figs.py
