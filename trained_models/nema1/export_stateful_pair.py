#!/usr/bin/env python3
"""Compatibility shim.

This exporter has moved to `new/export_stateful_pair.py`.
Please update your calls to use the new path.
"""
import sys
from pathlib import Path

shim_target = Path(__file__).resolve().parents[2] / 'new' / 'export_stateful_pair.py'
print('[export_stateful_pair] DEPRECATED: use `new/export_stateful_pair.py` instead.', file=sys.stderr)
if not shim_target.exists():
    print(f'[export_stateful_pair] ERROR: Missing {shim_target}', file=sys.stderr)
    sys.exit(1)

with open(shim_target, 'rb') as fh:
    code = compile(fh.read(), str(shim_target), 'exec')
    exec(code, globals(), globals())
