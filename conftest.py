"""
conftest.py — project root
pytest loads this automatically before any test file.
Adding rootdir to sys.path here means test files never need their own
sys.path manipulation, and `from data import ...` always resolves.
"""
import sys
from pathlib import Path

# Insert the project root (the directory containing this file) so that
# `data`, `models`, `probes`, `scripts` are all importable as top-level packages.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
