"""Make the demo app's pure-logic modules importable from tests."""

import sys
from pathlib import Path

DEMO_DIR = str(Path(__file__).resolve().parents[1] / "examples" / "research_agent")
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)
