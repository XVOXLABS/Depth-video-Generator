"""Put the vendored Video-Depth-Anything package on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))
