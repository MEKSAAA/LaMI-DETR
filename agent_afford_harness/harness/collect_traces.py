#!/usr/bin/env python3
"""List all trace JSON files under outputs/traces (for batch analysis)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_afford_harness.paths import outputs_dir


def main():
    d = outputs_dir("traces")
    files = sorted(d.glob("*.json"))
    meta = [{"path": str(f), "sample_id": f.stem} for f in files]
    print(json.dumps({"traces": meta, "count": len(meta)}, indent=2))


if __name__ == "__main__":
    main()
