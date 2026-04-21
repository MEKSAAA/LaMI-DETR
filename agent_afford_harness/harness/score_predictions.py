#!/usr/bin/env python3
"""Score a predictions JSON against masks (RoboAfford accuracy)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_afford_harness.data.eval_wrapper import calculate_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="JSON with final_points list or batch")
    ap.add_argument("--mask", help="Single mask path (if predictions is one sample)")
    args = ap.parse_args()

    with open(args.predictions, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "final_points" in data and args.mask:
        acc = calculate_accuracy(data["final_points"], args.mask)
        print(json.dumps({"accuracy": acc}))
        return

    raise SystemExit("Provide --predictions with final_points + --mask for v1 scorer")


if __name__ == "__main__":
    main()
