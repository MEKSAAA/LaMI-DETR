#!/usr/bin/env python3
"""Run agentic affordance pipeline on one image + question."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

# Allow `python harness/run_single_case.py` from package dir
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_afford_harness.agent.orchestrator import run_pipeline
from agent_afford_harness.data.eval_wrapper import evaluate_trace_points
from agent_afford_harness.paths import load_env_overrides, outputs_dir


def main():
    load_env_overrides()
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Image path")
    ap.add_argument("--question", required=True)
    ap.add_argument("--sample-id", default="single_case")
    ap.add_argument("--category", default=None, help="Benchmark category override")
    ap.add_argument("--mask", default=None, help="Optional mask for scoring")
    ap.add_argument("--trace-out", default=None)
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    points, trace = run_pipeline(
        img,
        args.question,
        args.sample_id,
        image_path=str(Path(args.image).resolve()),
        benchmark_category=args.category,
    )

    out = trace.to_dict()
    out["final_points"] = points

    if args.mask:
        ev = evaluate_trace_points(points, args.mask)
        out["reward"] = ev["accuracy"]
        trace.reward = ev["accuracy"]

    tpath = args.trace_out or str(outputs_dir("traces") / f"{args.sample_id}.json")
    Path(tpath).parent.mkdir(parents=True, exist_ok=True)
    with open(tpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"final_points": points, "trace": tpath}, indent=2))


if __name__ == "__main__":
    main()
