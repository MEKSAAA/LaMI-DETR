#!/usr/bin/env python3
"""Run showcase or custom subset; save traces + optional scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_afford_harness.agent.orchestrator import run_pipeline
from agent_afford_harness.data.eval_wrapper import evaluate_trace_points
from agent_afford_harness.data.roboafford_loader import load_showcase_samples
from agent_afford_harness.paths import load_env_overrides, outputs_dir


def main():
    load_env_overrides()
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-json", default=None, help="Override sample_subset.json path")
    ap.add_argument("--output", default=None, help="Summary json path")
    args = ap.parse_args()

    samples = load_showcase_samples(Path(args.subset_json) if args.subset_json else None)
    pred_dir = outputs_dir("predictions")
    trace_dir = outputs_dir("traces")

    summary = []
    for s in samples:
        if not Path(s.image_path).is_file():
            row = {"sample_id": s.sample_id, "error": f"missing_image: {s.image_path}", "skipped": True}
            summary.append(row)
            continue
        img = Image.open(s.image_path).convert("RGB")
        points, trace = run_pipeline(
            img,
            s.question,
            s.sample_id,
            image_path=s.image_path,
            benchmark_category=s.category,
        )
        ev = evaluate_trace_points(points, s.mask_path)
        trace.reward = ev["accuracy"]
        out = trace.to_dict()
        out["final_points"] = points
        out["reward"] = ev["accuracy"]
        tfile = trace_dir / f"{s.sample_id}.json"
        with open(tfile, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        summary.append(
            {
                "sample_id": s.sample_id,
                "reward": ev["accuracy"],
                "task_type": trace.task_type,
                "trace": str(tfile),
            }
        )

    outp = args.output or str(pred_dir / "subset_summary.json")
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump({"samples": summary}, f, indent=2)
    print(json.dumps({"written": outp, "count": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
