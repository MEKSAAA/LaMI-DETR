#!/usr/bin/env python3
"""Evaluate trace outputs against RoboAfford-style GT masks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_afford_harness.data.eval_wrapper import calculate_accuracy
from agent_afford_harness.paths import harness_root


def _norm_path_str(p: str) -> str:
    return str(Path(p).resolve())


def _flatten_grouped(groups: Any) -> List[List[float]]:
    out: List[List[float]] = []
    if not isinstance(groups, list):
        return out
    for g in groups:
        if not isinstance(g, list):
            continue
        for p in g:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append([float(p[0]), float(p[1])])
    return out


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_gt_index(gt_data: List[Dict[str, Any]], data_root: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    idx: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in gt_data:
        q = str(row.get("question", "")).strip()
        img_rel = str(row.get("img", "")).strip()
        category = str(row.get("category", "")).strip().lower()
        if not q or not img_rel or not category:
            continue
        img_name = Path(img_rel).name
        idx[(q, img_name, category)] = row
    return idx


def _collect_trace_files(traces_dir: Path) -> List[Path]:
    return sorted([p for p in traces_dir.glob("*.json") if p.is_file()])


def evaluate_traces(
    traces_dir: Path,
    gt_file: Path,
    data_root: Path,
    output_file: Path,
) -> Dict[str, Any]:
    gt_data = _read_json(gt_file)
    if not isinstance(gt_data, list):
        raise RuntimeError(f"GT file must be a list JSON: {gt_file}")
    gt_idx = _build_gt_index(gt_data, data_root)
    rows: List[Dict[str, Any]] = []
    cat_scores: Dict[str, List[float]] = {}
    matched = 0

    for tf in _collect_trace_files(traces_dir):
        tr = _read_json(tf)
        question = str(tr.get("question", "")).strip()
        img_path = str(tr.get("image_path", "")).strip()
        category = str(tr.get("category", "")).strip().lower()
        if not question or not img_path:
            rows.append(
                {
                    "trace": str(tf),
                    "sample_id": tr.get("sample_id", ""),
                    "matched": False,
                    "error": "missing question/image_path in trace",
                }
            )
            continue

        img_name = Path(img_path).name
        key = (question, img_name, category)
        gt = gt_idx.get(key)
        if gt is None:
            rows.append(
                {
                    "trace": str(tf),
                    "sample_id": tr.get("sample_id", ""),
                    "matched": False,
                    "error": "no gt matched by (question, image_filename, category)",
                    "question": question,
                    "image_path": img_path,
                    "image_filename": img_name,
                    "category": category,
                }
            )
            continue

        mask_rel = str(gt.get("mask", "")).strip()
        mask_abs = data_root / mask_rel
        if not mask_rel or not mask_abs.is_file():
            rows.append(
                {
                    "trace": str(tf),
                    "sample_id": tr.get("sample_id", ""),
                    "matched": False,
                    "error": f"mask missing: {mask_abs}",
                }
            )
            continue

        grouped = tr.get("final_point_groups")
        final_points = _flatten_grouped(grouped)
        if not final_points:
            raw_pts = tr.get("final_points") or []
            if isinstance(raw_pts, list):
                final_points = [
                    [float(p[0]), float(p[1])]
                    for p in raw_pts
                    if isinstance(p, (list, tuple)) and len(p) >= 2
                ]

        acc = calculate_accuracy(final_points, str(mask_abs))
        category = str(gt.get("category", "unknown"))
        cat_scores.setdefault(category, []).append(acc)
        matched += 1
        rows.append(
            {
                "trace": str(tf),
                "sample_id": tr.get("sample_id", ""),
                "matched": True,
                "category": category,
                "accuracy": acc,
                "num_points": len(final_points),
                "num_groups": len(grouped) if isinstance(grouped, list) else 0,
                "mask_path": str(mask_abs),
            }
        )

    overall = 0.0
    if matched > 0:
        overall = sum(r["accuracy"] for r in rows if r.get("matched")) / matched
    by_category = {
        c: (sum(v) / len(v) if v else 0.0)
        for c, v in sorted(cat_scores.items(), key=lambda kv: kv[0])
    }

    result = {
        "traces_dir": str(traces_dir),
        "gt_file": str(gt_file),
        "data_root": str(data_root),
        "total_traces": len(rows),
        "matched_traces": matched,
        "overall_accuracy": overall,
        "by_category_accuracy": by_category,
        "rows": rows,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--traces-dir",
        default=str(harness_root() / "outputs" / "traces"),
        help="Directory containing trace json files",
    )
    ap.add_argument(
        "--gt-file",
        default="/data9/data/miaojw/projects26/RoboAfford/annotations_absxy.json",
        help="Ground-truth annotation json",
    )
    ap.add_argument(
        "--data-root",
        default="/data9/data/miaojw/projects26/RoboAfford",
        help="Root directory for img/mask relative paths in GT",
    )
    ap.add_argument(
        "--output-file",
        default=str(harness_root() / "outputs" / "predictions" / "trace_eval_summary.json"),
        help="Where to save eval summary json",
    )
    args = ap.parse_args()

    result = evaluate_traces(
        traces_dir=Path(args.traces_dir),
        gt_file=Path(args.gt_file),
        data_root=Path(args.data_root),
        output_file=Path(args.output_file),
    )
    print(
        json.dumps(
            {
                "written": args.output_file,
                "overall_accuracy": result["overall_accuracy"],
                "matched_traces": result["matched_traces"],
                "total_traces": result["total_traces"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

