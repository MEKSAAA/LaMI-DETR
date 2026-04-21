#!/usr/bin/env python3
"""Overlay final points (+ optional LaMI boxes) from a trace JSON on the image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="Trace JSON from harness")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.trace, "r", encoding="utf-8") as f:
        tr = json.load(f)

    img_path = tr.get("image_path")
    if not img_path or not Path(img_path).is_file():
        raise SystemExit("trace must contain valid image_path")
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)

    for tc in tr.get("tool_call_history") or []:
        if tc.get("name") == "lami_detr_grounder":
            out = tc.get("output") or {}
            for b in out.get("boxes") or []:
                bb = b.get("bbox")
                if bb and len(bb) == 4:
                    draw.rectangle(bb, outline="cyan", width=3)

    for p in tr.get("final_points") or []:
        x, y = float(p[0]) * W, float(p[1]) * H
        r = 6
        draw.ellipse([x - r, y - r, x + r, y + r], fill="red", outline="white")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    im.save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
