"""LaMI-DETR grounding tool — candidate boxes in pixel space."""

from __future__ import annotations

import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from agent_afford_harness.paths import default_lami_config, lami_det_repo_root
from agent_afford_harness.utils.coord_1000 import bbox_1000_to_pixel, bbox_pixel_to_1000


_LOCK = threading.Lock()
_SINGLETON: Optional["LaMIDetrGrounder"] = None


def get_grounder(cfg_path: Optional[Path] = None) -> "LaMIDetrGrounder":
    global _SINGLETON
    if _SINGLETON is None:
        with _LOCK:
            if _SINGLETON is None:
                _SINGLETON = LaMIDetrGrounder(cfg_path=cfg_path)
    return _SINGLETON


class LaMIDetrGrounder:
    """Loads LaMI once; runs forward with names + visual_descs (see LaMI JSON convention)."""

    def __init__(self, cfg_path: Optional[Path] = None):
        repo = lami_det_repo_root()
        if str(repo) not in sys.path:
            sys.path.append(str(repo))

        self.cfg_file = Path(
            os.environ.get("AGENT_HARNESS_LAMI_CONFIG", "") or cfg_path or default_lami_config()
        )
        self._lami_detr_dir = repo / "lami_detr"
        if str(self._lami_detr_dir) not in sys.path:
            sys.path.append(str(self._lami_detr_dir))
        self._orig_cwd = os.getcwd()

        self._cfg = None
        self._model = None
        self._augmentation = None
        self._device = "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        ckpt = lami_det_repo_root() / "pretrained_models" / "lami_convnext_large_12ep_vg" / "model_final.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"LaMI checkpoint not found: {ckpt}")

        os.chdir(str(self._lami_detr_dir))
        try:
            from detectron2.config import LazyConfig, instantiate
            from detectron2.checkpoint import DetectionCheckpointer
            cfg = LazyConfig.load(str(self.cfg_file.resolve()))
            aug = instantiate(cfg.dataloader.test.mapper.augmentation)
            model = instantiate(cfg.model)
            model.to(cfg.train.device)
            model.eval()
            DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

            self._cfg = cfg
            self._augmentation = aug
            self._model = model
            self._device = cfg.train.device
        except Exception as e:
            raise RuntimeError(f"Failed to load LaMI model from {self.cfg_file}") from e
        finally:
            os.chdir(self._orig_cwd)

    @staticmethod
    def _pil_to_tensor_bgr(cv2_bgr: np.ndarray, augmentation) -> "torch.Tensor":
        import cv2
        import torch
        from detectron2.data import transforms as T

        b, g, r = cv2.split(cv2_bgr)
        img_eq = cv2.merge([cv2.equalizeHist(x) for x in [b, g, r]])
        img_aug, _ = T.apply_transform_gens(augmentation, img_eq)
        return torch.as_tensor(np.ascontiguousarray(img_aug.transpose(2, 0, 1))).unsqueeze(0)

    def infer(
        self,
        image: Image.Image,
        lami_classes: Dict[str, List[str]],
        nms_iou: float = 0.7,
        score_threshold: float = 0.25,
        topk: int = 8,
    ) -> Dict[str, Any]:
        """Return dict with boxes in pixel XYXY, scores, and class name per box."""
        w, h = image.size
        names = list(lami_classes.keys())
        visual_descs = {k: list(v) for k, v in lami_classes.items()}
        if not names:
            return {
                "prompt_summary": "",
                "names": [],
                "boxes": [],
                "image_size": [w, h],
                "mock": False,
                "note": "empty lami_classes",
            }

        self._ensure_loaded()

        import cv2
        import torch
        from detectron2.layers import batched_nms

        img_rgb = image.convert("RGB")
        img_cv = cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2BGR)
        os.chdir(str(self._lami_detr_dir))
        try:
            tensor = self._pil_to_tensor_bgr(img_cv, self._augmentation)
            with torch.no_grad():
                outputs = self._model(tensor.to(self._device), names, visual_descs, [image.size])[0]
            instances = outputs["instances"]
            if len(instances) == 0:
                return {
                    "prompt_summary": "|".join(names),
                    "names": names,
                    "visual_descs": visual_descs,
                    "boxes": [],
                    "image_size": [w, h],
                    "mock": False,
                    "coord_space": "pixel_xyxy",
                    "note": "no_detection",
                }
            boxes = instances.pred_boxes.tensor.cpu()
            scores = instances.scores.cpu()
            labels = instances.pred_classes.cpu()
        finally:
            os.chdir(self._orig_cwd)

        label_names = [names[i] for i in labels.tolist()]
        keep_inds = batched_nms(boxes, scores, labels, nms_iou)
        boxes = boxes[keep_inds]
        scores = scores[keep_inds]
        labels = labels[keep_inds]
        label_names = [label_names[i] for i in keep_inds]

        mask = scores > score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        label_names = [label_names[i] for i, flag in enumerate(mask.tolist()) if flag]

        out_boxes: List[Dict[str, Any]] = []
        for i in range(min(len(boxes), topk)):
            x1, y1, x2, y2 = boxes[i].tolist()
            bbox_px = [float(x1), float(y1), float(x2), float(y2)]
            out_boxes.append(
                {
                    "bbox": bbox_px,
                    "bbox_1000": bbox_pixel_to_1000(bbox_px, (w, h)),
                    "score": float(scores[i].item()),
                    "class_name": label_names[i],
                }
            )

        return {
            "prompt_summary": "|".join(names),
            "names": names,
            "visual_descs": visual_descs,
            "boxes": out_boxes,
            "image_size": [w, h],
            "mock": False,
            "coord_space": "pixel_xyxy",
        }


def _pick_doubao_prompt(prompt_info: Dict[str, Any]) -> str:
    for k in ("spatial_query", "part_prompt", "object_prompt"):
        v = prompt_info.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    classes = prompt_info.get("lami_classes") or {}
    if isinstance(classes, dict) and classes:
        return " ; ".join(classes.keys())
    return "Locate target object and output <bbox>x1 y1 x2 y2</bbox>"


def _parse_bbox_1000(text: str) -> List[List[float]]:
    # Supports: <bbox>175 98 791 476</bbox>
    out: List[List[float]] = []
    for m in re.finditer(r"<bbox>\s*([0-9\.\s,]+)\s*</bbox>", text or "", flags=re.I):
        nums = re.findall(r"\d+\.?\d*", m.group(1))
        if len(nums) >= 4:
            out.append([float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])])
    # Fallback: first 4 numbers in text
    if not out:
        nums = re.findall(r"\d+\.?\d*", text or "")
        if len(nums) >= 4:
            out.append([float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])])
    return out


def _doubao_grounding(
    image: Image.Image,
    prompt_info: Dict[str, Any],
    topk: int = 1,
) -> Dict[str, Any]:
    from openai import OpenAI
    import base64
    import io

    api_key = (
        os.environ.get("ARK_API_KEY")
        or os.environ.get("AGENT_HARNESS_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("Missing ARK_API_KEY for doubao grounding")

    model = os.environ.get("AGENT_HARNESS_DOUBAO_GROUNDING_MODEL", "doubao-seed-2-0-lite-260215")
    base_url = os.environ.get("AGENT_HARNESS_DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    client = OpenAI(api_key=api_key, base_url=base_url)

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    prompt = _pick_doubao_prompt(prompt_info)
    user_text = (
        f"{prompt}\n"
        "请只输出 bounding box，格式严格为 <bbox>x1 y1 x2 y2</bbox>。"
        "坐标使用归一化到1000*1000网格（范围0-999）。不要输出其他文本。"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": user_text},
                ],
            }
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content or ""
    b1000_list = _parse_bbox_1000(content)[: max(1, topk)]
    w, h = image.size
    boxes: List[Dict[str, Any]] = []
    for i, b1000 in enumerate(b1000_list):
        bpx = bbox_1000_to_pixel(b1000, (w, h))
        boxes.append(
            {
                "bbox": [float(x) for x in bpx],
                "bbox_1000": [float(x) for x in b1000],
                "score": max(0.1, 1.0 - 0.1 * i),
                "class_name": "doubao_grounded",
            }
        )
    return {
        "prompt_summary": prompt,
        "names": ["doubao_grounded"],
        "boxes": boxes,
        "image_size": [w, h],
        "mock": False,
        "coord_space": "pixel_xyxy",
        "raw_doubao_response": content,
        "backend": "doubao_grounding",
    }


def tool_lami_detr_grounder(
    image: Image.Image,
    prompt_info: Dict[str, Any],
    *,
    nms_iou: float = 0.7,
    score_threshold: float = 0.25,
    topk: int = 8,
    backend: str = "lami",
    grounder: Optional[LaMIDetrGrounder] = None,
) -> Dict[str, Any]:
    if (backend or "lami").lower() in ("doubao", "doubao_grounding", "ark"):
        return _doubao_grounding(image, prompt_info, topk=topk)
    g = grounder or get_grounder()
    classes = prompt_info.get("lami_classes") or {}
    return g.infer(
        image,
        classes,
        nms_iou=nms_iou,
        score_threshold=score_threshold,
        topk=topk,
    )
