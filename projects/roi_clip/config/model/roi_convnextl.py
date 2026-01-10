import copy
import torch.nn as nn

from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.backbone import ConvNeXt

from projects.roi_clip.modeling import (
    ROI_CLIP,
)


model = L(ROI_CLIP)(
    query_path= "dataset/metadata/coco_clip_convnextl_imagenet.npy",# not for train
    eval_query_path = "dataset/metadata/coco_clip_convnextl_imagenet.npy",
    clip_head_path = "pretrained_models/clip_convnext_large_head.pth",
    pixel_mean=[122.7709383, 116.7460125, 104.09373615000001],
    pixel_std=[68.5005327, 66.6321579, 70.32316304999999],
    device="cuda",
    backbone=L(ConvNeXt)(
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=(1, 2, 3),
        frozen_stages=4,
        score_ensemble=True,
    ),
)

