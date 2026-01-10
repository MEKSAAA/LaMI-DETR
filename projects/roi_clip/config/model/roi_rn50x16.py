import copy
import torch.nn as nn

from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.backbone import ModifiedResNet, CLIPStem

from projects.roi_clip.modeling import (
    ROI_CLIP,
)


model = L(ROI_CLIP)(
    query_path= "dataset/metadata/lvis_clip_rn50x4_imagenet.npy",# not for train
    eval_query_path = "dataset/metadata/lvis_clip_rn50x4_imagenet.npy",
    pixel_mean=[122.7709383, 116.7460125, 104.09373615000001],
    pixel_std=[68.5005327, 66.6321579, 70.32316304999999],
    device="cuda",
    backbone=L(ModifiedResNet)(
        stem=L(CLIPStem)(in_channels=3, out_channels=80, norm="FrozenBN"),
        stages=L(ModifiedResNet.make_default_stages)(
            depth='50x4',
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res3", "res4", "res5"],
        freeze_at=4,
    ),
)

