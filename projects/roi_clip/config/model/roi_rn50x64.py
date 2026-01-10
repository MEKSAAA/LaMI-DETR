import copy
import torch.nn as nn

from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.backbone import ModifiedResNet, CLIPStem, AttentionPool2d

from projects.roi_clip.modeling import (
    ROI_CLIP,
)


model = L(ROI_CLIP)(
    query_path= "dataset/metadata/lvis_visual_desc_rn50x64.npy",# not for train
    eval_query_path = "dataset/metadata/lvis_visual_desc_rn50x64.npy",
    pixel_mean=[122.7709383, 116.7460125, 104.09373615000001],
    pixel_std=[68.5005327, 66.6321579, 70.32316304999999],
    device="cuda",
    backbone=L(ModifiedResNet)(
        stem=L(CLIPStem)(in_channels=3, out_channels=128, norm="FrozenBN"),
        stages=L(ModifiedResNet.make_default_stages)(
            depth='50x64',
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res3", "res4", "res5"],
        freeze_at=4,
        output_dim=1024,
        attnpool=L(AttentionPool2d)(
            spacial_dim=9, 
            embed_dim="${..stem.out_channels}", 
            num_heads=32, 
            output_dim="${..output_dim}",
        )
    ),
    
)

