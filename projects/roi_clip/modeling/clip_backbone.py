# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import math
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import (inverse_sigmoid, is_dist_avail_and_initialized,
                          load_class_freq, get_fed_loss_inds)

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils import comm
import torchvision

class ROI_CLIP(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        query_path,
        eval_query_path,
        device="cuda",
        pixel_mean: List[float] = [122.7709383, 116.7460125, 104.09373615000001],
        pixel_std: List[float] = [68.5005327, 66.6321579, 70.32316304999999],
        roi_resolution: int = 30,
        clip_head_path=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.resolution = roi_resolution

        if not 'ResNet' in type(self.backbone).__name__:
            self.clip_head = torch.load(clip_head_path)
            self.identical, self.thead = self.clip_head[0]
            self.head = self.clip_head[1]
        
        content_query_embedding = torch.tensor(np.load(query_path), dtype=torch.float32, device=device).contiguous()
        self.content_query_embedding = F.normalize(content_query_embedding, p=2, dim=1)
        eval_content_query_embedding = torch.tensor(np.load(eval_query_path), dtype=torch.float32, device=device).contiguous()
        self.eval_content_query_embedding = F.normalize(eval_content_query_embedding, p=2, dim=1)

 
    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        batch_size, _, H, W = images.tensor.shape
        img_masks = images.tensor.new_zeros(batch_size, H, W)
        img_sizes = [[1.0, 1.0] for _ in range(batch_size)]

        # original features
        if 'ResNet' in type(self.backbone).__name__:
            features = self.backbone(images.tensor)  # output feature dict
        else:
            _, features = self.backbone(images.tensor)  # output feature dict

        if 'ResNet' in type(self.backbone).__name__:
            h, w = features['res5'].shape[-2:]# 25 37
        else:
            h, w = features['p2'].shape[-2:]# 50 75
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets = self.prepare_targets(gt_instances)

        rpn_boxes = [copy.deepcopy(target['boxes']) for target in targets]

        for i in range(len(rpn_boxes)):
            rpn_boxes[i][:,[0,2]] = rpn_boxes[i][:,[0,2]] * img_sizes[i][0] * w
            rpn_boxes[i][:,[1,3]] = rpn_boxes[i][:,[1,3]] * img_sizes[i][1] * h
        if 'ResNet' in type(self.backbone).__name__:
            roi_features = torchvision.ops.roi_align(
                features['res5'],# [1, 2048, 25, 37]
                rpn_boxes,
                output_size=(self.backbone.attnpool.spacial_dim, self.backbone.attnpool.spacial_dim),
                spatial_scale=1.0,
                aligned=True)  # [900, 1024, 14, 14]
            # roi_features = self.backbone.stages[-1](roi_features)# [900, 2048, 7, 7]
            roi_features = self.backbone.attnpool(roi_features)# [900, 1024]
        else:
            roi_features = torchvision.ops.roi_align(
                features['p2'],
                rpn_boxes,
                output_size=(30, 30),
                spatial_scale=1.0,
                aligned=True)  # (bs * num_queries, c, 14, 14) [900, 768, 30, 30]
        
            roi_features = self.backbone.downsample_layers[3](roi_features)# [33, 768, 30, 30]->[33, 1536, 15, 15]
            roi_features = self.backbone.stages[3](roi_features)# [900, 1536, 15, 15]
            roi_features = self.identical(roi_features)# [900, 1536, 15, 15]
            roi_features = self.thead(roi_features)# [900, 1536]
            roi_features = self.head(roi_features)# [900, 768] TODO:
        roi_features = nn.functional.normalize(roi_features, dim=-1)# [50, 768]
        # roi_features = roi_features / roi_features.norm(dim=-1, keepdim=True)
        output = {}
        outputs_class = roi_features @ self.eval_content_query_embedding.t()# [1, 900, 1203]

        # prepare for loss computation
        output["pred_logits"] = outputs_class


        box_cls = output["pred_logits"]
        box_pred = [copy.deepcopy(target['boxes']) for target in targets]
        # labels = [copy.deepcopy(target['labels']) for target in targets]
        box_cls = box_cls.softmax(dim=-1)# [20, 80]
        pred = torch.argmax(box_cls, dim=-1)# [20]
        return pred

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_scores = targets_per_image.gt_scores
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            # gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "scores": gt_scores})
        return new_targets
