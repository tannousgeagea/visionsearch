# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data

from .coco import build as build_coco
from .objects365 import build as build_objects365
from .torchvision_datasets import CocoDetection


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == "objects365":
        return build_objects365(image_set, args)
    if args.dataset_file == "coco":
        return build_coco(image_set, args)
    if args.dataset_file == "coco_panoptic":
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic

        return build_coco_panoptic(image_set, args)
    raise ValueError(f"dataset {args.dataset_file} not supported")
