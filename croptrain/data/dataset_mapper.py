# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
from PIL import Image
import torch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from croptrain.data.detection_utils import read_image


class DatasetMapperDensityCrop(DatasetMapper):
    """
    Reads the images, and if it is a crop, get the corresponding crop from the image

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        super(DatasetMapperDensityCrop, self).__init__(cfg, is_train)
        augmentations = utils.build_augmentation(cfg, is_train)
        self.augmentations = T.AugmentationList(augmentations)
        augmentations_crop = build_augmentation_crop(cfg, is_train)
        self.augmentations_crop = T.AugmentationList(augmentations_crop)
        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train
        self.cfg = cfg

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = read_image(dataset_dict)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.StandardAugInput(image, sem_seg=None)
        if dataset_dict['full_image']:
            transforms = self.augmentations(aug_input)
        else:
            if "dota" in self.cfg.DATASETS.TRAIN[0]  or "dota" in self.cfg.DATASETS.TEST[0]:
                if dataset_dict["two_stage_crop"]:
                    transforms = self.augmentations_crop(aug_input)
                else:
                    transforms = self.augmentations(aug_input)
            else:
                transforms = self.augmentations_crop(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)    

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        
        return dataset_dict


def build_augmentation_crop(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    min_size = cfg.CROPTRAIN.CROPSIZE
    max_size = cfg.CROPTRAIN.MAX_CROPSIZE
    if is_train:
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        sample_style = "choice"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    return augmentation
