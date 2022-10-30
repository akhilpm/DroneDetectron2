# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from email.mime import image
import logging
import cv2
import numpy as np
import torchvision.transforms as transforms
from croptrain.data.transforms.augmentation_impl import (
    GaussianBlur,
)


def build_strong_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

        randcrop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                ),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                ),
                transforms.RandomErasing(
                    p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                ),
                transforms.ToPILImage(),
            ]
        )
        augmentation.append(randcrop_transform)

        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)


def read_image(dataset_dict):
    file_name = dataset_dict['file_name']
    image = cv2.imread(file_name)

    if not dataset_dict['full_image']:
        crop_area = dataset_dict['crop_area']
        x1, y1, x2, y2 = crop_area[0], crop_area[1], crop_area[2], crop_area[3]
        image = image[y1:y2, x1:x2]
        if dataset_dict["two_stage_crop"]:
            crop_area = dataset_dict['inner_crop_area']
            x1, y1, x2, y2 = crop_area[0], crop_area[1], crop_area[2], crop_area[3]
            image = image[y1:y2, x1:x2]

    if len(image.shape) == 2:
        image = image[:,:,np.newaxis]
        image = np.concatenate((image,image,image), axis=2)
    image = image.astype(np.float32, copy=False)
    return image