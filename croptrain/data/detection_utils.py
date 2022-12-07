# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import cv2
import numpy as np

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