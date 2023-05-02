# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import json
import os
import torch.utils.data
from detectron2.data.common import (
    DatasetFromList,
    MapDataset,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.data.build import (
    trivial_batch_collator,
    worker_init_reset_seed,
    get_detection_dataset_dicts,
    build_batch_data_loader,
)

from croptrain.data.dataset_mapper import DatasetMapperDensityCrop


"""
This file contains the default logic to build a dataloader for training or testing.
"""


def divide_label_unlabel(dataset_dicts, cfg):
    dataset_name = cfg.DATASETS.TRAIN[0].split("_")[0]
    seed_file = os.path.join("dataseed", dataset_name + "_filenames.txt")
    with open(seed_file) as f:
        file_names = json.load(f)
    num_all = len(file_names["imagenames"])
    num_label = int(cfg.DATALOADER.SUP_PERCENT / 100.0 * num_all)

    # generate a permutation of images
    np.random.seed(cfg.DATALOADER.RANDOM_DATA_SEED)
    random_perm_index = np.random.permutation(num_all)
    shuffled_images = [file_names["imagenames"][x] for x in random_perm_index]
    labeled_image_ids = shuffled_images[:num_label]

    label_dicts = []
    unlabel_dicts = []

    for i in range(len(dataset_dicts)):
        file_name = dataset_dicts[i]["file_name"].split('/')[-1]
        #print(file_name)
        if file_name in labeled_image_ids:
            label_dicts.append(dataset_dicts[i])
        else:
            unlabel_dicts.append(dataset_dicts[i])
    return label_dicts, unlabel_dicts


# uesed by supervised-only baseline trainer
def build_detection_train_loader(cfg, mapper=None):

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # Divide into labeled and unlabeled sets according to supervision percentage
    label_dicts, unlabel_dicts = divide_label_unlabel(
        dataset_dicts,
        cfg
    )

    dataset = DatasetFromList(label_dicts, copy=False)

    if cfg.CROPTRAIN.USE_CROPS:
        mapper = DatasetMapperDensityCrop(cfg, True)
    if "dota" in cfg.DATASETS.TRAIN[0] or "dota" in cfg.DATASETS.TEST[0]:
        mapper = DatasetMapperDensityCrop(cfg, True)
    if "teldrone" in cfg.DATASETS.TRAIN[0] or "teldrone" in cfg.DATASETS.TEST[0]:
        mapper = DatasetMapperDensityCrop(cfg, True)
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))

    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = (
            RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                label_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
            )
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    # list num of labeled and unlabeled
    logger.info("Number of training samples " + str(len(dataset)))
    logger.info("Supervision percentage " + str(cfg.DATALOADER.SUP_PERCENT))

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# uesed by evaluation
def build_detection_test_loader(cfg, dataset_name, mapper=None):
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[
                list(cfg.DATASETS.TEST).index(dataset_name)
            ]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    dataset = DatasetFromList(dataset_dicts)
    if "dota" in cfg.DATASETS.TRAIN[0] or "dota" in cfg.DATASETS.TEST[0]:
        mapper = DatasetMapperDensityCrop(cfg, False)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


