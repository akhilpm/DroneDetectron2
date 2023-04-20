# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_croptrainer_config(cfg):
    _C = cfg
    _C.CROPTRAIN = CN()
    _C.CROPTRAIN.USE_CROPS = False
    _C.CROPTRAIN.CLUSTER_THRESHOLD = 0.1
    _C.CROPTRAIN.CROPSIZE = (320, 476, 512, 640)
    _C.CROPTRAIN.MAX_CROPSIZE = 800
    _C.CROPTEST = CN()
    _C.CROPTEST.CLUS_THRESH = 0.3
    _C.CROPTEST.MAX_CLUSTER = 5
    _C.CROPTEST.CROPSIZE = 800
    _C.CROPTEST.DETECTIONS_PER_IMAGE = 800
    _C.MODEL.CUSTOM = CN()
    _C.MODEL.CUSTOM.FOCAL_LOSS_GAMMAS = []
    _C.MODEL.CUSTOM.FOCAL_LOSS_ALPHAS = []

    _C.MODEL.CUSTOM.CLS_WEIGHTS = []
    _C.MODEL.CUSTOM.REG_WEIGHTS = []
     
    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 42  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"
