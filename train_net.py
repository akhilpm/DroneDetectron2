#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import GeneralizedRCNN

from croptrain import add_croptrainer_config, add_ubteacher_config
from croptrain.engine.trainer import UBTeacherTrainer, BaselineTrainer
# hacky way to register
from croptrain.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from croptrain.modeling.proposal_generator.rpn import PseudoLabRPN
from croptrain.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import croptrain.data.datasets.builtin
from croptrain.data.datasets.visdrone import register_visdrone

from croptrain.modeling.meta_arch.ts_ensemble import EnsembleTSModel


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_croptrainer_config(cfg)
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu'
    if cfg.SEMISUPNET.USE_SEMISUP:
        Trainer = UBTeacherTrainer
    else:
        Trainer = BaselineTrainer

    if cfg.CROPTRAIN.USE_CROPS:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES += 1
    if "visdrone" in cfg.DATASETS.TRAIN[0] or "visdrone" in cfg.DATASETS.TEST[0]:
        data_dir = os.path.join(os.environ['SLURM_TMPDIR'], "VisDrone")
        if not args.eval_only:
            register_visdrone(cfg.DATASETS.TRAIN[0], data_dir, cfg, True)
        register_visdrone(cfg.DATASETS.TEST[0], data_dir, cfg, False)

    if args.eval_only:
        if cfg.SEMISUPNET.USE_SEMISUP:
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            if cfg.CROPTRAIN.USE_CROPS:
                res = Trainer.test_crop(cfg, model)
            else:
                res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("No of gpus used: {}".format(args.num_gpus))
    print("Cuda detected {} gpus".format(torch.cuda.device_count()))

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
