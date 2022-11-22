from __future__ import annotations
import copy
import os
import contextlib
import io
import logging
from itertools import compress
from tkinter import image_names
import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager
from detectron2.structures import Boxes, BoxMode
from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, Boxes, pairwise_iou
from utils.box_utils import compute_crops, bbox_inside


logger = logging.getLogger(__name__)

def get_overlapping_sliding_window(data_dict, stepSize=1000, windowSize=1500):
    height, width = data_dict["height"], data_dict["width"]
    new_boxes = np.zeros((0, 4), dtype=np.int32)
    for y in range(0, height-320, stepSize):
        for x in range(0, width-320, stepSize):
            ymax, xmax = y+windowSize, x+windowSize
            if ymax>height:
                ymax = height
            if xmax>width:
                xmax = width
            new_boxes = np.append(new_boxes, np.array([x, y, xmax, ymax]).reshape(1, -1), axis=0)
    return new_boxes


def get_datadicts_from_sliding_windows(new_boxes, data_dict):
    if len(data_dict["annotations"])==0:
        return None
    gt_boxes = np.vstack([obj['bbox'] for obj in data_dict["annotations"]])
    new_data_dicts = []
    #extract boxes inside each cluster
    for i in range(len(new_boxes)):
        isect_boxes, cluster_components = bbox_inside(new_boxes[i], gt_boxes)
        data_dict_crop = copy.deepcopy(data_dict)
        data_dict_crop['full_image'] = False
        data_dict_crop["two_stage_crop"] = False
        data_dict_crop['crop_area'] = new_boxes[i]
        data_dict_crop["inner_crop_area"] = np.array([-1, -1, -1, -1], dtype=np.float32)
        data_dict_crop['height'] = new_boxes[i, 3] - new_boxes[i, 1]
        data_dict_crop['width'] = new_boxes[i, 2] - new_boxes[i, 0]
        if "annotations" in data_dict:
            x1, y1 = new_boxes[i, 0], new_boxes[i, 1]
            ref_point = np.array([x1, y1, x1, y1], dtype=np.int32)
            data_dict_crop['annotations'] =  list(compress(data_dict_crop['annotations'], cluster_components))
            if len(data_dict_crop['annotations'])==0:
                continue
            for j, obj in enumerate(data_dict_crop['annotations']):
                obj['bbox'] = list(isect_boxes[j] - ref_point)
        new_data_dicts.append(data_dict_crop)
    return new_data_dicts

def  extract_crops_from_image(dataset_dicts, cfg):
    old_dataset_dicts = []
    new_dataset_dicts = []
    for i, data_dict in enumerate(dataset_dicts):
        updated_dict, crop_dicts = compute_crops(data_dict, cfg,  cluster_id=2, inner_crop=True)
        new_dataset_dicts += crop_dicts
        old_dataset_dicts.append(updated_dict)
    total_dicts = old_dataset_dicts + new_dataset_dicts
    return total_dicts


def load_teldrone_instances(dataset_name, data_dir, cfg, is_train, extra_annotation_keys=None):
    split = dataset_name.split("_")[-1]
    json_file = os.path.join(data_dir, "annotations_TelDrone_%s.json" % split)
    image_path = os.path.join(data_dir, split, "images")
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
         coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        if cfg.CROPTRAIN.USE_CROPS and is_train:
            cats.append({'id':3, 'name':'cluster', 'supercategory':'none'})
            cat_ids.append(3)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.set(thing_classes=thing_classes)
        

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning("""
                    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
                    """
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.set(thing_dataset_id_to_contiguous_id=id_map)
        
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_path, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        record["full_image"] = True
        record["crop_area"] = np.array([-1, -1, -1, -1], dtype=np.float32)
        record["two_stage_crop"] = False

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if obj["category_id"] in (0, 11):
                continue
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )
            obj['bbox'] = BoxMode.convert(obj['bbox'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)    
            obj["bbox_mode"] = BoxMode.XYXY_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        if  is_train:
            new_boxes = get_overlapping_sliding_window(record)
            new_data_dicts = get_datadicts_from_sliding_windows(new_boxes, record)
            if new_data_dicts:
                dataset_dicts += new_data_dicts
        else:        
            dataset_dicts.append(record)
    if cfg.CROPTRAIN.USE_CROPS and is_train:
        dataset_dicts  = extract_crops_from_image(dataset_dicts, cfg)
    return dataset_dicts


def register_teldrone(dataset_name, data_dir, cfg, is_train):
    from pycocotools.coco import COCO
    metadata = {}

    # 1. register a function which returns dicts
    DatasetCatalog.register(dataset_name, lambda: load_teldrone_instances(dataset_name, data_dir, cfg, is_train))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    split = dataset_name.split("_")[-1]
    json_file = os.path.join(data_dir, "annotations_TelDrone_%s.json" % split)
    image_root = os.path.join(data_dir, split, "images")
    coco_api = COCO(json_file)
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    if cfg.CROPTRAIN.USE_CROPS and is_train:
        cats.append({'id':3, 'name':'cluster', 'supercategory':'none'})
        cat_ids.append(3)
    # The categories in a custom json file may not be sorted.
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    id_map = {v: i for i, v in enumerate(cat_ids)}
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco",
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=id_map, **metadata
    )