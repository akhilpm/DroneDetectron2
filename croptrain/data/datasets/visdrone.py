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


logger = logging.getLogger(__name__)
thing_classes = None
id_map = None

def bbox_inside(box, other_boxes):
    x_inside_min = box[0] < other_boxes[:, 0]
    y_inside_min = box[1] < other_boxes[:, 1]
    x_inside_max = box[2] > other_boxes[:, 2]
    y_inside_max = box[3] > other_boxes[:, 3]
    inside_box = x_inside_min & x_inside_max & y_inside_min & y_inside_max
    return inside_box

def bbox_scale(boxes, height, width):
    x_min, y_min = boxes[:, 0]-20, boxes[:, 1]-20
    x_max, y_max = boxes[:, 2]+20, boxes[:, 3]+20
    x_min, y_min = x_min.clip(min=0), y_min.clip(min=0)
    x_max, y_max = x_max.clip(max=width), y_max.clip(max=height)
    scaled_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
    return scaled_boxes


def compute_one_stage_clusters(data_dict, bboxes, seg_areas, cfg, stage=1):
    bboxes = Boxes(bboxes)
    overlaps = pairwise_iou(bboxes, bboxes)
    connectivity = (overlaps > cfg.CROPTRAIN.CLUSTER_THRESHOLD)
    new_boxes = np.zeros((0, 4), dtype=np.int32)
    new_seg_areas = []
    image_area = data_dict["height"] * data_dict["width"]
    while len(connectivity)>0:
        connections = connectivity.sum(dim=1)
        max_connected, max_connections = torch.argmax(connections), torch.max(connections)
        if max_connections==1:
            break
        cluster_components = torch.nonzero(connectivity[max_connected]).view(-1)
        other_boxes = torch.nonzero(~connectivity[max_connected]).view(-1)
        cluster_member_areas = seg_areas[cluster_components]
        cluster_member_areas = cluster_member_areas / float(image_area)

        #if the bounding boxes inside a cluster are sufficiently big, detect it from the original image itself.
        if cluster_member_areas.min()>0.2:
            bboxes.tensor = bboxes.tensor[other_boxes]
            seg_areas = seg_areas[other_boxes]
            connectivity = connectivity[:, other_boxes]
            connectivity = connectivity[other_boxes, :]
            if stage==1:
                data_dict['annotations'] = list(compress(data_dict["annotations"], other_boxes))
            continue

        cluster_members = bboxes.tensor[cluster_components]
        x1, y1 = cluster_members[:, 0].min()-20, cluster_members[:, 1].min()-20
        x2, y2 = cluster_members[:, 2].max()+20, cluster_members[:, 3].max()+20
        x1, y1 = torch.clamp(x1, min=0), torch.clamp(y1, min=0)
        x2, y2 = torch.clamp(x2, max=data_dict['width']), torch.clamp(y2, max=data_dict['height'])
        crop_area = np.array([int(x1), int(y1), int(x2), int(y2)]).astype(np.int32)
        bboxes.tensor = bboxes.tensor[other_boxes]
        seg_areas = seg_areas[other_boxes]

        if stage==1:
            data_dict['annotations'] = list(compress(data_dict["annotations"], other_boxes))
        new_boxes = np.append(new_boxes, crop_area.reshape(1, -1), axis=0)
        new_seg_areas.append((x2-x1) * (y2- y1))
        connectivity = connectivity[:, other_boxes]
        connectivity = connectivity[other_boxes, :]

    return data_dict, new_boxes, np.array(new_seg_areas)



def compute_crops(data_dict, cfg):
    data_dict_this_image = copy.deepcopy(data_dict)
    new_data_dicts = []
    gt_boxes = np.vstack([obj['bbox'] for obj in data_dict_this_image["annotations"]])
    scaled_boxes = bbox_scale(gt_boxes.copy(), data_dict_this_image['height'], data_dict_this_image['width'])
    inside_flag = np.ones(len(data_dict['annotations'])).astype(np.bool8)
    seg_areas = Boxes(gt_boxes).area()

    #stage 1 - merging
    data_dict_this_image, new_boxes, new_seg_areas = compute_one_stage_clusters(data_dict_this_image, scaled_boxes, seg_areas, cfg, stage=1)
    #stage 2 - merging
    data_dict_this_image, new_boxes, new_seg_areas = compute_one_stage_clusters(data_dict_this_image, new_boxes, new_seg_areas, cfg, stage=2)

    #extract boxes inside each cluster
    for i in range(len(new_boxes)):
        data_dict_crop = copy.deepcopy(data_dict)
        data_dict_crop['full_image'] = False
        data_dict_crop['crop_area'] = new_boxes[i]
        data_dict_crop['height'] = new_boxes[i, 3] - new_boxes[i, 1]
        data_dict_crop['width'] = new_boxes[i, 2] - new_boxes[i, 0]
        x1, y1 = new_boxes[i, 0], new_boxes[i, 1]
        cluster_components = bbox_inside(new_boxes[i], gt_boxes)
        ref_point = np.array([x1, y1, x1, y1], dtype=np.int32)
        data_dict_crop['annotations'] =  list(compress(data_dict_crop['annotations'], cluster_components))
        for obj in data_dict_crop['annotations']:
            obj['bbox'] = obj['bbox'] - ref_point
        new_data_dicts.append(data_dict_crop)
        inside_flag &= (~cluster_components)

    #finally change the original datadict by adding new cluster classes and the corresponding boxes
    data_dict["annotations"] = list(compress(data_dict["annotations"], inside_flag))
    for i in range(len(new_boxes)):
        crop_annotation = copy.deepcopy(data_dict["annotations"][0])
        crop_annotation['category_id'] = 11
        crop_annotation['bbox'] = list(new_boxes[i])
        data_dict["annotations"].append(crop_annotation)

    return data_dict, new_data_dicts


def  extract_crops_from_image(dataset_dicts, cfg):
    old_dataset_dicts = []
    new_dataset_dicts = []
    for i, data_dict in enumerate(dataset_dicts):
        updated_dict, crop_dicts = compute_crops(data_dict, cfg)
        new_dataset_dicts += crop_dicts
        old_dataset_dicts.append(updated_dict)

    return old_dataset_dicts + new_dataset_dicts


def load_visdrone_instances(dataset_name, data_dir, cfg, is_train, extra_annotation_keys=None):
    split = dataset_name.split("_")[-1]
    json_file = os.path.join(data_dir, "annotations_VisDrone_%s.json" % split)
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
        cat_ids = cat_ids[:-1] #ingore the last "others" class
        cats = coco_api.loadCats(cat_ids)
        if cfg.CROPTRAIN.USE_CROPS:
            cats.append({'id':11, 'name':'cluster', 'supercategory':'none'})
            cat_ids.append(11)
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
        dataset_dicts.append(record)
    if cfg.CROPTRAIN.USE_CROPS and is_train:
        dataset_dicts  = extract_crops_from_image(dataset_dicts, cfg)
    return dataset_dicts


def register_visdrone(dataset_name, data_dir, cfg, is_train):
    from pycocotools.coco import COCO
    metadata = {}

    # 1. register a function which returns dicts
    DatasetCatalog.register(dataset_name, lambda: load_visdrone_instances(dataset_name, data_dir, cfg, is_train))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    split = dataset_name.split("_")[-1]
    json_file = os.path.join(data_dir, "annotations_VisDrone_%s.json" % split)
    image_root = os.path.join(data_dir, split, "images")
    coco_api = COCO(json_file)
    cat_ids = sorted(coco_api.getCatIds())
    cat_ids = cat_ids[:-1] #ingore the last "others" class
    cats = coco_api.loadCats(cat_ids)
    if cfg.CROPTRAIN.USE_CROPS:
        cats.append({'id':11, 'name':'cluster', 'supercategory':'none'})
        cat_ids.append(11)
    # The categories in a custom json file may not be sorted.
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    id_map = {v: i for i, v in enumerate(cat_ids)}
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco",
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=id_map, **metadata
    )