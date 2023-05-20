import numpy as np
from itertools import compress
import copy
import torch
from detectron2.structures import BoxMode, Boxes, pairwise_iou

def bbox_enclose(box, other_boxes):
    x_inside_min = box[0] < other_boxes[:, 0]
    y_inside_min = box[1] < other_boxes[:, 1]
    x_inside_max = box[2] > other_boxes[:, 2]
    y_inside_max = box[3] > other_boxes[:, 3]
    inside_box = x_inside_min & x_inside_max & y_inside_min & y_inside_max
    return other_boxes[inside_box], inside_box

def bbox_inside(box, other_boxes):
    ixmin = np.maximum(other_boxes[:, 0], box[0])
    iymin = np.maximum(other_boxes[:, 1], box[1])
    ixmax = np.minimum(other_boxes[:, 2], box[2])
    iymax = np.minimum(other_boxes[:, 3], box[3])
    intersection = np.stack([ixmin, iymin, ixmax, iymax], axis=1).astype(np.int32)
    inters = np.maximum(ixmax - ixmin, 0.) * np.maximum(iymax - iymin, 0.)
    inside_box = (inters>50)
    return intersection[inside_box], inside_box

def bbox_scale(boxes, height, width):
    scale_pixels = 20
    x_min, y_min = boxes[:, 0]-scale_pixels, boxes[:, 1]-scale_pixels
    x_max, y_max = boxes[:, 2]+scale_pixels, boxes[:, 3]+scale_pixels
    x_min, y_min = x_min.clip(min=0), y_min.clip(min=0)
    x_max, y_max = x_max.clip(max=width), y_max.clip(max=height)
    scaled_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
    return scaled_boxes

def bbox_scale_by_factor(boxes, im_height, im_width):
    scale_factor = 10.0
    x_min, y_min = boxes[:, 0], boxes[:, 1]
    x_max, y_max = boxes[:, 2], boxes[:, 3]
    width, height = x_max-x_min, y_max-y_min
    cx, cy = x_min + width/2, y_min + height/2
    width, height = width * scale_factor, height * scale_factor
    x_min, x_max = cx - width/2, cx + width/2
    y_min, y_max = cy - height/2, cy + height/2
    x_min, y_min = x_min.clip(min=0), y_min.clip(min=0)
    x_max, y_max = x_max.clip(max=im_width), y_max.clip(max=im_height)
    scaled_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
    return scaled_boxes

def uniform_cropping(data_dict):
    height, width = data_dict["height"], data_dict["width"]
    new_boxes = np.zeros((0, 4), dtype=np.int32)
    mid_x, mid_y = width//2, height//2
    new_boxes = np.append(new_boxes, np.arrray([0, 0, mid_x, mid_y]).reshape(1, -1), axis=0)
    new_boxes = np.append(new_boxes, np.arrray([mid_x, 0, width, mid_y]).reshape(1, -1), axis=0)
    new_boxes = np.append(new_boxes, np.arrray([0, mid_y, mid_x, height]).reshape(1, -1), axis=0)
    new_boxes = np.append(new_boxes, np.arrray([mid_x, mid_y, width, height]).reshape(1, -1), axis=0)
    return new_boxes


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


def compute_crops(data_dict, cfg, cluster_id=10, inner_crop=False):
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
        isect_boxes, cluster_components = bbox_inside(new_boxes[i], gt_boxes)
        #check = (inside_flag&cluster_components)
        #if check.sum()==len(gt_boxes):
        #    continue
        data_dict_crop = copy.deepcopy(data_dict)
        data_dict_crop['full_image'] = False
        if inner_crop:
            data_dict_crop['inner_crop_area'] = new_boxes[i]
            data_dict_crop["two_stage_crop"] = True
        else:
            data_dict_crop['crop_area'] = new_boxes[i]    
        data_dict_crop['height'] = new_boxes[i, 3] - new_boxes[i, 1]
        data_dict_crop['width'] = new_boxes[i, 2] - new_boxes[i, 0]
        x1, y1 = new_boxes[i, 0], new_boxes[i, 1]
        ref_point = np.array([x1, y1, x1, y1], dtype=np.int32)
        data_dict_crop['annotations'] =  list(compress(data_dict_crop['annotations'], cluster_components))
        for j, obj in enumerate(data_dict_crop['annotations']):
            obj['bbox'] = list(isect_boxes[j] - ref_point)
        new_data_dicts.append(data_dict_crop)
        inside_flag &= (~cluster_components)

    #finally change the original datadict by adding new cluster classes and the corresponding boxes
    #if inside_flag.sum()!=0:
    #   data_dict["annotations"] = list(compress(data_dict["annotations"], inside_flag))
    for i in range(len(new_boxes)):
        crop_annotation = copy.deepcopy(data_dict["annotations"][0])
        crop_annotation['category_id'] = cluster_id
        crop_annotation['bbox'] = list(new_boxes[i])
        crop_annotation['iscrowd'] = 0
        data_dict["annotations"].append(crop_annotation)

    return data_dict, new_data_dicts