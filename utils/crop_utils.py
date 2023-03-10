import copy
import numpy as np
import torch
from torchvision.transforms import Resize
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes, pairwise_iou

def get_dict_from_crops(crops, input_dict, CROPSIZE=512, inner_crop=False, with_image=True):
    from croptrain.data.detection_utils import read_image
    if len(crops)==0:
        return []
    if isinstance(crops, Instances):
        crops = crops.pred_boxes.tensor.cpu().numpy().astype(np.int32)
    if with_image:
        transform = Resize(CROPSIZE)
    crop_dicts = []
    for i in range(len(crops)):
        x1, y1, x2, y2 = crops[i, 0], crops[i, 1], crops[i, 2], crops[i, 3]
        crop_size_min = min(x2-x1, y2-y1)
        if crop_size_min<=0:
            continue
        crop_dict = copy.deepcopy(input_dict)
        crop_dict['full_image'] = False
        if inner_crop:
            crop_dict["two_stage_crop"] = True
            crop_dict["inner_crop_area"] = np.array([x1, y1, x2, y2]).astype(np.int32)
        else:
            crop_dict['crop_area'] = np.array([x1, y1, x2, y2]).astype(np.int32)
        if with_image:
            crop_region = read_image(crop_dict)
            crop_region = torch.as_tensor(np.ascontiguousarray(crop_region.transpose(2, 0, 1)))
            crop_region = transform(crop_region)
            crop_dict["image"] = crop_region
        crop_dict["height"] = (y2-y1)
        crop_dict["width"] = (x2-x1)
        crop_dicts.append(crop_dict)
    return crop_dicts


def project_boxes_to_image(data_dict, crop_sizes, boxes):
    num_bbox_reg_classes = boxes.shape[1] // 4
    output_height, output_width = data_dict.get("height"), data_dict.get("width")
    new_size = (output_height, output_width)
    scale_x, scale_y = (
        output_width / crop_sizes[1],
        output_height / crop_sizes[0],
    )
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.scale(scale_x, scale_y)
    boxes.clip(new_size)
    boxes = boxes.tensor

    #shift to the proper position of the crop in the image
    if not data_dict["full_image"]:
        if data_dict["two_stage_crop"]:
            x1, y1 = data_dict['inner_crop_area'][0], data_dict['inner_crop_area'][1]
            ref_point = torch.tensor([x1, y1, x1, y1]).to(boxes.device)
            boxes = boxes + ref_point
        x1, y1 = data_dict["crop_area"][0], data_dict["crop_area"][1]
        ref_point = torch.tensor([x1, y1, x1, y1]).to(boxes.device)
        boxes = boxes + ref_point
    boxes = boxes.view(-1, num_bbox_reg_classes * 4) # R x C.4
    return boxes


def merge_cluster_boxes(cluster_boxes, cfg):
    if len(cluster_boxes)==0:
        return None
    if len(cluster_boxes)==1:
        box = cluster_boxes.pred_boxes.tensor.cpu().numpy().astype(np.int32).reshape(1, -1)
        return box

    overlaps = pairwise_iou(cluster_boxes.pred_boxes, cluster_boxes.pred_boxes)
    connectivity = (overlaps > cfg.CROPTRAIN.CLUSTER_THRESHOLD)
    new_boxes = np.zeros((0, 4), dtype=np.int32)
    while len(connectivity)>0:
        connections = connectivity.sum(dim=1)
        max_connected, max_connections = torch.argmax(connections), torch.max(connections)
        cluster_components = torch.nonzero(connectivity[max_connected]).view(-1)
        other_boxes = torch.nonzero(~connectivity[max_connected]).view(-1)
        if max_connections==1:
            box = cluster_boxes.pred_boxes.tensor[max_connected]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        else:
            cluster_members = cluster_boxes.pred_boxes.tensor[cluster_components]
            x1, y1 = cluster_members[:, 0].min(), cluster_members[:, 1].min()
            x2, y2 = cluster_members[:, 2].max(), cluster_members[:, 3].max()
        crop_area = np.array([int(x1), int(y1), int(x2), int(y2)]).astype(np.int32)
        new_boxes = np.append(new_boxes, crop_area.reshape(1, -1), axis=0)
        connectivity = connectivity[:, other_boxes]
        connectivity = connectivity[other_boxes, :]

    return new_boxes