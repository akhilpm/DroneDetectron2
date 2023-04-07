from turtle import width
from croptrain.data.datasets.dota import get_datadicts_from_sliding_windows, get_sliding_window_patches, get_overlapping_sliding_window
import numpy as np
import torch
from detectron2.evaluation import DatasetEvaluator
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator, DatasetEvaluators
import os
import datetime
import time
import copy
from contextlib import ExitStack, contextmanager
import logging
from torchvision.transforms import Resize
from detectron2.structures.instances import Instances
from utils.box_utils import bbox_inside_old
from utils.plot_utils import plot_detections
from detectron2.utils.logger import log_every_n_seconds
from utils.box_utils import compute_crops
from utils.crop_utils import get_dict_from_crops
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
logging.basicConfig(level=logging.INFO)


def prune_boxes_inside_cluster(cluster_dicts, boxes, scores):
    #num_classes = (boxes.shape[1] // 4) + 1
    boxes = boxes[0].reshape(-1, 4)
    scores = scores[0]
    for cluster_dict in cluster_dicts:
        crop_area = cluster_dict["crop_area"]
        inside_boxes = bbox_inside_old(crop_area, boxes)
        boxes = boxes[~inside_boxes]
        scores = scores[~inside_boxes]
    return [boxes], [scores]


def inference_dota(model, data_loader, evaluator, cfg, iter):
    from detectron2.utils.comm import get_world_size
    dataset_dicts = get_detection_dataset_dicts(cfg.DATASETS.TEST, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS)

    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()
    save_dir = os.path.join(cfg.OUTPUT_DIR, "detections")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    cluster_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES - 1
    with ExitStack() as stack:
        if isinstance(model, torch.nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            #new_boxes = get_sliding_window_patches(dataset_dicts[idx])
            new_boxes = get_overlapping_sliding_window(dataset_dicts[idx])
            new_data_dicts = get_dict_from_crops(new_boxes, inputs[0], cfg.INPUT.MIN_SIZE_TEST)
            image_shapes = [(dataset_dicts[idx].get("height"), dataset_dicts[idx].get("width"))]
            boxes, scores = torch.zeros(0, cfg.MODEL.ROI_HEADS.NUM_CLASSES*4).to(model.device), torch.zeros(0, cfg.MODEL.ROI_HEADS.NUM_CLASSES+1).to(model.device)
            for data_dict in new_data_dicts:
                if cfg.CROPTRAIN.USE_CROPS:
                    outputs = model.inference(batched_inputs=[data_dict])
                    cluster_class_indices = (outputs[0]["instances"].pred_classes==cluster_class)
                    cluster_boxes = outputs[0]["instances"][cluster_class_indices]
                    cluster_boxes = cluster_boxes[cluster_boxes.scores>0.7]
                else:
                    cluster_boxes = []
                #_, clus_dicts = compute_crops(dataset_dicts[idx], cfg)
                #cluster_boxes = np.array([item['crop_area'] for item in clus_dicts]).reshape(-1, 4)
                
                if len(cluster_boxes)!=0:
                    #cluster_boxes = merge_cluster_boxes(cluster_boxes, cfg)
                    cluster_dicts = get_dict_from_crops(cluster_boxes, data_dict, cfg.CROPTEST.CROPSIZE, inner_crop=True)
                    boxes_patch, scores_patch = model([data_dict], cluster_inputs=cluster_dicts, infer_on_crops=True)
                else:
                    boxes_patch, scores_patch = model([data_dict], None, infer_on_crops=True)
                boxes = torch.cat([boxes, boxes_patch[0]], dim=0)
                scores = torch.cat([scores, scores_patch[0]], dim=0)
            pred_instances, _ = fast_rcnn_inference([boxes], [scores], image_shapes, cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, \
                                    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST, cfg.CROPTEST.DETECTIONS_PER_IMAGE)
            pred_instances = pred_instances[0]
            pred_instances = pred_instances[pred_instances.pred_classes!=cluster_class]
            all_outputs = [{"instances": pred_instances}]
            
            #if idx%100==0:
            #    plot_detections(pred_instances.to("cpu"), cluster_boxes, inputs[0], evaluator._metadata, cfg, iter)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, all_outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results



@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


