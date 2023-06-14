from turtle import width
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
from utils.plot_utils import plot_detections
from detectron2.utils.logger import log_every_n_seconds
from utils.box_utils import compute_crops
from utils.crop_utils import get_dict_from_crops
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
logging.basicConfig(level=logging.INFO)

def inference_with_crops(model, data_loader, evaluator, cfg, iter):
    from detectron2.utils.comm import get_world_size
    #dataset_dicts = get_detection_dataset_dicts(cfg.DATASETS.TEST, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS)

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
    #last_class = evaluator._metadata.get("thing_classes")[-1]
    #if last_class=="cluster":
    #    all_classes = evaluator._metadata.get("thing_classes")
    #    evaluator._metadata.__dict__['thing_classes'] =  all_classes[:-1]
    #    evaluator._metadata.__dict__['thing_dataset_id_to_contiguous_id'].pop(cfg.MODEL.ROI_HEADS.NUM_CLASSES)

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
            #_, crop_dicts = compute_crops(dataset_dicts[idx], cfg)
            #crop_boxes = np.array([item['crop_area'] for item in crop_dicts]).reshape(-1, 4)
            all_outputs = model(inputs, infer_on_crops=True, cfg=cfg)
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

