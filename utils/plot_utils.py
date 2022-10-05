from detectron2.utils.visualizer import Visualizer
from PIL import Image
import numpy as np
import os

def plot_detections(predictions, cluster_boxes, data_dict, metadata, cfg, iter):
    img = np.array(Image.open(data_dict["file_name"]))
    visualizer = Visualizer(img, metadata=metadata)
    #visualizer.draw_instance_predictions(predictions)
    #vis = visualizer.overlay_instances(boxes=cluster_boxes.pred_boxes.tensor.cpu())
    vis = visualizer.overlay_instances(boxes=cluster_boxes)
    save_path = os.path.join(cfg.OUTPUT_DIR, "detections", str(iter)+'_'+data_dict["file_name"].split('/')[-1])
    vis.save(save_path)

def plot_image(data_dict, cfg, metadata):
    img = np.array(Image.open(data_dict["file_name"]))
    visualizer = Visualizer(img, metadata=metadata)
    vis = visualizer.draw_dataset_dict(data_dict)
    save_path = os.path.join(cfg.OUTPUT_DIR, "input", data_dict["file_name"].split('/')[-1])
    vis.save(save_path)