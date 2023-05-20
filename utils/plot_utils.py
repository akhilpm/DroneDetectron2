from detectron2.utils.visualizer import Visualizer
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from detectron2.structures.instances import Instances
from matplotlib.patches import Rectangle

def plot_detections(predictions, cluster_boxes, data_dict, metadata, cfg, iter):
    img = np.array(Image.open(data_dict["file_name"]))
    visualizer = Visualizer(img, metadata=metadata)
    #visualizer.draw_instance_predictions(predictions)
    if len(cluster_boxes)==0:
        print("No clusters")
        return
    vis = visualizer.overlay_instances(boxes=cluster_boxes.pred_boxes.tensor.cpu())
    save_path = os.path.join(cfg.OUTPUT_DIR, "detections", str(iter)+'_'+data_dict["file_name"].split('/')[-1])
    vis.save(save_path)

def plot_image(data_dict, cfg, metadata):
    img = np.array(Image.open(data_dict["file_name"]))
    visualizer = Visualizer(img, metadata=metadata)
    vis = visualizer.draw_dataset_dict(data_dict)
    save_path = os.path.join(cfg.OUTPUT_DIR, "input", data_dict["file_name"].split('/')[-1])
    vis.save(save_path)

def plot_detection_boxes(image, predictions, cfg):
    #cluster_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES - 1
    cluster_class = cfg.MODEL.FCOS.NUM_CLASSES - 1
    plt.axis('off')
    plt.imshow(image)
    ax = plt.gca()
    cluster_boxes = predictions[predictions.pred_classes==cluster_class]
    predictions = predictions[predictions.pred_classes!=cluster_class]
    if len(predictions)!=0:
        predictions = predictions[predictions.scores>0.6]
        predictions = predictions.pred_boxes.tensor.cpu()
        for bbox in predictions:
            x1, y1 = bbox[0], bbox[1]
            h, w = bbox[3]-bbox[1], bbox[2]-bbox[0]
            rect = Rectangle((x1, y1), w, h, linewidth=1.5, edgecolor='orange', facecolor='none')
            ax.add_patch(rect)
    if len(cluster_boxes)!=0:
        cluster_boxes = cluster_boxes[cluster_boxes.scores>0.6]
        cluster_boxes = cluster_boxes.pred_boxes.tensor.cpu()
        for bbox in cluster_boxes:
            x1, y1 = bbox[0], bbox[1]
            h, w = bbox[3]-bbox[1], bbox[2]-bbox[0]
            rect = Rectangle((x1, y1), w, h, linewidth=1.5, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
    plt.savefig(os.path.join(os.getcwd(), "temp", "example_output.jpg"), dpi=150, bbox_inches='tight')
    #plt.show()
    plt.clf()