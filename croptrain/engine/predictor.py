import torch
import numpy as np
import os
import cv2
from detectron2.engine import DefaultPredictor
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from utils.plot_utils import plot_detection_boxes
from PIL import Image

class DroneDetPredictor(DefaultPredictor):

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")
        self.aug_crop = T.ResizeShortestEdge(cfg.CROPTRAIN.CROPSIZE, cfg.CROPTRAIN.MAX_CROPSIZE, sample_style="choice")

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        file_name = os.path.join(os.getcwd(), "temp", "example_1.jpg")
        original_image = cv2.imread(file_name)
        with torch.no_grad(): 
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]    
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": height, "width": width}]

            #outputs = self.model(inputs, infer_on_crops=True, cfg=self.cfg)[0]
            outputs = self.model(inputs)[0]
            predictions = outputs["instances"]
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        #pil_image = Image.open(file_name)
        plot_detection_boxes(original_image, predictions, self.cfg)