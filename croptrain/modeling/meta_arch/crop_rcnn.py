# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures.boxes import Boxes
from typing import Dict, List, Optional, Tuple
from detectron2.utils.events import get_event_storage
from detectron2.structures import Instances
from utils.crop_utils import project_boxes_to_image

@META_ARCH_REGISTRY.register()
class CropRCNN(GeneralizedRCNN):

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], 
        cluster_inputs: Optional[List[Dict[str, torch.Tensor]]] = None,
        infer_on_crops: bool = False,
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            if infer_on_crops:
                return self.infer_on_image_and_crops(batched_inputs, cluster_inputs)
            else:    
                return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


    def get_box_predictions(self, features, proposals):
        features = [features[f] for f in self.roi_heads.box_in_features]
        box_features = self.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.roi_heads.box_head(box_features)
        predictions = self.roi_heads.box_predictor(box_features)
        del box_features
        boxes = self.roi_heads.box_predictor.predict_boxes(predictions, proposals)
        scores = self.roi_heads.box_predictor.predict_probs(predictions, proposals)
        return list(boxes), list(scores)

    def infer_on_image_and_crops(self, input_dicts, cluster_dicts):
        assert len(input_dicts)==1, "Only one image per inference is supported!"
        assert not self.training
        images_original = self.preprocess_image(input_dicts)
        features_original = self.backbone(images_original.tensor)
        proposals_original, _ = self.proposal_generator(images_original, features_original, None)
        #get detections from full image and project it to original image size
        boxes, scores = self.get_box_predictions(features_original, proposals_original)
        num_bbox_reg_classes = boxes[0].shape[1] // 4
        boxes[0] = project_boxes_to_image(input_dicts[0], images_original.image_sizes[0], boxes[0])
        del features_original

        if cluster_dicts:
            for i, cluster_dict in enumerate(cluster_dicts):
                images_crop = self.preprocess_image([cluster_dict])
                features_crop = self.backbone(images_crop.tensor)
                proposals_crop, _ = self.proposal_generator(images_crop, features_crop, None)
                #get detections from crop and project it to wrt to original image size
                boxes_crop, scores_crop = self.get_box_predictions(features_crop, proposals_crop)
                boxes_crop = project_boxes_to_image(cluster_dict, images_crop.image_sizes[0], boxes_crop[0])
                boxes[0] = torch.cat([boxes[0], boxes_crop], dim=0)
                scores[0] = torch.cat([scores[0], scores_crop[0]], dim=0)
        return boxes, scores
