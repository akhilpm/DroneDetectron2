import torch
from torch import Tensor, nn
import math
from detectron2.layers import get_norm
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.fcos import FCOS
from detectron2.structures import Boxes, Instances
from typing import Dict, List, Optional, Tuple
from detectron2.utils.events import get_event_storage
from detectron2.structures import Instances
from detectron2.config import configurable
from utils.crop_utils import project_boxes_to_image
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone import build_backbone
from detectron2.layers import ShapeSpec, batched_nms
from detectron2.modeling.postprocessing import detector_postprocess
from utils.crop_utils import project_boxes_to_image, get_dict_from_crops


@META_ARCH_REGISTRY.register()
class CROP_FCOS(FCOS):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        heads: nn.Module,
        head_in_features,
        num_classes,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        max_detections_per_image: int,
        crop_size: int,
    ):  
        super().__init__(backbone=backbone, head=heads, head_in_features=head_in_features, num_classes=num_classes,
            max_detections_per_image=max_detections_per_image, pixel_mean=pixel_mean, pixel_std=pixel_std)
        self.CROPSIZE = crop_size    


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.FCOS.IN_FEATURES]        
        head = FCOSHead(cfg, feature_shapes)
        return {
            "backbone": backbone,
            "heads": head,
            "head_in_features": cfg.MODEL.FCOS.IN_FEATURES,
            "num_classes": cfg.MODEL.FCOS.NUM_CLASSES,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "crop_size": cfg.CROPTEST.CROPSIZE,
        }


    def forward(self, batched_inputs: List[Dict[str, Tensor]],
        orig_image_size: Optional[Tuple[int]] = None,
        infer_on_crops: bool = False,
    ):
        if infer_on_crops:
            return self.infer_on_image_and_crops(batched_inputs, orig_image_size, self.CROPSIZE)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        predictions = self.head(features)

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            return self.forward_training(images, features, predictions, gt_instances)
        else:
            results = self.forward_inference(images, features, predictions)
            if torch.jit.is_scripting():
                return results

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def infer_on_image_and_crops(self, input_dicts, orig_image_size, CROPSIZE=512):
        assert not self.training
        all_preds: List[Instances] = []
        same_image = all(x["file_name"]==input_dicts[0]["file_name"] for x in input_dicts)
        assert same_image,  "Only one image per inference is supported!"
        images = self.preprocess_image(input_dicts)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        predictions = self.head(features)

        pred_logits, pred_anchor_deltas, pred_centerness = self._transpose_dense_predictions(
            predictions, [self.num_classes, 4, 1]
        )
        anchors = self.anchor_generator(features)
        for img_idx, image_size in enumerate(images.image_sizes):
            scores_per_image = [torch.sqrt(x[img_idx].sigmoid_() * y[img_idx].sigmoid_()) 
                for  x, y in zip(pred_logits, pred_centerness)]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            pred = self._decode_multi_level_predictions(anchors, scores_per_image, deltas_per_image, 
                self.test_score_thresh, self.test_topk_candidates, orig_image_size)
            pred.pred_boxes.tensor = project_boxes_to_image(input_dicts[img_idx], image_size, pred.pred_boxes.tensor)
            all_preds.append(pred)
        del features
        full_image_pred = Instances.cat(all_preds)
        keep = batched_nms(
            full_image_pred.pred_boxes.tensor, full_image_pred.scores, 
            full_image_pred.pred_classes, self.test_nms_thresh
        )
        full_image_pred = full_image_pred[keep]
        #extract cluster boxes
        cluster_class = self.num_classes-1
        cluster_class_indices = (full_image_pred.pred_classes==cluster_class)
        cluster_boxes = full_image_pred[cluster_class_indices]
        cluster_boxes = cluster_boxes[cluster_boxes.scores>0.7]

        if len(cluster_boxes)!=0:
            cluster_dicts = get_dict_from_crops(cluster_boxes, input_dicts[0], CROPSIZE)
            images_crop = self.preprocess_image(cluster_dicts)
            features_crop = self.backbone(images_crop.tensor)
            features_crop = [features_crop[f] for f in self.head_in_features]
            predictions_crop = self.head(features_crop)

            pred_logits, pred_anchor_deltas, pred_centerness = self._transpose_dense_predictions(
                predictions_crop, [self.num_classes, 4, 1]
            )
            anchors_crop = self.anchor_generator(features_crop)
            for img_idx, image_size in enumerate(images_crop.image_sizes):
                scores_per_crop = [
                # Multiply and sqrt centerness & classification scores
                # (See eqn. 4 in https://arxiv.org/abs/2006.09214)
                torch.sqrt(x[img_idx].sigmoid_() * y[img_idx].sigmoid_())
                for x, y in zip(pred_logits, pred_centerness)
                ]
                deltas_per_crop = [x[img_idx] for x in pred_anchor_deltas]
                # use original image size for pred instances on crop to facilitate concatenation
                pred_crop = self._decode_multi_level_predictions(anchors_crop, scores_per_crop, deltas_per_crop, 
                    self.test_score_thresh, self.test_topk_candidates, orig_image_size)
                pred_crop.pred_boxes.tensor = project_boxes_to_image(cluster_dicts[img_idx], image_size, pred_crop.pred_boxes.tensor)
                all_preds.append(pred_crop)
            del features_crop    

        all_preds = Instances.cat(all_preds)
        keep = batched_nms(
            all_preds.pred_boxes.tensor, all_preds.scores, all_preds.pred_classes, self.test_nms_thresh
        )
        all_preds = all_preds[keep[: self.max_detections_per_image]]
        #all_preds = all_preds[all_preds.pred_classes!=cluster_class]
        results = [{"instances": all_preds}]
        return results



class FCOSHead(nn.Module):
    """
    The head used in :paper:`fcos`. It adds an additional centerness
    prediction branch on top of :class:`RetinaNetHead`.
    """
    @configurable
    def __init__(
        self,
         *, 
        input_shape: List[ShapeSpec], 
        num_classes, 
        conv_dims: List[int], 
        norm="", 
        prior_prob=0.01
    ):
        super().__init__()

        self._num_features = len(input_shape)
        self.num_classes = num_classes
        cls_subnet = []
        bbox_subnet = []
        for in_channels, out_channels in zip(
            [input_shape[0].channels] + list(conv_dims), conv_dims
        ):
            cls_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                cls_subnet.append(get_norm(norm, out_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                bbox_subnet.append(get_norm(norm, out_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            conv_dims[-1], num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            conv_dims[-1], 4, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)
        # Unlike original FCOS, we do not add an additional learnable scale layer
        # because it's found to have no benefits after normalizing regression targets by stride.
        self.ctrness = nn.Conv2d(conv_dims[-1], 1, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.ctrness.weight, std=0.01)
        torch.nn.init.constant_(self.ctrness.bias, 0)


    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        return {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.FCOS.NUM_CLASSES,
            "conv_dims": [input_shape[0].channels] * cfg.MODEL.FCOS.NUM_CONVS,
            "norm": cfg.MODEL.FCOS.NORM,
        }

    def forward(self, features):
        assert len(features) == self._num_features
        logits = []
        bbox_reg = []
        ctrness = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_feature = self.bbox_subnet(feature)
            bbox_reg.append(self.bbox_pred(bbox_feature))
            ctrness.append(self.ctrness(bbox_feature))
        return logits, bbox_reg, ctrness
