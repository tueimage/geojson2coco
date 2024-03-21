import json
import os
from detectron2.data.datasets import register_coco_panoptic, register_coco_panoptic_separated
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, COCOPanopticEvaluator
from detectron2.engine import DefaultTrainer, EvalHook
from detectron2.config import get_cfg
import wandb
from detectron2.layers import batched_nms
from pycocotools.coco import COCO
from detectron2.model_zoo import model_zoo
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
from detectron2.utils.visualizer import Visualizer
import numpy as np
from detectron2.structures import BoxMode, Boxes
import pycocotools.mask as mask_util
from detectron2.engine import hooks
import numpy as np
import torch
from detectron2.structures import pairwise_iou
from collections import defaultdict


# Non-maximum suppression to remove duplicate predictions for the same GT annotation
def apply_nms_to_predictions(predictions, iou_threshold=0.3):
    if len(predictions) == 0:
        return predictions
    boxes = [pred["bbox"] for pred in predictions]
    scores = [pred["score"] for pred in predictions]
    classes = [pred["category_id"] for pred in predictions]
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    classes = torch.tensor(classes)

    keep = batched_nms(boxes, scores, classes, iou_threshold)

    return [predictions[i] for i in keep]


def custom_eval_function(trainer):
    evaluator = COCOPanopticEvaluator(dataset_name="PUMA_test_separated", input_dir="./output_inference")
    val_loader = build_detection_test_loader(trainer.cfg, "PUMA_test_separated")
    eval_results = inference_on_dataset(trainer.model, val_loader, evaluator)
    coco_gt = COCO(instance_json_test)
    category_id_to_cell_name = {c["id"]: c["name"] for c in categories if c["supercategory"] == "Cell"}

    model_predictions_test_json = 'output_inference/coco_instances_results.json'
    with open(model_predictions_test_json, "r") as predictions_json_file:
        predictions = json.load(predictions_json_file)

    # counters
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    # TODO: Change to centroid distance (distance=15)
    iou_threshold = 0.5 

    # iterate over each image and compute precision/recall/f1-score
    for img_id in coco_gt.getImgIds():
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        gt_boxes = Boxes(torch.tensor([ann['bbox'] for ann in gt_anns], dtype=torch.float32).reshape(-1, 4))
        gt_matched = [False] * len(gt_anns)
        pred_anns = [p for p in predictions if p['image_id'] == img_id]
        pred_anns_nms = apply_nms_to_predictions(pred_anns, iou_threshold)

        for pred in pred_anns_nms:
            pred_box = Boxes(torch.tensor([pred['bbox']], dtype=torch.float32).reshape(-1, 4))
            ious = pairwise_iou(pred_box, gt_boxes)

            max_iou, max_iou_index = torch.max(ious, dim=1)
            if max_iou >= iou_threshold:
                if gt_matched[max_iou_index]:
                    continue  # prediction is already matched
                class_tp[gt_anns[max_iou_index]['category_id']] += 1
                gt_matched[max_iou_index] = True
            else:
                class_fp[pred['category_id']] += 1

        for idx, matched in enumerate(gt_matched):
            if not matched:
                class_fn[gt_anns[idx]['category_id']] += 1

    class_recall = {}
    class_precision = {}
    class_f1_score = {}
    for cat_id, cat_name in category_id_to_cell_name.items():
        cat_name = category_id_to_cell_name[cat_id]
        tp = class_tp[cat_id]
        fp = class_fp[cat_id]
        fn = class_fn[cat_id]

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_recall[cat_id] = recall
        class_precision[cat_id] = precision
        class_f1_score[cat_id] = f1_score

        wandb.log({
            f"recall_class_{cat_name}": recall,
            f"precision_class_{cat_name}": precision,
            f"f1_score_class_{cat_name}": f1_score
        }, step=trainer.iter)

    # overall recall and precision
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    wandb.log({
        "recall_overall": overall_recall,
        "precision_overall": overall_precision,
        "f1_score_overall": overall_f1_score
    }, step=trainer.iter)

    for key, value in eval_results.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                wandb.log({f"{key}/{sub_key}": sub_value}, step=trainer.iter)
        else:
            wandb.log({key: value}, step=trainer.iter)

    return


class WandbHook(HookBase):
    def __init__(self, log_interval):
        self.log_interval = log_interval

    def after_step(self):
        if (self.trainer.iter + 1) % self.log_interval == 0:
            metrics = self.trainer.storage.latest()
            filtered_metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}
            wandb.log(filtered_metrics, step=self.trainer.iter)


class MyTrainer(DefaultTrainer):
    def build_hooks(self):
        hooks = super(MyTrainer, self).build_hooks()
        eval_period = 100  # evaluate every "eval_period" iterations
        hooks.append(EvalHook(eval_period, lambda: custom_eval_function(self)))
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)


wandb.login(key='a3e4f0230bd4bfac99da84a8b2f365acb93b592e')
wandb.init(project="PanopticFPN", name="panoptic_iou_0.3")

input_dir = 'train/input'

dataset_dir_full = os.path.join(os.getcwd(), 'geojson_to_coco', 'dataset')
panoptic_img_dir = os.path.join(input_dir, 'panoptic_coco_RLE')

panoptic_json_train = os.path.join(input_dir, 'train_panoptic.json')
panoptic_json_test = os.path.join(input_dir, 'test_panoptic.json')
panoptic_json_full = os.path.join(input_dir, 'full_panoptic.json')

instance_json_train = os.path.join(input_dir, 'train_instance.json')
instance_json_test = os.path.join(input_dir, 'test_instance.json')
instance_json_full = os.path.join(input_dir, 'full_instance.json')

sem_seg_root_full = os.path.join(input_dir, 'panoptic_stuff_full')
categories_json = os.path.join(input_dir, 'categories.json')
with open(categories_json, "r") as json_file:
    categories = json.load(json_file)

thing_classes = [c["name"] for c in categories if c["isthing"] == 1]
stuff_classes = [c["name"] for c in categories if c["isthing"] == 0]
thing_colors = [c["color"] for c in categories if c["isthing"] == 1]
stuff_colors = [c["color"] for c in categories if c["isthing"] == 0]

thing_dataset_id_to_contiguous_id = {c["id"]: idx for idx, c in enumerate(categories) if c["isthing"] == 1}
stuff_dataset_id_to_contiguous_id = {c["id"]: idx - len(thing_classes) + 1 for idx, c in enumerate(categories) if c["isthing"] == 0}

metadata = {
    "thing_classes": thing_classes,
    "stuff_classes": stuff_classes,
    "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
    "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
    "thing_colors": thing_colors,
    "stuff_colors": stuff_colors
}

register_coco_panoptic_separated(
    name="PUMA_train",
    metadata=metadata,
    image_root=dataset_dir_full,
    panoptic_root=panoptic_img_dir,
    sem_seg_root=sem_seg_root_full,
    panoptic_json=panoptic_json_train,
    instances_json=instance_json_train,
)

register_coco_panoptic_separated(
    name="PUMA_test",
    metadata=metadata,
    image_root=dataset_dir_full,
    panoptic_root=panoptic_img_dir,
    sem_seg_root=sem_seg_root_full,
    panoptic_json=panoptic_json_test,
    instances_json=instance_json_test,
)

#config_file = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
cfg = get_cfg()
cfg.merge_from_file("maskdino_config.yaml")
cfg.DATASETS.TRAIN = ("PUMA_train_separated", )
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.0005
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16  # faster (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 7  # 6 stuff classes + 1 background
cfg.INPUT.MASK_FORMAT = "bitmask"
cfg.SOLVER.AMP.ENABLED = True  # Mixed-precision training
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
#cfg.TEST.EVAL_PERIOD = 100
#cfg.MODEL.DEVICE = 'cpu'
wandb.config.update(cfg)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

evaluator = COCOPanopticEvaluator(dataset_name="PUMA_test_separated", input_dir="./output_inference")
val_loader = build_detection_test_loader(cfg, "PUMA_test_separated")
eval_results = inference_on_dataset(trainer.model, val_loader, evaluator)
wandb.log({"eval": eval_results})
wandb.finish()


