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
import torch


# Non-maximum suppression to remove duplicate predictions for the same GT annotation (not used currently)
def apply_nms_to_predictions(predictions, iou_threshold=0.5):
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


def custom_eval(trainer):
    evaluator = COCOPanopticEvaluator(dataset_name="PUMA_test_separated", output_dir="./output_inference_panoptic")
    val_loader = build_detection_test_loader(trainer.cfg, "PUMA_test_separated")
    eval_results = inference_on_dataset(trainer.model, val_loader, evaluator)

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
        eval_period = 2  # evaluate every "eval_period" iterations
        hooks.append(EvalHook(eval_period, lambda: custom_eval(self)))
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)


wandb.login(key='')
wandb.init(project="PanopticFPN", name="panoptic_dice_precision_recall_f1")

output_dir = 'geojson_to_coco/output'

dataset_dir_full = os.path.join(os.getcwd(), 'geojson_to_coco', 'dataset_open_source')
panoptic_img_dir = os.path.join(output_dir, 'panoptic_coco_RLE')

panoptic_json_train = os.path.join(output_dir, 'train_dataset_panoptic.json')
panoptic_json_val = os.path.join(output_dir, 'val_dataset_panoptic.json')
panoptic_json_test = os.path.join(output_dir, 'test_dataset_panoptic.json')

instance_json_train = os.path.join(output_dir, 'train_dataset_instance_things.json')
instance_json_val = os.path.join(output_dir, 'val_dataset_instance_things.json')
instance_json_test = os.path.join(output_dir, 'test_dataset_instance_things.json')

sem_seg_root_full = os.path.join(output_dir, 'panoptic_stuff_full')

categories_json = os.path.join(output_dir, 'categories.json')
with open(categories_json, "r") as categories_json_file:
    categories = json.load(categories_json_file)
    category_names = {}
    for category in categories:
        category_names[category['id']] = category['name']

thing_classes = [c["name"] for c in categories if c["isthing"] == 1]
stuff_classes = [c["name"] for c in categories if c["isthing"] == 0]
thing_colors = [c["color"] for c in categories if c["isthing"] == 1]
stuff_colors = [c["color"] for c in categories if c["isthing"] == 0]
#thing_dataset_id_to_contiguous_id = {c["id"]: idx for idx, c in enumerate(categories) if c["isthing"] == 1}
stuff_dataset_id_to_contiguous_id = {c["id"]: idx + 1 - len(thing_classes) for idx, c in enumerate(categories) if c["isthing"] == 0}

metadata = {
    "thing_classes": thing_classes,
    "stuff_classes": stuff_classes,
    #"thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
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
    name="PUMA_val",
    metadata=metadata,
    image_root=dataset_dir_full,
    panoptic_root=panoptic_img_dir,
    sem_seg_root=sem_seg_root_full,
    panoptic_json=panoptic_json_val,
    instances_json=instance_json_val,
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

config_file = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
# config_file = "maskdino_config.yaml"
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.WEIGHTS = "output_inference_panoptic/model_final.pth"
cfg.DATASETS.TRAIN = ("PUMA_train_separated", )
#cfg.DATASETS.TEST = ("PUMA_test_separated")
cfg.DATALOADER.NUM_WORKERS = 4
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

DatasetCatalog.get("PUMA_train_separated")
DatasetCatalog.get("PUMA_test_separated")

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

wandb.finish()


