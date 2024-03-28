# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict, defaultdict
from typing import Optional
from PIL import Image
from tabulate import tabulate
import pycocotools.mask as mask_util

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.visualizer import Visualizer
from scipy.spatial import cKDTree

from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class COCOPanopticEvaluator(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)

    def reset(self):
        self._predictions = []
        self._predictions_instances = {}

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img_tensor = panoptic_img.cpu()
            panoptic_img = panoptic_img.cpu().numpy()

            # instances = output["instances"]
            # instances = instances.to("cpu")
            # v = Visualizer(np.zeros_like(panoptic_img, dtype=np.uint8), self._metadata, scale=1.2)
            # v = v.draw_panoptic_seg(panoptic_img_tensor, segments_info)
            # v = v.draw_instance_predictions(predictions=instances)

            # combined_img = v.get_image()
            # file_name = os.path.basename(input["file_name"])
            # file_name_png = os.path.splitext(file_name)[0] + "_predictions.png"
            # output_path = os.path.join(self._output_dir, file_name_png)
            # Image.fromarray(combined_img).save(output_path)
            # print(f"Saved prediction image to {output_path}")

            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]

                # map "id" to "category_id", each cell is now annotated with the predicted category_id
                converted_panoptic_img = np.zeros_like(panoptic_img)
                for segment_info in segments_info:
                    converted_panoptic_img[panoptic_img == segment_info["id"]] = segment_info["category_id"]

                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                        "panoptic_img": converted_panoptic_img,
                    }
                )

                instance_category_ids = output["instances"]._fields["pred_classes"].tolist()
                instance_category_ids = [id + 1 for id in instance_category_ids]  # correct for 0-index
                instance_masks = output["instances"]._fields["pred_masks"].cpu().numpy()
                encoded_masks = [np.argwhere(mask) for mask in instance_masks]
                category_id_mask_tuples = [(category, mask) for category, mask in zip(instance_category_ids, encoded_masks)]
                self._predictions_instances[input["image_id"]] = category_id_mask_tuples


    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)
        dice_scores = defaultdict(lambda: {'score': 0, 'count': 0})

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = [{k: v for k, v in prediction.items() if k != "panoptic_img"} for prediction in
                                        self._predictions]

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from panopticapi.evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        #############################################################################################
        # extra code to compute DICE score
        #############################################################################################

        def rgb_to_panoptic_id(panoptic_rgb_image):
            panoptic_id_image = panoptic_rgb_image[:, :, 0].astype(np.uint32) + \
                                panoptic_rgb_image[:, :, 1].astype(np.uint32) * 256 + \
                                panoptic_rgb_image[:, :, 2].astype(np.uint32) * (256 ** 2)
            return panoptic_id_image

        def get_id_to_category_map(annotations, image_id):
            for ann in annotations:
                if ann['image_id'] == image_id:
                    return {segment['id']: segment['category_id'] for segment in ann['segments_info']}
            return {}

        def dice_coefficient(y_true, y_pred):
            intersection = np.sum(y_true * y_pred)
            return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

        with open(gt_json, 'r') as f:
            panoptic_gt_json = json.load(f)

        for prediction in self._predictions:
            image_id = prediction['image_id']
            image_file_name = prediction['file_name']
            predicted_img = prediction['panoptic_img']  # already in category_id format

            # load the GT panoptic image, convert to RGB and then to category_id
            gt_image_path = os.path.join(gt_folder, image_file_name)  # Update this path as needed
            gt_panoptic_rgb = np.array(Image.open(gt_image_path))
            gt_panoptic_id = rgb_to_panoptic_id(gt_panoptic_rgb)
            id_to_category_map = get_id_to_category_map(panoptic_gt_json['annotations'], image_id)
            gt_category_id_image = np.vectorize(lambda x: 0 if x == 0 else id_to_category_map.get(x, 0),
                                                otypes=[np.int32])(gt_panoptic_id)

            # compute DICE score for each category
            stuff_categories = set(self._stuff_contiguous_id_to_dataset_id.values())
            for category_id in np.unique(np.concatenate((predicted_img, gt_category_id_image))):
                if category_id in stuff_categories:
                    pred_mask = (predicted_img == category_id)
                    gt_mask = (gt_category_id_image == category_id)
                    dice_score = dice_coefficient(gt_mask, pred_mask)
                    dice_scores[category_id]['score'] += dice_score
                    dice_scores[category_id]['count'] += 1

        #############################################################################################
        # extra code to compute precision/recall/f1-score for instances (centroid distance 15)
        #############################################################################################

        def calculate_centroid(mask):
            coords = np.argwhere(mask)
            return coords.mean(axis=0) if len(coords) > 0 else None

        gt_json_instance = PathManager.get_local_path(self._metadata.json_file)
        with open(gt_json_instance, 'r') as f:
            instance_gt_json = json.load(f)

        class_tp = defaultdict(int)
        class_fp = defaultdict(int)
        class_fn = defaultdict(int)

        for image_id in self._predictions_instances.keys():
            gt_image_annotations = [ann for ann in instance_gt_json['annotations'] if ann['image_id'] == image_id]
            gt_centroids_by_category = defaultdict(list)

            for ann in gt_image_annotations:
                gt_mask = mask_util.decode(ann['segmentation'])
                gt_centroid = calculate_centroid(gt_mask)
                gt_centroids_by_category[ann['category_id']].append(gt_centroid)

            # lists to np-arrays for cKDTree, build KD-tree for each category
            for cat_id in gt_centroids_by_category:
                gt_centroids_by_category[cat_id] = np.array(gt_centroids_by_category[cat_id])

            kd_trees = {cat_id: cKDTree(centroids) for cat_id, centroids in gt_centroids_by_category.items()}

            for pred_category_id, encoded_mask in self._predictions_instances[image_id]:
                decoded_mask = np.zeros((1024,1024), dtype=bool)
                decoded_mask[tuple(encoded_mask.T)] = True
                pred_centroid = calculate_centroid(decoded_mask)
                if pred_centroid is None:
                    continue

                distance_threshold = 15

                # check nearest GT centroid in same category
                if pred_category_id in kd_trees:  # ensure category had a KD-tree (i.e. had GT annotations)
                    dist, idx = kd_trees[pred_category_id].query(pred_centroid[:2])
                    if dist < distance_threshold:
                        class_tp[pred_category_id] += 1
                    else:
                        class_fp[pred_category_id] += 1
                else:
                    class_fp[pred_category_id] += 1

            # check for unmatched GT annotations
            for cat_id, centroids in gt_centroids_by_category.items():
                if cat_id in kd_trees:
                    class_fn[cat_id] += len(centroids)  # not matched (FN)

        class_precision = {cat: class_tp[cat] / (class_tp[cat] + class_fp[cat]) if class_tp[cat] + class_fp[cat] > 0
            else 0 for cat in set(class_tp) | set(class_fp)}

        class_recall = {cat: class_tp[cat] / (class_tp[cat] + class_fn[cat]) if class_tp[cat] + class_fn[cat] > 0
            else 0 for cat in set(class_tp) | set(class_fn)}

        class_f1 = {cat: 2 * (class_precision.get(cat, 0) * class_recall.get(cat, 0)) / (class_precision.get(cat, 0)
                  + class_recall.get(cat, 0)) if (class_precision.get(cat, 0) + class_recall.get(cat, 0)) > 0 else 0
                    for cat in set(class_precision) | set(class_recall)}

        res = {}
        categories_json = 'geojson_to_coco/output/categories.json'
        with open(categories_json, "r") as categories_json_file:
            categories = json.load(categories_json_file)
            category_names = {category['id']: category['name'] for category in categories}

        for category_id, info in class_precision.items():  # all thing classes
            category_name = category_names.get(category_id, 'Unknown')
            res[f'Precision_{category_name}'] = class_precision[category_id]
            res[f'Recall_{category_name}'] = class_recall[category_id]
            res[f'F1-score_{category_name}'] = class_f1[category_id]

        for category_id, info in dice_scores.items():
            avg_dice_score = info['score'] / info['count'] if info['count'] > 0 else 0
            category_name = category_names.get(category_id, 'Unknown')
            res[f'DICE_{category_name}'] = avg_dice_score
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)

        return results


def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results:\n" + table)


if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json, args.pred_json, gt_folder=args.gt_dir, pred_folder=args.pred_dir
        )
        _print_panoptic_results(pq_res)
