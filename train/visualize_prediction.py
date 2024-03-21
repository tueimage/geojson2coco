import json
import os

import cv2
from pycocotools.mask import decode

categories_json = 'geojson_to_coco/output/categories.json'
with open(categories_json, "r") as categories_json_file:
    categories = json.load(categories_json_file)
    category_colors = {}
    category_names = {}
    for category in categories:
        color = category['color']
        category_colors[category['id']] = color
        category_names[category['id']] = category['name']

model_predictions_test_json = 'output_inference/coco_instances_results.json'
with open(model_predictions_test_json, "r") as predictions_json_file:
    predictions = json.load(predictions_json_file)

gt_instances_test_json = 'geojson_to_coco/output/test_instance.json'
with open(gt_instances_test_json, "r") as gt_instances_json_file:
    gt_instances = json.load(gt_instances_json_file)

# image_001.png, image_002.png, etc
img_dir = 'geojson_to_coco/dataset/'

output_dir = 'output_image_predictions'
os.makedirs(output_dir, exist_ok=True)


# draws predicted bounding boxes and segmentation masks on images
def draw_predictions(image_path, predictions, ground_truth):
    image = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    for prediction in predictions:
        bbox = prediction['bbox']
        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 150), 1)
        mask = decode(prediction['segmentation'])
        mask = mask.astype(bool)

        # create overlay/adjust transparency
        overlay = image.copy()
        overlay[mask] = category_colors[prediction['category_id']]
        image = cv2.addWeighted(src1=overlay, alpha=0.5, src2=image, beta=0.5, gamma=0)

        # add text
        class_name = category_names[prediction['category_id']]
        text_size = cv2.getTextSize(class_name, font, font_scale, font_thickness)[0]
        text_x = x + int((w - text_size[0]) / 2)
        text_y = y - 5
        cv2.putText(image, class_name, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

    for annotation in ground_truth:
        bbox = annotation['bbox']
        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 150, 0), 1)

        # add text
        class_name = category_names[annotation['category_id']]
        text_size = cv2.getTextSize(class_name, font, font_scale, font_thickness)[0]
        text_x = x + int((w - text_size[0]) / 2)
        text_y = y - 15
        cv2.putText(image, class_name, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)

    return image


for filename in os.listdir(img_dir):
    if filename.endswith('101.png'):  # change to visualize different images
        image_path = os.path.join(img_dir, filename)
        image_name = os.path.splitext(filename)[0]
        image_number = int(image_name[6:9])
        predictions_for_image = [prediction for prediction in predictions if prediction['image_id'] == image_number]
        gt_for_image = [annotation for annotation in gt_instances['annotations'] if annotation['image_id'] == image_number]
        image_with_boxes = draw_predictions(image_path, predictions_for_image, gt_for_image)

        output_path = os.path.join(output_dir, f"{image_name}_predictions.png")
        cv2.imwrite(output_path, image_with_boxes)

print("Done")
