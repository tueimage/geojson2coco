import json
import os

output_dir = '../../output'

# input files
train_dataset = output_dir + '/train_dataset_instance.json'
val_dataset = output_dir + '/val_dataset_instance.json'
test_dataset = output_dir + '/test_dataset_instance.json'
categories_json = output_dir + '/categories.json'

# output files
train_dataset_things = output_dir + '/train_dataset_instance_things.json'
val_dataset_things = output_dir + '/val_dataset_instance_things.json'
test_dataset_things = output_dir + '/test_dataset_instance_things.json'

# Load categories from categories.json
with open(categories_json, 'r') as f:
    categories = json.load(f)

cell_category_ids = []
for category in categories:
    if category['supercategory'] == 'Cell':
        cell_category_ids.append(category["id"])


def process_coco_file(input_path, output_path):
    with open(input_path, 'r') as f:
        coco_data = json.load(f)
    filtered_annotations = [annotation for annotation in coco_data["annotations"] if annotation['category_id'] in cell_category_ids]
    filtered_coco_data = {
        "images": coco_data.get("images", []),
        "annotations": filtered_annotations,
        "categories": [category for category in categories if category['id'] in cell_category_ids],
    }
    with open(output_path, 'w') as f:
        json.dump(filtered_coco_data, f)


process_coco_file(train_dataset, train_dataset_things)
process_coco_file(val_dataset, val_dataset_things)
process_coco_file(test_dataset, test_dataset_things)
