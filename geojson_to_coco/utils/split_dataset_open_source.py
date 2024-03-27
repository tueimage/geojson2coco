import json
import os

coco = "../../output/panoptic_coco_RLE_open_source.json"

# Load your original COCO dataset
with open(coco, 'r') as f:
    coco_dataset = json.load(f)

# Separate images into primary and metastasis
primary_images = [img for img in coco_dataset['images'] if 'primary' in img['file_name']]
metastasis_images = [img for img in coco_dataset['images'] if 'metastasis' in img['file_name']]

# Split images based on the requirements
train_images = primary_images[:60] + metastasis_images[:60]
val_images = primary_images[60:80] + metastasis_images[60:80]
test_images = primary_images[80:] + metastasis_images[80:]

# Function to filter annotations by image ids
def filter_annotations_by_images(selected_images, annotations):
    selected_image_ids = set([img['id'] for img in selected_images])
    return [ann for ann in annotations if ann['image_id'] in selected_image_ids]

# Split annotations based on the images selected
train_annotations = filter_annotations_by_images(train_images, coco_dataset['annotations'])
val_annotations = filter_annotations_by_images(val_images, coco_dataset['annotations'])
test_annotations = filter_annotations_by_images(test_images, coco_dataset['annotations'])

# Reconstruct dataset splits
def reconstruct_dataset(images, annotations, categories):
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

train_dataset = reconstruct_dataset(train_images, train_annotations, coco_dataset['categories'])
val_dataset = reconstruct_dataset(val_images, val_annotations, coco_dataset['categories'])
test_dataset = reconstruct_dataset(test_images, test_annotations, coco_dataset['categories'])

# Generate paths for the new JSON files based on the original COCO dataset's path
coco_dir = os.path.dirname(coco)
train_json_path = os.path.join(coco_dir, 'train_dataset.json')
val_json_path = os.path.join(coco_dir, 'val_dataset.json')
test_json_path = os.path.join(coco_dir, 'test_dataset.json')

# Save the splits to new JSON files
with open(train_json_path, 'w') as f:
    json.dump(train_dataset, f)

with open(val_json_path, 'w') as f:
    json.dump(val_dataset, f)

with open(test_json_path, 'w') as f:
    json.dump(test_dataset, f)
