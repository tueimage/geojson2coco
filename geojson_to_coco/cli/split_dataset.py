# Assuming 'instance_json_full' and 'panoptic_json_full' contain the paths to the COCO formatted JSON files for instance and panoptic annotations respectively.
# Splitting the instance.json into training and test sets.
import json
import os


def split_coco_instance_json(instance_json_path, train_json_path, test_json_path, split_ratio):
    with open(instance_json_path, 'r') as f:
        data = json.load(f)

    num_images = len(data['images'])
    split_index = int(num_images * split_ratio)

    train_images = data['images'][:split_index]
    test_images = data['images'][split_index:]

    train_image_ids = set([image['id'] for image in train_images])
    test_image_ids = set([image['id'] for image in test_images])

    train_annotations = [anno for anno in data['annotations'] if anno['image_id'] in train_image_ids]
    test_annotations = [anno for anno in data['annotations'] if anno['image_id'] in test_image_ids]

    categories = data['categories']

    train_data = {'images': train_images, 'annotations': train_annotations, 'categories': categories}
    test_data = {'images': test_images, 'annotations': test_annotations, 'categories': categories}

    with open(train_json_path, 'w') as f:
        json.dump(train_data, f)

    with open(test_json_path, 'w') as f:
        json.dump(test_data, f)


def split_coco_panoptic_json(panoptic_json_path, train_json_path, test_json_path, split_ratio):
    with open(panoptic_json_path, 'r') as f:
        data = json.load(f)

    num_images = len(data['images'])
    split_index = int(num_images * split_ratio)

    train_images = data['images'][:split_index]
    test_images = data['images'][split_index:]

    train_image_ids = set([image['id'] for image in train_images])
    test_image_ids = set([image['id'] for image in test_images])

    train_annotations = [anno for anno in data['annotations'] if anno['image_id'] in train_image_ids]
    test_annotations = [anno for anno in data['annotations'] if anno['image_id'] in test_image_ids]

    train_data = {'images': train_images, 'annotations': train_annotations, 'categories': data['categories']}
    test_data = {'images': test_images, 'annotations': test_annotations, 'categories': data['categories']}

    with open(train_json_path, 'w') as f:
        json.dump(train_data, f)

    with open(test_json_path, 'w') as f:
        json.dump(test_data, f)

output_dir = '../../output/'
instance_json_path = os.path.join(output_dir, 'coco_format_RLE_instance.json')
panoptic_json_path = os.path.join(output_dir, 'panoptic_coco_RLE.json')
train_instance_json_path = os.path.join(output_dir, 'train_instance.json')
test_instance_json_path = os.path.join(output_dir, 'test_instance.json')
train_panoptic_json_path = os.path.join(output_dir, 'train_panoptic.json')
test_panoptic_json_path = os.path.join(output_dir, 'test_panoptic.json')

# current ratio for the training set (100 out of 150 images => 2/3)
split_ratio = 100 / 150
split_coco_instance_json(instance_json_path, train_instance_json_path, test_instance_json_path, split_ratio)
split_coco_panoptic_json(panoptic_json_path, train_panoptic_json_path, test_panoptic_json_path, split_ratio)
