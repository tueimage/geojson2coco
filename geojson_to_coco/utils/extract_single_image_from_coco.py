import json

def filter_coco_json(input_json_file, output_json_file, image_id):
    with open(input_json_file, 'r') as f:
        coco_data = json.load(f)

    filtered_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    filtered_images = [image for image in coco_data['images'] if image['id'] == image_id]

    filtered_coco_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': coco_data.get('categories', []),
    }

    with open(output_json_file, 'w') as f:
        json.dump(filtered_coco_data, f)


image_id = 181  # adjust accordingly

input_json_file = '../../output/coco_open_source.json'
output_json_file = f'../../output/coco_open_source_{image_id}.json'
filter_coco_json(input_json_file, output_json_file, image_id)
