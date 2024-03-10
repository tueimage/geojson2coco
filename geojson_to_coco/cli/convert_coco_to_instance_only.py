import json

input_json_path = '../../output/coco_format_RLE.json'
output_json_path = '../../output/coco_format_RLE_instance.json'

with open(input_json_path, 'r') as file:
    data = json.load(file)

tissue_category_ids = [category['id'] for category in data['categories'] if category['supercategory'] == 'Tissue']

filtered_annotations = [annotation for annotation in data['annotations'] if annotation['category_id'] not in tissue_category_ids]
data['annotations'] = filtered_annotations

filtered_categories = [category for category in data['categories'] if category['supercategory'] != 'Tissue']
data['categories'] = filtered_categories

with open(output_json_path, 'w') as file:
    json.dump(data, file)

print('Successfully removed semantic (stuff) annotations.')
