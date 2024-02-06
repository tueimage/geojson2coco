import geojson
import json
import os
from PIL import Image
import numpy as np

def convert_geojson_to_coco_simple(image_dir, geojson_input_dir, coco_output_path, categories):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    annotation_id = 1
    image_id = 1
    category_name_to_id = {category['name']: category['id'] for category in categories}

    for geojson_filename in os.listdir(geojson_input_dir):
        if geojson_filename.endswith('.geojson'):
            # Infer image filename by replacing the GeoJSON extension with the image extension
            image_filename = geojson_filename.replace('.geojson', '.png')
            
            geojson_path = os.path.join(geojson_input_dir, geojson_filename)
            with open(geojson_path) as f:
                geojson_data = geojson.load(f)

            image_path = os.path.join(image_dir, image_filename)
            with Image.open(image_path) as img:
                width, height = img.size
            coco_data['images'].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": image_filename
            })

            for feature in geojson_data['features']:
                classification_name = feature['properties']['classification']['name']
                category_id = category_name_to_id.get(classification_name)
                if category_id is None:
                    continue  # Skip annotations with unknown classification
                
                # Convert GeoJSON polygon to COCO polygon format (ignoring holes)
                coco_polygon = sum(feature['geometry']['coordinates'][0], [])
                # Compute bounding box and area
                min_x, min_y = np.min(coco_polygon[::2]), np.min(coco_polygon[1::2])
                max_x, max_y = np.max(coco_polygon[::2]), np.max(coco_polygon[1::2])
                bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
                area = np.abs(np.dot(coco_polygon[::2], coco_polygon[1::2][::-1]) - np.dot(coco_polygon[1::2], coco_polygon[::2][::-1])) / 2
                
                coco_data['annotations'].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [coco_polygon],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })
                
                annotation_id += 1

            image_id += 1  # Increment image_id for the next image

    # Save the COCO dataset to the specified output JSON file
    with open(coco_output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

