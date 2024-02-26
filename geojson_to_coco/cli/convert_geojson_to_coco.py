import os
from PIL import Image, ImageDraw, ImageChops
import geojson
import json
import numpy as np
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
from skimage.segmentation import watershed
from scipy.ndimage import center_of_mass, distance_transform_edt
from tqdm import tqdm


# used to determine the marker for each cell/tissue in watershedding
def find_center(mask):
    center_y, center_x = center_of_mass(mask)
    return (round(center_y), round(center_x))


# auxiliary function to plot a mask
def plot_mask(mask, txt="mask"):
    plt.imshow(mask, cmap='gray')
    plt.title(txt)
    plt.colorbar()
    plt.show()


def process_image(image_id, image_path, cell_geojson_path, tissue_geojson_path, coco_data, cell_category_name_to_id,
                  tissue_category_name_to_id, annotation_id):
    with open(cell_geojson_path) as f_cell, open(tissue_geojson_path) as f_tissue:
        cell_geojson_data = geojson.load(f_cell)
        tissue_geojson_data = geojson.load(f_tissue)

    with Image.open(image_path) as img:
        width, height = img.size
        img_grayscale = img.convert('L')
        img_array = np.array(img_grayscale)
        coco_data['images'].append({
            "id": int(image_id),
            "width": width,
            "height": height,
            "file_name": os.path.basename(image_path)
        })

    cell_ids = []
    cell_mask_img = Image.new('1', (width, height))  # stores all cell masks combined
    tissue_mask_img = Image.new('1', (width, height))  # stores all tissue masks combined
    cell_centers = np.zeros((1024, 1024), dtype=np.uint16)  # pixel of cell center = annotation ID
    tissue_centers = np.zeros((1024, 1024), dtype=np.uint16)  # pixel of tissue center = annotation ID
    annotation_to_mask = {}
    pixel_to_annotation = {}
    overlaps_map = np.zeros((1024, 1024), dtype=np.uint16)

    # iterate over all features of an image
    for feature in tqdm((cell_geojson_data['features'] + tissue_geojson_data['features'])):
        is_tissue = False
        is_cell = False
        classification_name = feature['properties']['classification']['name']
        if classification_name in cell_category_name_to_id:
            category_id = cell_category_name_to_id[classification_name]
            is_cell = True
            cell_ids.append(annotation_id)
        elif classification_name in tissue_category_name_to_id:
            category_id = tissue_category_name_to_id[classification_name]
            is_tissue = True
        else:
            continue  # annotation with unknown classification

        geometry = feature["geometry"]
        geometry_type = feature["geometry"]["type"]
        if geometry_type == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry_type == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue  # geometry type is not supported

        mask_img = Image.new('1', (width, height))

        for polygon_coords in polygons:
            exterior_coords = polygon_coords[0]
            interior_coords = polygon_coords[1:]
            exterior_ring = [tuple(coord) for coord in exterior_coords]
            interior_rings = [[tuple(coord) for coord in interior] for interior in interior_coords]

            draw = ImageDraw.Draw(mask_img)
            draw.polygon(exterior_ring, outline=1, fill=1)  # draw the outer shape on the mask

            for interior_ring in interior_rings:
                draw.polygon(interior_ring, outline=0, fill=0)  # erase the holes from the mask

        if is_cell:  # add mask to global cell mask, get center of cell
            cell_mask_img = ImageChops.logical_or(mask_img, cell_mask_img)
            center_y, center_x = find_center(np.array(mask_img))
            cell_centers[center_y, center_x] = np.uint16(annotation_id)
        elif is_tissue:
            tissue_mask_img = ImageChops.logical_or(mask_img, tissue_mask_img)
            center_y, center_x = find_center(np.array(mask_img))
            tissue_centers[center_y, center_x] = np.uint16(annotation_id)

            # subtract cell mask from tissue mask
            mask_img = ImageChops.logical_and(mask_img, ImageChops.invert(cell_mask_img))

        mask = np.array(mask_img)
        overlaps_map += mask

        # update the annotations per pixel
        for coord in np.argwhere(mask > 0):
            coord_tuple = tuple(coord)
            if coord_tuple not in pixel_to_annotation:
                pixel_to_annotation[coord_tuple] = []
            pixel_to_annotation[coord_tuple].append(annotation_id)

        annotation_to_mask[annotation_id] = mask
        mask_encoded = maskUtils.encode(np.asfortranarray(mask))

        RLE = {
            "counts": mask_encoded["counts"].decode('utf-8'),  # Corrected access to counts
            "size": [width, height]
        }

        coco_data['annotations'].append({
            "id": annotation_id,
            "image_id": int(image_id),
            "category_id": category_id,
            "segmentation": RLE,
            "area": int(mask.sum()),  # Area calculation based on mask
            "bbox": [int(x) for x in list(maskUtils.toBbox(mask_encoded))],  # Bounding box calculation based on mask
            "iscrowd": 0
        })
        annotation_id += 1

    # marker-controlled watershed segmentation
    cell_segmentation = watershed(-img_array, markers=cell_centers, mask=np.array(cell_mask_img))
    tissue_segmentation = watershed(-img_array, markers=tissue_centers, mask=np.array(tissue_mask_img))
    annotations_to_delete = set()
    overlap_groups = {frozenset(group) for group in pixel_to_annotation.values() if len(group) >= 2}

    # for each overlap, check for duplicate annotations by Mark
    for overlap_group in overlap_groups:
        seen_masks = {}
        duplicate_annotation = False
        for segment_id in overlap_group:  # overlap_group usually has length 2 or 3
            mask = annotation_to_mask[segment_id]
            mask_bytes = mask.tobytes()
            if mask_bytes in seen_masks:  # duplicate annotation
                duplicate_segment_id = seen_masks[mask_bytes]
                annotations_to_delete.add(segment_id)
                print("Deleted duplicate annotation: ", segment_id, " which is a duplicate of: ", duplicate_segment_id)
                duplicate_annotation = True
                break  # unsafe, no longer needed when duplicate annotations are resolved
            else:
                seen_masks[mask_bytes] = segment_id
        if duplicate_annotation:
            continue

        # for each overlap, construct new cell/tissue masks
        overlap_mask = np.logical_and.reduce([annotation_to_mask[segment_id] for segment_id in overlap_group])

        for segment_id in overlap_group:
            mask = annotation_to_mask[segment_id]

            # prune the cell/tissue by subtracting the overlap
            pruned_mask = np.logical_and(mask, ~overlap_mask)
            if segment_id in cell_ids:
                segmented_mask = np.logical_and(cell_segmentation == segment_id, mask)
            else:
                segmented_mask = np.logical_and(tissue_segmentation == segment_id, mask)

            # the new mask is the pruned mask + the pixels assigned during watershedding
            new_mask = np.logical_or(pruned_mask, segmented_mask)
            annotation_to_mask[segment_id] = new_mask

            if not np.any(new_mask):
                print("Empty mask: " + str(segment_id))

            # update the segment's RLE
            mask_encoded = maskUtils.encode(np.asfortranarray(new_mask))
            coco_data['annotations'][segment_id - 1]['segmentation'] = {
                "counts": mask_encoded["counts"].decode('utf-8'),
                "size": [width, height]
            }

    # here the duplicate annotations are actually deleted
    for annotation_id in annotations_to_delete:
        del coco_data['annotations'][annotation_id - 1]

    return annotation_id


def convert_geojson_to_coco(dataset_dir, coco_output_path, cell_categories, tissue_categories):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": cell_categories + tissue_categories
    }

    annotation_id = 1
    image_id = 1
    cell_category_name_to_id = {category['name']: category['id'] for category in cell_categories}
    tissue_category_name_to_id = {category['name']: category['id'] for category in tissue_categories}
    image_paths = []
    cell_geojson_paths = []
    tissue_geojson_paths = []

    # iterate over all files in the dataset
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".tiff") and not filename.endswith("_context.tiff"):
            image_filename = os.path.join(dataset_dir, filename)
            cell_filename = filename.replace(".tiff", "_cell.geojson")
            tissue_filename = filename.replace(".tiff", "_tissue.geojson")

            # check if the corresponding _cell and _tissue GeoJSONs exist, and if not, skip the image
            # can be removed because dataset is now complete (150, 150, 150)
            if tissue_filename in os.listdir(dataset_dir) and cell_filename in os.listdir(dataset_dir):
                image_paths.append(image_filename)
                cell_geojson_paths.append(os.path.join(dataset_dir, cell_filename))
                tissue_geojson_paths.append(os.path.join(dataset_dir, tissue_filename))

    # iterate over all images
    for cell_geojson_path, tissue_geojson_path, image_path in tqdm(
            zip(cell_geojson_paths, tissue_geojson_paths, image_paths), desc=f"Processing {len(image_paths)} images"):
        if image_id > 1:
            break  # TEMPORARY, PROCESS 3 IMAGES ONLY
        annotation_id = process_image(image_id, image_path, cell_geojson_path, tissue_geojson_path, coco_data, cell_category_name_to_id,
                      tissue_category_name_to_id, annotation_id)
        image_id += 1

    # save to JSON file
    with open(coco_output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))  # current directory of this script
    dataset_dir = os.path.join(script_dir, "..", "..", "dataset")
    coco_output_path = os.path.join(script_dir, "..", "..", "output/coco_format_RLE.json")

    cell_categories = [
        {"id": 1, "name": "cell_lymphocyte", "supercategory": "Cell"},
        {"id": 2, "name": "cell_macrophage", "supercategory": "Cell"},
        {"id": 3, "name": "cell_stroma", "supercategory": "Cell"},
        {"id": 4, "name": "cell_melanophage", "supercategory": "Cell"},
        {"id": 5, "name": "cell_other", "supercategory": "Cell"},
        {"id": 6, "name": "cell_endothelium", "supercategory": "Cell"},
        {"id": 7, "name": "cell_plasma_cell", "supercategory": "Cell"},
        {"id": 8, "name": "cell_tumor", "supercategory": "Cell"},
        {"id": 9, "name": "cell_epithelium", "supercategory": "Cell"},
        {"id": 10, "name": "cell_neutrophil", "supercategory": "Cell"},
        {"id": 11, "name": "cell_necrosis", "supercategory": "Cell"},
    ]

    tissue_categories = [
        {"id": 12, "name": "tissue_blood_vessel", "supercategory": "Tissue"},
        {"id": 13, "name": "tissue_stroma", "supercategory": "Tissue"},
        {"id": 14, "name": "tissue_tumor", "supercategory": "Tissue"},
        {"id": 15, "name": "tissue_epidermis", "supercategory": "Tissue"},
        {"id": 16, "name": "tissue_white_background", "supercategory": "Tissue"},
        {"id": 17, "name": "tissue_necrosis", "supercategory": "Tissue"}
    ]

    convert_geojson_to_coco(dataset_dir, coco_output_path, cell_categories, tissue_categories)
    print("Done")

    '''
    Run the command below (at top-level of this GitHub repo) to convert from a COCO JSON file to panoptic COCO.

    python panopticapi/converters/detection2panoptic_coco_format.py --input_json_file output/coco_format_RLE.json --output_json_file output/panoptic_coco_RLE.json --categories_json_file output/panoptic_coco_categories.json
    '''
