import os
from PIL import Image, ImageDraw, ImageChops
import geojson
import json
import numpy as np
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
from skimage.measure import find_contours
from skimage.segmentation import watershed
from scipy.ndimage import center_of_mass, distance_transform_edt
from tqdm import tqdm


# used to convert a mask (RLE) to [polygons]
def mask_to_polygons(mask):
    polygons = []
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    contours = find_contours(padded_mask, 0.5)
    for contour in contours:
        contour = np.subtract(contour, 1)  # subtracting the padding
        contour = np.maximum(contour, 0)
        contour = np.flip(contour, axis=1)  # flipping x and y axis for COCO format
        segmentation = contour.ravel().tolist()
        segmentation = [float(point) for point in segmentation]
        polygons.append(segmentation)
        if len(segmentation) >= 6:  # polygon consists of at least 3 points
            polygons.append(segmentation)
    return polygons


# used to determine the marker for each cell in watershedding
def find_center(mask):
    center_y, center_x = center_of_mass(mask)
    return (round(center_y), round(center_x))


# auxiliary function to plot a mask
def plot_mask(mask, txt="mask"):
    plt.imshow(mask, cmap='gray')
    plt.title(txt)
    plt.colorbar()
    plt.show()


def process_image(image_id, image_path, cell_geojson_path, coco_data, cell_category_name_to_id, annotation_id):
    with open(cell_geojson_path) as f_cell:
        cell_geojson_data = geojson.load(f_cell)

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
    cell_centers = np.zeros((1024, 1024), dtype=np.uint32)  # pixel of cell center = annotation ID
    annotation_to_mask = {}
    pixel_to_annotation = {}
    overlaps_map = np.zeros((1024, 1024), dtype=np.uint16)

    # iterate over all cell (instance) features of an image
    for feature in tqdm((cell_geojson_data['features'])):
        classification_name = feature['properties']['classification']['name']
        if classification_name in cell_category_name_to_id:
            category_id = cell_category_name_to_id[classification_name]
            cell_ids.append(annotation_id)
        else:
            print("Unkown classification type")

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

        cell_mask_img = ImageChops.logical_or(mask_img, cell_mask_img)
        center_y, center_x = find_center(np.array(mask_img))
        cell_centers[center_y, center_x] = np.uint32(annotation_id)

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

        overlap_mask = np.logical_and.reduce([annotation_to_mask[segment_id] for segment_id in overlap_group])

        for segment_id in overlap_group:
            mask = annotation_to_mask[segment_id]

            # prune the cells by subtracting the overlap
            pruned_mask = np.logical_and(mask, ~overlap_mask)
            segmented_mask = np.logical_and(cell_segmentation == segment_id, mask)

            # the new mask is the pruned mask + the pixels assigned during watershedding
            new_mask = np.logical_or(pruned_mask, segmented_mask)

            if not np.any(new_mask):
                raise ValueError(f"Empty mask in image {str(image_id)}: {str(segment_id)}")

            # update the segment's RLE
            annotation_to_mask[segment_id] = new_mask
            coco_data['annotations'][segment_id - 1]['segmentation'] = mask_to_polygons(new_mask)

    # here the duplicate annotations are actually deleted
    for annotation_id in annotations_to_delete:
        del coco_data['annotations'][annotation_id - 1]

    return annotation_id


def convert_geojson_to_coco(dataset_dir, coco_output_path, cell_categories):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": cell_categories
    }

    annotation_id = 1
    image_id = 1
    cell_category_name_to_id = {category['name']: category['id'] for category in cell_categories}
    image_paths = []
    cell_geojson_paths = []

    # iterate over all files in the dataset
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".tiff") and not filename.endswith("_context.tiff"):
            image_filename = os.path.join(dataset_dir, filename)
            cell_filename = filename.replace(".tiff", "_cell.geojson")

            # check if the corresponding _cell GeoJSONs exist, and if not, skip the image
            # can be removed because dataset is now complete (150, 150, 150)
            if cell_filename in os.listdir(dataset_dir):
                image_paths.append(image_filename)
                cell_geojson_paths.append(os.path.join(dataset_dir, cell_filename))

    # iterate over all images
    for cell_geojson_path, image_path in tqdm(
            zip(cell_geojson_paths, image_paths), desc=f"Processing {len(image_paths)} images"):
        if image_id == 1:
            annotation_id = process_image(image_id, image_path, cell_geojson_path, coco_data, cell_category_name_to_id,
                                annotation_id)
        image_id += 1

    # save to JSON file
    with open(coco_output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))  # current directory of this script
    dataset_dir = os.path.join(script_dir, "..", "..", "dataset")
    coco_output_path = os.path.join(script_dir, "..", "..", "output/coco_format_instance_RLE_1.json")
    categories_path = os.path.join(script_dir, "..", "..", "output/panoptic_coco_categories.json")
    cell_categories = []

    with open(categories_path, 'r') as f:
        categories = json.load(f)

    for item in categories:
        category = {
            "id": item["id"],
            "name": item["name"],
            "supercategory": item["supercategory"]
        }
        if item["supercategory"] == "Cell":
            cell_categories.append(category)

    convert_geojson_to_coco(dataset_dir, coco_output_path, cell_categories)
    print("Done")


    '''
    Run the command below (at top-level of this GitHub repo) to convert from a COCO JSON file to panoptic COCO.

    python panopticapi/converters/detection2panoptic_coco_format.py --input_json_file output/coco_format_RLE.json --output_json_file output/panoptic_coco_RLE.json --categories_json_file output/panoptic_coco_categories.json
    '''
