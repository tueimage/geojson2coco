import pytest


def test_convert_geojson_to_coco_simple():
    # Define the directory containing your test data
    test_data_dir = 'test_data/'

    # Define the output path for the COCO formatted JSON
    coco_output_path = 'test_data/coco_conversion.json'

    # Define the categories based on your GeoJSON classification names
    categories = [
        {"id": 1, "name": "endothelium"},
        {"id": 2, "name": "macrophage"},
        {"id": 3, "name": "melanophage"},
        {"id": 4, "name": "epithelium"},
        {"id": 5, "name": "Tumor"},
        {"id": 5, "name": "lymphocyte"},
    ]

    # Call the conversion function
    convert_geojson_to_coco_simple(
        image_dir=test_data_dir,
        geojson_input_dir=test_data_dir,
        coco_output_path=coco_output_path,
        categories=categories
    )

    # Verify the output
    assert os.path.exists(coco_output_path), "COCO conversion output file does not exist."
