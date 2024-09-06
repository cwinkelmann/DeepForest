"""
take a folder of images and annotations and tile it into smaller images
based on the nest_detection example notebook
"""
from pathlib import Path

# load the modules
import pandas as pd
from deepforest import preprocess


base_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/deepforest/dataset_nest_spatially_disjunct")

paths = [
    "Horus",
    "Jerrod",
    "JetportNew_A",
    "JetportNew_B",
    "JetportNew_C",
]

for extract_folder in [base_path / x for x in paths]:
    print(extract_folder)
    image_names = ["JetPortNew_03_029_2022_DJI_0443.JPG"]

    image_names = list(extract_folder.glob("*.JPG"))

    annotation_path = extract_folder / "ground_truth.csv"
    crop_dir = extract_folder / "crops"
    all_annotations = []

    for image in image_names:
        image_path = extract_folder / image
        annotations = preprocess.split_raster(
            path_to_raster=image_path,
            annotations_file=str(annotation_path),
            patch_size=400,
            patch_overlap=0.0, # avoid data leakage here
            base_dir=str(crop_dir),
        )

        all_annotations.append(annotations)
    train_annotations = pd.concat(all_annotations, ignore_index=True)

    print(train_annotations)

    train_annotations.to_csv(crop_dir / "gt.csv", index=False)