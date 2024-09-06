from pathlib import Path

import pandas as pd

base_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/deepforest/dataset_nest_spatially_disjunct")

paths = [
    "Horus",
    "Jerrod",
    "JetportNew_A",
    "JetportNew_B",
    "JetportNew_C",
]

paths = [base_path / x for x in paths]

df_gt = pd.read_csv("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/deepforest/dataset_nest/nest_data.csv")

for extract_folder in paths:
    site_images = [x.name for x in list(extract_folder.glob("*.JPG"))]
    df_filtered = df_gt[df_gt.image_path.isin( site_images )]

    df_filtered.to_csv(extract_folder / "ground_truth.csv", index=False)