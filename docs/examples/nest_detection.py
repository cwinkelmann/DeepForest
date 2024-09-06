
# load the modules
import os

from deepforest import main

extract_folder = "/home/christian/hnee/DeepForest/docs/examples/nest_images/dataset"


from pathlib import Path
# Add a path to an image to test the model on
raster_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/deepforest/dataset_nest")


model = main.deepforest.load_from_checkpoint(
    os.path.join("/Users/christian/Downloads", "checkpoint_cosine_lr.pl")
)

model.model.score_thresh = 0.8

for image_name in ["JetPortNew_03_029_2022_DJI_0437.JPG", "Horus_04_27_2022_DJI_0327.JPG", "Horus_04_27_2022_DJI_0213.JPG"]:

    df_pred = model.predict_tile(
        raster_path / image_name, return_plot=False, patch_size=400, patch_overlap=0.25
    )

    print(df_pred)


