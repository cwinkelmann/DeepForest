
# load the modules
import comet_ml
import os
import time
import numpy as np
import pandas as pd
import torch
from deepforest import main
from deepforest import get_data
from deepforest import utilities
from deepforest import preprocess
from tqdm import tqdm
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.loggers import WandbLogger
import zipfile
import matplotlib.pyplot as plt
import subprocess

extract_folder = "/home/christian/hnee/DeepForest/docs/examples/nest_images/dataset"
crop_dir = os.path.join(os.getcwd(), "train_data_folder")
annotation_path = os.path.join(extract_folder, "nest_data.csv")

# save to file and create the file dir
annotations_file = os.path.join(crop_dir, "train.csv")
validation_file = os.path.join(crop_dir, "valid.csv")

# initialize the model and change the corresponding config file
m = main.deepforest(label_dict={"Nest": 0})
# m = main.deepforest()
# m = main.deepforest()

m.config["batch_size"] = 18

# move to GPU and use all the GPU resources
m.config["gpus"] = "-1"
m.config["train"]["csv_file"] = annotations_file
m.config["train"]["root_dir"] = os.path.dirname(annotations_file)

# Define the learning scheduler type
m.config["train"]["scheduler"]["type"] = "cosine" # TODO This is to prevent an error with torch.optim.lr_scheduler.ReduceLROnPlateau which can't monitor "val_classification"
m.config["score_thresh"] = 0.4
m.config["train"]["epochs"] = epochs = 100
m.config["train"]["check_val_every_n_epoch"] = 20
m.config["validation"]["csv_file"] = validation_file
m.config["validation"]["root_dir"] = os.path.dirname(validation_file)
wandb_logger = WandbLogger()
m.create_trainer(logger=wandb_logger)
# m.create_trainer()
# load the latest! release model (RetinaNet)
# m.use_release(check_release=True)

m.use_bird_release(check_release=True)

# Start the training
start_time = time.time()
m.trainer.fit(m)
print(f"--- Training on GPU: {(time.time() - start_time):.2f} seconds ---")


# time.sleep(30)

test_file = os.path.join(crop_dir, "test.csv")
save_dir = os.path.join(os.getcwd(), "pred_result_test")
results = m.evaluate(
    test_file, os.path.dirname(test_file), iou_threshold=0.4, savedir=save_dir
)

print(f"results: {results}")

from pathlib import Path
# Add a path to an image to test the model on
from PIL import Image
raster_path = Path("/home/christian/hnee/DeepForest/docs/examples/nest_images/dataset/")
for image_name in ["JetPortNew_03_029_2022_DJI_0437.JPG", "Horus_04_27_2022_DJI_0327.JPG", "Horus_04_27_2022_DJI_0213.JPG"]:
    predicted_raster = m.predict_tile(
        raster_path / image_name, return_plot=True, patch_size=400, patch_overlap=0.25
    )

im = Image.fromarray(predicted_raster)
im.save(f"predicted_raster_{image_name}")

m.trainer.save_checkpoint(
    os.path.join("models", f"checkpoint_epochs_{epochs}_cosine_lr_retinanet.pl")
)
torch.save(m.model.state_dict(), os.path.join("models", "weights_cosine_lr"))