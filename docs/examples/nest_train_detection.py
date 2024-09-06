
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


# initialize the model and change the corresponding config file
m = main.deepforest(label_dict={"Nest": 0})
m.model.device = "cpu"


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