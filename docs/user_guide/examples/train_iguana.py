import os
import sys
import comet_ml
from pathlib import Path
# Split the first 4 images into test
import pandas as pd
import torch
import numpy as np

import os
import sys
import comet_ml
from pathlib import Path
# Split the first 4 images into test
import pandas as pd
import torch
import numpy as np
import argparse
import os
from typing import Optional

from hydra import compose, initialize_config_dir, initialize
from omegaconf import DictConfig, OmegaConf
from deepforest import main
import os
from deepforest.utilities import read_file
from deepforest import visualize

from deepforest import main
import os
from deepforest.utilities import read_file
from deepforest import visualize
import comet_ml
from pytorch_lightning.loggers import CometLogger
# Comment out if comet-ml is not installed
comet_logger = CometLogger(api_key="bVOa3vnaXoP7OIstSDdblokzb")



# in pascal voc format
# eikelboom_640_path = Path("/home/cwinkelmann/work/deepforest/data/eikelboom_640")
# label_dict={"Giraffe": 0, "Elephant": 1, "Zebra": 2, "Bird": 3}
# model_path = eikelboom_640_path / "models"
#
# train_annotations_path = "/home/cwinkelmann/work/deepforest/data/eikelboom_640/eikelboom_train/detection_train_0_640/deep_forest_format__640_0_crops.csv"
# train_root_dir =  "/home/cwinkelmann/work/deepforest/data/eikelboom_640/eikelboom_train/detection_train_0_640/crops_640_numNone_overlap0"
#
# test_annotations_path = "/home/cwinkelmann/work/deepforest/data/eikelboom_640/eikelboom_test/detection_test_0_640/deep_forest_format__640_0_crops.csv"
# test_root_dir =  "/home/cwinkelmann/work/deepforest/data/eikelboom_640/eikelboom_test/detection_test_0_640/crops_640_numNone_overlap0"
#
#
# eikelboom_640_path = Path("/home/cwinkelmann/work/deepforest/data/iguana_floreana_640")
# label_dict={"iguana": 0}
# model_path = eikelboom_640_path / "models"
#
# train_annotations_path = "/home/cwinkelmann/work/deepforest/data/iguana_floreana_640/detection_train_0_640/deep_forest_format__640_0_crops.csv"
# train_root_dir =  "/home/cwinkelmann/work/deepforest/data/iguana_floreana_640/detection_train_0_640/crops_640_numNone_overlap0"
#
# test_annotations_path = "/home/cwinkelmann/work/deepforest/data/iguana_floreana_640/detection_val_0_640/deep_forest_format__640_0_crops.csv"
# test_root_dir =  "/home/cwinkelmann/work/deepforest/data/iguana_floreana_640/detection_val_0_640/crops_640_numNone_overlap0"


label_dict={"Bird": 0}
michigan_1000_path = Path("/home/cwinkelmann/work/deepforest/data/michigan/")
model_path = michigan_1000_path / "models"

train_annotations_path = "/home/cwinkelmann/work/deepforest/data/michigan/michigan_train.csv"
train_root_dir =  "/home/cwinkelmann/work/deepforest/data/michigan/"

test_annotations_path = "/home/cwinkelmann/work/deepforest/data/michigan/michigan_test.csv"
test_root_dir =  "/home/cwinkelmann/work/deepforest/data/michigan/"



# train_annotations_path = "/home/cwinkelmann/work/deepforest/data/DeepWaterHorizon/train_annotations.csv"
# train_root_dir =  "/home/cwinkelmann/work/deepforest/data/DeepWaterHorizon"
#
# test_annotations_path = "/home/cwinkelmann/work/deepforest/data/DeepWaterHorizon/train_annotations.csv"
# test_root_dir =  "/home/cwinkelmann/work/deepforest/data/DeepWaterHorizon"


config_name = "config_train_iguana"

initialize(version_base=None, config_path="pkg://deepforest.conf")
config = compose(config_name=config_name, overrides=[])

model = main.deepforest(config=config, num_classes=len(label_dict.keys()), label_dict=label_dict)
#model.load_model(model_name="weecology/deepforest-bird", label_dict=label_dict) # FIXME why can't I reuse this model when I want to add more classes?

# model.load_model( model_name="weecology/deepforest-tree", label_dict=label_dict)
model.config

model.config["train"]["csv_file"] = train_annotations_path
model.config["train"]["root_dir"] = train_root_dir
model.config["validation"]["csv_file"] = test_annotations_path
model.config["validation"]["root_dir"] = test_root_dir
model.config["validation"]["val_accuracy_interval"] = 2
model.config["train"]["epochs"] = 150
model.config["train"]["lr"] = 0.0001
model.config["batch_size"] = 5

# Train model for just a few steps for show on cpu (model.create_trainer(max_steps=5)), takes 1 min on GPU
model.create_trainer(logger=comet_logger)
model.trainer.fit(model)

model.trainer.save_checkpoint(
    os.path.join(model_path, "checkpoint_eikelboom_cosine_lr_retinanet.pl"))
torch.save(model.model.state_dict(), os.path.join(model_path, "weights_cosine_lr"))

loaded_model = main.deepforest.load_from_checkpoint(
    os.path.join(model_path, "checkpoint_eikelboom_cosine_lr_retinanet.pl"))

fine_tuned_predictions = []
for image in Path(test_root_dir).glob("*.jpg"):
    relative_path = str(michigan_1000_path / image)
    # predictions = model.predict_image(path=relative_path)
    predictions = model.predict_tile(path=relative_path)

    print(f"average score: {np.mean([s for s in predictions.score])}")
    filtered_predictions = predictions[predictions["score"] > -2.0]
    filtered_predictions.root_dir = predictions.root_dir

    #
    print(filtered_predictions)


    if len(filtered_predictions) == 0:
        print(f"predictions: {filtered_predictions}")
        continue
    else:
        fine_tuned_predictions.append(filtered_predictions)
        # visualize.plot_results(predictions, ground_truth=test_annotations, thickness=7)
        visualize.plot_results(filtered_predictions, thickness=7)

# Evaluate on test set
results = model.evaluate(
    csv_file=str(michigan_1000_path / 'test_annotations.csv'),
    root_dir=str(michigan_1000_path),
)

box_precision = results["box_precision"]
box_recall = results["box_recall"]
print(f"box precision: {results['box_precision']} and recall: {results['box_recall']}")
# save the results to a csv file
results["results"].to_csv(model_path / "results_test_lr_cosine.csv", index=False)

checkpoint_path = model_path / f"checkpoint_eikelboom_retinanet.pl"
weights_path = os.path.join(model_path, f"weights_eikelboom_retinanet")

model.trainer.save_checkpoint(checkpoint_path)

torch.save(model.model.state_dict(), weights_path )

loaded_model = main.deepforest.load_from_checkpoint(checkpoint_path)