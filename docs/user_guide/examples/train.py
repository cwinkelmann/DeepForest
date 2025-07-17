import os
from pathlib import Path

import numpy as np
# Split the first 4 images into test
import torch
from hydra import compose, initialize

from deepforest import main
from deepforest import visualize
from deepforest.utilities import read_file

label_dict = {"Bird": 0, "Giraffe": 1, "Elephant": 2, "Zebra": 3}
label_dict = {"Bird": 0}

config_name = "config_train_iguana"

initialize(version_base=None, config_path="pkg://deepforest.conf")
config = compose(config_name=config_name, overrides=[])

config["architecture"] = 'retinanet'
config["architecture"] = 'DeformableDetr'

model = main.deepforest(config=config, label_dict=label_dict, num_classes=len(label_dict))
# model.load_model(model_name="weecology/deepforest-bird", label_dict=label_dict)

model.config

# in pascal voc format
deepwater_horizon_path = Path("/home/cwinkelmann/work/deepforest/data/DeepWaterHorizon")
model_path = deepwater_horizon_path / "models"

annotations = read_file(str(deepwater_horizon_path / "annotations.csv"))
annotations.head()

images = deepwater_horizon_path.glob("*.jpg")
images_names = [image.name for image in images]

images = annotations.image_path.unique()
test_images = images[:4]
train_images = images[4:]

print(f"train_images: {train_images}")
print(f"test_images: {test_images}")

annotations = annotations[annotations.image_path.isin(images_names)]

annotations = annotations[annotations.label == "Bird"]
test_annotations = annotations[annotations.image_path.isin(test_images)]
train_annotations = annotations[~annotations.image_path.isin(test_images)]

# Save the annotations
test_annotations.to_csv(deepwater_horizon_path / "test_annotations.csv", index=False)
train_annotations.to_csv(deepwater_horizon_path / "train_annotations.csv", index=False)

# Comment out if comet-ml is not installed
from pytorch_lightning.loggers import CometLogger

comet_logger = CometLogger(api_key="bVOa3vnaXoP7OIstSDdblokzb",
                           name=f"deepforest_{config.architecture}_deepwater_horizon", )

# model.load_model("joshvm/milliontrees-detr")

# config = utilities.load_config()
# config.architecture = "DeformableDetr"


model.config["train"]["csv_file"] = str(deepwater_horizon_path / "train_annotations.csv")
model.config["train"]["root_dir"] = str(deepwater_horizon_path)
model.config["validation"]["csv_file"] = str(deepwater_horizon_path / "test_annotations.csv")
model.config["validation"]["root_dir"] = str(deepwater_horizon_path)
model.config["validation"]["val_accuracy_interval"] = 2
# model.config["train"]["epochs"] = 20
model.config["train"]["lr"] = 0.001
model.config["batch_size"] = 15

# Train model for just a few steps for show on cpu (model.create_trainer(max_steps=5)), takes 1 min on GPU
model.create_trainer(logger=comet_logger)
model.trainer.fit(model)

model.trainer.save_checkpoint(
    os.path.join(model_path, "checkpoint_epochs_10_cosine_lr_retinanet.pl"))
torch.save(model.model.state_dict(), os.path.join(model_path, "weights_cosine_lr"))

loaded_model = main.deepforest.load_from_checkpoint(
    os.path.join(model_path, "checkpoint_epochs_10_cosine_lr_retinanet.pl"))

fine_tuned_predictions = []
for image in test_images:
    relative_path = str(deepwater_horizon_path / image)
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
    csv_file=str(deepwater_horizon_path / 'test_annotations.csv'),
    root_dir=str(deepwater_horizon_path),
)

box_precision = results["box_precision"]
box_recall = results["box_recall"]
print(f"box precision: {results['box_precision']} and recall: {results['box_recall']}")
# save the results to a csv file
results["results"].to_csv(model_path / "results_test_lr_cosine.csv", index=False)

checkpoint_path = model_path / f"checkpoint_retinanet.pl"
weights_path = os.path.join(model_path, f"weights_retinanet")

model.trainer.save_checkpoint(checkpoint_path)

torch.save(model.model.state_dict(), weights_path)

loaded_model = main.deepforest.load_from_checkpoint(checkpoint_path)
