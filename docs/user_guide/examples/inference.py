from deepforest import main
import os
from deepforest.utilities import read_file
from deepforest import visualize


model = main.deepforest()
import torch
root_folder = "data/DeepWaterHorizon/models"

annotations = read_file("data/DeepWaterHorizon/annotations.csv")
images = annotations.image_path.unique()

test_images = images[:4]
annotations.head()

annotations = annotations[annotations.label == "Bird"]
test_annotations = annotations[annotations.image_path.isin(test_images)]
train_annotations = annotations[~annotations.image_path.isin(test_images)]


# /home/cwinkelmann/work/deepforest/data/DeepWaterHorizon/models/checkpoint_retinanet_recall_0.0097__precision0.0563.pl
loaded_model = main.deepforest.load_from_checkpoint(
    #os.path.join(root_folder, "checkpoint_epochs_10_cosine_lr_retinanet.pl"))
    os.path.join(root_folder, "checkpoint_retinanet_recall_0.0097__precision0.0563.pl"))
# Add a path to an image to test the model on
fine_tuned_predictions = []
for image in test_images:
    relative_path = os.path.join("data/DeepWaterHorizon", image)
    predictions = loaded_model.predict_image(path=relative_path)
    fine_tuned_predictions.append(predictions)

    filtered_predictions = predictions[predictions.score > 0.2]
    filtered_predictions.root_dir = predictions.root_dir

    visualize.plot_results(predictions, ground_truth=test_annotations, thickness=7)