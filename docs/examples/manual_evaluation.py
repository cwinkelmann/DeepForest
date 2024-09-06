"""
- take a folder of images and the ground truth,
- apply predict_tile on it
- either calculate the MAE based on the counts or the F1 score based on the iou of the bounding boxes
"""
import numpy as np

import pandas as pd


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Each box is represented as [xmin, ymin, xmax, ymax]
    """
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    # Calculate intersection area
    inter_width = max(0, x2_min - x1_max)
    inter_height = max(0, y2_min - y1_max)
    inter_area = inter_width * inter_height

    # Calculate areas of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area
    union_area = box1_area + box2_area - inter_area

    # Return IoU
    if union_area == 0:
        return 0
    return inter_area / union_area



def calculate_precision_recall(ground_truth_file, predictions_file, iou_threshold=0.5):
    """
    Calculate precision and recall given ground truth and predictions CSV files.
    """
    # Read both CSV files
    ground_truth_df = pd.read_csv(ground_truth_file)
    predictions_df = pd.read_csv(predictions_file)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Loop over ground truth data by image
    for image_path in ground_truth_df['image_path'].unique():
        # Get ground truth and predictions for the current image
        ground_truth_boxes = ground_truth_df[ground_truth_df['image_path'] == image_path]
        predicted_boxes = predictions_df[predictions_df['image_path'] == image_path]

        # Track matches to avoid duplicates
        matched_ground_truth = set()
        matched_predictions = set()

        # Compare each prediction to ground truth boxes
        for idx_pred, pred_row in predicted_boxes.iterrows():
            pred_box = [pred_row['xmin'], pred_row['ymin'], pred_row['xmax'], pred_row['ymax']]
            matched = False
            for idx_gt, gt_row in ground_truth_boxes.iterrows():
                if idx_gt in matched_ground_truth:
                    continue

                gt_box = [gt_row['xmin'], gt_row['ymin'], gt_row['xmax'], gt_row['ymax']]
                iou = calculate_iou(pred_box, gt_box)

                if iou >= iou_threshold:
                    true_positives += 1
                    matched_ground_truth.add(idx_gt)
                    matched_predictions.add(idx_pred)
                    matched = True
                    break

            if not matched:
                false_positives += 1

        # Any ground truth boxes not matched are false negatives
        false_negatives += len(ground_truth_boxes) - len(matched_ground_truth)

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall


def filter_ground_truth(df_gt: pd.DataFrame, image_list: list[str]):
    """

    """

def calculate_mae(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate the mean absolute error between the predicted and the ground truth
    :param predicted: the predicted counts
    :param ground_truth: the ground truth counts
    :return: the mean absolute error
    """
    return np.mean(np.abs(predicted - ground_truth))


if __name__ == '__main__':
    # Example usage
    ground_truth_file = '../../deepforest/data/example_gt.csv'  # Replace with your ground truth CSV file
    predictions_file = '../../deepforest/data/example_pred.csv'  # Replace with your predictions CSV file
    precision, recall = calculate_precision_recall(ground_truth_file, predictions_file)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

