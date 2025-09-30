"""
Metrics for robot trajectory prediction evaluation.
"""

import torch
import numpy as np


def robot_trajectory_accuracy(output, target, threshold=150.0):
    """
    Calculate accuracy metrics for robot trajectory prediction.

    Args:
        output (torch.Tensor): Predicted robot trajectories [batch_size, 12] (4 points x 3 coords)
        target (torch.Tensor): Ground truth robot trajectories [batch_size, 1, 4, 3]
        threshold (float): Distance threshold in mm for considering a prediction correct

    Returns:
        tuple: (pck, mean_ap) where pck is per-point accuracy and mean_ap is mean average precision
    """
    # Reshape output to match target format
    batch_size = output.shape[0]
    output = output.view(batch_size, 4, 3)  # [batch_size, 4, 3]
    target = target.squeeze(1)  # [batch_size, 4, 3]

    # Calculate Euclidean distances between predicted and target points
    distances = torch.sqrt(torch.sum((output - target) ** 2, dim=2))  # [batch_size, 4]

    # Calculate PCK (Percentage of Correct Keypoints) for each robot point
    correct_predictions = distances < threshold  # [batch_size, 4]
    pck = torch.mean(correct_predictions.float(), dim=0)  # [4] - accuracy per robot point

    # Calculate mean Average Precision (mAP)
    # For robot trajectory, we use the average accuracy across all points
    mean_ap = torch.mean(pck)

    return pck, mean_ap


def robot_trajectory_distance(output, target):
    """
    Calculate mean distance error for robot trajectory prediction.

    Args:
        output (torch.Tensor): Predicted robot trajectories [batch_size, 12] (4 points x 3 coords)
        target (torch.Tensor): Ground truth robot trajectories [batch_size, 1, 4, 3]

    Returns:
        torch.Tensor: Mean distance error per robot point [4]
    """
    # Reshape output to match target format
    batch_size = output.shape[0]
    output = output.view(batch_size, 4, 3)  # [batch_size, 4, 3]
    target = target.squeeze(1)  # [batch_size, 4, 3]

    # Calculate Euclidean distances
    distances = torch.sqrt(torch.sum((output - target) ** 2, dim=2))  # [batch_size, 4]

    # Return mean distance per robot point
    return torch.mean(distances, dim=0)  # [4]


def evaluate_robot_trajectory_batch(outputs, targets, threshold=150.0):
    """
    Evaluate a batch of robot trajectory predictions.

    Args:
        outputs (torch.Tensor): Predicted robot trajectories [batch_size, 12]
        targets (torch.Tensor): Ground truth robot trajectories [batch_size, 1, 4, 3]
        threshold (float): Distance threshold for PCK calculation

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    pck, mean_ap = robot_trajectory_accuracy(outputs, targets, threshold)
    distances = robot_trajectory_distance(outputs, targets)

    return {
        "pck": pck.detach().cpu().numpy(),
        "mean_ap": mean_ap.detach().cpu().item(),
        "mean_distances": distances.detach().cpu().numpy(),
        "threshold": threshold
    }