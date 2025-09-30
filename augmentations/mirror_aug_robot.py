"""
Mirror augmentation for robot trajectory data.
"""

import torch
from augmentations.augmentation import Augmentation
from const.robot_trajectory import flip_robot_sides


class MirrorAugRobot(Augmentation):
    """Class to represent a mirror augmentation for robot trajectory data."""

    def __init__(self, p_prob=1.0, p_axes=None, **kwargs):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_axes (list of bool): Axes to apply the augmentation.
        """
        if p_axes is None:
            p_axes = [True, False, False]

        super().__init__(p_prob)
        self.p_axes_ = p_axes
        self.probs_ = []

    def set_probs(self, probs):
        """Set the probabilities for mirroring along each axis."""
        self.probs_ = probs

    def __compute_augmentation__(self, p_pts, p_gt_tensor, apply_on_gt):
        """
        Apply mirror augmentation to robot trajectory data.

        Args:
            p_pts (torch.Tensor): Point cloud tensor [N, 3].
            p_gt_tensor (torch.Tensor): Robot trajectory ground truth tensor [4, 3].
            apply_on_gt (bool): Whether to apply augmentation on ground truth.

        Returns:
            tuple: (augmented_points, augmentation_params, augmented_gt)
        """
        mirror_vec = torch.ones(3)

        # Apply mirroring based on probabilities and enabled axes
        for i in range(3):
            if self.p_axes_[i] and self.probs_[i] <= self.prob_:
                mirror_vec[i] = -1

        # Apply mirroring to point cloud
        aug_pts = p_pts * mirror_vec

        # Apply mirroring to robot trajectory ground truth if requested
        if apply_on_gt:
            augmented_gt_tensor = p_gt_tensor * mirror_vec.view(1, -1)

            # If flipping in X or Z dimension, exchange positions of left and right robot arms
            # Flipping in both X and Z dim cancels out
            flip = torch.logical_xor(mirror_vec[0] == -1, mirror_vec[2] == -1).item()
            if flip:
                augmented_gt_tensor = flip_robot_sides(augmented_gt_tensor)
        else:
            augmented_gt_tensor = p_gt_tensor

        return aug_pts, (mirror_vec,), augmented_gt_tensor