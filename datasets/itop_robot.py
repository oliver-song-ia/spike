"""
Module for loading the ITOP dataset with robot trajectory labels.
"""

import os
import numpy as np
import h5py
import torch
import tqdm
from torch.utils.data import Dataset
from augmentations.aug_pipeline_robot import AugPipelineRobot
from const import robot_trajectory


class ITOPRobot(Dataset):
    """ITOP dataset class for robot trajectory prediction."""

    def __init__(
        self,
        root,
        data_output_path=None,
        frames_per_clip=16,
        num_points=2048,
        train=True,
        use_valid_only=False,
        aug_list=None,
        target_frame="middle",
    ):
        super().__init__()

        self.target_frame = target_frame
        self.frames_per_clip = frames_per_clip
        self.num_points = num_points
        self.train = train
        self.root = root  # dataset_path - contains arm_labels.h5
        self.data_output_path = data_output_path or root  # data_output_path - contains train/test folders
        self.num_robot_coords = robot_trajectory.NUM_ROBOT_COORDS

        self._load_data(use_valid_only)

        if aug_list is not None:
            self.aug_pipeline = AugPipelineRobot()
            self.aug_pipeline.create_pipeline(aug_list)
        else:
            self.aug_pipeline = None

    def _get_valid_robot_coords(self, use_valid_only, robot_coords_dict, point_clouds_dict):
        """Create clips of valid robot coordinates and their corresponding frames."""

        valid_robot_coords_dict = {}
        list_coords_items = list(robot_coords_dict.items())

        for identifier, robot_coords in list_coords_items:
            if not use_valid_only:
                # Use all frames when not filtering for valid only
                middle_frame_starting_index = int(identifier[3:]) - self.frames_per_clip // 2
                if middle_frame_starting_index >= 0:
                    frames = [
                        point_clouds_dict.get(
                            identifier[:3]
                            + str(middle_frame_starting_index + i).zfill(5),
                            None,
                        )
                        for i in range(self.frames_per_clip)
                    ]
                    if all(frame is not None for frame in frames):
                        valid_robot_coords_dict[identifier] = (robot_coords, frames)
            else:
                # Filter for valid robot coordinates only
                if robot_coords is not None and not np.any(np.isnan(robot_coords)):
                    middle_frame_starting_index = int(identifier[3:]) - self.frames_per_clip // 2
                    if middle_frame_starting_index >= 0:
                        frames = [
                            point_clouds_dict.get(
                                identifier[:3]
                                + str(middle_frame_starting_index + i).zfill(5),
                                None,
                            )
                            for i in range(self.frames_per_clip)
                        ]
                        if all(frame is not None for frame in frames):
                            valid_robot_coords_dict[identifier] = (robot_coords, frames)

        return valid_robot_coords_dict

    def _load_data(self, use_valid_only):
        """Load the data from the dataset."""

        # Point clouds are in data_output_path/train or data_output_path/test
        point_clouds_folder = os.path.join(self.data_output_path, "train" if self.train else "test")

        # Load robot trajectory labels from arm_labels files in data_output_path
        labels_file_name = "arm_labels.h5" if self.train else "arm_labels_test.h5"
        full_labels_path = os.path.join(self.data_output_path, labels_file_name)

        print(f"DEBUG: point_clouds_folder = {point_clouds_folder}")
        print(f"DEBUG: full_labels_path = {full_labels_path}")
        print(f"DEBUG: Labels file exists: {os.path.exists(full_labels_path)}")
        print(f"DEBUG: Point clouds folder exists: {os.path.exists(point_clouds_folder)}")

        labels_file = h5py.File(full_labels_path, "r")

        identifiers = labels_file["id"][:]

        # Load robot coordinates from the arm labels file
        # Actual structure: left_arm_coords (N, 2, 3), right_arm_coords (N, 2, 3)
        robot_coords_list = []
        left_coords = labels_file["left_arm_coords"][:]   # (N, 2, 3)
        right_coords = labels_file["right_arm_coords"][:]  # (N, 2, 3)

        for i, identifier in enumerate(identifiers):
            # Combine left and right arm coordinates into flat array
            left_L1 = left_coords[i, 0, :]    # [x, y, z]
            left_L2 = left_coords[i, 1, :]    # [x, y, z]
            right_R1 = right_coords[i, 0, :]  # [x, y, z]
            right_R2 = right_coords[i, 1, :]  # [x, y, z]

            # Concatenate to create 12-element array
            coords = np.concatenate([left_L1, left_L2, right_R1, right_R2])
            robot_coords_list.append(coords)

        labels_file.close()

        # Load point clouds
        point_cloud_names = sorted(
            os.listdir(point_clouds_folder), key=lambda x: int(x.split(".")[0])
        )
        point_clouds = []

        for pc_name in tqdm.tqdm(
            point_cloud_names,
            f"Loading {'train' if self.train else 'test'} point clouds",
        ):
            point_clouds.append(
                np.load(os.path.join(point_clouds_folder, pc_name))["arr_0"]
            )

        point_clouds_dict = {
            identifier.decode("utf-8"): point_clouds[i]
            for i, identifier in enumerate(identifiers)
        }
        robot_coords_dict = {
            identifier.decode("utf-8"): robot_coords_list[i]
            for i, identifier in enumerate(identifiers)
        }

        self.valid_robot_coords_dict = self._get_valid_robot_coords(
            use_valid_only, robot_coords_dict, point_clouds_dict
        )
        self.valid_identifiers = list(self.valid_robot_coords_dict.keys())

        if use_valid_only:
            print(
                f"Using only frames with valid robot coordinates. From the total of {len(point_clouds)} "
                f"{'train' if self.train else 'test'} frames using {len(self.valid_identifiers)} valid coordinates"
            )

    def _random_sample_pc(self, p):
        """Randomly sample points from the point cloud."""
        if p.shape[0] > self.num_points:
            r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
        elif p.shape[0] < self.num_points:
            repeat, residue = divmod(self.num_points, p.shape[0])
            r = np.concatenate(
                [np.arange(p.shape[0])] * repeat
                + [np.random.choice(p.shape[0], size=residue, replace=False)],
                axis=0,
            )
        else:
            return p
        return p[r, :]

    def __len__(self):
        return len(self.valid_identifiers)

    def __getitem__(self, idx):
        identifier = self.valid_identifiers[idx]
        robot_coords, clip = self.valid_robot_coords_dict.get(
            identifier, (None, [None] * self.frames_per_clip)
        )

        if robot_coords is None or any(frame is None for frame in clip):
            raise ValueError(f"Invalid robot coordinates or frames for identifier {identifier}")

        clip = [self._random_sample_pc(p) for p in clip]
        clip = torch.FloatTensor(clip)
        robot_coords = torch.FloatTensor(robot_coords).view(1, -1, 3)

        if self.aug_pipeline:
            clip, _, robot_coords = self.aug_pipeline.augment(clip, robot_coords)
        robot_coords = robot_coords.view(1, -1, 3)

        return clip, robot_coords, np.array([tuple(map(int, identifier.split("_")))])