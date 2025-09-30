"""
Module for loading the ITOP dataset.
"""

import os
import numpy as np
import h5py
import torch
import tqdm
from torch.utils.data import Dataset
from augmentations.aug_pipeline import AugPipeline
from const import skeleton_joints


class ITOP(Dataset):
    """ITOP dataset class."""

    def __init__(
        self,
        root,
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
        self.root = root
        self.num_coord_joints = skeleton_joints.NUM_COORD_JOINTS

        self._load_data(use_valid_only)

        if aug_list is not None:
            self.aug_pipeline = AugPipeline()
            self.aug_pipeline.create_pipeline(aug_list)
        else:
            self.aug_pipeline = None

    def _get_valid_joints(self, use_valid_only, joints_dict, point_clouds_dict):
        """Cumbersome but necessary logic to create clips of only valid joints and their corresponding frames."""

        valid_joints_dict = {}
        list_joints_items = list(joints_dict.items())

        # Check if using simple numeric IDs (e.g., '0', '1', '2') or ITOP format (e.g., '10300001')
        sample_id = list(joints_dict.keys())[0] if joints_dict else "0"
        is_simple_format = len(sample_id) <= 6  # Simple numeric IDs are typically shorter

        # Process joints based on if we are inputting only past timestamps (target_frame == 'last')
        # or past and future timestamps (target_frame == 'middle')
        if self.target_frame == "last":
            for identifier, (joints, is_valid) in joints_dict.items():
                frame_idx = int(identifier)

                # Check if we have enough previous frames
                if frame_idx < self.frames_per_clip - 1:
                    continue

                # Check validity if required
                if use_valid_only and not is_valid:
                    continue

                # For simple format, collect consecutive frames
                if is_simple_format:
                    frames = [
                        point_clouds_dict.get(str(frame_idx - self.frames_per_clip + 1 + i), None)
                        for i in range(self.frames_per_clip)
                    ]
                else:
                    # Original ITOP format logic
                    frames = [
                        point_clouds_dict.get(
                            identifier[:3]
                            + str(
                                int(identifier[-5:]) - self.frames_per_clip + 1 + i
                            ).zfill(5),
                            None,
                        )
                        for i in range(self.frames_per_clip)
                    ]

                # Only add if all frames exist
                if all(frame is not None for frame in frames):
                    valid_joints_dict[identifier] = (joints, frames)
        # If we are considering past and future frames, we need to ensure that we have enough frames before and after,
        # and that they belong to the same person (see ITOP naming convention for more details)
        elif self.target_frame == "middle":
            for i, (identifier, (joints, is_valid)) in enumerate(list_joints_items):
                frame_idx = int(identifier)
                half_clip = self.frames_per_clip // 2

                # Check if we have enough frames before and after
                if frame_idx < half_clip or frame_idx + half_clip >= len(list_joints_items):
                    continue

                # Check validity if required
                if use_valid_only and not is_valid:
                    continue

                # For simple format, collect centered frames
                if is_simple_format:
                    middle_frame_starting_index = frame_idx - half_clip
                    frames = [
                        point_clouds_dict.get(str(middle_frame_starting_index + j), None)
                        for j in range(self.frames_per_clip)
                    ]
                else:
                    # Original ITOP format logic
                    next_half_frames_per_clip_id_person, _ = (
                        list_joints_items[i + half_clip]
                        if i + half_clip < len(list_joints_items)
                        else (None, None)
                    )
                    if next_half_frames_per_clip_id_person is None:
                        continue

                    # Check that frames belong to same person
                    if int(identifier[:2]) != int(next_half_frames_per_clip_id_person[:2]):
                        continue

                    middle_frame_starting_index = int(identifier[-5:]) - half_clip
                    frames = [
                        point_clouds_dict.get(
                            identifier[:3] + str(middle_frame_starting_index + j).zfill(5),
                            None,
                        )
                        for j in range(self.frames_per_clip)
                    ]

                # Only add if all frames exist
                if all(frame is not None for frame in frames):
                    valid_joints_dict[identifier] = (joints, frames)
        return valid_joints_dict

    def _load_data(self, use_valid_only):
        """Load the data from the dataset."""

        point_clouds_folder = os.path.join(self.root, "train" if self.train else "test")
        labels_file = h5py.File(
            os.path.join(
                self.root, "train_labels.h5" if self.train else "test_labels.h5"
            ),
            "r",
        )
        identifiers = labels_file["id"][:]
        joints = labels_file["real_world_coordinates"][:]
        # Check if is_valid field exists, otherwise assume all are valid
        if "is_valid" in labels_file:
            is_valid_flags = labels_file["is_valid"][:]
        else:
            is_valid_flags = np.ones(len(identifiers), dtype=bool)
            print(f"Warning: 'is_valid' field not found in labels file. Assuming all frames are valid.")
        labels_file.close()

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
        joints_dict = {
            identifier.decode("utf-8"): (joints[i], is_valid_flags[i])
            for i, identifier in enumerate(identifiers)
        }

        self.valid_joints_dict = self._get_valid_joints(
            use_valid_only, joints_dict, point_clouds_dict
        )
        self.valid_identifiers = list(self.valid_joints_dict.keys())

        if use_valid_only:
            print(
                f"Using only frames labeled as valid. From the total of {len(point_clouds)} "
                f"{'train' if self.train else 'test'} frames using {len(self.valid_identifiers)} valid joints"
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
        joints, clip = self.valid_joints_dict.get(
            identifier, (None, [None] * self.frames_per_clip)
        )

        if joints is None or any(frame is None for frame in clip):
            raise ValueError(f"Invalid joints or frames for identifier {identifier}")

        clip = [self._random_sample_pc(p) for p in clip]
        clip = torch.FloatTensor(clip)
        joints = torch.FloatTensor(joints).view(1, -1, 3)

        if self.aug_pipeline:
            clip, _, joints = self.aug_pipeline.augment(clip, joints)
        joints = joints.view(1, -1, 3)

        # For simple format, just return the frame index; for ITOP format, parse person_id and frame_id
        if "_" in identifier:
            return clip, joints, np.array([tuple(map(int, identifier.split("_")))])
        else:
            return clip, joints, np.array([[int(identifier)]])
