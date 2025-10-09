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

    def _get_valid_joints(self, use_valid_only, joints_dict, point_clouds_dict, traj_id_dict=None):
        """Create clips with padding for early frames in each trajectory."""

        valid_joints_dict = {}
        list_joints_items = list(joints_dict.items())

        # Check if using simple numeric IDs (e.g., '0', '1', '2') or ITOP format (e.g., '10300001')
        sample_id = list(joints_dict.keys())[0] if joints_dict else "0"
        is_simple_format = len(sample_id) <= 6  # Simple numeric IDs are typically shorter

        if self.target_frame != "last":
            raise ValueError(f"Unsupported target_frame mode: {self.target_frame}. Only 'last' is supported.")

        # Build trajectory start index mapping
        traj_start_indices = {}
        if traj_id_dict:
            current_traj = None
            for idx, identifier in enumerate(sorted(joints_dict.keys(), key=lambda x: int(x))):
                traj_id = traj_id_dict.get(identifier)
                if traj_id != current_traj:
                    current_traj = traj_id
                    traj_start_indices[traj_id] = idx

        for identifier, (joints, is_valid) in joints_dict.items():
            # Check validity if required
            if use_valid_only and not is_valid:
                continue

            # Parse person_id and global frame number
            if '_' in identifier:
                person_id, global_frame_str = identifier.split('_')
                global_frame_num = int(global_frame_str)
            else:
                person_id = ""
                global_frame_num = int(identifier)

            # Collect frames with padding for early frames
            frames = []
            for i in range(self.frames_per_clip):
                target_global_num = global_frame_num - self.frames_per_clip + 1 + i

                # Construct target identifier
                if person_id:
                    target_identifier = f"{person_id}_{target_global_num:05d}"
                else:
                    target_identifier = str(target_global_num)

                # Get frame path, with padding for negative or missing frames
                frame_path = point_clouds_dict.get(target_identifier, None)

                # If frame doesn't exist (before trajectory start), use first frame of current trajectory for padding
                if frame_path is None:
                    # Use current frame identifier as padding
                    frame_path = point_clouds_dict.get(identifier, None)

                frames.append(frame_path)

            # Only add if at least the target frame exists
            if all(frame is not None for frame in frames):
                valid_joints_dict[identifier] = (joints, frames)

        return valid_joints_dict

    def _load_data(self, use_valid_only):
        """Load the data from the dataset."""

        self.point_clouds_folder = os.path.join(self.root, "train" if self.train else "test")
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

        # Load traj_id if available for padding logic
        traj_ids = labels_file["traj_id"][:] if "traj_id" in labels_file else None
        labels_file.close()

        point_cloud_names = sorted(
            os.listdir(self.point_clouds_folder), key=lambda x: int(x.split(".")[0])
        )

        # Store file paths instead of loading all data into memory
        point_clouds_dict = {
            identifier.decode("utf-8"): os.path.join(self.point_clouds_folder, point_cloud_names[i])
            for i, identifier in enumerate(identifiers)
        }
        joints_dict = {
            identifier.decode("utf-8"): (joints[i], is_valid_flags[i])
            for i, identifier in enumerate(identifiers)
        }
        traj_id_dict = {
            identifier.decode("utf-8"): traj_ids[i].decode("utf-8") if isinstance(traj_ids[i], bytes) else traj_ids[i]
            for i, identifier in enumerate(identifiers)
        } if traj_ids is not None else None

        self.valid_joints_dict = self._get_valid_joints(
            use_valid_only, joints_dict, point_clouds_dict, traj_id_dict
        )
        self.valid_identifiers = list(self.valid_joints_dict.keys())

        if use_valid_only:
            print(
                f"Using only frames labeled as valid. From the total of {len(point_cloud_names)} "
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
        joints, clip_paths = self.valid_joints_dict.get(
            identifier, (None, [None] * self.frames_per_clip)
        )

        if joints is None or any(frame_path is None for frame_path in clip_paths):
            raise ValueError(f"Invalid joints or frames for identifier {identifier}")

        # Load point clouds on demand
        clip = [np.load(frame_path)["arr_0"] for frame_path in clip_paths]
        clip = [self._random_sample_pc(p) for p in clip]
        clip = torch.FloatTensor(np.array(clip))
        joints = torch.FloatTensor(joints).view(1, -1, 3)

        if self.aug_pipeline:
            clip, _, joints = self.aug_pipeline.augment(clip, joints)
        joints = joints.view(1, -1, 3)

        # For simple format, just return the frame index; for ITOP format, parse person_id and frame_id
        if "_" in identifier:
            return clip, joints, np.array([tuple(map(int, identifier.split("_")))])
        else:
            return clip, joints, np.array([[int(identifier)]])
