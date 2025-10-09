"""
Module for training and evaluating the SPiKE model on the ITOP dataset.
"""

from tqdm import tqdm
import torch
from torch import nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from datasets.itop import ITOP
from utils import metrics, scheduler


def train_one_epoch(
    model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, threshold, scaler=None
):

    model.train()
    header = f"Epoch: [{epoch}]"
    total_loss = 0.0
    total_pck = np.zeros(15)
    total_map = 0.0
    use_amp = scaler is not None

    for clip, target, _ in tqdm(data_loader, desc=header):
        clip, target = clip.to(device), target.to(device)

        optimizer.zero_grad()

        # Use automatic mixed precision if scaler is provided
        if use_amp:
            with autocast(device_type='cuda'):
                output = model(clip).reshape(target.shape)
                loss = criterion(output, target)

            # Scale loss and backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(clip).reshape(target.shape)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Update learning rate scheduler after optimizer step
        lr_scheduler.step()

        pck, mean_ap = metrics.joint_accuracy(output, target, threshold)
        total_pck += pck.detach().cpu().numpy()
        total_map += mean_ap.detach().cpu().item()
        total_loss += loss.item()

    total_loss /= len(data_loader)
    total_map /= len(data_loader)
    total_pck /= len(data_loader)

    return total_loss, total_pck, total_map


def evaluate(model, criterion, data_loader, device, threshold):
    model.eval()
    total_loss = 0.0
    total_pck = np.zeros(15)
    total_map = 0.0

    with torch.no_grad():
        for clip, target, _ in tqdm(
            data_loader, desc="Validation" if data_loader.dataset.train else "Test"
        ):
            clip, target = clip.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )
            output = model(clip).reshape(target.shape)
            loss = criterion(output, target)

            pck, mean_ap = metrics.joint_accuracy(output, target, threshold)
            total_pck += pck.detach().cpu().numpy()
            total_map += mean_ap.detach().cpu().item()
            total_loss += loss.item()

    total_loss /= len(data_loader)
    total_map /= len(data_loader)
    total_pck /= len(data_loader)

    return total_loss, total_pck, total_map


def load_data(config, mode="train"):
    """
    Load the ITOP dataset.

    Args:
        config (dict): The configuration dictionary.
        mode (str): The mode to load the data in ("train" or "test").

    Returns:
        tuple: A tuple containing the data loader(s) and the number of coordinate joints.
    """
    dataset_params = {
        "root": config.get("data_output_path", config["dataset_path"]),
        "frames_per_clip": config["frames_per_clip"],
        "num_points": config["num_points"],
        "use_valid_only": config["use_valid_only"],
        "target_frame": config["target_frame"],
    }

    if mode == "train":
        dataset = ITOP(
            train=True, aug_list=config["PREPROCESS_AUGMENT_TRAIN"], **dataset_params
        )
        dataset_test = ITOP(
            train=False, aug_list=config["PREPROCESS_TEST"], **dataset_params
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["workers"],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,  # Increased from 2 to 4 for better data pipeline
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=config["batch_size"], num_workers=config["workers"]
        )
        return data_loader, data_loader_test, dataset.num_coord_joints

    dataset_test = ITOP(
        train=False, aug_list=config["PREPROCESS_TEST"], **dataset_params
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=config["batch_size"], num_workers=config["workers"]
    )
    return data_loader_test, dataset_test.num_coord_joints


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss that applies different weights to different joints.
    Supports both group-based weights (shoulder, elbow, hand) and individual joint weights.
    """
    def __init__(self, joint_weights_config=None):
        super().__init__()
        from const import skeleton_joints

        # Joint indices from skeleton_joints.py:
        # 0: Head, 1: Neck, 2: R Shoulder, 3: L Shoulder, 4: R Elbow, 5: L Elbow,
        # 6: R Hand, 7: L Hand, 8: Torso, 9: R Hip, 10: L Hip, 11: R Knee,
        # 12: L Knee, 13: R Foot, 14: L Foot

        # Default weights - all joints equal weight
        if joint_weights_config is None:
            joint_weights_config = {
                'shoulder': 1.0,
                'elbow': 1.0,
                'hand': 1.0,
                'default': 1.0
            }

        # Create weight tensor for 15 joints
        weights = torch.ones(15)

        # Set default weight for all joints first
        default_weight = joint_weights_config.get('default', 1.0)
        weights.fill_(default_weight)

        # Apply group-based weights
        shoulder_weight = joint_weights_config.get('shoulder', default_weight)
        elbow_weight = joint_weights_config.get('elbow', default_weight)
        hand_weight = joint_weights_config.get('hand', default_weight)
        hip_weight = joint_weights_config.get('hip', default_weight)
        knee_weight = joint_weights_config.get('knee', default_weight)
        foot_weight = joint_weights_config.get('foot', default_weight)

        # Apply group weights
        weights[2] = shoulder_weight  # R Shoulder
        weights[3] = shoulder_weight  # L Shoulder
        weights[4] = elbow_weight     # R Elbow
        weights[5] = elbow_weight     # L Elbow
        weights[6] = hand_weight      # R Hand
        weights[7] = hand_weight      # L Hand
        weights[9] = hip_weight       # R Hip
        weights[10] = hip_weight      # L Hip
        weights[11] = knee_weight     # R Knee
        weights[12] = knee_weight     # L Knee
        weights[13] = foot_weight     # R Foot
        weights[14] = foot_weight     # L Foot

        # Apply individual joint weights (overrides group weights if specified)
        individual_weights = joint_weights_config.get('individual', {})
        for joint_name, weight in individual_weights.items():
            if joint_name in skeleton_joints.joint_indices.values():
                # Find joint index by name
                joint_idx = None
                for idx, name in skeleton_joints.joint_indices.items():
                    if name == joint_name:
                        joint_idx = idx
                        break
                if joint_idx is not None:
                    weights[joint_idx] = weight
                    print(f"  Individual weight: {joint_name} (index {joint_idx}) = {weight}")

        print(f"WeightedL1Loss initialized with:")
        print(f"  Group weights:")
        print(f"    Shoulder: {shoulder_weight}")
        print(f"    Elbow: {elbow_weight}")
        print(f"    Hand: {hand_weight}")
        print(f"    Hip: {hip_weight}")
        print(f"    Knee: {knee_weight}")
        print(f"    Foot: {foot_weight}")
        print(f"    Default: {default_weight}")

        # Print final weight distribution
        print(f"  Final joint weights:")
        for i, weight in enumerate(weights):
            joint_name = skeleton_joints.joint_indices.get(i, f"Unknown_{i}")
            print(f"    {i:2d}: {joint_name:12s} = {weight:.3f}")

        # Expand to match coordinate dimensions (15 joints * 3 coordinates = 45)
        self.register_buffer('joint_weights', weights.repeat_interleave(3))
        
    def forward(self, output, target):
        """
        Compute weighted L1 loss.
        
        Args:
            output (torch.Tensor): Model predictions
            target (torch.Tensor): Ground truth
            
        Returns:
            torch.Tensor: Weighted L1 loss
        """
        # Flatten to ensure consistent shape handling
        output_flat = output.view(-1, 45)  # [batch_size, 45]
        target_flat = target.view(-1, 45)  # [batch_size, 45]
        
        # Compute absolute differences
        abs_diff = torch.abs(output_flat - target_flat)
        
        # Ensure weights are on the same device as input tensors
        weights = self.joint_weights.to(abs_diff.device)
        
        # Apply weights - broadcast weights to match batch dimensions  
        weighted_diff = abs_diff * weights.view(1, -1)
        
        # Return mean of weighted differences
        return torch.mean(weighted_diff)


def create_criterion(config):
    """
    Create the loss criterion based on the configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        torch.nn.Module: The loss function.
    """
    loss_type = config.get("loss_type", "std_cross_entropy")

    if loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "weighted_l1":
        joint_weights_config = config.get("joint_weights", None)
        return WeightedL1Loss(joint_weights_config)
    elif loss_type == "mse":
        return nn.MSELoss()
    raise ValueError("Invalid loss type. Supported types: 'l1', 'weighted_l1', 'mse'.")


def create_optimizer_and_scheduler(config, model, data_loader):
    lr = config["lr"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    warmup_iters = config["lr_warmup_epochs"] * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in config["lr_milestones"]]
    lr_scheduler = scheduler.WarmupMultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=config["lr_gamma"],
        warmup_iters=warmup_iters,
        warmup_factor=1e-5,
    )
    return optimizer, lr_scheduler
