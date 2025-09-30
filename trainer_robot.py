"""
Module for training and evaluating the SPiKE robot trajectory model.
"""

from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from datasets.itop_robot import ITOPRobot
from utils import metrics_robot, scheduler


def train_one_epoch(
    model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, threshold
):
    """Train the robot trajectory model for one epoch."""

    model.train()
    header = f"Epoch: [{epoch}]"
    total_loss = 0.0
    total_pck = np.zeros(4)  # 4 robot trajectory points
    total_map = 0.0

    for clip, target, _ in tqdm(data_loader, desc=header):
        clip, target = clip.to(device), target.to(device)
        output = model(clip).reshape(target.shape[0], -1)  # [batch_size, 12]

        # Reshape target for loss calculation
        target_flat = target.view(target.shape[0], -1)  # [batch_size, 12]
        loss = criterion(output, target_flat)

        # Calculate accuracy metrics
        pck, mean_ap = metrics_robot.robot_trajectory_accuracy(output, target, threshold)
        total_pck += pck.detach().cpu().numpy()
        total_map += mean_ap.detach().cpu().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        lr_scheduler.step()

    total_loss /= len(data_loader)
    total_map /= len(data_loader)
    total_pck /= len(data_loader)

    return total_loss, total_pck, total_map


def evaluate(model, criterion, data_loader, device, threshold):
    """Evaluate the robot trajectory model."""

    model.eval()
    total_loss = 0.0
    total_pck = np.zeros(4)  # 4 robot trajectory points
    total_map = 0.0

    with torch.no_grad():
        for clip, target, _ in tqdm(data_loader, desc="Evaluating"):
            clip, target = clip.to(device), target.to(device)
            output = model(clip).reshape(target.shape[0], -1)  # [batch_size, 12]

            # Reshape target for loss calculation
            target_flat = target.view(target.shape[0], -1)  # [batch_size, 12]
            loss = criterion(output, target_flat)

            # Calculate accuracy metrics
            pck, mean_ap = metrics_robot.robot_trajectory_accuracy(output, target, threshold)
            total_pck += pck.detach().cpu().numpy()
            total_map += mean_ap.detach().cpu().item()
            total_loss += loss.item()

    total_loss /= len(data_loader)
    total_map /= len(data_loader)
    total_pck /= len(data_loader)

    return total_loss, total_pck, total_map


def load_data(config, mode="train"):
    """
    Load the ITOP robot trajectory dataset.

    Args:
        config (dict): The configuration dictionary.
        mode (str): The mode to load the data in ("train" or "test").

    Returns:
        tuple: A tuple containing the data loader(s) and the number of robot coordinates.
    """
    dataset_params = {
        "root": config["dataset_path"],  # dataset_path for labels location reference
        "data_output_path": config.get("data_output_path", config["dataset_path"]),  # data_output_path for point clouds
        "frames_per_clip": config["frames_per_clip"],
        "num_points": config["num_points"],
        "use_valid_only": config["use_valid_only"],
        "target_frame": config["target_frame"],
    }

    if mode == "train":
        dataset = ITOPRobot(
            train=True, aug_list=config["PREPROCESS_AUGMENT_TRAIN"], **dataset_params
        )
        dataset_test = ITOPRobot(
            train=False, aug_list=config["PREPROCESS_TEST"], **dataset_params
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["workers"],
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=config["batch_size"], num_workers=config["workers"]
        )
        return data_loader, data_loader_test, dataset.num_robot_coords

    elif mode == "test":
        dataset_test = ITOPRobot(
            train=False, aug_list=config["PREPROCESS_TEST"], **dataset_params
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=config["batch_size"], num_workers=config["workers"]
        )
        return data_loader_test, dataset_test.num_robot_coords


def create_criterion(config):
    """Create loss criterion for robot trajectory prediction."""
    return nn.MSELoss()


def create_optimizer_and_scheduler(config, model, data_loader):
    """Create optimizer and learning rate scheduler for robot trajectory training."""

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"]
    )

    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(data_loader) - 1)

    lr_scheduler = scheduler.WarmupMultiStepLR(
        optimizer,
        milestones=config["lr_milestones"],
        gamma=config["lr_gamma"],
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iters,
        warmup_method="linear",
    )

    return optimizer, lr_scheduler