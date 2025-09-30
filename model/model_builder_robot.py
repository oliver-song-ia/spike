"""
Module for building the SPiKE robot trajectory model.
"""

import torch
from .spike_robot import SPiKERobot


def create_robot_model(config, num_robot_coords):
    """
    Create a SPiKE robot trajectory model based on the given configuration.

    Args:
        config (dict): The configuration dictionary.
        num_robot_coords (int): The number of robot coordinate outputs.

    Returns:
        SPiKERobot: The SPiKE robot trajectory model.
    """
    model_params = {
        "radius": config["radius"],
        "nsamples": config["nsamples"],
        "spatial_stride": config["spatial_stride"],
        "dim": config["dim"],
        "depth": config["depth"],
        "heads": config["heads"],
        "dim_head": config["dim_head"],
        "mlp_dim": config["mlp_dim"],
        "num_robot_coords": num_robot_coords,
        "dropout1": config.get("dropout1", 0.0),
        "dropout2": config.get("dropout2", 0.0),
    }

    model = SPiKERobot(**model_params)

    if config.get("compile", False):
        model = torch.compile(model)

    return model