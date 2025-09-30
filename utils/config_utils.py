"""
Module for configuration utilities.
"""

import sys
import os
import yaml
import numpy as np
import torch


def load_yaml_file(file_path):
    """Load a YAML file and return its content."""
    try:
        with open(file_path, "r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit(1)


def convert_to_float(config, keys):
    """Convert specific values in the config to floats."""
    for key in keys:
        config[key] = float(config[key])


def set_dataset_paths(config):
    """Set the dataset paths in the config based on the dataset name."""
    # Use paths from config instead of const/path.py
    if config["dataset"] == "CUSTOM" or config["dataset"]=="ROBOT":
        if "dataset_path" not in config:
            raise ValueError("dataset_path not specified in config for CUSTOM dataset")
        config["dataset_root"] = config["dataset_path"]
    else:
        raise ValueError(f"Dataset {config['dataset']} not found.")


def set_output_dir(config, config_file):
    """Set the output directory in the config and create it if it doesn't exist."""
    if config.get("log_dir"):
        # Use experiments_path from config instead of const/path.py
        if "experiments_path" not in config:
            raise ValueError("experiments_path not specified in config")
        config["output_dir"] = os.path.join(
            config["experiments_path"], config_file, config["log_dir"]
        )
        print(f"Log dir: {config['output_dir']}")
        os.makedirs(config["output_dir"], exist_ok=True)
        if os.listdir(config["output_dir"]) and config.get("mode_train"):
            raise FileExistsError(
                f"Directory {config['output_dir']} already exists and is not empty"
            )


def load_config(config_file):
    """Load a config file and return its content."""
    # First load config to get experiments_path
    if os.path.isabs(config_file):
        # Absolute path provided
        path_config_file = os.path.join(config_file, "config.yaml")
    else:
        # Relative path - need to determine base path
        # Check if config_file contains experiments_path or use default
        temp_config_path = os.path.join(config_file, "config.yaml")
        if os.path.exists(temp_config_path):
            path_config_file = temp_config_path
        else:
            # Fall back to using current directory structure
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path_config_file = os.path.join(current_dir, config_file, "config.yaml")

    config = load_yaml_file(path_config_file)
    convert_to_float(config, ["momentum", "weight_decay"])
    set_dataset_paths(config)
    set_output_dir(config, config_file)
    return config


def set_random_seed(seed):
    """Set random seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
