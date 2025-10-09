"""
Module for training the SPiKE model on the ITOP dataset.
"""

import os
import sys
import argparse
import torch
import torch.utils.data
from torch import nn
import wandb
from model import model_builder
from trainer_itop import (
    train_one_epoch,
    evaluate,
    load_data,
    create_criterion,
    create_optimizer_and_scheduler,
)
from utils.config_utils import load_config, set_random_seed
from utils.distrib_utils import is_main_process

def main(arguments):
    config = load_config(arguments.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_args"])
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    device = torch.device(0)

    set_random_seed(config["seed"])

    print(f"Loading data from {config['dataset_path']}")
    data_loader, data_loader_test, num_coord_joints = load_data(config)

    model = model_builder.create_model(config, num_coord_joints)
    model_without_ddp = model

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    criterion = create_criterion(config)
    optimizer, lr_scheduler = create_optimizer_and_scheduler(config, model, data_loader)

    if config["resume"]:
        print(f"Loading model from {config['resume']}")
        checkpoint = torch.load(config["resume"], map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=True)

        # Check if finetuning mode
        if config.get("finetune", False):
            print("Finetuning mode: only loading model weights, resetting optimizer and scheduler")
        else:
            # Resume training: load optimizer and scheduler states
            print(f"Resume mode: loading all training states from checkpoint epoch {checkpoint['epoch']}")
            config["start_epoch"] = checkpoint["epoch"] + 1
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            optimizer.load_state_dict(checkpoint["optimizer"])

    # Initialize wandb with offline mode as fallback
    try:
        wandb.init(
            project=config["wandb_project"],
            name=arguments.config,
            settings=wandb.Settings(start_method="thread")
        )
        wandb.config.update(config)
        wandb.watch_called = False
    except Exception as e:
        print(f"Warning: wandb initialization failed: {e}")
        print("Continuing training without wandb logging...")
        wandb.init(mode="disabled")

    # Initialize GradScaler for automatic mixed precision
    from torch.amp import GradScaler
    scaler = GradScaler('cuda')
    print("Using automatic mixed precision (AMP) for training")

    print("Start training")
    min_loss = sys.maxsize
    eval_thresh = config["threshold"]

    for epoch in range(config["start_epoch"], config["epochs"]):
        train_clip_loss, train_pck, train_map = train_one_epoch(
            model,
            criterion,
            optimizer,
            lr_scheduler,
            data_loader,
            device,
            epoch,
            eval_thresh,
            scaler,
        )
        val_clip_loss, val_pck, val_map = evaluate(
            model, criterion, data_loader_test, device=device, threshold=eval_thresh
        )

        # Log to wandb with error handling
        try:
            data1 = [(idx, train_pck[idx]) for idx in range(len(train_pck))]
            data2 = [(idx, val_pck[idx]) for idx in range(len(val_pck))]
            table1 = wandb.Table(data=data1, columns=["joint", "pck"])
            table2 = wandb.Table(data=data2, columns=["joint", "pck"])

            wandb.log(
                {
                    "Train loss": train_clip_loss,
                    "Train mAP": train_map,
                    "Train PCK": wandb.plot.bar(table1, "joint", "pck", title="Train PCK"),
                    "Val loss": val_clip_loss,
                    "Val mAP": val_map,
                    "Val PCK": wandb.plot.bar(table2, "joint", "pck", title="Val PCK"),
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
        except Exception as e:
            print(f"Warning: wandb logging failed at epoch {epoch}: {e}")
            print("Continuing training without logging this epoch...")
            # Try logging basic metrics without plots as fallback
            try:
                wandb.log(
                    {
                        "Train loss": train_clip_loss,
                        "Train mAP": train_map,
                        "Val loss": val_clip_loss,
                        "Val mAP": val_map,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
            except:
                pass  # Continue training even if basic logging fails

        print(f"Epoch {epoch} - Train Loss: {train_clip_loss:.4f}")
        print(f"Epoch {epoch} - Train mAP: {train_map:.4f}")
        print(f"Epoch {epoch} - Train PCK: {train_pck}")
        print(f"Epoch {epoch} - Validation Loss: {val_clip_loss:.4f}")
        print(f"Epoch {epoch} - Validation mAP: {val_map:.4f}")
        print(f"Epoch {epoch} - Validation PCK: {val_pck}")

        if config["output_dir"] and is_main_process():
            model_to_save = (
                model_without_ddp.module
                if isinstance(model_without_ddp, nn.DataParallel)
                else model_without_ddp
            )

            checkpoint = {
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": config,
            }
            torch.save(checkpoint, os.path.join(config["output_dir"], "checkpoint.pth"))

            if val_clip_loss < min_loss:
                min_loss = val_clip_loss
                torch.save(
                    checkpoint, os.path.join(config["output_dir"], "best_model.pth")
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPiKE Training on custom dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/Custom/1",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()
    main(args)
