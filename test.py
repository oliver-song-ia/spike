"""
Evaluate SPiKE on ITOP with subset sampling across multiple experiments.

- For each x in {1,2,3,4}, we load:
    config: experiments/ITOP-SIDE/x
    model : experiments/ITOP-SIDE/x/log/best_model.pth

- Randomly sample up to K=100 clips from the test split and evaluate ONLY on these.
- Print metrics in the SAME format as the official eval script:
    Validation Loss: ...
    Validation mAP: ...
    Validation PCK: [...]
- Also print a compact summary table at the end across all 4 experiments.
"""

from __future__ import print_function
import os
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

from model import model_builder
from trainer_itop import load_data, create_criterion
from utils.config_utils import load_config, set_random_seed
from utils.metrics import joint_accuracy


def evaluate_on_indices(model, criterion, dataset, indices, device, threshold, show_progress=False):
    """
    Evaluate ONLY on the given dataset indices (list[int]) using the same protocol
    as the official evaluation script (per-sample loss, PCK, mAP; then average).
    """
    model.eval()
    total_loss = 0.0
    total_map = 0.0
    total_pck = None
    clip_losses = []

    iterator = indices
    if show_progress:
        iterator = tqdm(indices, desc="Subset Eval")

    with torch.no_grad():
        for i in iterator:
            clip, target, video_id = dataset[i]

            clip_t = clip.unsqueeze(0).to(device, non_blocking=True)
            target_t = target.unsqueeze(0).to(device, non_blocking=True)

            output_t = model(clip_t)
            # Align shape to target (same as official script)
            output_t = output_t.reshape(target_t.shape)

            loss = criterion(output_t, target_t)
            pck_t, map_t = joint_accuracy(output_t, target_t, threshold)

            loss_item = float(loss.item())
            map_item = float(map_t.detach().cpu().item())
            pck_np = pck_t.detach().cpu().numpy()

            total_loss += loss_item
            total_map += map_item
            if total_pck is None:
                total_pck = np.zeros_like(pck_np, dtype=np.float64)
            total_pck += pck_np

            # Keep clip_losses format aligned with official script
            clip_losses.append(
                (
                    np.array(video_id),            # video_id as numpy
                    loss_item,
                    clip.cpu().numpy(),
                    target.cpu().numpy(),
                    output_t.cpu().numpy(),
                )
            )

    n = max(1, len(indices))
    val_loss = total_loss / n
    val_map = total_map / n
    val_pck = (total_pck / n) if total_pck is not None else np.array([])

    # Sort like the official script
    clip_losses.sort(key=lambda x: x[1], reverse=True)

    return clip_losses, val_loss, val_map, val_pck


def pick_indices(dataset_len, k, rng):
    """Pick up to k unique indices from [0, dataset_len)."""
    k_eff = min(k, dataset_len)
    return rng.sample(range(dataset_len), k_eff)


def run_one_experiment(exp_root, exp_id, sample_k, seed):
    """
    exp_root: e.g., 'experiments/ITOP-SIDE'
    exp_id  : one of {1,2,3,4}
    """
    cfg_dir = os.path.join(exp_root, str(exp_id))
    model_path = os.path.join(exp_root, str(exp_id), "log", "best_model.pth")

    # Load config
    config = load_config(cfg_dir)
    # Respect device selection in config (same style as original)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("device_args", 0))
    print(f"\n=== Experiment {exp_id} ===")
    print(f"Config: {cfg_dir}")
    print(f"Model : {model_path}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    device = torch.device(0)
    set_random_seed(config["seed"])

    # Data
    print(f"Loading data from {config['dataset_path']}")
    data_loader_test, num_coord_joints = load_data(config, mode="test")
    dataset = data_loader_test.dataset
    num_samples = len(dataset)
    if num_samples == 0:
        raise RuntimeError(f"Empty test set for experiment {exp_id}")

    # Indices to evaluate
    rng = random.Random(config["seed"])
    indices = pick_indices(num_samples, sample_k, rng)
    print(f"[Info] Evaluating on subset of size {len(indices)} (randomly sampled)")

    # Model + criterion
    model = model_builder.create_model(config, num_coord_joints).to(device)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    criterion = create_criterion(config)

    threshold = config.get("threshold", 0.1)

    # Evaluate on subset
    losses, val_loss, val_map, val_pck = evaluate_on_indices(
        model, criterion, dataset, indices, device, threshold, show_progress=False
    )

    # Print EXACTLY like the official eval script
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation mAP: {val_map:.4f}")
    print(f"Validation PCK: {val_pck}")

    # Return for summary
    return {"exp": exp_id, "loss": val_loss, "mAP": val_map, "PCK": val_pck, "n": len(indices)}


def main():
    parser = argparse.ArgumentParser(description="SPiKE Subset Evaluation over ITOP-SIDE/1~4")
    parser.add_argument("--experiments_root", type=str, default="experiments/ITOP-SIDE",
                        help="Root folder that contains subfolders 1,2,3,4")
    parser.add_argument("--sample_k", type=int, default=100,
                        help="Number of random samples to evaluate per experiment (cap at dataset size)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global seed to override config seed for sampling only (optional)")
    args = parser.parse_args()

    exp_root = args.experiments_root
    sample_k = args.sample_k

    results = []
    for x in [1, 2, 3, 4]:
        try:
            # We keep per-experiment configs' seed for model/data reproducibility.
            # The subset sampling uses that same seed; if a global seed is provided, use it instead.
            res = run_one_experiment(exp_root, x, sample_k, seed=args.seed)
            results.append(res)
        except Exception as e:
            print(f"[Error] Experiment {x} failed: {e}")

    # Compact summary
    if results:
        print("\n=== Summary (subset evaluation) ===")
        # Align columns
        for r in results:
            exp = r["exp"]
            n = r["n"]
            loss = r["loss"]
            mAP = r["mAP"]
            # Print PCK mean for quick glance; full vector may be long
            pck_mean = float(np.mean(r["PCK"])) if isinstance(r["PCK"], np.ndarray) and r["PCK"].size else float("nan")
            print(f"Exp {exp}: n={n:3d}  Loss={loss:.4f}  mAP={mAP:.4f}  PCK(mean)={pck_mean:.2f}")
    else:
        print("[Summary] No successful experiment runs.")


if __name__ == "__main__":
    main()
