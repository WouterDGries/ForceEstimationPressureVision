"""Evaluates a fine-tuned checkpoint (or the zero-shot PressureVisionNet) on
a split - by default the held-out test sessions, densely (every usable
frame).

Writes to <checkpoint_dir>/eval_<split>/ (or zero_shot_<split>/):
    metrics.json            overall + per-session metrics, trivial baseline
    predictions.csv         one row per frame: session, frame, time, true / predicted force
    timeline_<session>.png  measured vs predicted force over time
    pressure_examples.png   crops with the predicted pressure map overlaid, across the force range

Usage (trajPipeline env, from the repo root):
    python Model/evaluate.py                                  # best_model.pt on the test split
    python Model/evaluate.py --split val --checkpoint Model/checkpoints/seed_7/best_model.pt
    python Model/evaluate.py --zero_shot                      # official weights, no fine-tuning
"""

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch

import dataset
import model as model_module
import train


def _timeline_plots(predictions, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for session, df in predictions.groupby("session"):
        t = (df["host_ts_ns"] - df["host_ts_ns"].min()) / 1e9
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, df["true_n"], ".", ms=1.5, label="measured")
        ax.plot(t, df["pred_n"], ".", ms=1.5, label="predicted")
        mae = np.mean(np.abs(df["pred_n"] - df["true_n"]))
        ax.set_title(f"{session}  MAE {mae:.2f} N")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("force (N)")
        ax.legend(markerscale=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"timeline_{session}.png"), dpi=90)
        plt.close(fig)


def _pressure_examples(pressure_maps, predictions, out_path):
    """Grid of crops with their predicted pressure map overlaid (inferno,
    0..max over the shown examples), labelled measured / predicted force.
    """
    if not pressure_maps:
        return
    max_kpa = max(float(p.max()) for _, p in pressure_maps.values()) or 1.0
    tiles = []
    for index in sorted(pressure_maps, key=lambda k: predictions["true_n"].iloc[k]):
        img, pressure = pressure_maps[index]
        rgb = (np.clip(img.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        heat = cv2.applyColorMap((np.clip(pressure / max_kpa, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        overlay = cv2.addWeighted(bgr, 0.55, heat, 0.45, 0)
        text = f"GT {predictions['true_n'].iloc[index]:.1f}N  pred {predictions['pred_n'].iloc[index]:.1f}N"
        for image in (bgr, overlay):
            cv2.putText(image, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.hstack([bgr, overlay]))
    cols = 2
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(out_path, grid)


def evaluate(checkpoint_path=None, split="test", zero_shot=False, stride=1, num_examples=12):
    """Input: checkpoint path (ignored with zero_shot), split name, zero_shot
    flag (official PressureVisionNet weights, gain 1, no fine-tuning), frame
    stride, number of pressure-map examples. Runs the evaluation and writes
    the outputs listed in the module docstring. Returns the metrics dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_config = dataset.load_config()
    if zero_shot:
        config = base_config
        net = model_module.build_model(config["model_training"], device, init_weights="pressurevision")
        net.eval()
        splits = dataset.resolve_splits(config["model_training"])
        out_dir = os.path.join(dataset.repo_path(config["model_training"]["training"]["checkpoint_dir"]),
                               f"zero_shot_{split}")
        label = "zero-shot PressureVisionNet"
    else:
        checkpoint_path = checkpoint_path or os.path.join(
            dataset.repo_path(base_config["model_training"]["training"]["checkpoint_dir"]), "best_model.pt")
        net, checkpoint = model_module.load_checkpoint(checkpoint_path, device)
        config = checkpoint["config"]
        splits = checkpoint["splits"]
        out_dir = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), f"eval_{split}")
        label = f"{checkpoint_path} (epoch {checkpoint['epoch']})"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[evaluate] {label} on {split}: {splits[split]}")

    data, loader = dataset.build_dataloader(split, config, shuffle=False, stride=stride,
                                            sessions=splits[split], augment=False)
    example_indices = set(np.argsort(data.labels_n())[np.linspace(0, len(data) - 1, num_examples).astype(int)].tolist())
    autocast_dtype, _ = train.select_precision(device)
    result = train.predict(net, loader, device, autocast_dtype, keep_pressure_for=example_indices)

    contact_threshold_n = config["model_training"]["training"]["contact_threshold_n"]
    predictions = pd.DataFrame({key: result[key] for key in
                                ("session", "raw_frame_index", "host_ts_ns", "true_n", "pred_n", "hard_pred_n")})
    predictions.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    train_labels = pd.concat([dataset.load_session_frames(
        os.path.join(dataset.repo_path(config["model_training"]["data_root"]), name),
        config["model_training"]["frames"], 1)["force_n"] for name in splits["train"]])
    mean_train_n = float(train_labels.mean())
    metrics = {
        "model": label,
        "split": split,
        "gain": float(net.gain().item()),
        "overall": train.regression_metrics(result["pred_n"], result["true_n"], contact_threshold_n),
        "overall_hard_argmax": train.regression_metrics(result["hard_pred_n"], result["true_n"], contact_threshold_n),
        "trivial_baseline_mean_train_force": train.regression_metrics(
            np.full(len(result["true_n"]), mean_train_n), result["true_n"], contact_threshold_n),
        "per_session": {
            session: train.regression_metrics(df["pred_n"].to_numpy(), df["true_n"].to_numpy(), contact_threshold_n)
            for session, df in predictions.groupby("session")
        },
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _timeline_plots(predictions, out_dir)
    _pressure_examples(result["pressure_maps"], predictions, os.path.join(out_dir, "pressure_examples.png"))

    overall, trivial = metrics["overall"], metrics["trivial_baseline_mean_train_force"]
    print(f"[evaluate] {split}: MAE {overall['mae_n']:.3f} N  RMSE {overall['rmse_n']:.3f} N  R2 {overall['r2']:.3f}  "
          f"contact-MAE {overall['mae_contact_n']:.3f} N  contact-acc {overall['contact_acc']:.3f}  "
          f"(trivial baseline MAE {trivial['mae_n']:.3f} N, {overall['num_frames']} frames)")
    for session, session_metrics in metrics["per_session"].items():
        print(f"  [evaluate] {session}: MAE {session_metrics['mae_n']:.3f} N  R2 {session_metrics['r2']:.3f}")
    print(f"[evaluate] Outputs written to {out_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, help="default: <checkpoint_dir>/best_model.pt")
    parser.add_argument("--split", default="test", choices=dataset.SPLIT_NAMES)
    parser.add_argument("--zero_shot", action="store_true", help="evaluate the official weights without fine-tuning")
    parser.add_argument("--stride", type=int, default=1, help="evaluate every n-th usable frame")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.split, args.zero_shot, args.stride)


if __name__ == "__main__":
    main()
