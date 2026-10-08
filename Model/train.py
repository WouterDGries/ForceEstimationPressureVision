"""Fine-tuning loop for PressureVisionNet with force-integral supervision.

Reads config.yaml's model_training block, builds the train/val
ForceFrameDataset + DataLoaders (dataset.py) and PressureVisionForceNet
(model.py), then trains with AdamW (separate learning rates for encoder,
decoder+head and the pressure gain), linear warmup + cosine decay, and a
Huber loss between the integrated pressure map and the measured force,
under bf16/fp16 autocast. Before the first epoch the untouched starting
model is validated too (epoch 0 = zero-shot PressureVisionNet when
init_weights is "pressurevision"). Checkpoints whenever validation MAE
improves; writes a metrics-history CSV. train() is called by mainModel.py;
running this file directly runs the overfit-one-batch sanity check instead.
"""

import copy
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset
import model as model_module

METRICS_FIELDS = ["epoch", "train_loss", "val_loss", "val_mae_n", "val_rmse_n", "val_r2", "val_mae_contact_n",
                  "val_contact_acc", "val_hard_mae_n", "gain", "lr_encoder", "epoch_time_s"]


def set_seed(seed):
    """Seeds python, numpy and torch (CPU + CUDA). Returns nothing."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_precision(device):
    """Input: torch.device. Returns (autocast_dtype, use_grad_scaler):
    bf16 without GradScaler on Ampere or newer, fp16 with GradScaler on
    older GPUs, (None, False) = plain float32 on CPU.
    """
    if device.type != "cuda":
        return None, False
    major, _ = torch.cuda.get_device_capability(device)
    return (torch.bfloat16, False) if major >= 8 else (torch.float16, True)


def build_optimizer(net, optimizer_config):
    """Input: PressureVisionForceNet and config['model_training']['training']
    ['optimizer']. AdamW with one (decay, no-decay) group pair per part:
    encoder, decoder (+ head), and the scalar gain (never decayed). Biases
    and normalization parameters get no weight decay. Returns the optimizer.
    """
    learning_rates = {"encoder": optimizer_config["encoder_lr"], "decoder": optimizer_config["decoder_lr"],
                      "gain": optimizer_config["gain_lr"]}
    param_groups = []
    for name, params in model_module.parameter_groups(net).items():
        decay = [p for p in params if p.ndim > 1]
        no_decay = [p for p in params if p.ndim <= 1]
        param_groups.append({"name": name, "params": decay, "lr": learning_rates[name],
                             "weight_decay": optimizer_config["weight_decay"]})
        param_groups.append({"name": name, "params": no_decay, "lr": learning_rates[name], "weight_decay": 0.0})
    return torch.optim.AdamW([group for group in param_groups if group["params"]])


def build_scheduler(optimizer, warmup_steps, total_steps):
    """Returns a LambdaLR stepped once per optimizer step: linear warmup,
    then cosine decay to 0.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def location_prior_loss(pressure_kpa, pixel_area_m2, true_n, radius_mm, contact_threshold_n):
    """Input: (B, H, W) pressure maps, (B,) pixel areas, (B,) labels, the
    prior radius and contact threshold. Returns the mean (over contact
    frames) of the pressure-weighted distance beyond radius_mm from the crop
    centre, in units of radius_mm - 0 when all pressure lies within the
    radius. Returns 0 if the batch has no contact frames.
    """
    contact = true_n > contact_threshold_n
    if not contact.any():
        return pressure_kpa.new_zeros(())
    pressure = pressure_kpa[contact]
    height, width = pressure.shape[-2:]
    ys = torch.arange(height, device=pressure.device, dtype=torch.float32) - (height - 1) / 2
    xs = torch.arange(width, device=pressure.device, dtype=torch.float32) - (width - 1) / 2
    distance_px = torch.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
    pixel_mm = pixel_area_m2[contact].sqrt() * 1000.0
    excess = torch.relu(distance_px[None] * pixel_mm[:, None, None] - radius_mm) / radius_mm
    share = (pressure * excess).sum(dim=(1, 2)) / (pressure.sum(dim=(1, 2)) + 1e-6)
    return share.mean()


def regression_metrics(pred_n, true_n, contact_threshold_n):
    """Input: 1D numpy arrays of predicted / true force (N) and the contact
    threshold. Returns a dict: mae, rmse, r2 over all frames, mae_contact
    over frames whose true force exceeds the threshold, and contact_acc
    (agreement of pred > threshold with true > threshold).
    """
    error = pred_n - true_n
    ss_tot = np.sum((true_n - true_n.mean()) ** 2)
    contact = true_n > contact_threshold_n
    return {
        "mae_n": float(np.mean(np.abs(error))),
        "rmse_n": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - np.sum(error ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "mae_contact_n": float(np.mean(np.abs(error[contact]))) if contact.any() else float("nan"),
        "contact_acc": float(np.mean((pred_n > contact_threshold_n) == contact)),
        "num_frames": int(len(true_n)),
    }


def predict(net, loader, device, autocast_dtype, loss_fn=None, keep_pressure_for=None):
    """Input: model, DataLoader, device, autocast dtype (or None), optional
    loss function, optional set of dataset indices whose pressure maps should
    be returned. Runs the loader under inference mode. Returns a dict with
    numpy arrays pred_n, hard_pred_n, true_n, session, raw_frame_index,
    host_ts_ns, the mean loss (if loss_fn), and pressure maps {index: (img, map)}.
    """
    net.eval()
    out = {"pred_n": [], "hard_pred_n": [], "true_n": [], "session": [], "raw_frame_index": [], "host_ts_ns": []}
    losses, pressure_maps, seen = [], {}, 0
    with torch.inference_mode():
        for batch in loader:
            img = batch["img"].to(device, non_blocking=True)
            area = batch["pixel_area_m2"].to(device, non_blocking=True)
            true = batch["force_n"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                force, pressure, logits = net(img, area)
            if loss_fn is not None:
                losses.append(loss_fn(force, true).item() * len(true))
            hard_force, _ = net.hard_force(logits, area)
            out["pred_n"].append(force.cpu().numpy())
            out["hard_pred_n"].append(hard_force.cpu().numpy())
            out["true_n"].append(true.cpu().numpy())
            out["session"].extend(batch["session"])
            out["raw_frame_index"].append(batch["raw_frame_index"].numpy())
            out["host_ts_ns"].append(batch["host_ts_ns"].numpy())
            if keep_pressure_for:
                for k in range(len(true)):
                    if seen + k in keep_pressure_for:
                        pressure_maps[seen + k] = (batch["img_original"][k].numpy(), pressure[k].float().cpu().numpy())
            seen += len(true)
    result = {key: (np.array(value) if key == "session" else np.concatenate(value)) for key, value in out.items()}
    result["loss"] = float(np.sum(losses) / max(seen, 1)) if loss_fn is not None else None
    result["pressure_maps"] = pressure_maps
    return result


def _append_metrics_row(path, row):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _save_checkpoint(path, net, config, epoch, val_metrics, splits):
    """Saves the weights plus everything needed to rebuild and use the
    model (full config, split lists, crop/pixel conventions). Returns nothing.
    """
    torch.save({
        "model_state_dict": net.state_dict(),
        "config": config,
        "epoch": epoch,
        "val_metrics": val_metrics,
        "splits": splits,
        "gain": float(net.gain().item()),
        "conventions": {
            "input": "RGB fingertip crop from preprocessing/dataProcessor.py, /255, PressureVision resnet_preprocessor",
            "crop_side_mm": config["processing"]["crop"]["side_mm"],
            "crop_output_px": config["processing"]["crop"]["output_px"],
            "force_label": "max(-Fz_filtered, 0) in N",
            "force_from_model": "sum(pressure_kpa) * 1000 * pixel_area_m2, pixel_area_m2 = (side_mm/output_px/1000)^2",
        },
    }, path)


def train(config=None):
    """Input: optionally a pre-loaded config dict. Runs the training loop
    described in the module docstring. Returns (best validation MAE in N,
    path of the best checkpoint).
    """
    config = copy.deepcopy(config or dataset.load_config())
    model_config = config["model_training"]
    training_config = model_config["training"]
    set_seed(training_config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    # Pin the split as explicit lists so datasets and checkpoint agree.
    splits = dataset.resolve_splits(model_config)
    model_config["splits"] = {f"{split}_sessions": names for split, names in splits.items()}
    for split, names in splits.items():
        print(f"[train] {split} sessions ({len(names)}): {names}")
    checkpoint_dir = dataset.repo_path(training_config["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    train_dataset, train_loader = dataset.build_dataloader("train", config)
    val_dataset, val_loader = dataset.build_dataloader("val", config)

    contact_threshold_n = training_config["contact_threshold_n"]
    mean_train_n = float(train_dataset.labels_n().mean())
    baseline = regression_metrics(np.full(len(val_dataset), mean_train_n), val_dataset.labels_n(), contact_threshold_n)
    print(f"[train] Trivial baseline (always predict the mean train force {mean_train_n:.2f} N): "
          f"val MAE = {baseline['mae_n']:.3f} N")

    net = model_module.build_model(model_config, device)
    optimizer = build_optimizer(net, training_config["optimizer"])
    batch_size = model_config["loader"]["batch_size"]
    samples_per_epoch = training_config["samples_per_epoch"] or len(train_dataset)
    steps_per_epoch = math.ceil(min(samples_per_epoch, len(train_dataset)) / batch_size)
    warmup_steps = int(training_config["schedule"]["warmup_epochs"] * steps_per_epoch)
    scheduler = build_scheduler(optimizer, warmup_steps, training_config["max_epochs"] * steps_per_epoch)

    autocast_dtype, use_grad_scaler = select_precision(device)
    scaler = torch.amp.GradScaler(device="cuda" if device.type == "cuda" else "cpu", enabled=use_grad_scaler)
    loss_fn = nn.HuberLoss(delta=training_config["loss"]["huber_delta_n"])
    prior_weight = training_config["loss"]["location_prior_weight"]
    prior_radius_mm = training_config["loss"]["location_prior_radius_mm"]
    metrics_path = os.path.join(checkpoint_dir, training_config["metrics_history_file"])
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    best_path = os.path.join(checkpoint_dir, "best_model.pt")

    def validate(epoch, train_loss, epoch_time_s):
        result = predict(net, val_loader, device, autocast_dtype, loss_fn)
        metrics = regression_metrics(result["pred_n"], result["true_n"], contact_threshold_n)
        hard_mae = float(np.mean(np.abs(result["hard_pred_n"] - result["true_n"])))
        print(f"[train] epoch {epoch}/{training_config['max_epochs']}  train_loss={train_loss:.4f}  "
              f"val_loss={result['loss']:.4f}  val_MAE={metrics['mae_n']:.3f}N  val_RMSE={metrics['rmse_n']:.3f}N  "
              f"val_R2={metrics['r2']:.3f}  contact-MAE={metrics['mae_contact_n']:.3f}N  "
              f"contact-acc={metrics['contact_acc']:.3f}  hard-MAE={hard_mae:.3f}N  gain={net.gain().item():.3f}  "
              f"({epoch_time_s:.0f}s)")
        _append_metrics_row(metrics_path, {
            "epoch": epoch, "train_loss": train_loss, "val_loss": result["loss"], "val_mae_n": metrics["mae_n"],
            "val_rmse_n": metrics["rmse_n"], "val_r2": metrics["r2"], "val_mae_contact_n": metrics["mae_contact_n"],
            "val_contact_acc": metrics["contact_acc"], "val_hard_mae_n": hard_mae, "gain": net.gain().item(),
            "lr_encoder": optimizer.param_groups[0]["lr"], "epoch_time_s": epoch_time_s,
        })
        return metrics

    print(f"[train] Epoch 0: validating the untrained starting model ({model_config['model']['init_weights']})")
    best_metrics = validate(0, float("nan"), 0.0)
    best_val_mae_n = best_metrics["mae_n"]
    _save_checkpoint(best_path, net, config, 0, best_metrics, splits)
    epochs_without_improvement = 0

    print(f"[train] Training: max_epochs={training_config['max_epochs']} batch_size={batch_size} "
          f"steps_per_epoch={steps_per_epoch} precision={autocast_dtype or torch.float32}")
    for epoch in range(1, training_config["max_epochs"] + 1):
        epoch_start = time.time()
        net.train()
        losses, samples = [], 0
        report_every = max(1, steps_per_epoch // 5)
        for step, batch in enumerate(train_loader):
            img = batch["img"].to(device, non_blocking=True)
            area = batch["pixel_area_m2"].to(device, non_blocking=True)
            true = batch["force_n"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                force, pressure, _ = net(img, area)
            loss = loss_fn(force, true)
            if prior_weight > 0:
                loss = loss + prior_weight * location_prior_loss(pressure, area, true, prior_radius_mm,
                                                                 contact_threshold_n)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if use_grad_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), training_config["grad_clip_max_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            losses.append(loss.item())
            samples += len(true)
            if (step + 1) % report_every == 0:
                print(f"  [train] epoch {epoch} step {step + 1}/{steps_per_epoch} "
                      f"loss={np.mean(losses[-report_every:]):.4f} lr_enc={optimizer.param_groups[0]['lr']:.2e} "
                      f"gain={net.gain().item():.3f}")
            if samples >= samples_per_epoch:
                break

        metrics = validate(epoch, float(np.mean(losses)), time.time() - epoch_start)
        if metrics["mae_n"] < best_val_mae_n:
            best_val_mae_n = metrics["mae_n"]
            epochs_without_improvement = 0
            _save_checkpoint(best_path, net, config, epoch, metrics, splits)
            print(f"  [train] validation MAE improved to {best_val_mae_n:.3f}N - saved {best_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config["early_stopping_patience"]:
                print(f"[train] Early stopping: no val MAE improvement in {epochs_without_improvement} epochs")
                break
    torch.save({"model_state_dict": net.state_dict(), "config": config, "epoch": epoch, "splits": splits},
               os.path.join(checkpoint_dir, "last_model.pt"))

    print(f"[train] Done. Best val MAE = {best_val_mae_n:.3f}N (trivial baseline {baseline['mae_n']:.3f}N). "
          f"Checkpoint: {best_path}  Metrics history: {metrics_path}")
    return best_val_mae_n, best_path


def check_overfit_one_batch(config, num_iterations=300, batch_size=8):
    """Sanity check: with augmentation off, a handful of frames must be fit
    almost perfectly within a few hundred iterations - otherwise something
    is wrong in the model, the force integral or the labels. Prints the loss
    trajectory. Returns the final loss.
    """
    model_config = config["model_training"]
    set_seed(model_config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = dataset.ForceFrameDataset(model_config, "train", augment=False)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(model_config["training"]["seed"]))
    batch = next(iter(loader))
    img, area, true = batch["img"].to(device), batch["pixel_area_m2"].to(device), batch["force_n"].to(device)

    net = model_module.build_model(model_config, device)
    net.train()
    optimizer = build_optimizer(net, model_config["training"]["optimizer"])
    loss_fn = nn.HuberLoss(delta=model_config["training"]["loss"]["huber_delta_n"])
    autocast_dtype, _ = select_precision(device)
    print(f"[train] Overfit-one-batch: labels {np.round(true.cpu().numpy(), 2).tolist()} N")
    for iteration in range(1, num_iterations + 1):
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            force, _, _ = net(img, area)
        loss = loss_fn(force, true)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % 25 == 0:
            print(f"  [train] iteration {iteration}: loss={loss.item():.4f}  "
                  f"pred={np.round(force.detach().cpu().numpy(), 2).tolist()}")
    verdict = "PASS" if loss.item() < 0.05 else "CHECK MANUALLY - did not fall near zero"
    print(f"[train] Overfit-one-batch final loss {loss.item():.4f} ({verdict})")
    return loss.item()


if __name__ == "__main__":
    check_overfit_one_batch(dataset.load_config())
