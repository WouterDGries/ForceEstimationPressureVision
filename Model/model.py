"""PressureVisionNet with a force-integral output, for training on scalar
force labels.

The network itself is PressureVision's, built by the cloned repo's own
prediction.model_builder.build_model: an FPN with an SE-ResNeXt50 encoder
whose 9-channel output is a per-pixel pressure-class map (classes =
config/paper.yml FORCE_THRESHOLDS, in kPa). Its weights come from either the
official PressureVisionNet checkpoint (paper_59.pt) or ImageNet (encoder
only), see config.yaml's model_training.model block.

We have no pressure map, only the total force, so the network is supervised
through the physics that connect the two:

    pressure_kpa[x, y] = gain * sum_k softmax(logits)[k, x, y] * class_centre_kpa[k]
    force_n            = sum_xy pressure_kpa[x, y] * 1000 * pixel_area_m2

The class centres are the ones PressureVision itself uses to turn classes
back into pressure (recording.util.classes_to_scalar), so with gain = 1 the
pretrained network gives a meaningful zero-shot force. `gain` is a single
learnable scalar for the systematic difference between pressing a rigid
sensor and soft skin. The force is a differentiable function of every
pixel, so a force loss trains the whole FPN - and the pressure map stays an
output (unsupervised spatially: only its integral is ever compared to data).
"""

import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml

import dataset

# Values from PressureVision/config/paper.yml - the official checkpoint's output layout.
PAPER_NUM_FORCE_CLASSES = 9
PAPER_FORCE_THRESHOLDS_KPA = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
IMAGENET_INIT_BACKGROUND_BIAS = 5.0   # imagenet init: class 0 ("no pressure") starts dominant everywhere


def _paper_config(pressurevision_root):
    """Reads PressureVision's config/paper.yml, so the model built here is
    the one the official checkpoint was trained as. Returns a SimpleNamespace
    with the fields build_model() reads.
    """
    with open(os.path.join(pressurevision_root, "config", "paper.yml")) as f:
        paper = yaml.safe_load(f)
    if paper["NUM_FORCE_CLASSES"] != PAPER_NUM_FORCE_CLASSES or paper["FORCE_THRESHOLDS"] != PAPER_FORCE_THRESHOLDS_KPA:
        raise ValueError("PressureVision/config/paper.yml no longer matches the official checkpoint layout")
    return SimpleNamespace(FORCE_CLASSIFICATION=True, NUM_FORCE_CLASSES=paper["NUM_FORCE_CLASSES"],
                           FORCE_THRESHOLDS=paper["FORCE_THRESHOLDS"], NETWORK_TYPE=paper["NETWORK_TYPE"],
                           NETWORK_INPUT_CHANNELS=paper["NETWORK_INPUT_CHANNELS"])


def _ensure_pressurevision_checkpoint(pressurevision_root, checkpoint_path):
    """Downloads the official paper_59.pt with PressureVision's own
    downloader if it is not on disk yet. Returns nothing.
    """
    if os.path.isfile(checkpoint_path):
        return
    from recording.downloader import download_model_checkpoint
    print(f"[model] {checkpoint_path} not found - downloading the official PressureVisionNet checkpoint")
    previous_dir = os.getcwd()
    os.chdir(pressurevision_root)   # the downloader saves to data/model/ relative to the working directory
    try:
        download_model_checkpoint()
    finally:
        os.chdir(previous_dir)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"download did not produce {checkpoint_path} - set model.pressurevision_checkpoint")


class PressureVisionForceNet(nn.Module):
    """PressureVision FPN + the force integral described in the module
    docstring. forward(img, pixel_area_m2) returns (force_n (B,),
    pressure_kpa (B, H, W), logits (B, 9, H, W)).
    """

    def __init__(self, fpn, force_thresholds_kpa, initial_gain, frozen_encoder_layers=()):
        super().__init__()
        from recording.util import classes_to_scalar
        self.fpn = fpn
        centres = classes_to_scalar(torch.arange(len(force_thresholds_kpa)), force_thresholds_kpa)
        self.register_buffer("class_centres_kpa", centres.float())
        self.force_thresholds_kpa = list(force_thresholds_kpa)
        self.log_gain = nn.Parameter(torch.tensor(math.log(initial_gain), dtype=torch.float32))
        self.frozen_modules = [getattr(fpn.encoder, name) for name in frozen_encoder_layers]
        for module in self.frozen_modules:
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """As nn.Module.train, but frozen encoder stages always stay in eval
        mode: requires_grad=False alone does not stop BatchNorm from updating
        its running statistics in training mode.
        """
        super().train(mode)
        for module in self.frozen_modules:
            module.eval()
        return self

    def gain(self):
        return self.log_gain.exp()

    def forward(self, img, pixel_area_m2):
        logits = self.fpn(img).float()   # softmax/integral in float32 even under autocast
        probabilities = torch.softmax(logits, dim=1)
        pressure_kpa = (probabilities * self.class_centres_kpa[None, :, None, None]).sum(dim=1) * self.gain()
        force_n = (pressure_kpa * 1000.0).sum(dim=(1, 2)) * pixel_area_m2
        return force_n, pressure_kpa, logits

    def hard_force(self, logits, pixel_area_m2):
        """PressureVision's own (non-differentiable) reading of the output:
        argmax class per pixel -> class centre (kPa) -> integrated force, times
        the gain. Used for reporting only. Returns (force_n (B,), pressure_kpa).
        """
        from recording.util import classes_to_scalar
        pressure_kpa = classes_to_scalar(torch.argmax(logits, dim=1), self.force_thresholds_kpa) * self.gain()
        return (pressure_kpa * 1000.0).sum(dim=(1, 2)) * pixel_area_m2, pressure_kpa


def build_model(model_config, device=torch.device("cpu"), init_weights=None):
    """Input: config['model_training'], the device, optional override of
    model.init_weights ("pressurevision" / "imagenet" / "none" - "none" is
    for loading a fine-tuned checkpoint right after). Builds PressureVision's
    network via its own model_builder, loads the requested weights, wraps it
    in PressureVisionForceNet. Returns the model on device.
    """
    pressurevision_root = dataset.add_pressurevision_to_path(model_config)
    from prediction.model_builder import build_model as pressurevision_build_model

    net_config = model_config["model"]
    init_weights = init_weights or net_config["init_weights"]
    paper_config = _paper_config(pressurevision_root)
    fpn = pressurevision_build_model(paper_config, torch.device("cpu"), phases=[])["model"]   # ImageNet encoder

    if init_weights == "pressurevision":
        checkpoint_path = dataset.repo_path(net_config["pressurevision_checkpoint"])
        _ensure_pressurevision_checkpoint(pressurevision_root, checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = state.state_dict() if isinstance(state, nn.Module) else state
        fpn.load_state_dict(state, strict=True)
        print(f"[model] Loaded official PressureVisionNet weights from {checkpoint_path}")
    elif init_weights == "imagenet":
        with torch.no_grad():
            fpn.segmentation_head[0].bias[0] += IMAGENET_INIT_BACKGROUND_BIAS
        print("[model] ImageNet encoder, randomly initialized FPN decoder and head")
    elif init_weights != "none":
        raise ValueError(f"unknown model.init_weights '{init_weights}'")

    frozen = net_config.get("frozen_encoder_layers") or []
    model = PressureVisionForceNet(fpn, paper_config.FORCE_THRESHOLDS, net_config["initial_gain"], frozen)
    if frozen:
        print(f"[model] Frozen encoder stages: {frozen}")
    return model.to(device)


def parameter_groups(model):
    """Input: PressureVisionForceNet. Returns {"encoder": [...], "decoder":
    [...], "gain": [...]} lists of the trainable parameters (decoder
    includes the segmentation head; frozen encoder stages are left out) for
    per-group learning rates.
    """
    groups = {
        "encoder": list(model.fpn.encoder.parameters()),
        "decoder": list(model.fpn.decoder.parameters()) + list(model.fpn.segmentation_head.parameters()),
        "gain": [model.log_gain],
    }
    return {name: [p for p in params if p.requires_grad] for name, params in groups.items()}


def load_checkpoint(path, device=torch.device("cpu")):
    """Input: a checkpoint written by train.py. Rebuilds the model with the
    config stored in it and loads the fine-tuned weights. Returns (model in
    eval mode, checkpoint dict).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"]["model_training"], device, init_weights="none")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


if __name__ == "__main__":
    cfg = dataset.load_config()["model_training"]
    net = build_model(cfg)
    net.eval()
    groups = parameter_groups(net)
    print("[model] trainable parameters: " + ", ".join(f"{k}={sum(p.numel() for p in v):,}" for k, v in groups.items())
          + f", total={sum(p.numel() for p in net.parameters() if p.requires_grad):,} "
          f"of {sum(p.numel() for p in net.parameters()):,}")
    print(f"[model] class centres (kPa): {net.class_centres_kpa.tolist()}")
    size = dataset.load_config()["processing"]["crop"]["output_px"]
    pixel_area = torch.tensor([(dataset.load_config()["processing"]["crop"]["side_mm"] / size / 1000) ** 2])
    with torch.no_grad():
        force, pressure, logits = net(torch.zeros(1, 3, size, size), pixel_area)
    print(f"[model] blank (mean-colour) image: logits {tuple(logits.shape)}, pressure {tuple(pressure.shape)}, "
          f"soft force {force.item():.3f} N, hard force {net.hard_force(logits, pixel_area)[0].item():.3f} N")
