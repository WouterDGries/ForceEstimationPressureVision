"""Per-frame dataset for fine-tuning PressureVisionNet on finger-on-forearm
presses.

Reads the sessions written by preprocessing/dataProcessor.py
(data/processed/session_*/frames.csv + frames/*.jpg; only sessions with a
PROCESSED_OK marker). Every usable frame (no force artifact, label below
max_force_n) is one sample: a fingertip-centred RGB crop, normalized exactly
like PressureVision does (recording.util.resnet_preprocessor, ImageNet
stats), plus its scalar force label in newtons and the physical area of one
pixel (m^2), which the model needs to integrate its pressure map into a force.

Splits are per session (see resolve_splits). Consumed by train.py and
evaluate.py via build_dataloader().
"""

import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
REPO_ROOT = os.path.dirname(os.path.abspath(CONFIG_PATH))
SPLIT_NAMES = ("train", "val", "test")
PROCESSED_MARKER = "PROCESSED_OK"


def load_config(path=CONFIG_PATH):
    """Input: path to config.yaml. Returns it parsed as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(path):
    """Returns path resolved against the repo root (absolute paths unchanged)."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def add_pressurevision_to_path(model_config):
    """Input: config['model_training']. Puts the cloned PressureVision repo
    on sys.path so its modules (prediction.*, recording.*) import as in that
    repo. Returns the repo's absolute path.
    """
    root = repo_path(model_config["model"]["pressurevision_root"])
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def discover_sessions(model_config):
    """Input: config['model_training']. Returns the sorted names of every
    session_* folder in data_root that dataProcessor.py finished.
    """
    data_root = repo_path(model_config["data_root"])
    return sorted(name for name in os.listdir(data_root)
                  if name.startswith("session_") and os.path.isfile(os.path.join(data_root, name, PROCESSED_MARKER)))


def resolve_splits(model_config):
    """Input: config['model_training']. Returns {"train": [...], "val":
    [...], "test": [...]} session names. Explicit <split>_sessions lists in
    the splits block are used as-is if any is present; otherwise all
    discovered sessions are shuffled with split_seed and divided by the
    fractions (val/test get at least one session each if their fraction is
    non-zero). Raises ValueError on bad fractions or too few sessions.
    """
    splits_config = model_config["splits"]
    if any(f"{split}_sessions" in splits_config for split in SPLIT_NAMES):
        return {split: list(splits_config.get(f"{split}_sessions") or []) for split in SPLIT_NAMES}

    fractions = {split: float(splits_config[f"{split}_fraction"]) for split in SPLIT_NAMES}
    if any(fraction < 0 for fraction in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must be >= 0 and sum to 1, got {fractions}")

    names = discover_sessions(model_config)
    rng = np.random.default_rng(splits_config.get("split_seed", 0))
    shuffled = [names[i] for i in rng.permutation(len(names))]

    def count(split):
        return 0 if fractions[split] == 0 else max(1, int(round(fractions[split] * len(shuffled))))

    num_val, num_test = count("val"), count("test")
    num_train = len(shuffled) - num_val - num_test
    if num_train < 1:
        raise ValueError(f"only {len(shuffled)} processed session(s) in {model_config['data_root']} - "
                         f"not enough for a train/val/test split with fractions {fractions}")
    return {
        "train": sorted(shuffled[:num_train]),
        "val": sorted(shuffled[num_train:num_train + num_val]),
        "test": sorted(shuffled[num_train + num_val:]),
    }


def load_session_frames(session_dir, frames_config, stride):
    """Input: processed session directory, config['model_training']
    ['frames'], frame stride. Returns frames.csv restricted to usable frames
    (no artifact, force_n <= max_force_n), every stride-th one, with an
    absolute image path column added.
    """
    df = pd.read_csv(os.path.join(session_dir, "frames.csv"))
    usable = (df["artifact"] == 0) & (df["force_n"] <= frames_config["max_force_n"])
    df = df[usable].iloc[::stride].reset_index(drop=True)
    df["image_path"] = [os.path.join(session_dir, image) for image in df["image"]]
    df["session"] = os.path.basename(session_dir)
    return df


class ForceFrameDataset(Dataset):
    """torch Dataset over single fingertip crops. `split` is "train", "val"
    or "test"; only "train" is augmented. `stride` overrides the configured
    frame stride (e.g. 1 for a dense evaluation).
    """

    def __init__(self, model_config, split, sessions=None, stride=None, augment=None):
        add_pressurevision_to_path(model_config)
        from recording.util import resnet_preprocessor   # PressureVision's own input normalization
        self.preprocessing_fn = resnet_preprocessor

        self.split = split
        self.augment = (split == "train") if augment is None else augment
        self.augmentation_config = model_config["augmentation"]
        frames_config = model_config["frames"]
        if stride is None:
            stride = frames_config["train_stride"] if split == "train" else frames_config["eval_stride"]

        session_names = resolve_splits(model_config)[split] if sessions is None else sessions
        data_root = repo_path(model_config["data_root"])
        print(f"[dataset] Loading {split} split: {len(session_names)} session(s), stride {stride}")
        tables = []
        for name in session_names:
            df = load_session_frames(os.path.join(data_root, name), frames_config, stride)
            print(f"  [dataset] {name}: {len(df)} frames, "
                  f"{(df['force_n'] > 0.5).mean() * 100:.0f}% in contact, max {df['force_n'].max():.1f} N")
            tables.append(df)
        self.frames = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        print(f"  [dataset] {split}: {len(self.frames)} frames total")

    def __len__(self):
        return len(self.frames)

    def labels_n(self):
        """Returns all force labels (N) of this dataset as a numpy array."""
        return self.frames["force_n"].to_numpy(dtype=np.float32)

    def _augment(self, image):
        """Input: (H, W, 3) float32 RGB in [0, 1]. Applies one random flip /
        rotation / scale / translation (one warpAffine), a brightness/contrast
        change, occasional blur and pixel noise. Returns (image, scale): scale > 1 means
        the content was enlarged, i.e. each pixel now covers 1/scale^2 of the
        original physical area.
        """
        aug = self.augmentation_config
        height, width = image.shape[:2]
        if np.random.rand() < aug["horizontal_flip_prob"]:
            image = image[:, ::-1]
        angle = np.random.uniform(-aug["rotation_max_deg"], aug["rotation_max_deg"])
        scale = np.random.uniform(*aug["scale_range"])
        max_shift = aug["translate_max_frac"] * width
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        matrix[:, 2] += np.random.uniform(-max_shift, max_shift, size=2)
        image = cv2.warpAffine(np.ascontiguousarray(image), matrix, (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        brightness = 1.0 + np.random.uniform(-aug["brightness_jitter"], aug["brightness_jitter"])
        contrast = 1.0 + np.random.uniform(-aug["contrast_jitter"], aug["contrast_jitter"])
        mean = image.mean()
        image = (image - mean) * contrast + mean * brightness
        if np.random.rand() < aug["blur_prob"]:
            sigma = np.random.uniform(0.0, aug["blur_max_sigma_px"])
            if sigma > 0.1:
                image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
        if aug["noise_std"] > 0:
            image = image + np.random.normal(0.0, aug["noise_std"], size=image.shape).astype(np.float32)
        return np.clip(image, 0.0, 1.0).astype(np.float32), scale

    def __getitem__(self, index):
        row = self.frames.iloc[index]
        image_bgr = cv2.imread(row["image_path"])
        if image_bgr is None:
            raise FileNotFoundError(row["image_path"])
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        scale = 1.0
        if self.augment:
            image, scale = self._augment(image)
        pixel_size_m = float(row["pixel_size_mm"]) / 1000.0 / scale

        network_image = self.preprocessing_fn(image).transpose(2, 0, 1).astype(np.float32)
        return {
            "img": torch.from_numpy(network_image),
            "img_original": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "force_n": torch.tensor(float(row["force_n"]), dtype=torch.float32),
            "pixel_area_m2": torch.tensor(pixel_size_m * pixel_size_m, dtype=torch.float32),
            "session": row["session"],
            "raw_frame_index": int(row["raw_frame_index"]),
            "host_ts_ns": int(row["host_ts_ns"]),
        }


def _seed_worker(worker_id):
    """Gives every DataLoader worker its own numpy seed (augmentation),
    derived from torch's per-worker seed.
    """
    np.random.seed(torch.initial_seed() % 2 ** 32)


def build_dataloader(split, config=None, shuffle=None, stride=None, sessions=None, augment=None):
    """Input: split name, optional config dict, optional shuffle / stride /
    session list / augment overrides. Returns (dataset, DataLoader) built
    from config['model_training']['loader'].
    """
    config = config or load_config()
    model_config = config["model_training"]
    dataset = ForceFrameDataset(model_config, split, sessions=sessions, stride=stride, augment=augment)
    loader_config = model_config["loader"]
    shuffle = (split == "train") if shuffle is None else shuffle
    seed = model_config["training"].get("seed")
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    loader = DataLoader(dataset, batch_size=loader_config["batch_size"], shuffle=shuffle,
                        num_workers=loader_config["num_workers"], pin_memory=True,
                        worker_init_fn=_seed_worker, generator=generator,
                        persistent_workers=loader_config["num_workers"] > 0)
    return dataset, loader


if __name__ == "__main__":
    cfg = load_config()
    splits = resolve_splits(cfg["model_training"])
    for split_name, names in splits.items():
        print(f"[dataset] {split_name}: {names}")
    train_dataset, train_loader = build_dataloader("train", cfg)
    batch = next(iter(train_loader))
    print(f"[dataset] batch img={tuple(batch['img'].shape)} force_n={batch['force_n'].tolist()}")
    print(f"  pixel_area_m2 range {batch['pixel_area_m2'].min():.3e}..{batch['pixel_area_m2'].max():.3e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    count = min(8, batch["img"].shape[0])
    cols = 4
    fig, axes = plt.subplots(math.ceil(count / cols), cols, figsize=(3 * cols, 3 * math.ceil(count / cols)))
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if k < count:
            ax.imshow(batch["img_original"][k].permute(1, 2, 0).numpy())
            ax.set_title(f"{batch['force_n'][k]:.1f} N")
    preview_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DatasetCheck", "batch_preview.png")
    os.makedirs(os.path.dirname(preview_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(preview_path, dpi=100)
    print(f"[dataset] augmented batch preview saved to {preview_path}")
