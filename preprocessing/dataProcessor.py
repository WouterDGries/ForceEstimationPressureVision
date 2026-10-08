"""Turns raw recordings into the per-frame dataset Model/dataset.py trains on.

Reads data/raw/session_<ts>/ (color.mkv, depth.mkv, frames_raw.csv,
force_raw.csv, meta_raw.json - as written by the acquisition pipeline) and
writes data/processed/session_<ts>/:

    frames/<raw_frame_index>.jpg   fingertip-centred RGB crop, output_px x output_px
    frames.csv                     one row per saved crop: timestamps, force label, crop geometry, flags
    meta.json                      settings used, crop scale, camera calibration
    qc/                            qc.json, force_plot.png, crop_strip.png, full_frame_check.jpg
    PROCESSED_OK                   written last; Model/dataset.py only reads sessions that have it

Per session:
  1. Force: zero-phase Butterworth low-pass of Fz, resampled onto every frame
     timestamp. Label force_n = max(-Fz, 0) (pressing reads negative). Frames
     where Fz goes strongly positive (forearm lifted off the plate, e.g. at
     the end of every session) are flagged as artifacts.
  2. Pass 1 (tracking): MediaPipe HandLandmarker in VIDEO mode on every full
     frame; the pressing hand's fingertip/DIP are smoothed zero-phase
     (Savitzky-Golay) and short detection gaps interpolated. Every n-th
     frame the fingertip depth is read from the (aligned) depth stream.
  3. Crop scale: one square crop side per session, side_mm at the session's
     median fingertip depth (contact frames preferred), so every crop covers
     the same physical area and pixel_size_mm is known for the force integral.
  4. Pass 2 (cropping): anti-aliased, sub-pixel crop around the smoothed
     fingertip, saved as JPEG.

Usage (trajPipeline env, from the repo root):
    python preprocessing/dataProcessor.py data/raw/session_x [...]
    python preprocessing/dataProcessor.py --all               # every raw session not processed yet
    python preprocessing/dataProcessor.py --all --force       # re-process everything (e.g. new crop settings)
    python preprocessing/dataProcessor.py --all --workers 4   # sessions in parallel
"""

import argparse
import glob
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, savgol_filter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthAlign  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")
PROCESSED_MARKER = "PROCESSED_OK"


def load_config(path=CONFIG_PATH):
    """Input: path to config.yaml. Returns it parsed as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(path):
    """Returns path resolved against the repo root (absolute paths unchanged)."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def read_video_frames(path, width, height, pix_fmt, channels, dtype):
    """Input: video path, frame size, ffmpeg output pixel format ("rgb24" or
    "gray16le"), channels per pixel and numpy dtype. Yields decoded frames
    one at a time, exactly as stored (no frame-rate conversion).
    """
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
               "-f", "rawvideo", "-pix_fmt", pix_fmt, "-fps_mode", "passthrough", "-"]
    shape = (height, width, channels) if channels > 1 else (height, width)
    frame_bytes = width * height * channels * np.dtype(dtype).itemsize
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            buffer = process.stdout.read(frame_bytes)
            if len(buffer) < frame_bytes:
                break
            yield np.frombuffer(buffer, dtype=dtype).reshape(shape)
    finally:
        process.stdout.close()
        process.kill()
        process.wait()


def _run_slices(mask):
    """Returns a list of (start, end_exclusive) for the True runs in mask."""
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[0::2], changes[1::2]))


def fill_and_smooth(values, tracking_config):
    """Input: (N, k) float array with NaN rows where there is no detection,
    config['processing']['hand_tracking']. Linearly interpolates gaps of at
    most max_gap_interp_frames with detections on both sides, then applies a
    zero-phase Savitzky-Golay filter within every continuous run. Returns
    (smoothed (N, k) with NaN rows left where untracked, interpolated (N,) bool).
    """
    values = values.astype(np.float64).copy()
    present = ~np.isnan(values).any(axis=1)
    interpolated = np.zeros(len(values), dtype=bool)

    for start, end in _run_slices(~present):
        if start == 0 or end == len(values) or end - start > tracking_config["max_gap_interp_frames"]:
            continue
        weights = (np.arange(start, end) - (start - 1)) / (end - (start - 1))
        values[start:end] = values[start - 1] + weights[:, None] * (values[end] - values[start - 1])
        interpolated[start:end] = True

    tracked = ~np.isnan(values).any(axis=1)
    window = tracking_config["smoothing_window_frames"]
    polyorder = tracking_config["smoothing_polyorder"]
    for start, end in _run_slices(tracked):
        run_window = min(window, (end - start) if (end - start) % 2 == 1 else (end - start) - 1)
        if run_window > polyorder:
            values[start:end] = savgol_filter(values[start:end], run_window, polyorder, axis=0)
    return values, interpolated


def frame_force_labels(force_raw, frame_ts_ns, force_config):
    """Input: force_raw.csv as a DataFrame, (N,) frame host timestamps,
    config['processing']['force']. Zero-phase low-pass filters Fz at its
    own sample rate and resamples it onto the frame timestamps (shifted by
    frame_time_offset_ms). Returns (fz_filtered_n, force_n, artifact), all
    (N,): filtered Fz in newtons, the label max(-Fz, 0), and a bool flag for
    frames within artifact_margin_s of Fz > artifact_positive_n.
    """
    force_ts_ns = force_raw["host_ts_ns"].to_numpy().astype(np.int64)
    sample_rate_hz = 1e9 / np.median(np.diff(force_ts_ns.astype(np.float64)))
    sos = butter(force_config["order"], force_config["cutoff_hz"], btype="low", fs=sample_rate_hz, output="sos")
    fz_filtered = sosfiltfilt(sos, force_raw["fz_N"].to_numpy())

    lookup_ts = frame_ts_ns.astype(np.float64) + force_config["frame_time_offset_ms"] * 1e6
    fz_frames = np.interp(lookup_ts, force_ts_ns.astype(np.float64), fz_filtered)

    # Artifact mask on the force timeline, dilated by the margin, then sampled per frame.
    margin_ns = force_config["artifact_margin_s"] * 1e9
    artifact_samples = fz_filtered > force_config["artifact_positive_n"]
    artifact = np.zeros(len(frame_ts_ns), dtype=bool)
    for start, end in _run_slices(artifact_samples):
        t0 = force_ts_ns[start] - margin_ns
        t1 = force_ts_ns[end - 1] + margin_ns
        artifact |= (lookup_ts >= t0) & (lookup_ts <= t1)
    # Frames outside the force recording cannot be labelled.
    artifact |= (lookup_ts < force_ts_ns[0]) | (lookup_ts > force_ts_ns[-1])

    force_n = np.clip(-fz_frames, 0.0, None)
    return fz_frames.astype(np.float32), force_n.astype(np.float32), artifact


def build_hand_landmarker(model_path):
    """Input: path to hand_landmarker.task. Returns a MediaPipe
    HandLandmarker in VIDEO mode (synchronous, one result per frame).
    """
    import mediapipe as mp
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def select_tracked_hand(result, hand_label):
    """Input: a HandLandmarkerResult and the handedness label to follow.
    Returns that hand's 21 normalized landmarks, or None.
    """
    if not result or not result.hand_landmarks or not result.handedness:
        return None
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        if handedness[0].category_name == hand_label:
            return hand_landmarks
    return None


def _center_depth_m(aligner, depth_raw, point, half=4):
    """Median valid aligned depth (m) in a (2*half+1)^2 box around point, or NaN."""
    cx, cy = int(round(point[0])), int(round(point[1]))
    roi = aligner.align_roi(depth_raw, (cx - half, cy - half, cx + half + 1, cy + half + 1))
    valid = roi[roi > 0]
    return float(np.median(valid)) * aligner.depth_scale_m if valid.size else np.nan


def crop_frame(color_rgb, center, side_px, output_px):
    """Input: full RGB frame, float crop center (x, y), square side length
    in source pixels, output size. Returns (crop (output_px, output_px, 3)
    uint8, outside_fraction): one sub-pixel warpAffine, anti-aliased before
    downsampling, border replicated; outside_fraction is the share of the
    crop that lies outside the image.
    """
    scale = side_px / output_px
    half = side_px / 2
    x0, y0 = int(math.floor(center[0] - half)) - 3, int(math.floor(center[1] - half)) - 3
    x1, y1 = int(math.ceil(center[0] + half)) + 4, int(math.ceil(center[1] + half)) + 4
    height, width = color_rgb.shape[:2]

    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if sx1 <= sx0 or sy1 <= sy0:
        return np.zeros((output_px, output_px, 3), dtype=np.uint8), 1.0
    roi = cv2.copyMakeBorder(color_rgb[sy0:sy1, sx0:sx1], sy0 - y0, y1 - sy1, sx0 - x0, x1 - sx1,
                             cv2.BORDER_REPLICATE)
    if scale > 1:
        roi = cv2.GaussianBlur(roi, (0, 0), sigmaX=0.45 * scale)

    # Output pixel p maps to source center + scale * (p - h), h = output center.
    h = (output_px - 1) / 2
    matrix = np.array([[scale, 0.0, center[0] - x0 - scale * h],
                       [0.0, scale, center[1] - y0 - scale * h]])
    crop = cv2.warpAffine(roi, matrix, (output_px, output_px),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

    inside_w = max(0.0, min(center[0] + half, width) - max(center[0] - half, 0))
    inside_h = max(0.0, min(center[1] + half, height) - max(center[1] - half, 0))
    outside_fraction = 1.0 - (inside_w * inside_h) / (side_px * side_px)
    return crop, float(outside_fraction)


def _write_qc(qc_dir, frames_df, all_frames, strip, full_frame_preview, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(qc_dir, exist_ok=True)
    with open(os.path.join(qc_dir, "qc.json"), "w") as f:
        json.dump(stats, f, indent=2)
    all_frames.to_csv(os.path.join(qc_dir, "frames_all.csv"), index=False)

    t0 = all_frames["host_ts_ns"].iloc[0]
    t_all = (all_frames["host_ts_ns"] - t0) / 1e9
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_all, all_frames["fz_filtered_n"], lw=0.6, color="0.6", label="Fz filtered (all frames)")
    usable = frames_df[~frames_df["artifact"].astype(bool)]
    ax.scatter((usable["host_ts_ns"] - t0) / 1e9, usable["force_n"], s=0.5, color="C0", label="label force_n (saved, usable)")
    artifact = all_frames["artifact"].to_numpy().astype(bool)
    for start, end in _run_slices(artifact):
        ax.axvspan(t_all.iloc[start], t_all.iloc[end - 1], color="C3", alpha=0.2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("N")
    ax.set_title(f"{stats['session']}: red = artifact (excluded)")
    ax.legend(loc="upper right", markerscale=10)
    fig.tight_layout()
    fig.savefig(os.path.join(qc_dir, "force_plot.png"), dpi=90)
    plt.close(fig)

    if strip:
        tiles = []
        for crop, label in strip:
            tile = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.putText(tile, f"{label:.1f}N", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(tile, f"{label:.1f}N", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        cols = 6
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(os.path.join(qc_dir, "crop_strip.png"), np.vstack(rows))
    if full_frame_preview is not None:
        cv2.imwrite(os.path.join(qc_dir, "full_frame_check.jpg"), full_frame_preview, [cv2.IMWRITE_JPEG_QUALITY, 90])


def process_session(raw_dir, config):
    """Input: raw session directory, full config dict. Writes the processed
    session to processing.output_root. Returns the processed session
    directory. Raises on failure (no PROCESSED_OK then).
    """
    started = time.time()
    proc_config = config["processing"]
    tracking_config = proc_config["hand_tracking"]
    crop_config = proc_config["crop"]
    output_px = crop_config["output_px"]
    if output_px % 32:
        raise ValueError(f"crop.output_px must be a multiple of 32 (FPN), got {output_px}")

    raw_dir = os.path.abspath(raw_dir)
    session_name = os.path.basename(raw_dir.rstrip("/"))
    out_dir = os.path.join(repo_path(proc_config["output_root"]), session_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir)

    with open(os.path.join(raw_dir, "meta_raw.json")) as f:
        meta_raw = json.load(f)
    calibration = meta_raw["camera"]["stream_calibration"]
    color_intrinsics = calibration["color_intrinsics"]
    depth_intrinsics = calibration["depth_intrinsics"]
    depth_scale_m = meta_raw["camera"]["depth"]["depth_scale_m"]
    width, height = color_intrinsics["width"], color_intrinsics["height"]
    hand_label = tracking_config["tracked_hand_label"] or meta_raw["hand_tracking"]["tracked_hand_label"]

    frames_raw = pd.read_csv(os.path.join(raw_dir, "frames_raw.csv"))
    num_frames = len(frames_raw)
    frame_ts_ns = frames_raw["host_ts_ns"].to_numpy().astype(np.int64)
    print(f"[process] {session_name}: {num_frames} raw frames at {width}x{height}, tracking the '{hand_label}' hand")

    force_raw = pd.read_csv(os.path.join(raw_dir, "force_raw.csv"))
    fz_filtered_n, force_n, artifact = frame_force_labels(force_raw, frame_ts_ns, proc_config["force"])

    aligner = depthAlign.DepthAligner(depth_intrinsics, color_intrinsics,
                                      calibration["depth_to_color_extrinsics"], depth_scale_m)

    def color_frames():
        return read_video_frames(os.path.join(raw_dir, "color.mkv"), width, height, "rgb24", 3, np.uint8)

    def depth_frames():
        return read_video_frames(os.path.join(raw_dir, "depth.mkv"), depth_intrinsics["width"],
                                 depth_intrinsics["height"], "gray16le", 1, np.dtype("<u2"))

    # Pass 1: tracking + sparse fingertip depth.
    import mediapipe as mp
    tip_index = tracking_config["fingertip_landmark_index"]
    dip_index = tracking_config["dip_landmark_index"]
    tip = np.full((num_frames, 2), np.nan)
    dip = np.full((num_frames, 2), np.nan)
    tip_depth_m = np.full(num_frames, np.nan)
    depth_every = crop_config["depth_sample_every_n"]
    landmarker = build_hand_landmarker(repo_path(tracking_config["model_path"]))
    last_timestamp_ms = -1
    decoded = 0
    try:
        for i, (color_rgb, depth_raw) in enumerate(zip(color_frames(), depth_frames())):
            if i >= num_frames:
                break
            decoded += 1
            timestamp_ms = max(int(frame_ts_ns[i] // 1_000_000), last_timestamp_ms + 1)  # must strictly increase
            last_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(color_rgb)), timestamp_ms)
            landmarks = select_tracked_hand(result, hand_label)
            if landmarks is not None:
                tip[i] = landmarks[tip_index].x * width, landmarks[tip_index].y * height
                dip[i] = landmarks[dip_index].x * width, landmarks[dip_index].y * height
                if i % depth_every == 0:
                    tip_depth_m[i] = _center_depth_m(aligner, depth_raw, tip[i])
            if (i + 1) % 1000 == 0:
                print(f"[process]   {session_name} pass 1 (tracking): {i + 1}/{num_frames}")
    finally:
        landmarker.close()
    if decoded != num_frames:
        raise RuntimeError(f"decoded {decoded} frames but frames_raw.csv has {num_frames}")

    detected = ~np.isnan(tip[:, 0])
    smoothed, interpolated = fill_and_smooth(np.hstack([tip, dip]), tracking_config)
    tip_s, dip_s = smoothed[:, :2], smoothed[:, 2:]
    tracked = ~np.isnan(tip_s[:, 0])
    centers = tip_s - crop_config["anchor_alpha"] * (tip_s - dip_s)

    # One crop scale per session, from the fingertip depth while pressing
    # (that is where the forearm surface is) - all frames if too few.
    depth_valid = ~np.isnan(tip_depth_m)
    contact_depth = depth_valid & (force_n > 1.0) & ~artifact
    depth_pool = tip_depth_m[contact_depth] if contact_depth.sum() >= 20 else tip_depth_m[depth_valid]
    if depth_pool.size:
        session_depth_m = float(np.median(depth_pool))
    else:
        session_depth_m = float(crop_config["fallback_depth_m"])
        print(f"[process]   {session_name}: WARNING no valid fingertip depth, using fallback {session_depth_m} m")
    side_px = crop_config["side_mm"] / 1000.0 * color_intrinsics["fx"] / session_depth_m
    pixel_size_mm = crop_config["side_mm"] / output_px

    # QC strip: evenly spread over the tracked, non-artifact frames.
    strip_pool = np.flatnonzero(tracked & ~artifact)
    strip_pool = strip_pool if strip_pool.size else np.flatnonzero(tracked)
    strip_n = min(proc_config["qc_strip_frames"], strip_pool.size)
    strip_indices = set(strip_pool[np.linspace(0, strip_pool.size - 1, strip_n).astype(int)].tolist()) if strip_n else set()

    # Pass 2: crop and save.
    rows, all_rows, strip_crops = [], [], {}
    tracked_indices = np.flatnonzero(tracked)
    preview_index = int(tracked_indices[len(tracked_indices) // 2]) if tracked_indices.size else None
    full_frame_preview = None
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, crop_config["jpeg_quality"]]
    for i, color_rgb in enumerate(color_frames()):
        if i >= num_frames:
            break
        saved = False
        outside_fraction = float("nan")
        if tracked[i]:
            crop, outside_fraction = crop_frame(color_rgb, centers[i], side_px, output_px)
            if outside_fraction <= crop_config["max_outside_fraction"]:
                image_name = f"{i:06d}.jpg"
                cv2.imwrite(os.path.join(frames_dir, image_name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), jpeg_params)
                rows.append({
                    "raw_frame_index": i,
                    "image": f"frames/{image_name}",
                    "host_ts_ns": int(frame_ts_ns[i]),
                    "fz_filtered_n": round(float(fz_filtered_n[i]), 4),
                    "force_n": round(float(force_n[i]), 4),
                    "artifact": int(artifact[i]),
                    "detected": int(detected[i]),
                    "interpolated": int(interpolated[i]),
                    "tip_x": round(float(tip_s[i][0]), 2), "tip_y": round(float(tip_s[i][1]), 2),
                    "crop_cx": round(float(centers[i][0]), 2), "crop_cy": round(float(centers[i][1]), 2),
                    "crop_side_px": round(float(side_px), 2),
                    "pixel_size_mm": pixel_size_mm,
                    "outside_fraction": round(outside_fraction, 4),
                })
                if i in strip_indices:
                    strip_crops[i] = crop
                saved = True
        if i == preview_index:
            preview = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            half = side_px / 2
            cv2.rectangle(preview, (int(centers[i][0] - half), int(centers[i][1] - half)),
                          (int(centers[i][0] + half), int(centers[i][1] + half)), (0, 255, 0), 2)
            cv2.circle(preview, tuple(np.round(tip_s[i]).astype(int)), 4, (0, 0, 255), -1)
            full_frame_preview = preview
        all_rows.append({
            "raw_frame_index": i, "host_ts_ns": int(frame_ts_ns[i]),
            "fz_filtered_n": float(fz_filtered_n[i]), "force_n": float(force_n[i]), "artifact": int(artifact[i]),
            "detected": int(detected[i]), "tracked": int(tracked[i]), "saved": int(saved),
            "tip_depth_m": tip_depth_m[i], "outside_fraction": outside_fraction,
        })
        if (i + 1) % 1000 == 0:
            print(f"[process]   {session_name} pass 2 (cropping): {i + 1}/{num_frames}")

    if not rows:
        raise RuntimeError("no frames saved - hand never tracked?")
    frames_df = pd.DataFrame(rows)
    frames_df.to_csv(os.path.join(out_dir, "frames.csv"), index=False)
    all_frames = pd.DataFrame(all_rows)

    strip = [(strip_crops[i], float(force_n[i])) for i in sorted(strip_crops)]

    usable = (frames_df["artifact"] == 0).to_numpy()
    stats = {
        "session": session_name,
        "raw_frames": num_frames,
        "saved_frames": len(frames_df),
        "usable_frames": int(usable.sum()),
        "detected_pct": float(detected.mean() * 100),
        "tracked_pct": float(tracked.mean() * 100),
        "artifact_pct": float(artifact.mean() * 100),
        "hand_label": hand_label,
        "session_fingertip_depth_m": session_depth_m,
        "depth_samples": int(depth_pool.size),
        "crop_side_px": side_px,
        "crop_side_mm": crop_config["side_mm"],
        "pixel_size_mm": pixel_size_mm,
        "force_n_percentiles_usable": dict(zip(["p5", "p50", "p95", "max"], np.percentile(
            frames_df["force_n"].to_numpy()[usable], [5, 50, 95, 100]).round(2).tolist())) if usable.any() else None,
        "contact_pct_usable": float((frames_df["force_n"].to_numpy()[usable] > 0.5).mean() * 100) if usable.any() else None,
        "processing_time_s": round(time.time() - started, 1),
    }
    _write_qc(os.path.join(out_dir, "qc"), frames_df, all_frames, strip, full_frame_preview, stats)

    meta = {
        "session_name": session_name,
        "raw_session_dir": raw_dir,
        "processing": proc_config,
        "hand_label": hand_label,
        "session_fingertip_depth_m": session_depth_m,
        "crop_side_px": side_px,
        "pixel_size_mm": pixel_size_mm,
        "output_px": output_px,
        "force_label": "force_n = max(-Fz_filtered, 0), newtons; Fz reads negative while pressing",
        "camera": meta_raw["camera"],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, PROCESSED_MARKER), "w") as f:
        json.dump({"saved_frames": len(frames_df), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    print(f"[process] {session_name}: saved {len(frames_df)}/{num_frames} crops ({stats['usable_frames']} usable), "
          f"depth {session_depth_m:.3f} m -> crop {side_px:.0f}px = {crop_config['side_mm']:.0f}mm, "
          f"{stats['processing_time_s']}s")
    return out_dir


def _process_one(raw_dir):
    """Worker entry point: returns (raw_dir, error string or None)."""
    try:
        process_session(raw_dir, load_config())
        return raw_dir, None
    except Exception as e:  # keep going with the other sessions
        traceback.print_exc()
        return raw_dir, repr(e)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_session_dirs", nargs="*", help="raw session_* folders to process")
    parser.add_argument("--all", action="store_true", help="process every raw session under processing.raw_root")
    parser.add_argument("--force", action="store_true", help="with --all: also re-process finished sessions")
    parser.add_argument("--workers", type=int, default=1, help="number of sessions processed in parallel")
    args = parser.parse_args()

    config = load_config()
    proc_config = config["processing"]
    raw_dirs = list(args.raw_session_dirs)
    if args.all:
        for raw_dir in sorted(glob.glob(os.path.join(repo_path(proc_config["raw_root"]), "session_*"))):
            done = os.path.exists(os.path.join(repo_path(proc_config["output_root"]),
                                               os.path.basename(raw_dir), PROCESSED_MARKER))
            if args.force or not done:
                raw_dirs.append(raw_dir)
    if not raw_dirs:
        parser.error("no raw sessions to process (pass folders or --all; finished sessions need --force)")

    print(f"[process] {len(raw_dirs)} session(s), {args.workers} worker(s)")
    if args.workers > 1:
        with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(_process_one, raw_dirs, chunksize=1)
    else:
        results = [_process_one(raw_dir) for raw_dir in raw_dirs]

    failures = [(raw_dir, error) for raw_dir, error in results if error]
    for raw_dir, error in failures:
        print(f"[process] FAILED {raw_dir}: {error}")
    print(f"[process] done: {len(results) - len(failures)} ok, {len(failures)} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
