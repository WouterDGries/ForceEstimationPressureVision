"""Review tool for the processed sessions written by dataProcessor.py.

For every saved frame it shows, side by side:
  * the raw full video frame with the crop box and fingertip drawn on it
    (decoded from the raw color.mkv the session was made from),
  * the saved crop exactly as stored (frames/<raw_frame_index>.jpg),
  * the network input as Model/dataset.py produces it - optionally with the
    random training augmentation - de-normalized for display, with the
    tensor's statistics, label and pixel area,
and underneath the session's force: filtered Fz (original sign, pressing is
negative), raw Fz, and the label force_n with frames marked by whether the
dataset uses them. A slider (or clicking the plot, or the arrow keys) moves
through the saved frames; the session box switches sessions and shows which
split each session falls in with the current config.

Read-only: nothing on disk is changed.

Run with the trajPipeline conda env, from the repo root:
    ~/anaconda3/envs/trajPipeline/bin/python preprocessing/dataViewer.py
    ~/anaconda3/envs/trajPipeline/bin/python preprocessing/dataViewer.py data/processed/session_20261002_143513
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "Model"))
import dataset  # noqa: E402  (Model/dataset.py - the exact input pipeline used for training)

FULL_FRAME_WIDTH_PX = 560          # on-screen width of the full-frame panel
CROP_DISPLAY_PX = 300              # on-screen size of the two crop panels
FULL_FRAME_DEBOUNCE_MS = 120       # video seeking is slow; only decode once the slider settles
RAW_FORCE_PLOT_EVERY_N = 5         # raw Fz (~1 kHz) is decimated for plotting


def rgb_to_pixmap(rgb, width, height=None):
    """Input: (H, W, 3) uint8 RGB, target on-screen width (and optional
    height). Returns a QPixmap scaled to fit, aspect ratio kept.
    """
    rgb = np.ascontiguousarray(rgb)
    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy()).scaled(width, height or width, Qt.KeepAspectRatio,
                                                  Qt.SmoothTransformation)


class SessionData:
    """One processed session: frames.csv, qc/frames_all.csv, meta.json, the
    raw force (if the raw session is still on disk) and the dataset's view
    of which frames are used.
    """

    def __init__(self, session_dir, model_config, split):
        self.session_dir = session_dir
        self.name = os.path.basename(session_dir)
        self.split = split
        with open(os.path.join(session_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.frames = pd.read_csv(os.path.join(session_dir, "frames.csv"))
        self.frames["image_path"] = [os.path.join(session_dir, image) for image in self.frames["image"]]
        self.frames["session"] = self.name
        self.all_frames = pd.read_csv(os.path.join(session_dir, "qc", "frames_all.csv"))
        self.t0 = int(self.all_frames["host_ts_ns"].iloc[0])
        self.frame_time_s = (self.frames["host_ts_ns"].to_numpy() - self.t0) / 1e9

        # Same rules as dataset.load_session_frames: usable, then every stride-th usable frame.
        frames_config = model_config["frames"]
        self.max_force_n = frames_config["max_force_n"]
        self.usable = ((self.frames["artifact"] == 0) & (self.frames["force_n"] <= self.max_force_n)).to_numpy()
        self.stride = frames_config["train_stride"] if split == "train" else frames_config["eval_stride"]
        position = np.cumsum(self.usable) - 1
        self.in_subsample = self.usable & (position % self.stride == 0)

        self.raw_dir = self.meta.get("raw_session_dir")
        self.raw_force = None
        force_path = os.path.join(self.raw_dir, "force_raw.csv") if self.raw_dir else None
        if force_path and os.path.exists(force_path):
            raw = pd.read_csv(force_path).iloc[::RAW_FORCE_PLOT_EVERY_N]
            self.raw_force = ((raw["host_ts_ns"].to_numpy() - self.t0) / 1e9, raw["fz_N"].to_numpy())
        video_path = os.path.join(self.raw_dir, "color.mkv") if self.raw_dir else None
        self.video = cv2.VideoCapture(video_path) if video_path and os.path.exists(video_path) else None
        self._next_video_frame = None

    def num_frames(self):
        return len(self.frames)

    def nearest_frame_index(self, time_s):
        index = int(np.clip(np.searchsorted(self.frame_time_s, time_s), 0, self.num_frames() - 1))
        if index > 0 and abs(self.frame_time_s[index - 1] - time_s) < abs(self.frame_time_s[index] - time_s):
            return index - 1
        return index

    def full_frame_rgb(self, raw_frame_index):
        """Returns the raw color frame (RGB) or None if the raw video is not
        available. Reads sequentially when possible, seeks otherwise.
        """
        if self.video is None:
            return None
        if raw_frame_index != self._next_video_frame:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, raw_frame_index)
        ok, bgr = self.video.read()
        self._next_video_frame = raw_frame_index + 1 if ok else None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    def usage_text(self, index):
        row = self.frames.iloc[index]
        if row["artifact"]:
            return "<b style='color:#c00'>EXCLUDED</b> (force artifact: arm lifted / outside force recording)"
        if row["force_n"] > self.max_force_n:
            return f"<b style='color:#c00'>EXCLUDED</b> (label > max_force_n = {self.max_force_n} N)"
        if self.in_subsample[index]:
            return f"<b style='color:#080'>USED</b> ({self.split}, stride {self.stride})"
        return f"usable, skipped by {self.split} stride {self.stride}"

    def close(self):
        if self.video is not None:
            self.video.release()


class ImagePanel(QWidget):
    def __init__(self, title, width, height):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(width, height)
        self.width_px, self.height_px = width, height
        layout.addWidget(title_label)
        layout.addWidget(self.image_label)

    def set_rgb(self, rgb):
        self.image_label.setPixmap(rgb_to_pixmap(rgb, self.width_px, self.height_px))

    def set_message(self, text):
        self.image_label.clear()
        self.image_label.setText(text)


class ViewerWindow(QMainWindow):
    def __init__(self, config, session_dirs, splits, start_session_dir):
        super().__init__()
        self.config = config
        self.model_config = config["model_training"]
        self.session_dirs = session_dirs
        self.splits = splits
        self.session = None

        # The dataset object is only used for its __getitem__ (load -> augment
        # -> PressureVision normalization); its frame table is swapped per session.
        self.input_dataset = dataset.ForceFrameDataset(self.model_config, "val", sessions=[], augment=False)
        self.rgb_mean = np.array([0.485, 0.456, 0.406])   # recording.util.resnet_preprocessor
        self.rgb_std = np.array([0.229, 0.224, 0.225])

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        top.addWidget(QLabel("Session:"))
        self.session_box = QComboBox()
        for session_dir in session_dirs:
            name = os.path.basename(session_dir)
            self.session_box.addItem(f"{name}   [{splits.get(name, 'not in split')}]", session_dir)
        top.addWidget(self.session_box, 1)
        root.addLayout(top)

        self.info_label = QLabel()
        self.info_label.setTextFormat(Qt.RichText)
        root.addWidget(self.info_label)

        images = QHBoxLayout()
        self.full_panel = ImagePanel("Raw frame + crop box (green) + fingertip (red)",
                                     FULL_FRAME_WIDTH_PX, int(FULL_FRAME_WIDTH_PX * 9 / 16))
        self.crop_panel = ImagePanel("Saved crop (frames/*.jpg)", CROP_DISPLAY_PX, CROP_DISPLAY_PX)
        self.input_panel = ImagePanel("Network input (de-normalized)", CROP_DISPLAY_PX, CROP_DISPLAY_PX)
        images.addWidget(self.full_panel)
        images.addWidget(self.crop_panel)
        input_column = QVBoxLayout()
        input_column.addWidget(self.input_panel)
        augment_row = QHBoxLayout()
        self.augment_box = QCheckBox("training augmentation")
        self.augment_box.toggled.connect(lambda _: self._show_network_input())
        self.redraw_button = QPushButton("re-draw")
        self.redraw_button.clicked.connect(self._show_network_input)
        augment_row.addWidget(self.augment_box)
        augment_row.addWidget(self.redraw_button)
        input_column.addLayout(augment_row)
        self.input_stats_label = QLabel()
        self.input_stats_label.setStyleSheet("font-family: monospace;")
        input_column.addWidget(self.input_stats_label)
        images.addLayout(input_column)
        root.addLayout(images)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Force", units="N")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend(offset=(10, 10))
        self.plot_widget.setMinimumHeight(260)
        root.addWidget(self.plot_widget)
        self.click_proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseClicked, slot=self._on_plot_clicked)

        controls = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.valueChanged.connect(self._show_frame)
        controls.addWidget(self.slider, 1)
        self.used_only_box = QCheckBox("step through used frames only")
        controls.addWidget(self.used_only_box)
        root.addLayout(controls)

        self.full_frame_timer = QTimer(self)
        self.full_frame_timer.setSingleShot(True)
        self.full_frame_timer.timeout.connect(self._show_full_frame)

        # Arrow keys step frames wherever the keyboard focus is.
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self._step(1))
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self._step(-1))

        start = session_dirs.index(start_session_dir) if start_session_dir in session_dirs else 0
        self.session_box.setCurrentIndex(start)
        self.session_box.currentIndexChanged.connect(self._on_session_changed)
        self._on_session_changed(start)
        self.resize(1300, 900)

    # ---- session / plot ----
    def _on_session_changed(self, combo_index):
        if self.session is not None:
            self.session.close()
        session_dir = self.session_box.itemData(combo_index)
        name = os.path.basename(session_dir)
        self.session = SessionData(session_dir, self.model_config, self.splits.get(name, "val"))
        self.input_dataset.frames = self.session.frames
        self.setWindowTitle(f"Processed Data Viewer - {name}")
        self._build_plot()
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.session.num_frames() - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._show_frame(0)

    def _build_plot(self):
        session = self.session
        plot = self.plot_widget
        plot.clear()
        all_t = (session.all_frames["host_ts_ns"].to_numpy() - session.t0) / 1e9
        if session.raw_force is not None:
            plot.plot(*session.raw_force, pen=pg.mkPen((200, 200, 200), width=1), name="raw Fz (sensor sign)")
        plot.plot(all_t, session.all_frames["fz_filtered_n"].to_numpy(), pen=pg.mkPen((90, 90, 90), width=1),
                  name="filtered Fz (sensor sign)")

        frames = session.frames
        used_label = np.where(session.usable, frames["force_n"].to_numpy(), np.nan)
        plot.plot(session.frame_time_s, used_label, pen=pg.mkPen((30, 100, 220), width=2), connect="finite",
                  name="label force_n (usable frames)")
        excluded = ~session.usable
        if excluded.any():
            plot.plot(session.frame_time_s[excluded], frames["force_n"].to_numpy()[excluded], pen=None,
                      symbol="o", symbolSize=3, symbolBrush=(220, 30, 30), symbolPen=None,
                      name="saved but excluded")
        not_saved = session.all_frames["saved"].to_numpy() == 0
        if not_saved.any():
            plot.plot(all_t[not_saved], np.full(not_saved.sum(), -1.0), pen=None, symbol="s", symbolSize=2,
                      symbolBrush=(240, 150, 0), symbolPen=None, name="no crop (hand not tracked), at -1 N")
        artifact_n = self.config["processing"]["force"]["artifact_positive_n"]
        plot.addItem(pg.InfiniteLine(pos=artifact_n, angle=0, pen=pg.mkPen((220, 30, 30), style=Qt.DashLine)))
        plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("k", width=1)))
        self.cursor_line = pg.InfiniteLine(pos=session.frame_time_s[0], angle=90, pen=pg.mkPen("r", width=2))
        plot.addItem(self.cursor_line)

    def _on_plot_clicked(self, event):
        mouse_event = event[0]
        if mouse_event.button() != Qt.LeftButton:
            return
        point = self.plot_widget.plotItem.vb.mapSceneToView(mouse_event.scenePos())
        self.slider.setValue(self.session.nearest_frame_index(point.x()))

    def _step(self, step):
        """Moves one saved frame forward/back, or to the next/previous used
        frame when "used frames only" is ticked.
        """
        index = self.slider.value() + step
        if self.used_only_box.isChecked():
            used = np.flatnonzero(self.session.in_subsample)
            candidates = used[used > self.slider.value()] if step > 0 else used[used < self.slider.value()][::-1]
            index = int(candidates[0]) if candidates.size else self.slider.value()
        self.slider.setValue(int(np.clip(index, 0, self.session.num_frames() - 1)))

    # ---- frame display ----
    def _show_frame(self, index):
        session = self.session
        row = session.frames.iloc[index]
        self.cursor_line.setPos(session.frame_time_s[index])

        crop = cv2.imread(row["image_path"])
        if crop is None:
            self.crop_panel.set_message(f"missing {row['image']}")
        else:
            self.crop_panel.set_rgb(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        self._show_network_input()
        self.full_frame_timer.start(FULL_FRAME_DEBOUNCE_MS)

        flags = []
        if row["interpolated"]:
            flags.append("fingertip interpolated")
        if not row["detected"]:
            flags.append("not detected this frame")
        self.info_label.setText(
            f"Frame {index + 1}/{session.num_frames()} (raw #{int(row['raw_frame_index'])})   "
            f"t = {session.frame_time_s[index]:.2f} s   "
            f"Fz = {row['fz_filtered_n']:+.2f} N  &rarr;  label <b>{row['force_n']:.2f} N</b>   "
            f"{session.usage_text(index)}<br>"
            f"crop centre ({row['crop_cx']:.0f}, {row['crop_cy']:.0f}) px, side {row['crop_side_px']:.0f} px "
            f"= {self.config['processing']['crop']['side_mm']:.0f} mm, {row['pixel_size_mm']:.3f} mm/px, "
            f"outside image {max(row['outside_fraction'], 0.0) * 100:.0f}%   "
            f"{', '.join(flags) if flags else ''}   split: <b>{session.split}</b>")

    def _show_network_input(self):
        index = self.slider.value()
        self.input_dataset.augment = self.augment_box.isChecked()
        self.redraw_button.setEnabled(self.augment_box.isChecked())
        try:
            sample = self.input_dataset[index]
        except Exception as error:
            self.input_panel.set_message(f"dataset error:\n{error}")
            return
        tensor = sample["img"].numpy()                      # (3, H, W), exactly what the network receives
        rgb = np.clip(tensor.transpose(1, 2, 0) * self.rgb_std + self.rgb_mean, 0, 1)
        self.input_panel.set_rgb((rgb * 255).astype(np.uint8))
        means = ", ".join(f"{v:+.2f}" for v in tensor.mean(axis=(1, 2)))
        stds = ", ".join(f"{v:.2f}" for v in tensor.std(axis=(1, 2)))
        pixel_mm = float(np.sqrt(sample["pixel_area_m2"].item())) * 1000
        self.input_stats_label.setText(
            f"tensor {tuple(tensor.shape)} {tensor.dtype}\n"
            f"range [{tensor.min():+.2f}, {tensor.max():+.2f}]\n"
            f"ch mean [{means}]\nch std  [{stds}]\n"
            f"label {sample['force_n'].item():.2f} N\n"
            f"pixel {pixel_mm:.3f} mm ({sample['pixel_area_m2'].item():.2e} m2)")

    def _show_full_frame(self):
        session = self.session
        row = session.frames.iloc[self.slider.value()]
        frame = session.full_frame_rgb(int(row["raw_frame_index"]))
        if frame is None:
            self.full_panel.set_message("raw color.mkv not available")
            return
        frame = frame.copy()
        half = row["crop_side_px"] / 2
        cv2.rectangle(frame, (int(row["crop_cx"] - half), int(row["crop_cy"] - half)),
                      (int(row["crop_cx"] + half), int(row["crop_cy"] + half)), (0, 255, 0), 2)
        cv2.circle(frame, (int(round(row["tip_x"])), int(round(row["tip_y"]))), 5, (255, 0, 0), -1)
        self.full_panel.set_rgb(frame)

    def closeEvent(self, event):
        if self.session is not None:
            self.session.close()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", nargs="?", default=None, help="processed session to open first")
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    config = dataset.load_config()
    model_config = config["model_training"]
    data_root = dataset.repo_path(model_config["data_root"])
    session_dirs = [os.path.join(data_root, name) for name in dataset.discover_sessions(model_config)]
    if not session_dirs:
        QMessageBox.critical(None, "No data", f"No processed sessions (with PROCESSED_OK) in {data_root}")
        return
    try:
        splits = {name: split for split, names in dataset.resolve_splits(model_config).items() for name in names}
    except ValueError as error:   # too few sessions for a split yet
        print(f"[viewer] {error}")
        splits = {}

    start = os.path.abspath(args.session_dir) if args.session_dir else None
    window = ViewerWindow(config, session_dirs, splits, start)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
