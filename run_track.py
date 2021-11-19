"""Run demo script with config loading similar to PKD."""

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import yaml

from tracker.multitracker import JDETracker
from utils import visualization as vis


def parse_model_config(config_path: Path) -> List[Dict[str, Any]]:
    """Parse model configuration with context manager"""
    with open(config_path) as infile:
        lines = [
            line
            for line in map(str.strip, infile.readlines())
            if line and not line.startswith("#")
        ]
    module_defs: List[Dict[str, Any]] = []
    for line in lines:
        if line.startswith("["):
            module_defs.append({})
            module_defs[-1]["type"] = line[1:-1].rstrip()
            if module_defs[-1]["type"] == "convolutional":
                module_defs[-1]["batch_normalize"] = 0
        else:
            key, value = tuple(map(str.strip, line.split("=")))
            if value.startswith("$"):
                value = module_defs[0].get(value.strip("$"), None)
            module_defs[-1][key] = value
    return module_defs


def preprocess(frame0, input_size):
    """Preprocesses input frame, padded resize and normalize RGB."""
    # Resizing input frame
    video_h, video_w = frame0.shape[:2]
    ratio_w, ratio_h = (float(input_size[0]) / video_w, float(input_size[1]) / video_h)
    ratio = min(ratio_w, ratio_h)
    width, height = int(video_w * ratio), int(video_h * ratio)
    frame0 = cv2.resize(frame0, (width, height))
    # Padded resize
    frame, _, _, _ = _letterbox(frame0, height=input_size[1], width=input_size[0])
    # Normalize RGB
    frame = frame[..., ::-1].transpose(2, 0, 1)
    frame = np.ascontiguousarray(frame, dtype=np.float32)
    frame /= 255.0

    return frame, frame0


def _letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):
    """Resizes a rectangular image to a padded rectangular."""
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    # new_shape = [width, height]
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    # padded rectangular
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, ratio, dw, dh


def main(opt):
    """Main function."""
    opt["output_dir"].mkdir(parents=True, exist_ok=True)
    (opt["output_dir"] / "frames").mkdir(parents=True, exist_ok=True)

    model_cfg = parse_model_config(
        opt["weights_dir"] / opt["config_files"][opt["model_type"]]
    )
    opt["input_size"] = [int(model_cfg[0]["width"]), int(model_cfg[0]["height"])]

    opt_compat = SimpleNamespace(**opt)
    opt_compat.cfg = str(opt["weights_dir"] / opt["config_files"][opt["model_type"]])
    opt_compat.weights = opt["weights_dir"] / opt["model_files"][opt["model_type"]]
    opt_compat.conf_thres = opt["score_threshold"]
    opt_compat.nms_thres = opt["nms_threshold"]
    opt_compat.img_size = opt["input_size"]
    tracker = JDETracker(opt_compat)

    cap = cv2.VideoCapture(str(opt["input_dir"]))
    ret, frame0 = cap.read()

    results = []
    frame_id = 0
    while ret:
        if frame_id % 20 == 0:
            print(f"Processing frame {frame_id}")
        # ==============================================================
        # Model
        # ==============================================================
        frame, frame0 = preprocess(frame0, opt["input_size"])
        blob = torch.from_numpy(frame).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, frame0)
        online_tlwhs = []
        online_ids = []

        for target in online_targets:
            tlwh = target.tlwh
            target_id = target.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt["min_box_area"] and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(target_id)
        results.append((frame_id + 1, online_tlwhs, online_ids))
        online_image = vis.plot_tracking(
            frame0, online_tlwhs, online_ids, frame_id=frame_id
        )
        # ==============================================================
        # Visualization
        # ==============================================================
        cv2.imwrite(
            str(opt["output_dir"] / "frames" / f"{frame_id:05d}.jpg"), online_image
        )

        ret, frame0 = cap.read()
        frame_id += 1


if __name__ == "__main__":
    root_dir = Path.cwd()
    with open(root_dir / "jde_config.yml") as config_file:
        config = yaml.safe_load(config_file.read())
    config["weights_dir"] = root_dir / config["weights_dir"]

    # Load input
    config["input_dir"] = root_dir / "data" / "video" / "mot-clip.mp4"
    # Prepare output
    config["output_dir"] = root_dir / "outputs" / time.strftime("%Y%m%d-%H%M%S")
    print(config)

    main(config)
