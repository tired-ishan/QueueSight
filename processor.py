from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

from detector import HOGPersonDetector
from tracker import CentroidTracker
from utils import (
    color_for_status,
    crowd_status,
    draw_text_box,
    ensure_dir,
    load_config,
    normalized_polygon_to_pixels,
    point_in_polygon,
    write_csv,
)


def _fourcc_writer(output_path: str | Path, fps: float, width: int, height: int):
    ext = str(output_path).lower()
    if ext.endswith(".avi"):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    csv_path: str | Path | None = None,
    config_path: str | Path = "config.yaml",
    show_progress: bool = True,
) -> Dict[str, Any]:
    config = load_config(config_path)
    detector_cfg = config.get("detector", {})
    track_cfg = config.get("tracking", {})
    display_cfg = config.get("display", {})
    crowd_cfg = config.get("crowd_thresholds", {})

    low_max = int(crowd_cfg.get("low_max", 2))
    medium_max = int(crowd_cfg.get("medium_max", 5))

    detector = HOGPersonDetector(
        hit_threshold=float(detector_cfg.get("hit_threshold", 0.0)),
        win_stride=tuple(detector_cfg.get("win_stride", [8, 8])),
        padding=tuple(detector_cfg.get("padding", [8, 8])),
        scale=float(detector_cfg.get("scale", 1.05)),
    )
    tracker = CentroidTracker(
        max_disappeared=int(track_cfg.get("max_disappeared", 18)),
        max_distance=int(track_cfg.get("max_distance", 80)),
    )
    smoothing_window = int(track_cfg.get("smoothing_window", 12))
    count_history = deque(maxlen=smoothing_window)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-2:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    writer = _fourcc_writer(output_path, fps, width, height)

    polygon = normalized_polygon_to_pixels(config.get("queue_region_normalized", []), width, height)
    rows: List[Dict[str, Any]] = []

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            detections = detector.detect(frame)

            queue_dets = []
            for box in detections:
                x, y, w, h = box
                center = (int(x + w / 2), int(y + h / 2))
                if point_in_polygon(center, polygon):
                    queue_dets.append(box)

            objects = tracker.update(queue_dets)
            active_ids = []
            for obj_id, centroid in objects.items():
                if point_in_polygon(centroid, polygon):
                    active_ids.append(obj_id)

            count = len(active_ids)
            count_history.append(count)
            smooth_count = int(round(sum(count_history) / max(1, len(count_history))))
            status = crowd_status(smooth_count, low_max, medium_max)

            rows.append(
                {
                    "frame": frame_idx,
                    "time_sec": round(frame_idx / fps, 2),
                    "raw_count": count,
                    "smoothed_count": smooth_count,
                    "status": status,
                }
            )

            overlay = frame.copy()

            if len(polygon) > 0:
                cv2.polylines(overlay, [polygon], isClosed=True, color=(255, 255, 0), thickness=3)

            for box in queue_dets:
                x, y, w, h = box
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for obj_id, centroid in objects.items():
                if point_in_polygon(centroid, polygon):
                    cx, cy = centroid
                    cv2.circle(overlay, (cx, cy), 5, (0, 255, 255), -1)
                    if display_cfg.get("show_ids", True):
                        cv2.putText(
                            overlay,
                            f"ID {obj_id}",
                            (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            float(display_cfg.get("font_scale", 0.7)),
                            (255, 255, 255),
                            int(display_cfg.get("thickness", 2)),
                            cv2.LINE_AA,
                        )

            status_color = color_for_status(status)
            cv2.circle(overlay, (50, 50), 18, status_color, -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            text_lines = [
                f"Queue Count: {smooth_count}",
                f"Raw Count: {count}",
                f"Status: {status}",
                f"Frame: {frame_idx}" + (f"/{total_frames}" if total_frames > 0 else ""),
            ]
            draw_text_box(
                frame,
                text_lines,
                origin=(15, 35),
                font_scale=float(display_cfg.get("font_scale", 0.7)),
                thickness=int(display_cfg.get("thickness", 2)),
            )

            writer.write(frame)

            if show_progress and frame_idx % 25 == 0:
                print(f"Processed {frame_idx}" + (f"/{total_frames}" if total_frames > 0 else "") + " frames")

    finally:
        cap.release()
        writer.release()

    if csv_path:
        csv_path = Path(csv_path)
        ensure_dir(csv_path.parent)
        write_csv(rows, csv_path)

    summary = {
        "frames_processed": frame_idx,
        "output_video": str(output_path),
        "csv_path": str(csv_path) if csv_path else None,
        "peak_count": max((r["smoothed_count"] for r in rows), default=0),
        "average_count": round(sum(r["smoothed_count"] for r in rows) / max(1, len(rows)), 2),
        "final_status": rows[-1]["status"] if rows else "N/A",
    }
    return summary


def plot_history(csv_path: str | Path, output_png: str | Path) -> str:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df["time_sec"], df["smoothed_count"], label="Smoothed Count")
    ax.plot(df["time_sec"], df["raw_count"], alpha=0.35, label="Raw Count")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("People in Queue")
    ax.set_title("Queue Count Over Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_png = Path(output_png)
    ensure_dir(output_png.parent)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return str(output_png)
