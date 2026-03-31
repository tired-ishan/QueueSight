from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict, Any

import cv2
import numpy as np
import yaml


Point = Tuple[int, int]


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalized_polygon_to_pixels(normalized_points: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    pts = []
    for x, y in normalized_points:
        pts.append([int(round(x * width)), int(round(y * height))])
    return np.array(pts, dtype=np.int32)


def point_in_polygon(point: Tuple[int, int], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def draw_text_box(
    frame: np.ndarray,
    text_lines: List[str],
    origin: Tuple[int, int] = (15, 30),
    font_scale: float = 0.6,
    thickness: int = 2,
    line_gap: int = 8,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX

    widths = []
    heights = []
    baseline_max = 0
    for line in text_lines:
        (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        widths.append(w)
        heights.append(h)
        baseline_max = max(baseline_max, baseline)

    box_w = max(widths) + 24
    box_h = sum(heights) + (len(text_lines) - 1) * line_gap + 24

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - 22), (x + box_w, y + box_h - 12), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    yy = y
    for i, line in enumerate(text_lines):
        (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.putText(frame, line, (x + 12, yy + h), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        yy += h + line_gap


def crowd_status(count: int, low_max: int, medium_max: int) -> str:
    if count <= low_max:
        return "LOW"
    if count <= medium_max:
        return "MEDIUM"
    return "HIGH"


def color_for_status(status: str) -> Tuple[int, int, int]:
    return {
        "LOW": (0, 200, 0),
        "MEDIUM": (0, 165, 255),
        "HIGH": (0, 0, 255),
    }.get(status, (255, 255, 255))


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
