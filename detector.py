from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


def _nms(boxes: List[BBox], weights: List[float], score_threshold: float = 0.0, iou_threshold: float = 0.35) -> List[BBox]:
    if not boxes:
        return []

    x1 = np.array([b[0] for b in boxes], dtype=np.float32)
    y1 = np.array([b[1] for b in boxes], dtype=np.float32)
    x2 = np.array([b[0] + b[2] for b in boxes], dtype=np.float32)
    y2 = np.array([b[1] + b[3] for b in boxes], dtype=np.float32)
    scores = np.array(weights, dtype=np.float32)

    idxs = np.argsort(scores)[::-1]
    keep = []

    while idxs.size > 0:
        i = idxs[0]
        if scores[i] < score_threshold:
            idxs = idxs[1:]
            continue

        keep.append(i)
        if idxs.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_j = (x2[idxs[1:]] - x1[idxs[1:]]) * (y2[idxs[1:]] - y1[idxs[1:]])
        union = area_i + area_j - inter + 1e-6
        iou = inter / union

        idxs = idxs[1:][iou <= iou_threshold]

    return [boxes[i] for i in keep]


@dataclass
class HOGPersonDetector:
    hit_threshold: float = 0.0
    win_stride: Tuple[int, int] = (8, 8)
    padding: Tuple[int, int] = (8, 8)
    scale: float = 1.05
    nms_iou_threshold: float = 0.35

    def __post_init__(self) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> List[BBox]:
        if frame is None or frame.size == 0:
            return []

        rects, weights = self.hog.detectMultiScale(
            frame,
            hitThreshold=self.hit_threshold,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
        )

        boxes: List[BBox] = []
        scores: List[float] = []
        for (x, y, w, h), weight in zip(rects, weights):
            boxes.append((int(x), int(y), int(w), int(h)))
            try:
                scores.append(float(weight))
            except Exception:
                scores.append(0.0)

        return _nms(boxes, scores, score_threshold=-999.0, iou_threshold=self.nms_iou_threshold)
