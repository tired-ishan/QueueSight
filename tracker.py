from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]


def _centroid(box: BBox) -> Tuple[int, int]:
    x, y, w, h = box
    return (int(x + w / 2), int(y + h / 2))


@dataclass
class CentroidTracker:
    max_disappeared: int = 18
    max_distance: int = 80
    next_object_id: int = 0
    objects: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    boxes: Dict[int, BBox] = field(default_factory=dict)
    disappeared: Dict[int, int] = field(default_factory=dict)

    def register(self, box: BBox) -> None:
        object_id = self.next_object_id
        self.objects[object_id] = _centroid(box)
        self.boxes[object_id] = box
        self.disappeared[object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        self.objects.pop(object_id, None)
        self.boxes.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    def update(self, boxes: List[BBox]) -> Dict[int, Tuple[int, int]]:
        if len(boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array([_centroid(b) for b in boxes], dtype=np.int32)

        if len(self.objects) == 0:
            for box in boxes:
                self.register(box)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()), dtype=np.int32)

        # pairwise distances between existing object centroids and new centroids
        D = np.linalg.norm(object_centroids[:, None, :] - input_centroids[None, :, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = tuple(input_centroids[col])
            self.boxes[object_id] = boxes[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        for col in unused_cols:
            self.register(boxes[col])

        return self.objects
