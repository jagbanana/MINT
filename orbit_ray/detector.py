from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np


@dataclass
class CalibrationSummary:
    frame_count: int
    dark_mean: float
    dark_std: float
    max_intensity: int
    static_hot_pixels: int


class RecurringCoordinateTracker:
    """Track flickering candidate coordinates and dynamically mask repeat offenders.

    Persistent bright pixels are masked when they remain bright in the next frame. Some
    camera artifacts only flash for a single frame but recur at the same pixel over and
    over. This tracker catches that second pattern using a rolling frame window.
    """

    def __init__(self, repeat_threshold: int, window_frames: int, mask_radius_pixels: int = 1):
        self.repeat_threshold = max(1, repeat_threshold)
        self.window_frames = max(1, window_frames)
        self.mask_radius_pixels = max(0, mask_radius_pixels)
        self._hits: dict[tuple[int, int], deque[int]] = {}

    def observe_and_mask(self, dynamic_mask: np.ndarray, candidate: CandidateEvent) -> int:
        """Record a verified candidate and mask its neighborhood if it recurs enough.

        Returns the number of newly masked pixels. Simulated candidates are ignored so
        deterministic simulation/demo runs do not self-suppress.
        """
        if candidate.simulated:
            return 0

        key = (int(round(candidate.centroid_y)), int(round(candidate.centroid_x)))
        frames = self._hits.setdefault(key, deque())
        cutoff = candidate.frame_index - self.window_frames
        while frames and frames[0] < cutoff:
            frames.popleft()
        frames.append(candidate.frame_index)

        if len(frames) < self.repeat_threshold:
            return 0

        additions = mask_neighborhood(dynamic_mask, key, self.mask_radius_pixels)
        frames.clear()
        return additions


def mask_neighborhood(dynamic_mask: np.ndarray, center_yx: tuple[int, int], radius: int) -> int:
    """Mask a square neighborhood around ``center_yx`` and return new additions."""
    y, x = center_yx
    h, w = dynamic_mask.shape[:2]
    top = max(0, y - radius)
    bottom = min(h, y + radius + 1)
    left = max(0, x - radius)
    right = min(w, x + radius + 1)
    before = int(dynamic_mask.sum())
    dynamic_mask[top:bottom, left:right] = True
    return int(dynamic_mask.sum()) - before


@dataclass
class CandidateEvent:
    frame_index: int
    coords: np.ndarray
    peak_intensity: int
    cluster_size: int
    centroid_x: float
    centroid_y: float
    bbox: tuple[int, int, int, int]
    sensor_label: str = "sensor_0"
    simulated: bool = False
    timestamp: str = ""

    def to_record(
        self,
        static_mask_count: int,
        dynamic_mask_count: int,
        camera_settings: dict,
        crop_path: str | None = None,
        score: dict | None = None,
        event_type: str = "single_sensor_candidate",
    ) -> dict:
        timestamp = self.timestamp or utc_timestamp_ms()
        return {
            "event_type": event_type,
            "timestamp": timestamp,
            "sensor_label": self.sensor_label,
            "frame_index": self.frame_index,
            "peak_intensity": self.peak_intensity,
            "cluster_size": self.cluster_size,
            "centroid": {"x": self.centroid_x, "y": self.centroid_y},
            "bbox": {
                "x_min": self.bbox[0],
                "y_min": self.bbox[1],
                "x_max": self.bbox[2],
                "y_max": self.bbox[3],
            },
            "simulated": self.simulated,
            "static_mask_count": static_mask_count,
            "dynamic_mask_count": dynamic_mask_count,
            "camera_settings": camera_settings,
            "crop_path": crop_path,
            "score": score,
        }


def utc_timestamp_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def calibrate(
    frames: Iterable[np.ndarray],
    hot_pixel_threshold: int,
    hot_pixel_fraction: float,
) -> tuple[np.ndarray, CalibrationSummary]:
    frame_list = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    if not frame_list:
        raise ValueError("Calibration requires at least one frame.")

    stack = np.stack(frame_list, axis=0)
    hot_counts = np.count_nonzero(stack > hot_pixel_threshold, axis=0)
    static_mask = hot_counts / stack.shape[0] > hot_pixel_fraction
    summary = CalibrationSummary(
        frame_count=stack.shape[0],
        dark_mean=float(stack.mean()),
        dark_std=float(stack.std()),
        max_intensity=int(stack.max()),
        static_hot_pixels=int(static_mask.sum()),
    )
    return static_mask, summary


def detect_clusters(
    gray: np.ndarray,
    static_mask: np.ndarray,
    dynamic_mask: np.ndarray,
    threshold: int,
    frame_index: int,
    max_cluster_size: int,
    simulated_coords: set[tuple[int, int]] | None = None,
    sensor_label: str = "sensor_0",
) -> list[CandidateEvent]:
    eligible = (gray > threshold) & ~static_mask & ~dynamic_mask
    ys, xs = np.nonzero(eligible)
    if len(xs) == 0:
        return []

    active = set(zip(ys.tolist(), xs.tolist()))
    visited: set[tuple[int, int]] = set()
    candidates: list[CandidateEvent] = []

    for start in list(active):
        if start in visited:
            continue
        coords = _walk_cluster(start, active, visited)
        if len(coords) > max_cluster_size:
            continue
        candidates.append(_candidate_from_coords(gray, coords, frame_index, sensor_label, simulated_coords))
    return candidates


def verify_candidate(candidate: CandidateEvent, next_gray: np.ndarray, threshold: int) -> bool:
    ys = candidate.coords[:, 0]
    xs = candidate.coords[:, 1]
    return not bool(np.any(next_gray[ys, xs] > threshold))


def apply_dynamic_mask(dynamic_mask: np.ndarray, candidate: CandidateEvent) -> int:
    ys = candidate.coords[:, 0]
    xs = candidate.coords[:, 1]
    before = int(dynamic_mask.sum())
    dynamic_mask[ys, xs] = True
    return int(dynamic_mask.sum()) - before


def crop_around(gray: np.ndarray, bbox: tuple[int, int, int, int], radius: int) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    h, w = gray.shape[:2]
    left = max(0, x_min - radius)
    top = max(0, y_min - radius)
    right = min(w, x_max + radius + 1)
    bottom = min(h, y_max + radius + 1)
    return gray[top:bottom, left:right].copy()


def _walk_cluster(
    start: tuple[int, int],
    active: set[tuple[int, int]],
    visited: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    stack = [start]
    coords: list[tuple[int, int]] = []
    while stack:
        point = stack.pop()
        if point in visited or point not in active:
            continue
        visited.add(point)
        coords.append(point)
        y, x = point
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = (y + dy, x + dx)
                if neighbor in active and neighbor not in visited:
                    stack.append(neighbor)
    return coords


def _candidate_from_coords(
    gray: np.ndarray,
    coords: list[tuple[int, int]],
    frame_index: int,
    sensor_label: str,
    simulated_coords: set[tuple[int, int]] | None,
) -> CandidateEvent:
    coord_array = np.array(coords, dtype=np.int64)
    ys = coord_array[:, 0]
    xs = coord_array[:, 1]
    intensities = gray[ys, xs].astype(np.float64)
    weight_sum = float(intensities.sum())
    centroid_x = float((xs * intensities).sum() / weight_sum) if weight_sum else float(xs.mean())
    centroid_y = float((ys * intensities).sum() / weight_sum) if weight_sum else float(ys.mean())
    coord_set = set(coords)
    simulated = bool(simulated_coords and coord_set.intersection(simulated_coords))
    return CandidateEvent(
        frame_index=frame_index,
        coords=coord_array,
        peak_intensity=int(intensities.max()),
        cluster_size=len(coords),
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        sensor_label=sensor_label,
        simulated=simulated,
        timestamp=utc_timestamp_ms(),
    )
