from __future__ import annotations

from dataclasses import dataclass
import random
import time

import numpy as np


@dataclass
class SimulationInjector:
    interval_seconds: float = 60.0
    intensity: int = 255
    max_cluster_pixels: int = 3
    last_injection_time: float = 0.0

    def maybe_inject(self, gray: np.ndarray, now: float | None = None) -> set[tuple[int, int]]:
        current = time.monotonic() if now is None else now
        if self.last_injection_time and current - self.last_injection_time < self.interval_seconds:
            return set()

        self.last_injection_time = current
        return self.inject(gray)

    def inject(self, gray: np.ndarray) -> set[tuple[int, int]]:
        h, w = gray.shape[:2]
        y = random.randint(1, max(1, h - 2))
        x = random.randint(1, max(1, w - 2))
        size = random.randint(1, max(1, self.max_cluster_pixels))
        neighbor_offsets = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        random.shuffle(neighbor_offsets)
        offsets = [(0, 0), *neighbor_offsets[: max(0, size - 1)]]
        coords: set[tuple[int, int]] = set()
        for dy, dx in offsets:
            yy = min(max(0, y + dy), h - 1)
            xx = min(max(0, x + dx), w - 1)
            gray[yy, xx] = self.intensity
            coords.add((yy, xx))
        return coords
