from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SensorSafetyState:
    label: str
    baseline_mean: float
    baseline_std: float
    baseline_max: int
    consecutive_unsafe_frames: int = 0
    last_sample: dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyEvaluation:
    safe: bool
    reasons: list[str]
    sample: dict[str, Any]


def initialize_safety_states(sensors) -> dict[str, SensorSafetyState]:
    return {
        sensor.label: SensorSafetyState(
            label=sensor.label,
            baseline_mean=float(sensor.calibration.dark_mean),
            baseline_std=float(sensor.calibration.dark_std),
            baseline_max=int(sensor.calibration.max_intensity),
        )
        for sensor in sensors
    }


def evaluate_frame_safety(
    gray: np.ndarray,
    state: SensorSafetyState,
    *,
    max_dark_mean: float,
    max_dark_std: float,
    max_bright_pixel_fraction: float,
    bright_pixel_threshold: int,
    max_dynamic_mask_count: int,
    dynamic_mask_count: int,
    max_dynamic_additions_per_minute: int,
    dynamic_additions_per_minute: float,
) -> SafetyEvaluation:
    sample = {
        "dark_mean": float(gray.mean()),
        "dark_std": float(gray.std()),
        "max_intensity": int(gray.max()),
        "bright_pixel_fraction": float(np.count_nonzero(gray >= bright_pixel_threshold) / gray.size),
        "bright_pixel_threshold": bright_pixel_threshold,
        "dynamic_mask_count": int(dynamic_mask_count),
        "dynamic_additions_per_minute": float(dynamic_additions_per_minute),
        "baseline_mean": state.baseline_mean,
        "baseline_std": state.baseline_std,
        "baseline_max": state.baseline_max,
    }
    reasons: list[str] = []
    if sample["dark_mean"] > max_dark_mean:
        reasons.append(f"dark_mean {sample['dark_mean']:.3f} > {max_dark_mean:.3f}")
    if sample["dark_std"] > max_dark_std:
        reasons.append(f"dark_std {sample['dark_std']:.3f} > {max_dark_std:.3f}")
    if sample["bright_pixel_fraction"] > max_bright_pixel_fraction:
        reasons.append(
            f"bright_pixel_fraction {sample['bright_pixel_fraction']:.6f} > {max_bright_pixel_fraction:.6f}"
        )
    if dynamic_mask_count > max_dynamic_mask_count:
        reasons.append(f"dynamic_mask_count {dynamic_mask_count} > {max_dynamic_mask_count}")
    if dynamic_additions_per_minute > max_dynamic_additions_per_minute:
        reasons.append(
            f"dynamic_additions_per_minute {dynamic_additions_per_minute:.1f} > {max_dynamic_additions_per_minute}"
        )

    state.last_sample = sample
    return SafetyEvaluation(safe=not reasons, reasons=reasons, sample=sample)
