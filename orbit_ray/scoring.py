from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .detector import CalibrationSummary, CandidateEvent


@dataclass
class ScoreWeights:
    signal_strength: float = 0.25
    cluster_shape: float = 0.20
    mask_isolation: float = 0.20
    calibration_stability: float = 0.15
    event_rate: float = 0.10
    dynamic_mask_activity: float = 0.10


@dataclass
class EventScore:
    candidate_quality_score: float
    artifact_risk_score: float
    confidence_class: str
    factors: dict[str, float]
    notes: list[str]

    def to_record(self) -> dict:
        return asdict(self)


def score_event(
    candidate: CandidateEvent,
    threshold: int,
    max_cluster_size: int,
    static_mask: np.ndarray,
    dynamic_mask: np.ndarray,
    calibration: CalibrationSummary,
    event_rate_per_minute: float,
    dynamic_mask_additions_per_minute: float,
    mask_isolation_radius: int,
    expected_max_events_per_minute: float,
    noisy_events_per_minute: float,
    high_confidence_threshold: float,
    medium_confidence_threshold: float,
) -> EventScore:
    factors = {
        "signal_strength": _signal_strength(candidate.peak_intensity, threshold),
        "cluster_shape": _cluster_shape(candidate, max_cluster_size),
        "mask_isolation": _mask_isolation(candidate, static_mask, dynamic_mask, mask_isolation_radius),
        "calibration_stability": _calibration_stability(calibration, threshold),
        "event_rate": _event_rate_score(
            event_rate_per_minute,
            expected_max_events_per_minute,
            noisy_events_per_minute,
        ),
        "dynamic_mask_activity": _dynamic_activity_score(dynamic_mask_additions_per_minute),
    }
    weights = ScoreWeights()
    quality = sum(factors[name] * getattr(weights, name) for name in factors)
    risk = 1.0 - quality
    notes = _notes(candidate, factors, event_rate_per_minute, dynamic_mask_additions_per_minute)

    if quality >= high_confidence_threshold and risk <= 0.35:
        confidence = "high"
    elif quality >= medium_confidence_threshold:
        confidence = "medium"
    else:
        confidence = "low"

    return EventScore(
        candidate_quality_score=round(float(quality), 3),
        artifact_risk_score=round(float(risk), 3),
        confidence_class=confidence,
        factors={key: round(float(value), 3) for key, value in factors.items()},
        notes=notes,
    )


def _signal_strength(peak_intensity: int, threshold: int) -> float:
    usable_range = max(1, 255 - threshold)
    return _clamp((peak_intensity - threshold) / usable_range)


def _cluster_shape(candidate: CandidateEvent, max_cluster_size: int) -> float:
    x_min, y_min, x_max, y_max = candidate.bbox
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    bbox_area = max(1, width * height)
    fill_ratio = candidate.cluster_size / bbox_area
    aspect = max(width, height) / max(1, min(width, height))

    if candidate.cluster_size <= 4:
        size_score = 1.0
    else:
        size_score = 1.0 - (candidate.cluster_size - 4) / max(1, max_cluster_size - 4)

    if bbox_area <= 9:
        morphology_score = 1.0
    elif aspect >= 2.0 and fill_ratio <= 0.75:
        morphology_score = 0.85
    elif fill_ratio > 0.8:
        morphology_score = 0.45
    else:
        morphology_score = 0.65

    return _clamp(0.6 * size_score + 0.4 * morphology_score)


def _mask_isolation(
    candidate: CandidateEvent,
    static_mask: np.ndarray,
    dynamic_mask: np.ndarray,
    radius: int,
) -> float:
    x_min, y_min, x_max, y_max = candidate.bbox
    h, w = static_mask.shape[:2]
    left = max(0, x_min - radius)
    top = max(0, y_min - radius)
    right = min(w, x_max + radius + 1)
    bottom = min(h, y_max + radius + 1)
    nearby_mask_count = int((static_mask[top:bottom, left:right] | dynamic_mask[top:bottom, left:right]).sum())
    if nearby_mask_count == 0:
        return 1.0
    if nearby_mask_count <= 2:
        return 0.65
    return 0.25


def _calibration_stability(calibration: CalibrationSummary, threshold: int) -> float:
    mean_score = 1.0 - min(1.0, calibration.dark_mean / 20.0)
    std_score = 1.0 - min(1.0, calibration.dark_std / 10.0)
    max_score = 1.0 if calibration.max_intensity < threshold else 0.5
    return _clamp(0.4 * mean_score + 0.4 * std_score + 0.2 * max_score)


def _event_rate_score(
    event_rate_per_minute: float,
    expected_max_events_per_minute: float,
    noisy_events_per_minute: float,
) -> float:
    if event_rate_per_minute <= expected_max_events_per_minute:
        return 1.0
    if event_rate_per_minute >= noisy_events_per_minute:
        return 0.2
    span = max(0.001, noisy_events_per_minute - expected_max_events_per_minute)
    return _clamp(1.0 - 0.8 * ((event_rate_per_minute - expected_max_events_per_minute) / span))


def _dynamic_activity_score(dynamic_mask_additions_per_minute: float) -> float:
    if dynamic_mask_additions_per_minute <= 10:
        return 1.0
    if dynamic_mask_additions_per_minute >= 500:
        return 0.1
    return _clamp(1.0 - 0.9 * ((dynamic_mask_additions_per_minute - 10) / 490))


def _notes(
    candidate: CandidateEvent,
    factors: dict[str, float],
    event_rate_per_minute: float,
    dynamic_mask_additions_per_minute: float,
) -> list[str]:
    notes: list[str] = []
    if candidate.simulated:
        notes.append("simulated event")
    if factors["mask_isolation"] < 0.7:
        notes.append("near existing masked pixels")
    if factors["cluster_shape"] < 0.6:
        notes.append("large or broad cluster shape")
    if factors["calibration_stability"] < 0.75:
        notes.append("noisy calibration baseline")
    if factors["event_rate"] < 0.75:
        notes.append(f"elevated event rate: {event_rate_per_minute:.2f}/min")
    if factors["dynamic_mask_activity"] < 0.75:
        notes.append(f"dynamic mask activity: {dynamic_mask_additions_per_minute:.2f}/min")
    return notes


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
