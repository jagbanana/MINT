from __future__ import annotations

from dataclasses import dataclass
import math

from .detector import CandidateEvent, utc_timestamp_ms


@dataclass
class CoincidenceMatch:
    primary: CandidateEvent
    secondary: CandidateEvent
    frame_delta: int
    centroid_distance_pixels: float


def match_coincidences(
    primary_events: list[CandidateEvent],
    secondary_events: list[CandidateEvent],
    max_frame_delta: int,
    max_centroid_distance_pixels: float,
) -> tuple[list[CoincidenceMatch], list[CandidateEvent], list[CandidateEvent]]:
    matches: list[CoincidenceMatch] = []
    used_secondary: set[int] = set()

    for primary in primary_events:
        best_index = None
        best_distance = None
        best_delta = None
        for index, secondary in enumerate(secondary_events):
            if index in used_secondary:
                continue
            frame_delta = abs(primary.frame_index - secondary.frame_index)
            if frame_delta > max_frame_delta:
                continue
            distance = centroid_distance(primary, secondary)
            if distance > max_centroid_distance_pixels:
                continue
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance
                best_delta = frame_delta
        if best_index is not None and best_distance is not None and best_delta is not None:
            used_secondary.add(best_index)
            matches.append(
                CoincidenceMatch(
                    primary=primary,
                    secondary=secondary_events[best_index],
                    frame_delta=best_delta,
                    centroid_distance_pixels=best_distance,
                )
            )

    matched_primary = {id(match.primary) for match in matches}
    matched_secondary = {id(match.secondary) for match in matches}
    unmatched_primary = [event for event in primary_events if id(event) not in matched_primary]
    unmatched_secondary = [event for event in secondary_events if id(event) not in matched_secondary]
    return matches, unmatched_primary, unmatched_secondary


def coincidence_record(
    match: CoincidenceMatch,
    primary_record: dict,
    secondary_record: dict,
    camera_settings: dict,
    track: dict | None = None,
) -> dict:
    return {
        "event_type": "coincidence_candidate",
        "timestamp": utc_timestamp_ms(),
        "frame_index": min(match.primary.frame_index, match.secondary.frame_index),
        "frame_delta": match.frame_delta,
        "centroid_distance_pixels": round(match.centroid_distance_pixels, 3),
        "simulated": match.primary.simulated or match.secondary.simulated,
        "confidence_class": "coincidence",
        "candidate_quality_score": 1.0,
        "artifact_risk_score": 0.0,
        "camera_settings": camera_settings,
        "track": track,
        "sensors": {
            match.primary.sensor_label: primary_record,
            match.secondary.sensor_label: secondary_record,
        },
        "notes": [
            "paired transient detected on two independent sensors",
            "coincidence strongly suppresses single-sensor noise",
        ],
    }


def centroid_distance(primary: CandidateEvent, secondary: CandidateEvent) -> float:
    return math.hypot(primary.centroid_x - secondary.centroid_x, primary.centroid_y - secondary.centroid_y)
