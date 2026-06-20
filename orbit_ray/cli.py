from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
import time

import numpy as np

from .calibration import load_detector_calibration, reconstruct_track, write_calibration_template
from .verification import (
    VerifiedTrackMatch,
    centroid_distance,
    match_verified_tracks,
    near_miss_record,
    verified_track_record,
)
from .config import AppConfig, CameraConfig, load_config
from .detector import (
    CalibrationSummary,
    CandidateEvent,
    RecurringCoordinateTracker,
    apply_dynamic_mask,
    calibrate,
    crop_around,
    detect_clusters,
    utc_timestamp_ms,
    verify_candidate,
)
from .logging_io import append_jsonl, ensure_output_dirs, write_json
from .safety import initialize_safety_states, evaluate_frame_safety
from .scoring import score_event
from .simulator import SimulationInjector


class SafetyShutdown(RuntimeError):
    """Raised when MINT detects sustained unsafe/noisy sensor behavior."""

    def __init__(self, message: str, status: dict | None = None):
        super().__init__(message)
        self.status = status or {}


@dataclass
class SensorRuntime:
    camera_config: CameraConfig
    capture: object
    camera_settings: dict
    static_mask: np.ndarray
    dynamic_mask: np.ndarray
    calibration: CalibrationSummary
    calibration_record: dict
    pending: list[tuple[CandidateEvent, np.ndarray]] = field(default_factory=list)
    dynamic_window_start: float = field(default_factory=time.monotonic)
    dynamic_window_additions: int = 0
    recurring_tracker: RecurringCoordinateTracker | None = None

    @property
    def label(self) -> str:
        return self.camera_config.label


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.write_calibration_template:
        write_calibration_template(args.write_calibration_template)
        print(f"Wrote calibration template: {args.write_calibration_template}")
        return 0

    config = load_config(args.config)
    apply_overrides(config, args)

    try:
        import cv2  # type: ignore
    except ImportError:
        print("OpenCV is required for camera capture. Install dependencies with: pip install -r requirements.txt")
        return 2

    output_dir = Path(config.output.dir)
    paths = ensure_output_dirs(output_dir)
    event_log = paths["root"] / config.output.event_log

    sensor_configs = [config.camera]
    if config.verification.enabled:
        sensor_configs.append(config.secondary_camera)

    detector_calibration = None
    if config.track_reconstruction.enabled:
        detector_calibration = load_detector_calibration(config.track_reconstruction.calibration_file)

    sensors = setup_sensors(cv2, config, paths["root"], sensor_configs)
    print("MINT started")
    print(f"Mode: {'two-sensor verification' if config.verification.enabled else 'single sensor'}")
    safety_states = initialize_safety_states(sensors)

    injector = SimulationInjector(
        interval_seconds=config.simulation.interval_seconds,
        intensity=config.simulation.intensity,
        max_cluster_pixels=config.simulation.max_cluster_pixels,
    )

    counters = {
        "frames": 0,
        "sensor_candidates": 0,
        "verified_sensor_events": 0,
        "verified_track_events": 0,
        "verification_near_miss_events": 0,
        "unmatched_sensor_events": 0,
        "persistent_dropped": 0,
        "dynamic_mask_additions": 0,
        "recurring_dropped": 0,
    }
    recent_events: list[dict] = []
    verification_history: dict[str, list[tuple[CandidateEvent, dict]]] = {}
    verified_event_keys: set[tuple[str, int, float, float]] = set()
    logged_near_miss_pairs: set[tuple[str, int, str, int]] = set()
    start = time.monotonic()
    last_status = start

    try:
        while True:
            frames = read_sensor_frames(cv2, config, sensors)
            counters["frames"] += 1
            frame_index = counters["frames"]

            check_safety_or_shutdown(
                frames=frames,
                sensors=sensors,
                config=config,
                safety_states=safety_states,
                counters=counters,
                start=start,
                recent_events=recent_events,
            )

            verified_by_sensor: dict[str, list[CandidateEvent]] = {}
            records_by_event_id: dict[int, dict] = {}

            for sensor in sensors:
                verified = verify_pending_events(
                    sensor,
                    frames[sensor.label],
                    paths["crops"],
                    config,
                    counters,
                    start,
                )
                kept_verified: list[CandidateEvent] = []
                for candidate in verified:
                    if maybe_drop_recurring_candidate(sensor, candidate, config, counters):
                        continue
                    kept_verified.append(candidate)
                    records_by_event_id[id(candidate)] = build_sensor_event_record(
                        sensor=sensor,
                        candidate=candidate,
                        candidate_frame=get_candidate_frame(sensor, candidate),
                        crops_dir=paths["crops"],
                        config=config,
                        counters=counters,
                        start=start,
                    )
                verified_by_sensor[sensor.label] = kept_verified
                sensor.pending = []

            if config.verification.enabled and len(sensors) == 2:
                new_records = log_verification_results(
                    sensors,
                    verified_by_sensor,
                    records_by_event_id,
                    config,
                    event_log,
                    counters,
                    detector_calibration,
                    verification_history,
                    verified_event_keys,
                    logged_near_miss_pairs,
                )
            else:
                new_records = []
                for candidates in verified_by_sensor.values():
                    for candidate in candidates:
                        record = records_by_event_id[id(candidate)]
                        append_jsonl(event_log, record)
                        new_records.append(record)

            if new_records:
                recent_events = (recent_events + new_records)[-5:]

            simulated_coords = maybe_inject_simulation(frames, sensors, injector, config)
            for sensor in sensors:
                candidates = detect_clusters(
                    frames[sensor.label],
                    sensor.static_mask,
                    sensor.dynamic_mask,
                    config.detection.trigger_threshold,
                    frame_index,
                    config.detection.max_cluster_size,
                    simulated_coords.get(sensor.label),
                    sensor.label,
                )
                counters["sensor_candidates"] += len(candidates)
                sensor.pending = [(candidate, frames[sensor.label].copy()) for candidate in candidates]

            update_dynamic_windows(sensors, config)

            now = time.monotonic()
            if now - last_status >= config.output.status_interval_seconds:
                status = build_status(counters, start, sensors, config, recent_events)
                if should_print_status(status, config):
                    print_status(status)
                write_json(paths["snapshots"] / "latest_status.json", status)
                last_status = now
    except KeyboardInterrupt:
        print("\nMINT stopped.")
    except SafetyShutdown as exc:
        print(f"\n{exc}")
        if exc.status:
            write_json(paths["snapshots"] / "latest_status.json", exc.status)
        return 3
    finally:
        for sensor in sensors:
            sensor.capture.release()
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MINT candidate particle event detector.")
    parser.add_argument("--config", help="Path to JSON or YAML config file.")
    parser.add_argument("--simulate", action="store_true", help="Inject simulated events into captured frames.")
    parser.add_argument("--threshold", type=int, help="Override trigger threshold.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument(
        "--write-calibration-template",
        help="Write a detector_calibration.json template and exit.",
    )
    return parser.parse_args(argv)


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    if args.simulate:
        config.simulation.enabled = True
    if args.threshold is not None:
        config.detection.trigger_threshold = args.threshold
    if args.output_dir:
        config.output.dir = args.output_dir


def setup_sensors(cv2, config: AppConfig, output_dir: Path, camera_configs: list[CameraConfig]) -> list[SensorRuntime]:
    sensors: list[SensorRuntime] = []
    for camera_config in camera_configs:
        capture = open_capture(cv2, camera_config)
        camera_settings = read_camera_settings(cv2, capture, camera_config)
        print(f"Camera settings for {camera_config.label}: {camera_settings}")
        calibration_frames = capture_calibration_frames(capture, cv2, config, camera_config.label)
        static_mask, calibration = calibrate(
            calibration_frames,
            config.detection.hot_pixel_threshold,
            config.detection.hot_pixel_fraction,
        )
        dynamic_mask = np.zeros_like(static_mask, dtype=bool)
        recurring_tracker = None
        if config.detection.recurring_coordinate_mask_enabled:
            recurring_tracker = RecurringCoordinateTracker(
                repeat_threshold=config.detection.recurring_coordinate_repeat_threshold,
                window_frames=config.detection.recurring_coordinate_window_frames,
                mask_radius_pixels=config.detection.recurring_coordinate_mask_radius_pixels,
            )
        calibration_record = {
            "timestamp": utc_timestamp_ms(),
            "sensor_label": camera_config.label,
            "calibration": calibration,
            "camera_settings": camera_settings,
            "config": asdict(config),
        }
        write_json(output_dir / f"calibration_summary_{camera_config.label}.json", calibration_record)
        print(f"Calibration complete for {camera_config.label}: {asdict(calibration)}")
        sensors.append(
            SensorRuntime(
                camera_config=camera_config,
                capture=capture,
                camera_settings=camera_settings,
                static_mask=static_mask,
                dynamic_mask=dynamic_mask,
                calibration=calibration,
                calibration_record=calibration_record,
                recurring_tracker=recurring_tracker,
            )
        )
    if len(sensors) == 1:
        write_json(output_dir / "calibration_summary.json", sensors[0].calibration_record)
    return sensors


def read_sensor_frames(cv2, config: AppConfig, sensors: list[SensorRuntime]) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for sensor in sensors:
        ok, frame = sensor.capture.read()
        if not ok:
            print(f"Camera read failed for {sensor.label}; attempting reconnect...")
            sensor.capture.release()
            time.sleep(2)
            sensor.capture = open_capture(cv2, sensor.camera_config)
            sensor.camera_settings = read_camera_settings(cv2, sensor.capture, sensor.camera_config)
            ok, frame = sensor.capture.read()
            if not ok:
                raise RuntimeError(f"Camera read failed after reconnect for {sensor.label}.")
        frames[sensor.label] = to_gray(cv2, frame)
    return frames


def verify_pending_events(
    sensor: SensorRuntime,
    gray: np.ndarray,
    crops_dir: Path,
    config: AppConfig,
    counters: dict,
    start: float,
) -> list[CandidateEvent]:
    del crops_dir, start
    verified: list[CandidateEvent] = []
    for candidate, _candidate_frame in sensor.pending:
        if verify_candidate(candidate, gray, config.detection.trigger_threshold):
            counters["verified_sensor_events"] += 1
            verified.append(candidate)
        else:
            additions = apply_dynamic_mask(sensor.dynamic_mask, candidate)
            sensor.dynamic_window_additions += additions
            counters["dynamic_mask_additions"] += additions
            counters["persistent_dropped"] += 1
    return verified


def maybe_drop_recurring_candidate(
    sensor: SensorRuntime,
    candidate: CandidateEvent,
    config: AppConfig,
    counters: dict,
) -> bool:
    if not config.detection.recurring_coordinate_mask_enabled or sensor.recurring_tracker is None:
        return False

    additions = sensor.recurring_tracker.observe_and_mask(sensor.dynamic_mask, candidate)
    if additions <= 0:
        return False

    sensor.dynamic_window_additions += additions
    counters["dynamic_mask_additions"] += additions
    counters["recurring_dropped"] += 1
    return True


def get_candidate_frame(sensor: SensorRuntime, candidate: CandidateEvent) -> np.ndarray:
    for pending_candidate, frame in sensor.pending:
        if pending_candidate is candidate:
            return frame
    raise RuntimeError(f"Missing candidate frame for {candidate.sensor_label} frame {candidate.frame_index}.")


def build_sensor_event_record(
    sensor: SensorRuntime,
    candidate: CandidateEvent,
    candidate_frame: np.ndarray,
    crops_dir: Path,
    config: AppConfig,
    counters: dict,
    start: float,
    event_type: str = "single_sensor_candidate",
) -> dict:
    crop_path = None
    if config.output.save_crops:
        crop = crop_around(candidate_frame, candidate.bbox, config.output.crop_radius)
        crop_path = save_crop(None, crops_dir, crop, candidate)

    elapsed_minutes = max(0.001, (time.monotonic() - start) / 60)
    event_rate_per_minute = counters["verified_sensor_events"] / elapsed_minutes
    dynamic_additions_per_minute = sensor.dynamic_window_additions / max(
        0.001,
        (time.monotonic() - sensor.dynamic_window_start) / 60,
    )
    score_record = None
    if config.scoring.enabled:
        score_record = score_event(
            candidate=candidate,
            threshold=config.detection.trigger_threshold,
            max_cluster_size=config.detection.max_cluster_size,
            static_mask=sensor.static_mask,
            dynamic_mask=sensor.dynamic_mask,
            calibration=sensor.calibration,
            event_rate_per_minute=event_rate_per_minute,
            dynamic_mask_additions_per_minute=dynamic_additions_per_minute,
            mask_isolation_radius=config.scoring.mask_isolation_radius,
            expected_max_events_per_minute=config.scoring.expected_max_events_per_minute,
            noisy_events_per_minute=config.scoring.noisy_events_per_minute,
            high_confidence_threshold=config.scoring.high_confidence_threshold,
            medium_confidence_threshold=config.scoring.medium_confidence_threshold,
        ).to_record()
    return candidate.to_record(
        static_mask_count=int(sensor.static_mask.sum()),
        dynamic_mask_count=int(sensor.dynamic_mask.sum()),
        camera_settings=sensor.camera_settings,
        crop_path=crop_path,
        score=score_record,
        event_type=event_type,
    )


def log_verification_results(
    sensors: list[SensorRuntime],
    verified_by_sensor: dict[str, list[CandidateEvent]],
    records_by_event_id: dict[int, dict],
    config: AppConfig,
    event_log: Path,
    counters: dict,
    detector_calibration,
    verification_history: dict[str, list[tuple[CandidateEvent, dict]]] | None = None,
    verified_event_keys: set[tuple[str, int, float, float]] | None = None,
    logged_near_miss_pairs: set[tuple[str, int, str, int]] | None = None,
) -> list[dict]:
    primary = sensors[0]
    secondary = sensors[1]
    matches, unmatched_primary, unmatched_secondary = match_verified_tracks(
        verified_by_sensor.get(primary.label, []),
        verified_by_sensor.get(secondary.label, []),
        config.verification.max_frame_delta,
        config.verification.max_centroid_distance_pixels,
    )
    written: list[dict] = []
    camera_settings = {
        primary.label: primary.camera_settings,
        secondary.label: secondary.camera_settings,
    }
    match_records: list[tuple[VerifiedTrackMatch, dict, dict]] = [
        (
            match,
            records_by_event_id[id(match.primary)],
            records_by_event_id[id(match.secondary)],
        )
        for match in matches
    ]
    for match in matches:
        if verified_event_keys is not None:
            verified_event_keys.add(event_key(match.primary))
            verified_event_keys.add(event_key(match.secondary))

    if verification_history is not None:
        history_matches = find_verified_history_matches(
            sensors=sensors,
            current_events=unmatched_primary + unmatched_secondary,
            records_by_event_id=records_by_event_id,
            history=verification_history,
            max_frame_delta=config.verification.max_frame_delta,
            max_centroid_distance_pixels=config.verification.max_centroid_distance_pixels,
            verified_event_keys=verified_event_keys,
        )
        matched_current_ids: set[int] = set()
        for match, primary_record, secondary_record in history_matches:
            matched_current_ids.add(id(match.primary))
            matched_current_ids.add(id(match.secondary))
            match_records.append((match, primary_record, secondary_record))
            if verified_event_keys is not None:
                verified_event_keys.add(event_key(match.primary))
                verified_event_keys.add(event_key(match.secondary))
        unmatched_primary = [event for event in unmatched_primary if id(event) not in matched_current_ids]
        unmatched_secondary = [event for event in unmatched_secondary if id(event) not in matched_current_ids]

    for match, primary_record, secondary_record in match_records:
        track = reconstruct_track(match, detector_calibration) if detector_calibration else None
        record = verified_track_record(
            match,
            primary_record,
            secondary_record,
            camera_settings,
            track,
        )
        append_jsonl(event_log, record)
        counters["verified_track_events"] += 1
        written.append(record)

    if config.verification.log_near_miss_events and verification_history is not None:
        near_misses = find_near_misses(
            sensors=sensors,
            current_events=unmatched_primary + unmatched_secondary,
            records_by_event_id=records_by_event_id,
            history=verification_history,
            max_frame_delta=config.verification.near_miss_max_frame_delta,
            max_centroid_distance_pixels=config.verification.near_miss_max_centroid_distance_pixels,
            max_records=config.verification.near_miss_max_records_per_cycle,
            logged_pairs=logged_near_miss_pairs,
        )
        for primary_event, secondary_event, primary_record, secondary_record in near_misses:
            record = near_miss_record(
                primary_event,
                secondary_event,
                primary_record,
                secondary_record,
                camera_settings,
            )
            append_jsonl(event_log, record)
            counters["verification_near_miss_events"] += 1
            written.append(record)

    if config.verification.log_unmatched_sensor_events:
        for candidate in unmatched_primary + unmatched_secondary:
            record = records_by_event_id[id(candidate)]
            record["event_type"] = "unmatched_sensor_candidate"
            append_jsonl(event_log, record)
            counters["unmatched_sensor_events"] += 1
            written.append(record)

    if verification_history is not None:
        update_verification_history(
            verification_history,
            verified_by_sensor,
            records_by_event_id,
            max_history_frames=max(
                config.verification.near_miss_history_frames,
                config.verification.near_miss_max_frame_delta,
                config.verification.max_frame_delta,
            ),
        )
    return written


def find_verified_history_matches(
    sensors: list[SensorRuntime],
    current_events: list[CandidateEvent],
    records_by_event_id: dict[int, dict],
    history: dict[str, list[tuple[CandidateEvent, dict]]],
    max_frame_delta: int,
    max_centroid_distance_pixels: float,
    verified_event_keys: set[tuple[str, int, float, float]] | None,
) -> list[tuple[VerifiedTrackMatch, dict, dict]]:
    if len(sensors) != 2:
        return []
    labels = {sensors[0].label, sensors[1].label}
    candidates: list[tuple[int, float, CandidateEvent, CandidateEvent, dict, dict]] = []
    for event in current_events:
        if verified_event_keys is not None and event_key(event) in verified_event_keys:
            continue
        event_record = records_by_event_id[id(event)]
        other_labels = labels - {event.sensor_label}
        if not other_labels:
            continue
        other_label = next(iter(other_labels))
        for other_event, other_record in history.get(other_label, []):
            if verified_event_keys is not None and event_key(other_event) in verified_event_keys:
                continue
            frame_delta = abs(event.frame_index - other_event.frame_index)
            if frame_delta > max_frame_delta:
                continue
            distance = centroid_distance(event, other_event)
            if distance > max_centroid_distance_pixels:
                continue
            candidates.append((frame_delta, distance, event, other_event, event_record, other_record))

    candidates.sort(key=lambda item: (item[0], item[1]))
    selected: list[tuple[VerifiedTrackMatch, dict, dict]] = []
    used_events: set[tuple[str, int, float, float]] = set()
    for frame_delta, distance, event, other_event, event_record, other_record in candidates:
        current_key = event_key(event)
        other_key = event_key(other_event)
        if current_key in used_events or other_key in used_events:
            continue
        used_events.add(current_key)
        used_events.add(other_key)
        selected.append((VerifiedTrackMatch(event, other_event, frame_delta, distance), event_record, other_record))
    return selected


def find_near_misses(
    sensors: list[SensorRuntime],
    current_events: list[CandidateEvent],
    records_by_event_id: dict[int, dict],
    history: dict[str, list[tuple[CandidateEvent, dict]]],
    max_frame_delta: int,
    max_centroid_distance_pixels: float,
    max_records: int,
    logged_pairs: set[tuple[str, int, str, int]] | None,
) -> list[tuple[CandidateEvent, CandidateEvent, dict, dict]]:
    if len(sensors) != 2 or max_records <= 0:
        return []
    labels = {sensors[0].label, sensors[1].label}
    candidates: list[tuple[int, float, CandidateEvent, CandidateEvent, dict, dict]] = []
    for event in current_events:
        event_record = records_by_event_id[id(event)]
        other_labels = labels - {event.sensor_label}
        if not other_labels:
            continue
        other_label = next(iter(other_labels))
        for other_event, other_record in history.get(other_label, []):
            frame_delta = abs(event.frame_index - other_event.frame_index)
            if frame_delta <= 0 or frame_delta > max_frame_delta:
                continue
            distance = centroid_distance(event, other_event)
            if distance > max_centroid_distance_pixels:
                continue
            pair_key = canonical_event_pair(event, other_event)
            if logged_pairs is not None and pair_key in logged_pairs:
                continue
            candidates.append((frame_delta, distance, event, other_event, event_record, other_record))

    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = []
    for _frame_delta, _distance, event, other_event, event_record, other_record in candidates:
        pair_key = canonical_event_pair(event, other_event)
        if logged_pairs is not None:
            logged_pairs.add(pair_key)
        selected.append((event, other_event, event_record, other_record))
        if len(selected) >= max_records:
            break
    return selected


def update_verification_history(
    history: dict[str, list[tuple[CandidateEvent, dict]]],
    verified_by_sensor: dict[str, list[CandidateEvent]],
    records_by_event_id: dict[int, dict],
    max_history_frames: int,
) -> None:
    latest_frame = 0
    for events in verified_by_sensor.values():
        for event in events:
            latest_frame = max(latest_frame, event.frame_index)
            history.setdefault(event.sensor_label, []).append((event, records_by_event_id[id(event)]))
    if latest_frame <= 0:
        return
    cutoff = latest_frame - max_history_frames
    for label, events in list(history.items()):
        history[label] = [(event, record) for event, record in events if event.frame_index >= cutoff]


def canonical_event_pair(first: CandidateEvent, second: CandidateEvent) -> tuple[str, int, str, int]:
    left = (first.sensor_label, first.frame_index)
    right = (second.sensor_label, second.frame_index)
    if left <= right:
        return left[0], left[1], right[0], right[1]
    return right[0], right[1], left[0], left[1]


def event_key(event: CandidateEvent) -> tuple[str, int, float, float]:
    return (event.sensor_label, event.frame_index, round(event.centroid_x, 3), round(event.centroid_y, 3))


def maybe_inject_simulation(
    frames: dict[str, np.ndarray],
    sensors: list[SensorRuntime],
    injector: SimulationInjector,
    config: AppConfig,
) -> dict[str, set[tuple[int, int]]]:
    simulated: dict[str, set[tuple[int, int]]] = {sensor.label: set() for sensor in sensors}
    if not config.simulation.enabled:
        return simulated

    first = sensors[0]
    coords = injector.maybe_inject(frames[first.label])
    if not coords:
        return simulated
    simulated[first.label] = coords
    if config.verification.enabled:
        for sensor in sensors[1:]:
            apply_simulated_coords(frames[sensor.label], coords, config.simulation.intensity)
            simulated[sensor.label] = set(coords)
    return simulated


def apply_simulated_coords(gray: np.ndarray, coords: set[tuple[int, int]], intensity: int) -> None:
    h, w = gray.shape[:2]
    for y, x in coords:
        if 0 <= y < h and 0 <= x < w:
            gray[y, x] = intensity


def check_safety_or_shutdown(
    frames: dict[str, np.ndarray],
    sensors: list[SensorRuntime],
    config: AppConfig,
    safety_states: dict[str, object],
    counters: dict,
    start: float,
    recent_events: list[dict],
) -> None:
    if not config.safety.enabled:
        return

    unsafe_messages: list[str] = []
    for sensor in sensors:
        state = safety_states[sensor.label]
        dynamic_additions_per_minute = sensor.dynamic_window_additions / max(
            0.001,
            (time.monotonic() - sensor.dynamic_window_start) / 60,
        )
        evaluation = evaluate_frame_safety(
            frames[sensor.label],
            state,
            max_dark_mean=config.safety.max_dark_mean,
            max_dark_std=config.safety.max_dark_std,
            max_bright_pixel_fraction=config.safety.max_bright_pixel_fraction,
            bright_pixel_threshold=config.safety.bright_pixel_threshold,
            max_dynamic_mask_count=config.safety.max_dynamic_mask_count,
            dynamic_mask_count=int(sensor.dynamic_mask.sum()),
            max_dynamic_additions_per_minute=config.safety.max_dynamic_additions_per_minute,
            dynamic_additions_per_minute=dynamic_additions_per_minute,
        )
        if evaluation.safe:
            state.consecutive_unsafe_frames = 0
            continue

        state.consecutive_unsafe_frames += 1
        unsafe_messages.append(
            f"{sensor.label}: {'; '.join(evaluation.reasons)} "
            f"({state.consecutive_unsafe_frames}/{config.safety.consecutive_unsafe_frames} unsafe frames)"
        )

    if not unsafe_messages:
        return

    message = "Safety monitor warning: possible overheat/noise/light-leak drift detected: " + " | ".join(unsafe_messages)
    print(message)
    if not config.safety.shutdown_on_unsafe:
        return

    tripped = [
        state
        for state in safety_states.values()
        if state.consecutive_unsafe_frames >= config.safety.consecutive_unsafe_frames
    ]
    if not tripped:
        return

    status = build_status(counters, start, sensors, config, recent_events)
    status["shutdown"] = {
        "reason": "overheat_potential_detected",
        "message": message,
        "tripped_sensors": [state.label for state in tripped],
    }
    raise SafetyShutdown(f"Overheat potential detected, shutting down. {message}", status)


def update_dynamic_windows(sensors: list[SensorRuntime], config: AppConfig) -> None:
    now = time.monotonic()
    for sensor in sensors:
        if now - sensor.dynamic_window_start >= 60:
            sensor.dynamic_window_start = now
            sensor.dynamic_window_additions = 0
        if sensor.dynamic_window_additions > config.detection.dynamic_mask_max_additions_per_minute:
            print(
                f"Warning: dynamic mask for {sensor.label} is growing quickly. "
                "Check for light leaks, heat drift, or camera setting changes."
            )
            sensor.dynamic_window_additions = 0


def open_capture(cv2, camera_config: CameraConfig):
    capture = cv2.VideoCapture(camera_config.index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(camera_config.index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_config.index}.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.height)
    capture.set(cv2.CAP_PROP_FPS, camera_config.fps)
    if camera_config.pixel_format:
        fourcc = cv2.VideoWriter_fourcc(*camera_config.pixel_format[:4])
        capture.set(cv2.CAP_PROP_FOURCC, fourcc)
    if camera_config.lock_auto_controls:
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        capture.set(cv2.CAP_PROP_AUTO_WB, 0)
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return capture


def read_camera_settings(cv2, capture, camera_config: CameraConfig) -> dict:
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> 8 * i) & 0xFF) for i in range(4)).strip()
    return {
        "requested": asdict(camera_config),
        "actual_width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        "actual_height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "actual_fps": capture.get(cv2.CAP_PROP_FPS),
        "actual_fourcc": fourcc,
        "exposure": capture.get(cv2.CAP_PROP_EXPOSURE),
        "gain": capture.get(cv2.CAP_PROP_GAIN),
    }


def capture_calibration_frames(
    capture,
    cv2,
    config: AppConfig,
    sensor_label: str,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    target = config.detection.calibration_frames
    print(f"[{local_timestamp_ms()}] Calibrating {sensor_label} with {target} dark frames...")
    while len(frames) < target:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Camera read failed during calibration for {sensor_label}.")
        frames.append(to_gray(cv2, frame))
        if len(frames) % 50 == 0:
            print(f"[{local_timestamp_ms()}] {sensor_label} calibration frames: {len(frames)}/{target}")
    return frames


def to_gray(cv2, frame) -> np.ndarray:
    if len(frame.shape) == 2:
        return frame.astype(np.uint8, copy=False)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def save_crop(_cv2, crops_dir: Path, crop: np.ndarray, candidate: CandidateEvent) -> str:
    import cv2  # type: ignore

    stamp = candidate.timestamp.replace(":", "").replace(".", "")
    sim = "sim" if candidate.simulated else "real"
    path = crops_dir / f"{stamp}_{candidate.sensor_label}_frame{candidate.frame_index}_{sim}.png"
    cv2.imwrite(str(path), crop)
    return str(path)


def build_status(
    counters: dict,
    start: float,
    sensors: list[SensorRuntime],
    config: AppConfig,
    recent_events: list[dict],
) -> dict:
    elapsed = max(0.001, time.monotonic() - start)
    return {
        "timestamp": utc_timestamp_ms(),
        "mode": "two-sensor verification" if config.verification.enabled else "single sensor",
        "runtime_seconds": elapsed,
        "fps_average": counters["frames"] / elapsed,
        "counters": counters,
        "sensors": {
            sensor.label: {
                "static_mask_count": int(sensor.static_mask.sum()),
                "dynamic_mask_count": int(sensor.dynamic_mask.sum()),
                "recurring_coordinate_mask_enabled": sensor.recurring_tracker is not None,
                "camera_settings": sensor.camera_settings,
                "calibration": sensor.calibration,
            }
            for sensor in sensors
        },
        "recent_events": recent_events,
    }


def print_status(status: dict) -> None:
    counters = status["counters"]
    latest = ""
    if status["recent_events"]:
        event = status["recent_events"][-1]
        if event.get("event_type") == "verified_track_candidate":
            latest = " latest=verified_track"
        else:
            score = event.get("score") or {}
            latest = (
                f" latest={score.get('confidence_class', 'n/a')}"
                f"/{score.get('candidate_quality_score', 'n/a')}"
            )
    mask_summary = ",".join(
        f"{label}:{data['static_mask_count']}+{data['dynamic_mask_count']}"
        for label, data in status["sensors"].items()
    )
    print(
        f"[{local_timestamp_ms()}] Status: "
        f"mode={status['mode']} "
        f"frames={counters['frames']} "
        f"fps={status['fps_average']:.2f} "
        f"candidates={counters['sensor_candidates']} "
        f"sensor_verified={counters['verified_sensor_events']} "
        f"verified_tracks={counters['verified_track_events']} "
        f"near_misses={counters.get('verification_near_miss_events', 0)} "
        f"unmatched={counters['unmatched_sensor_events']} "
        f"persistent={counters['persistent_dropped']} "
        f"recurring={counters.get('recurring_dropped', 0)} "
        f"masks={mask_summary}"
        f"{latest}"
    )


def should_print_status(status: dict, config: AppConfig) -> bool:
    """Return True when a periodic status line is worth appending to stdout logs.

    MINT always rewrites snapshots/latest_status.json. This only suppresses long-running
    zero-candidate stdout/status-log growth after a configurable troubleshooting window.
    """
    ttl = config.output.zero_candidate_status_ttl_seconds
    if ttl is None or ttl < 0:
        return True
    if status["runtime_seconds"] <= ttl:
        return True
    counters = status["counters"]
    candidate_keys = (
        "sensor_candidates",
        "verified_sensor_events",
        "verified_track_events",
        "unmatched_sensor_events",
        "persistent_dropped",
        "dynamic_mask_additions",
        "recurring_dropped",
    )
    return any(counters.get(key, 0) > 0 for key in candidate_keys)


def local_timestamp_ms() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"


if __name__ == "__main__":
    raise SystemExit(main())
