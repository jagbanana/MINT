from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from .config import AppConfig, load_config
from .detector import (
    CandidateEvent,
    apply_dynamic_mask,
    calibrate,
    crop_around,
    detect_clusters,
    utc_timestamp_ms,
    verify_candidate,
)
from .logging_io import append_jsonl, ensure_output_dirs, write_json
from .scoring import score_event
from .simulator import SimulationInjector


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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

    capture = open_capture(cv2, config)
    camera_settings = read_camera_settings(cv2, capture, config)
    print("Orbit-Ray started")
    print(f"Camera settings: {camera_settings}")

    calibration_frames = capture_calibration_frames(capture, cv2, config)
    static_mask, calibration = calibrate(
        calibration_frames,
        config.detection.hot_pixel_threshold,
        config.detection.hot_pixel_fraction,
    )
    dynamic_mask = np.zeros_like(static_mask, dtype=bool)
    calibration_record = {
        "timestamp": utc_timestamp_ms(),
        "calibration": calibration,
        "camera_settings": camera_settings,
        "config": asdict(config),
    }
    write_json(paths["root"] / "calibration_summary.json", calibration_record)
    print(f"Calibration complete: {asdict(calibration)}")

    injector = SimulationInjector(
        interval_seconds=config.simulation.interval_seconds,
        intensity=config.simulation.intensity,
        max_cluster_pixels=config.simulation.max_cluster_pixels,
    )

    counters = {
        "frames": 0,
        "candidates": 0,
        "verified": 0,
        "persistent_dropped": 0,
        "dynamic_mask_additions": 0,
    }
    recent_events: list[dict] = []
    pending: list[tuple[CandidateEvent, np.ndarray]] = []
    start = time.monotonic()
    last_status = start
    dynamic_window_start = start
    dynamic_window_additions = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera read failed; attempting reconnect...")
                capture.release()
                time.sleep(2)
                capture = open_capture(cv2, config)
                continue

            gray = to_gray(cv2, frame)
            counters["frames"] += 1
            frame_index = counters["frames"]

            for candidate, candidate_frame in pending:
                if verify_candidate(candidate, gray, config.detection.trigger_threshold):
                    crop_path = None
                    if config.output.save_crops:
                        crop = crop_around(candidate_frame, candidate.bbox, config.output.crop_radius)
                        crop_path = save_crop(cv2, paths["crops"], crop, candidate)
                    elapsed_minutes = max(0.001, (time.monotonic() - start) / 60)
                    event_rate_per_minute = counters["verified"] / elapsed_minutes
                    dynamic_additions_per_minute = dynamic_window_additions / max(
                        0.001,
                        (time.monotonic() - dynamic_window_start) / 60,
                    )
                    score_record = None
                    if config.scoring.enabled:
                        score_record = score_event(
                            candidate=candidate,
                            threshold=config.detection.trigger_threshold,
                            max_cluster_size=config.detection.max_cluster_size,
                            static_mask=static_mask,
                            dynamic_mask=dynamic_mask,
                            calibration=calibration,
                            event_rate_per_minute=event_rate_per_minute,
                            dynamic_mask_additions_per_minute=dynamic_additions_per_minute,
                            mask_isolation_radius=config.scoring.mask_isolation_radius,
                            expected_max_events_per_minute=config.scoring.expected_max_events_per_minute,
                            noisy_events_per_minute=config.scoring.noisy_events_per_minute,
                            high_confidence_threshold=config.scoring.high_confidence_threshold,
                            medium_confidence_threshold=config.scoring.medium_confidence_threshold,
                        ).to_record()
                    record = candidate.to_record(
                        static_mask_count=int(static_mask.sum()),
                        dynamic_mask_count=int(dynamic_mask.sum()),
                        camera_settings=camera_settings,
                        crop_path=crop_path,
                        score=score_record,
                    )
                    append_jsonl(event_log, record)
                    counters["verified"] += 1
                    recent_events = (recent_events + [record])[-5:]
                else:
                    additions = apply_dynamic_mask(dynamic_mask, candidate)
                    dynamic_window_additions += additions
                    counters["dynamic_mask_additions"] += additions
                    counters["persistent_dropped"] += 1
            pending = []

            simulated_coords: set[tuple[int, int]] = set()
            if config.simulation.enabled:
                simulated_coords = injector.maybe_inject(gray)

            candidates = detect_clusters(
                gray,
                static_mask,
                dynamic_mask,
                config.detection.trigger_threshold,
                frame_index,
                config.detection.max_cluster_size,
                simulated_coords,
            )
            counters["candidates"] += len(candidates)
            pending = [(candidate, gray.copy()) for candidate in candidates]

            now = time.monotonic()
            if now - dynamic_window_start >= 60:
                dynamic_window_start = now
                dynamic_window_additions = 0
            if dynamic_window_additions > config.detection.dynamic_mask_max_additions_per_minute:
                print("Warning: dynamic mask is growing quickly. Check for light leaks or camera setting changes.")
                dynamic_window_additions = 0

            if now - last_status >= config.output.status_interval_seconds:
                status = build_status(
                    counters,
                    start,
                    static_mask,
                    dynamic_mask,
                    camera_settings,
                    calibration_record,
                    recent_events,
                )
                print_status(status)
                write_json(paths["snapshots"] / "latest_status.json", status)
                last_status = now
    except KeyboardInterrupt:
        print("\nOrbit-Ray stopped.")
    finally:
        capture.release()
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orbit-Ray candidate particle event detector.")
    parser.add_argument("--config", help="Path to JSON or YAML config file.")
    parser.add_argument("--simulate", action="store_true", help="Inject simulated events into captured frames.")
    parser.add_argument("--threshold", type=int, help="Override trigger threshold.")
    parser.add_argument("--output-dir", help="Override output directory.")
    return parser.parse_args(argv)


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    if args.simulate:
        config.simulation.enabled = True
    if args.threshold is not None:
        config.detection.trigger_threshold = args.threshold
    if args.output_dir:
        config.output.dir = args.output_dir


def open_capture(cv2, config: AppConfig):
    capture = cv2.VideoCapture(config.camera.index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(config.camera.index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera index {config.camera.index}.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
    capture.set(cv2.CAP_PROP_FPS, config.camera.fps)
    if config.camera.pixel_format:
        fourcc = cv2.VideoWriter_fourcc(*config.camera.pixel_format[:4])
        capture.set(cv2.CAP_PROP_FOURCC, fourcc)
    if config.camera.lock_auto_controls:
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        capture.set(cv2.CAP_PROP_AUTO_WB, 0)
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return capture


def read_camera_settings(cv2, capture, config: AppConfig) -> dict:
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> 8 * i) & 0xFF) for i in range(4)).strip()
    return {
        "requested": asdict(config.camera),
        "actual_width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        "actual_height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "actual_fps": capture.get(cv2.CAP_PROP_FPS),
        "actual_fourcc": fourcc,
        "exposure": capture.get(cv2.CAP_PROP_EXPOSURE),
        "gain": capture.get(cv2.CAP_PROP_GAIN),
    }


def capture_calibration_frames(capture, cv2, config: AppConfig) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    target = config.detection.calibration_frames
    print(f"[{local_timestamp_ms()}] Calibrating with {target} dark frames...")
    while len(frames) < target:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Camera read failed during calibration.")
        frames.append(to_gray(cv2, frame))
        if len(frames) % 50 == 0:
            print(f"[{local_timestamp_ms()}] Calibration frames: {len(frames)}/{target}")
    return frames


def to_gray(cv2, frame) -> np.ndarray:
    if len(frame.shape) == 2:
        return frame.astype(np.uint8, copy=False)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def save_crop(cv2, crops_dir: Path, crop: np.ndarray, candidate: CandidateEvent) -> str:
    stamp = candidate.timestamp.replace(":", "").replace(".", "")
    sim = "sim" if candidate.simulated else "real"
    path = crops_dir / f"{stamp}_frame{candidate.frame_index}_{sim}.png"
    cv2.imwrite(str(path), crop)
    return str(path)


def build_status(
    counters: dict,
    start: float,
    static_mask: np.ndarray,
    dynamic_mask: np.ndarray,
    camera_settings: dict,
    calibration_record: dict,
    recent_events: list[dict],
) -> dict:
    elapsed = max(0.001, time.monotonic() - start)
    return {
        "timestamp": utc_timestamp_ms(),
        "runtime_seconds": elapsed,
        "fps_average": counters["frames"] / elapsed,
        "counters": counters,
        "static_mask_count": int(static_mask.sum()),
        "dynamic_mask_count": int(dynamic_mask.sum()),
        "camera_settings": camera_settings,
        "calibration": calibration_record["calibration"],
        "recent_events": recent_events,
    }


def print_status(status: dict) -> None:
    counters = status["counters"]
    latest_score = ""
    if status["recent_events"]:
        score = status["recent_events"][-1].get("score") or {}
        latest_score = (
            f" latest={score.get('confidence_class', 'n/a')}"
            f"/{score.get('candidate_quality_score', 'n/a')}"
        )
    print(
        f"[{local_timestamp_ms()}] Status: "
        f"frames={counters['frames']} "
        f"fps={status['fps_average']:.2f} "
        f"candidates={counters['candidates']} "
        f"verified={counters['verified']} "
        f"persistent={counters['persistent_dropped']} "
        f"masks={status['static_mask_count']}+{status['dynamic_mask_count']}"
        f"{latest_score}"
    )


def local_timestamp_ms() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"


if __name__ == "__main__":
    raise SystemExit(main())
