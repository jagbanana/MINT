import numpy as np

from orbit_ray.calibration import (
    DetectorCalibration,
    load_detector_calibration,
    reconstruct_track,
    write_calibration_template,
)
from orbit_ray.detector import (
    RecurringCoordinateTracker,
    apply_dynamic_mask,
    calibrate,
    crop_around,
    detect_clusters,
    verify_candidate,
)
from orbit_ray.verification import verified_track_record, match_verified_tracks
from orbit_ray.cli import should_print_status
from orbit_ray.config import load_config
from orbit_ray.safety import SensorSafetyState, evaluate_frame_safety
from orbit_ray.scoring import score_event
from orbit_ray.simulator import SimulationInjector


def test_calibration_masks_persistent_hot_pixel():
    frames = []
    for _ in range(10):
        frame = np.zeros((4, 4), dtype=np.uint8)
        frame[2, 1] = 220
        frames.append(frame)

    mask, summary = calibrate(frames, hot_pixel_threshold=180, hot_pixel_fraction=0.05)

    assert mask[2, 1]
    assert summary.static_hot_pixels == 1
    assert summary.max_intensity == 220


def test_detect_clusters_groups_contiguous_pixels_and_verifies_transient():
    gray = np.zeros((8, 8), dtype=np.uint8)
    gray[3, 3] = 230
    gray[3, 4] = 240
    static = np.zeros_like(gray, dtype=bool)
    dynamic = np.zeros_like(gray, dtype=bool)

    events = detect_clusters(gray, static, dynamic, 200, 42, 10)

    assert len(events) == 1
    assert events[0].cluster_size == 2
    assert events[0].peak_intensity == 240
    assert verify_candidate(events[0], np.zeros_like(gray), 200)


def test_persistent_candidate_updates_dynamic_mask():
    gray = np.zeros((5, 5), dtype=np.uint8)
    gray[1, 1] = 255
    static = np.zeros_like(gray, dtype=bool)
    dynamic = np.zeros_like(gray, dtype=bool)
    event = detect_clusters(gray, static, dynamic, 200, 1, 5)[0]

    assert not verify_candidate(event, gray, 200)
    additions = apply_dynamic_mask(dynamic, event)

    assert additions == 1
    assert dynamic[1, 1]


def test_simulator_injected_event_flows_through_detection():
    gray = np.zeros((12, 12), dtype=np.uint8)
    coords = SimulationInjector(interval_seconds=1, max_cluster_pixels=3).inject(gray)
    static = np.zeros_like(gray, dtype=bool)
    dynamic = np.zeros_like(gray, dtype=bool)

    events = detect_clusters(gray, static, dynamic, 200, 10, 10, coords)

    assert len(events) == 1
    assert events[0].simulated



def test_recurring_coordinate_tracker_masks_flickering_pixel():
    gray = np.zeros((8, 8), dtype=np.uint8)
    gray[3, 4] = 230
    static = np.zeros_like(gray, dtype=bool)
    dynamic = np.zeros_like(gray, dtype=bool)
    tracker = RecurringCoordinateTracker(repeat_threshold=3, window_frames=10, mask_radius_pixels=1)

    additions = []
    for frame_index in (1, 4, 7):
        event = detect_clusters(gray, static, dynamic, 200, frame_index, 10)[0]
        additions.append(tracker.observe_and_mask(dynamic, event))

    assert additions[:2] == [0, 0]
    assert additions[2] == 9
    assert dynamic[3, 4]
    assert dynamic[2, 3]
    assert dynamic[4, 5]
    assert not detect_clusters(gray, static, dynamic, 200, 8, 10)


def test_recurring_coordinate_tracker_respects_frame_window():
    gray = np.zeros((8, 8), dtype=np.uint8)
    gray[3, 4] = 230
    static = np.zeros_like(gray, dtype=bool)
    dynamic = np.zeros_like(gray, dtype=bool)
    tracker = RecurringCoordinateTracker(repeat_threshold=3, window_frames=5, mask_radius_pixels=0)

    for frame_index in (1, 10, 20):
        event = detect_clusters(gray, static, dynamic, 200, frame_index, 10)[0]
        assert tracker.observe_and_mask(dynamic, event) == 0

    assert not dynamic[3, 4]

def test_crop_around_clamps_to_frame_edges():
    gray = np.arange(25, dtype=np.uint8).reshape(5, 5)

    crop = crop_around(gray, (0, 0, 1, 1), radius=2)

    assert crop.shape == (4, 4)


def test_scoring_adds_quality_risk_and_confidence_class():
    frames = [np.zeros((8, 8), dtype=np.uint8) for _ in range(10)]
    static, calibration = calibrate(frames, hot_pixel_threshold=180, hot_pixel_fraction=0.05)
    dynamic = np.zeros_like(static, dtype=bool)
    gray = np.zeros((8, 8), dtype=np.uint8)
    gray[4, 4] = 255
    event = detect_clusters(gray, static, dynamic, 200, 1, 10)[0]

    score = score_event(
        candidate=event,
        threshold=200,
        max_cluster_size=10,
        static_mask=static,
        dynamic_mask=dynamic,
        calibration=calibration,
        event_rate_per_minute=1.0,
        dynamic_mask_additions_per_minute=0.0,
        mask_isolation_radius=2,
        expected_max_events_per_minute=5.0,
        noisy_events_per_minute=20.0,
        high_confidence_threshold=0.75,
        medium_confidence_threshold=0.45,
    )
    record = event.to_record(0, 0, {}, score=score.to_record())

    assert score.candidate_quality_score >= 0.75
    assert score.artifact_risk_score <= 0.25
    assert score.confidence_class == "high"
    assert record["score"]["factors"]["signal_strength"] == 1.0


def test_config_defaults_to_single_sensor_with_secondary_available():
    config = load_config(None)

    assert not config.verification.enabled
    assert config.camera.index == 0
    assert config.secondary_camera.index == 1
    assert config.safety.enabled
    assert config.safety.shutdown_on_unsafe


def test_zero_candidate_status_printing_is_suppressed_after_ttl():
    config = load_config(None)
    config.output.zero_candidate_status_ttl_seconds = 10
    status = {
        "runtime_seconds": 11,
        "counters": {
            "sensor_candidates": 0,
            "verified_sensor_events": 0,
            "verified_track_events": 0,
            "unmatched_sensor_events": 0,
            "persistent_dropped": 0,
            "dynamic_mask_additions": 0,
        },
    }

    assert not should_print_status(status, config)


def test_status_printing_continues_after_ttl_when_candidates_exist():
    config = load_config(None)
    config.output.zero_candidate_status_ttl_seconds = 10
    status = {
        "runtime_seconds": 11,
        "counters": {
            "sensor_candidates": 1,
            "verified_sensor_events": 0,
            "verified_track_events": 0,
            "unmatched_sensor_events": 0,
            "persistent_dropped": 0,
            "dynamic_mask_additions": 0,
        },
    }

    assert should_print_status(status, config)


def test_safety_monitor_flags_bright_noisy_frames():
    gray = np.zeros((20, 20), dtype=np.uint8)
    gray[:, :10] = 120
    state = SensorSafetyState(label="top", baseline_mean=0.0, baseline_std=0.0, baseline_max=0)

    evaluation = evaluate_frame_safety(
        gray,
        state,
        max_dark_mean=10.0,
        max_dark_std=20.0,
        max_bright_pixel_fraction=0.001,
        bright_pixel_threshold=80,
        max_dynamic_mask_count=5000,
        dynamic_mask_count=0,
        max_dynamic_additions_per_minute=500,
        dynamic_additions_per_minute=0.0,
    )

    assert not evaluation.safe
    assert any("dark_mean" in reason for reason in evaluation.reasons)
    assert any("bright_pixel_fraction" in reason for reason in evaluation.reasons)



def test_safety_monitor_accepts_clean_dark_frames():
    gray = np.zeros((20, 20), dtype=np.uint8)
    state = SensorSafetyState(label="bottom", baseline_mean=0.0, baseline_std=0.0, baseline_max=0)

    evaluation = evaluate_frame_safety(
        gray,
        state,
        max_dark_mean=10.0,
        max_dark_std=20.0,
        max_bright_pixel_fraction=0.001,
        bright_pixel_threshold=80,
        max_dynamic_mask_count=5000,
        dynamic_mask_count=0,
        max_dynamic_additions_per_minute=500,
        dynamic_additions_per_minute=0.0,
    )

    assert evaluation.safe
    assert evaluation.reasons == []



def test_verification_matching_pairs_nearby_dual_sensor_events():
    static = np.zeros((8, 8), dtype=bool)
    dynamic = np.zeros_like(static)
    top = np.zeros((8, 8), dtype=np.uint8)
    bottom = np.zeros((8, 8), dtype=np.uint8)
    top[4, 4] = 255
    bottom[4, 5] = 255

    top_event = detect_clusters(top, static, dynamic, 200, 12, 10, sensor_label="top")[0]
    bottom_event = detect_clusters(bottom, static, dynamic, 200, 12, 10, sensor_label="bottom")[0]

    matches, unmatched_top, unmatched_bottom = match_verified_tracks(
        [top_event],
        [bottom_event],
        max_frame_delta=1,
        max_centroid_distance_pixels=2.0,
    )

    assert len(matches) == 1
    assert not unmatched_top
    assert not unmatched_bottom
    record = verified_track_record(matches[0], top_event.to_record(0, 0, {}), bottom_event.to_record(0, 0, {}), {})
    assert record["event_type"] == "verified_track_candidate"
    assert record["confidence_class"] == "verified_track"


def test_calibration_template_round_trips(tmp_path):
    path = tmp_path / "detector_calibration.json"

    write_calibration_template(path)
    calibration = load_detector_calibration(path)

    assert calibration.geometry.sensor_separation_mm == 12.0
    assert calibration.pose.top_sensor_label == "top"


def test_reconstruct_track_returns_local_sky_direction():
    static = np.zeros((8, 8), dtype=bool)
    dynamic = np.zeros_like(static)
    top = np.zeros((8, 8), dtype=np.uint8)
    bottom = np.zeros((8, 8), dtype=np.uint8)
    top[3, 3] = 255
    bottom[4, 3] = 255
    top_event = detect_clusters(top, static, dynamic, 200, 1, 10, sensor_label="top")[0]
    bottom_event = detect_clusters(bottom, static, dynamic, 200, 1, 10, sensor_label="bottom")[0]
    matches, _, _ = match_verified_tracks([top_event], [bottom_event], 1, 5)
    calibration = DetectorCalibration()
    calibration.geometry.active_width_px = 8
    calibration.geometry.active_height_px = 8
    calibration.geometry.pixel_pitch_um = 3
    calibration.geometry.sensor_separation_mm = 12

    track = reconstruct_track(matches[0], calibration)

    assert "azimuth_degrees_from_true_north" in track
    assert track["incoming_local_enu_unit"]["up"] > 0.99
