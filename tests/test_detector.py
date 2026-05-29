import numpy as np

from orbit_ray.detector import (
    apply_dynamic_mask,
    calibrate,
    crop_around,
    detect_clusters,
    verify_candidate,
)
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
