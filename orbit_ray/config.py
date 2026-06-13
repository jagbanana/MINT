from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class CameraConfig:
    index: int = 0
    label: str = "sensor_0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    pixel_format: str = "YUY2"
    lock_auto_controls: bool = True


@dataclass
class DetectionConfig:
    trigger_threshold: int = 200
    hot_pixel_threshold: int = 180
    hot_pixel_fraction: float = 0.05
    calibration_frames: int = 300
    max_cluster_size: int = 25
    dynamic_mask_max_additions_per_minute: int = 500


@dataclass
class ScoringConfig:
    enabled: bool = True
    mask_isolation_radius: int = 2
    expected_max_events_per_minute: float = 5.0
    noisy_events_per_minute: float = 20.0
    high_confidence_threshold: float = 0.75
    medium_confidence_threshold: float = 0.45


@dataclass
class CoincidenceConfig:
    enabled: bool = False
    max_frame_delta: int = 1
    max_centroid_distance_pixels: float = 12.0
    log_unmatched_sensor_events: bool = True


@dataclass
class TrackReconstructionConfig:
    enabled: bool = False
    calibration_file: str = "detector_calibration.json"


@dataclass
class OutputConfig:
    dir: str = "orbit_ray_output"
    event_log: str = "cosmic_events.jsonl"
    save_crops: bool = True
    crop_radius: int = 12
    status_interval_seconds: int = 5
    zero_candidate_status_ttl_seconds: float = 14400.0


@dataclass
class SafetyConfig:
    enabled: bool = True
    shutdown_on_unsafe: bool = True
    consecutive_unsafe_frames: int = 30
    max_dark_mean: float = 10.0
    max_dark_std: float = 20.0
    bright_pixel_threshold: int = 80
    max_bright_pixel_fraction: float = 0.001
    max_dynamic_mask_count: int = 5000
    max_dynamic_additions_per_minute: int = 500


@dataclass
class SimulationConfig:
    enabled: bool = False
    interval_seconds: float = 60.0
    intensity: int = 255
    max_cluster_pixels: int = 3


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    secondary_camera: CameraConfig = field(default_factory=lambda: CameraConfig(index=1, label="sensor_1"))
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    coincidence: CoincidenceConfig = field(default_factory=CoincidenceConfig)
    track_reconstruction: TrackReconstructionConfig = field(default_factory=TrackReconstructionConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None) -> AppConfig:
    config = AppConfig()
    if not path:
        return config

    config_path = Path(path)
    raw = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML. Use JSON or install PyYAML.") from exc
        data = yaml.safe_load(raw) or {}
    else:
        data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("Config root must be an object.")
    return _merge_dataclass(config, data)
