from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from .verification import VerifiedTrackMatch


@dataclass
class SiteCalibration:
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    elevation_m: float | None = None
    location_name: str = ""


@dataclass
class DetectorPose:
    yaw_degrees_from_true_north: float = 0.0
    pitch_degrees: float = 0.0
    roll_degrees: float = 0.0
    top_sensor_label: str = "top"
    bottom_sensor_label: str = "bottom"


@dataclass
class SensorGeometry:
    active_width_px: int = 1280
    active_height_px: int = 800
    pixel_pitch_um: float = 3.0
    sensor_separation_mm: float = 12.0


@dataclass
class SensorAlignment:
    x_offset_px: float = 0.0
    y_offset_px: float = 0.0
    rotation_degrees: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0


@dataclass
class DetectorCalibration:
    version: int = 1
    notes: list[str] = field(default_factory=lambda: [
        "Orient the detector so yaw_degrees_from_true_north describes the top sensor +Y pixel axis.",
        "Use identical stacked sensors for the first pass; alignment can be refined later.",
        "Pixel coordinates are converted to local sky direction for verification candidates only.",
        "Pitch and roll are recorded for the model, but the first reconstruction pass applies yaw only.",
    ])
    site: SiteCalibration = field(default_factory=SiteCalibration)
    pose: DetectorPose = field(default_factory=DetectorPose)
    geometry: SensorGeometry = field(default_factory=SensorGeometry)
    bottom_to_top_alignment: SensorAlignment = field(default_factory=SensorAlignment)


def default_calibration() -> DetectorCalibration:
    return DetectorCalibration()


def write_calibration_template(path: str | Path) -> None:
    calibration = default_calibration()
    Path(path).write_text(json.dumps(asdict(calibration), indent=2) + "\n", encoding="utf-8")


def load_detector_calibration(path: str | Path) -> DetectorCalibration:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _from_dict(DetectorCalibration(), data)


def reconstruct_track(match: VerifiedTrackMatch, calibration: DetectorCalibration) -> dict[str, Any]:
    top_event, bottom_event = _ordered_events(match, calibration)
    top_x_mm, top_y_mm = _pixel_to_sensor_mm(
        top_event.centroid_x,
        top_event.centroid_y,
        calibration.geometry,
    )
    bottom_x, bottom_y = _align_bottom_to_top_pixels(
        bottom_event.centroid_x,
        bottom_event.centroid_y,
        calibration.bottom_to_top_alignment,
    )
    bottom_x_mm, bottom_y_mm = _pixel_to_sensor_mm(bottom_x, bottom_y, calibration.geometry)

    detector_vector = (
        bottom_x_mm - top_x_mm,
        bottom_y_mm - top_y_mm,
        -calibration.geometry.sensor_separation_mm,
    )
    incoming_vector = tuple(-component for component in detector_vector)
    local_vector = _detector_to_local_enu(incoming_vector, calibration.pose)
    azimuth, elevation = _azimuth_elevation(local_vector)

    return {
        "site": asdict(calibration.site),
        "top_sensor_label": top_event.sensor_label,
        "bottom_sensor_label": bottom_event.sensor_label,
        "top_hit_mm": {"x": round(top_x_mm, 4), "y": round(top_y_mm, 4), "z": 0.0},
        "bottom_hit_mm": {
            "x": round(bottom_x_mm, 4),
            "y": round(bottom_y_mm, 4),
            "z": -calibration.geometry.sensor_separation_mm,
        },
        "incoming_local_enu_unit": _unit_vector_record(local_vector),
        "azimuth_degrees_from_true_north": round(azimuth, 3),
        "elevation_degrees": round(elevation, 3),
        "assumptions": [
            "Direction is inferred as top-to-bottom muon travel reversed back into the local sky.",
            "This estimates local arrival direction, not the astrophysical source of the primary cosmic ray.",
        ],
    }


def _ordered_events(match: VerifiedTrackMatch, calibration: DetectorCalibration):
    if match.primary.sensor_label == calibration.pose.top_sensor_label:
        return match.primary, match.secondary
    if match.secondary.sensor_label == calibration.pose.top_sensor_label:
        return match.secondary, match.primary
    return match.primary, match.secondary


def _pixel_to_sensor_mm(x_px: float, y_px: float, geometry: SensorGeometry) -> tuple[float, float]:
    pitch_mm = geometry.pixel_pitch_um / 1000.0
    x_center = (geometry.active_width_px - 1) / 2
    y_center = (geometry.active_height_px - 1) / 2
    return (x_px - x_center) * pitch_mm, (y_center - y_px) * pitch_mm


def _align_bottom_to_top_pixels(x_px: float, y_px: float, alignment: SensorAlignment) -> tuple[float, float]:
    x = x_px * alignment.scale_x + alignment.x_offset_px
    y = y_px * alignment.scale_y + alignment.y_offset_px
    angle = math.radians(alignment.rotation_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


def _detector_to_local_enu(vector: tuple[float, float, float], pose: DetectorPose) -> tuple[float, float, float]:
    x, y, z = vector
    yaw = math.radians(pose.yaw_degrees_from_true_north)
    # Detector +Y points toward yaw; detector +X is 90 degrees clockwise from +Y.
    east = x * math.cos(yaw) + y * math.sin(yaw)
    north = -x * math.sin(yaw) + y * math.cos(yaw)
    up = z
    # Pitch/roll are reserved in the model; for now document rather than apply them.
    return east, north, up


def _azimuth_elevation(local_enu: tuple[float, float, float]) -> tuple[float, float]:
    east, north, up = local_enu
    horizontal = math.hypot(east, north)
    azimuth = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
    elevation = math.degrees(math.atan2(up, horizontal))
    return azimuth, elevation


def _unit_vector_record(vector: tuple[float, float, float]) -> dict[str, float]:
    length = math.sqrt(sum(component * component for component in vector)) or 1.0
    return {
        "east": round(vector[0] / length, 6),
        "north": round(vector[1] / length, 6),
        "up": round(vector[2] / length, 6),
    }


def _from_dict(instance, values: dict):
    for key, value in values.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            setattr(instance, key, _from_dict(current, value))
        else:
            setattr(instance, key, value)
    return instance
