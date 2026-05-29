# MINT

**Muon Ionization Sensor Tracker**

MINT is a nerdy little detector experiment that uses a fully dark-shielded webcam as a sensor for candidate ionizing-particle events. The idea is simple: if the camera is sealed from visible light, then sudden bright pixels or tiny clusters in a dark frame may be interesting. They might be caused by cosmic ray muons, local ionizing radiation, sensor noise, electronics weirdness, or plain old artifacts. MINT tries to separate the fleeting stuff from the boring stuff and logs what survives.

This project does **not** claim that a single webcam can confirm cosmic rays. It identifies candidate transient sensor events and assigns a heuristic quality score. Stronger confirmation would require coincidence detection across two or more independent sensors.

## What It Does

MINT:

* Captures frames from the default webcam with OpenCV.
* Converts frames to 8-bit grayscale dark frames.
* Runs a startup calibration phase to estimate baseline noise and static hot pixels.
* Masks static hot pixels and runtime persistent pixels.
* Detects high-intensity pixel clusters above a configurable threshold.
* Verifies candidates by checking that the same cluster disappears in the next frame.
* Logs verified candidate events to append-safe JSON Lines.
* Saves small grayscale crop images around each event.
* Writes periodic status snapshots for troubleshooting.
* Supports simulation mode so the whole pipeline can be tested without waiting for a real event.
* Scores each event with a heuristic candidate quality score, artifact risk score, and confidence class.

## How It Works

The detector pipeline is intentionally direct:

```text
[Webcam Stream]
        |
        v
[Grayscale Dark Frame]
        |
        v
[Calibration: Noise Stats + Static Hot-Pixel Mask]
        |
        v
[Runtime Masking: Static + Dynamic Masks]
        |
        v
[Threshold Filtering]
        |
        v
[Cluster Extraction]
        |
        v
[Transient Verification: Frame t+1]
        |
        v
[JSONL Event Log + Crop Images + Status Snapshots]
```

During calibration, MINT captures a stack of dark frames and marks pixels that are repeatedly bright as static hot pixels. During capture, it looks for pixels above the trigger threshold, groups contiguous pixels into clusters, and holds each candidate for one frame. If the same coordinates are still bright in frame `t+1`, the candidate is treated as persistent noise and added to the runtime mask. If the signal disappears, it is logged as a verified candidate transient event.

## Event Scoring

Each logged event includes a heuristic score:

* `candidate_quality_score`: 0.0 to 1.0, where higher means cleaner candidate.
* `artifact_risk_score`: 0.0 to 1.0, where higher means more likely artifact.
* `confidence_class`: `low`, `medium`, or `high`.

The score currently considers:

* Signal strength above threshold.
* Cluster size and shape.
* Isolation from existing masked pixels.
* Calibration baseline stability.
* Recent event rate.
* Recent dynamic mask activity.

Treat this as an artifact-rejection quality score, not a cosmic-ray probability.

## Setup

Install Python 3.10+ and dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional YAML config support:

```powershell
pip install PyYAML
```

## Run

Run against the default webcam:

```powershell
python -m orbit_ray.cli
```

Run with simulated injected events:

```powershell
python -m orbit_ray.cli --simulate
```

Run with an explicit config:

```powershell
python -m orbit_ray.cli --config config.example.json
```

If installed as a package, the CLI command is:

```powershell
mint --simulate
```

## Output

By default, MINT writes runtime output to `orbit_ray_output/`:

* `calibration_summary.json`
* `cosmic_events.jsonl`
* `crops/*.png`
* `snapshots/latest_status.json`

Runtime output is ignored by Git.

## Physical Setup

Before collecting real candidate events:

* Fully cover the webcam lens or sensor with opaque material.
* Start MINT only after the camera is dark, because calibration happens first.
* Watch the calibration summary for unusually high mean, standard deviation, or hot-pixel count.
* If the dynamic mask grows quickly, check for light leaks or camera auto-exposure/gain changes.

## Configuration

Start with [config.example.json](config.example.json). Common settings include:

* `camera.width`, `camera.height`, and `camera.fps`
* `detection.trigger_threshold`
* `detection.calibration_frames`
* `output.status_interval_seconds`
* `output.save_crops`
* `simulation.enabled`
* `scoring.enabled`

## Tests

```powershell
pip install pytest
python -m pytest
```

## Roadmap

Likely next steps:

* Timed capture sessions with `--duration`.
* Better offline review tools for JSONL events and crop images.
* Heatmap generation.
* Linux/headless camera support.
* Multi-camera coincidence detection.
* Dedicated astronomy camera support.

## License

MINT is released under the MIT License. See [LICENSE](LICENSE).
