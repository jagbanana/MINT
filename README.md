# MINT

**Muon Ionization Sensor Tracker**

MINT is a homebrew cosmic-ray telescope experiment built from ordinary camera sensors. It starts as a dark-frame detector for a single webcam, then grows into a two-sensor coincidence detector that can separate one-camera noise from events that pass through two independent layers of silicon.

## Why Cosmic Rays Are Wonderful

Cosmic rays are high-energy particles that arrive at Earth from space. By the time they reach sea level, many of the particles we can realistically detect are **secondary cosmic rays**, especially muons: short-lived, highly penetrating particles created when primary cosmic rays slam into atoms high in the atmosphere.

The eerie part is that the universe is still not done explaining itself. Scientists know some cosmic rays come from violent astrophysical places and processes, including supernova remnants, active galaxies, and other extreme environments. But the highest-energy cosmic rays remain partly mysterious. Something out there is accelerating particles to absurd energies, far beyond what human-built accelerators can reach, and we are still piecing together exactly where and how.

MINT is a tiny way to touch that mystery from a desk, server shelf, garage, or spare laptop. It will not replace a scientific detector array, but it lets you build a little listening post for rare, invisible events passing through ordinary matter all the time.

## What It Does

MINT:

* Captures frames from one or two OpenCV-compatible cameras.
* Converts frames to 8-bit grayscale dark frames.
* Runs startup calibration for each sensor to estimate baseline noise and static hot pixels.
* Masks static hot pixels and runtime persistent pixels.
* Detects high-intensity pixel clusters above a configurable threshold.
* Verifies candidates by checking that the same cluster disappears in the next frame.
* Logs verified candidate events to append-safe JSON Lines.
* Saves small grayscale crop images around each event.
* Writes periodic status snapshots for troubleshooting.
* Supports simulation mode so the whole pipeline can be tested without waiting for rare events.
* Scores each single-sensor event with a heuristic quality score, artifact risk score, and confidence class.
* In two-sensor mode, promotes paired hits to `coincidence_candidate` events.

## Detector Modes

### Single-Sensor Mode

Single-sensor mode is the default. It is great for experimentation, calibration, and learning the noise behavior of your camera.

```powershell
python -m orbit_ray.cli
```

A single dark-covered webcam can identify candidate transient sensor events. It cannot, by itself, confirm that an event was a cosmic ray. Single-camera detections may be real particle interactions, thermal noise, hot pixels, electronics artifacts, or environmental radiation.

### Two-Sensor Coincidence Mode

Two-sensor mode opens two cameras, calibrates them independently, runs the same transient detector on both streams, and then looks for events that appear on both sensors within a configurable frame and centroid-distance window.

This technique is a small version of a **coincidence telescope**. Random thermal noise on one sensor should not happen at nearly the same time and place on a second independent sensor. A penetrating muon, however, can pass through both sensor layers.

Enable it in config:

```json
{
  "coincidence": {
    "enabled": true,
    "max_frame_delta": 1,
    "max_centroid_distance_pixels": 12.0,
    "log_unmatched_sensor_events": true
  },
  "camera": {
    "index": 0,
    "label": "top"
  },
  "secondary_camera": {
    "index": 1,
    "label": "bottom"
  }
}
```

Then run:

```powershell
python -m orbit_ray.cli --config config.example.json
```

In two-sensor mode, JSONL records may include:

* `coincidence_candidate`: paired transient detected on both sensors.
* `unmatched_sensor_candidate`: transient detected on only one sensor.

## How It Works

The single-sensor pipeline:

```text
[Camera Stream]
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

The two-sensor pipeline runs that detector independently for each camera, then adds:

```text
[Verified Top Sensor Transients]      [Verified Bottom Sensor Transients]
                 |                         |
                 v                         v
             [Frame + Centroid Coincidence Matcher]
                             |
                             v
                  [Coincidence Event Log Record]
```

During calibration, MINT captures a stack of dark frames and marks pixels that are repeatedly bright as static hot pixels. During capture, it looks for pixels above the trigger threshold, groups contiguous pixels into clusters, and holds each candidate for one frame. If the same coordinates are still bright in frame `t+1`, the candidate is treated as persistent noise and added to the runtime mask. If the signal disappears, it becomes a verified transient. In two-sensor mode, verified transients from both sensors are compared for coincidence.

## Hardware Recommendation

For a first dedicated two-sensor build, the **InnoMaker U20CAM-9281M / OV9281 Global Shutter Mono USB Camera Module** is a strong starting point. InnoMaker lists it as a 1MP monochrome global-shutter OV9281 UVC module with Windows, Linux, and macOS plug-and-play support, YUY2/MJPG output, and 1280x800/1280x720 modes. It is also small enough to stack mechanically.

At the time of writing, the manufacturer page listed the U20CAM-9281M at about **$33 per module**, and Amazon pricing commonly floats around the low-to-mid `$30s`. Plan on buying two matched modules.

Why this sensor family is attractive:

* **Monochrome:** no Bayer color interpolation, so bright pixels are cleaner luminance events.
* **Global shutter:** the full sensor exposes at once, which makes timing between stacked cameras more meaningful.
* **UVC USB:** should appear as ordinary video devices on Windows and Linux.
* **Small board:** easier to stack with nylon standoffs.
* **Low power:** practical for 24/7 logging if you manage heat.

Product reference: [InnoMaker U20CAM-9281M](https://www.inno-maker.com/product/u20cam-9281m/)

## Two-Sensor Physical Build

Stack the sensors vertically, not side-by-side. You want one high-energy particle to pass through both silicon layers in sequence.

```text
        Incoming muon path
               |
               v
   +-------------------------+
   | Top lightproof cap      |
   | Top OV9281 sensor board |
   +-------------------------+
        10-15 mm spacer
   +-------------------------+
   | Bottom OV9281 board     |
   | Bottom lightproof layer |
   +-------------------------+
```

Build notes:

* Use non-conductive M3 nylon screws, nuts, and 10-15 mm standoffs.
* Align the sensors on the X/Y axes as closely as possible.
* Keep the boards parallel and rigid so vibration does not ruin alignment.
* Route USB cables so they do not pull on the frame.
* Prefer a small lightproof project box over wrapping whole boards in tape.
* Block light at the lens/sensor path, but leave room for heat to escape.

Spacing matters. A very wide separation creates a narrower acceptance angle and may reduce the hit rate dramatically. A tight 10-15 mm separation is a better first build because it gives the code a fighting chance to see coincidences while you are still experimenting.

## Thermal Notes

Heat creates dark current: thermally generated electrons inside the sensor that can look like faint signal. Over long runs, rising temperature can increase noise and hot-pixel behavior.

Practical mitigations:

* Do not wrap the entire board in insulating tape.
* Use a tiny adhesive copper or aluminum heatsink on the back of each camera board if it fits safely.
* Keep airflow around the boards when possible.
* Watch dynamic mask growth and calibration stats.
* Future work should add rolling dark-baseline adjustment for 24/7 deployments.

## Event Scoring

Each single-sensor event includes a heuristic score:

* `candidate_quality_score`: 0.0 to 1.0, where higher means cleaner candidate.
* `artifact_risk_score`: 0.0 to 1.0, where higher means more likely artifact.
* `confidence_class`: `low`, `medium`, or `high`.

The score currently considers signal strength, cluster shape, isolation from masked pixels, calibration stability, recent event rate, and dynamic mask activity.

Treat this as an artifact-rejection quality score, not a cosmic-ray probability. In two-sensor mode, `coincidence_candidate` records are the stronger evidence path.

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

Single camera:

```powershell
python -m orbit_ray.cli
```

Simulated events:

```powershell
python -m orbit_ray.cli --simulate
```

Explicit config:

```powershell
python -m orbit_ray.cli --config config.example.json
```

If installed as a package, the CLI command is:

```powershell
mint --simulate
```

## Output

By default, MINT writes runtime output to `orbit_ray_output/`:

* `calibration_summary.json` for single-sensor compatibility.
* `calibration_summary_<sensor_label>.json` for each sensor.
* `cosmic_events.jsonl`
* `crops/*.png`
* `snapshots/latest_status.json`

Runtime output is ignored by Git.

## Configuration

Start with [config.example.json](config.example.json). Common settings include:

* `camera.index`, `camera.label`, `camera.width`, `camera.height`, and `camera.fps`
* `secondary_camera.index` and `secondary_camera.label`
* `coincidence.enabled`
* `coincidence.max_frame_delta`
* `coincidence.max_centroid_distance_pixels`
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
* Hardware-trigger synchronization for supported cameras.
* Rolling dark-baseline adjustment for temperature drift.
* Dedicated astronomy camera support.

## License

MINT is released under the MIT License. See [LICENSE](LICENSE).
