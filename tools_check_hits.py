#!/usr/bin/env python3
"""Summarize current MINT run status and candidate/event hits."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="orbit_ray_output/two_cam")
parser.add_argument("--last", type=int, default=5)
args = parser.parse_args()

root = Path(args.output_dir)
status_path = root / "snapshots" / "latest_status.json"
event_path = root / "cosmic_events.jsonl"

status = None
run_started_at = None
status_time = None
if status_path.exists():
    status = json.loads(status_path.read_text())
    counters = status.get("counters", {})
    status_timestamp = status.get("timestamp")
    runtime_seconds = status.get("runtime_seconds", 0)
    if status_timestamp:
        try:
            status_time = datetime.fromisoformat(status_timestamp.replace("Z", "+00:00"))
            run_started_at = status_time - timedelta(
                seconds=runtime_seconds
            )
        except ValueError:
            run_started_at = None
    print("STATUS")
    print(f"  timestamp: {status_timestamp}")
    if run_started_at:
        print(f"  estimated_run_started_at: {run_started_at.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')}")
    print(f"  runtime_minutes: {runtime_seconds / 60:.1f}")
    print(f"  fps: {status.get('fps_average', 0):.2f}")
    print(f"  sensor_candidates: {counters.get('sensor_candidates', 0)}")
    print(f"  verified_sensor_events: {counters.get('verified_sensor_events', 0)}")
    print(f"  verified_track_events: {counters.get('verified_track_events', 0)}")
    print(f"  unmatched_sensor_events: {counters.get('unmatched_sensor_events', 0)}")
    print(f"  persistent_dropped: {counters.get('persistent_dropped', 0)}")
    print(f"  dynamic_mask_additions: {counters.get('dynamic_mask_additions', 0)}")
    print(f"  recurring_dropped: {counters.get('recurring_dropped', 0)}")
    shutdown = status.get("shutdown")
    if shutdown:
        print(f"  SHUTDOWN: {shutdown.get('reason')} - {shutdown.get('message')}")
else:
    print(f"No status file found: {status_path}")

print("\nEVENT LOG")
if not event_path.exists():
    print(f"  no event log yet: {event_path}")
    raise SystemExit(0)

records = []
with event_path.open(encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass

print(f"  total_records: {len(records)}")
counts = Counter(record.get("event_type", "unknown") for record in records)
for event_type, count in sorted(counts.items()):
    print(f"  {event_type}: {count}")

if run_started_at:
    current_run_records = []
    current_snapshot_records = []
    newer_than_status = 0
    for record in records:
        timestamp = record.get("timestamp")
        if not timestamp:
            continue
        try:
            record_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            continue
        if record_time >= run_started_at:
            current_run_records.append(record)
            if status_time and record_time <= status_time:
                current_snapshot_records.append(record)
            elif status_time and record_time > status_time:
                newer_than_status += 1
    current_counts = Counter(record.get("event_type", "unknown") for record in current_run_records)
    snapshot_counts = Counter(record.get("event_type", "unknown") for record in current_snapshot_records)
    print("\nCURRENT RUN ESTIMATE")
    print(f"  records_since_estimated_start: {len(current_run_records)}")
    for event_type, count in sorted(current_counts.items()):
        print(f"  {event_type}: {count}")
    if status_time:
        print(f"  records_since_start_through_status_snapshot: {len(current_snapshot_records)}")
        for event_type, count in sorted(snapshot_counts.items()):
            print(f"  snapshot_{event_type}: {count}")
        if newer_than_status:
            print(f"  records_newer_than_status_snapshot: {newer_than_status}")
    if len(current_run_records) != len(records):
        print("  note: EVENT LOG totals include earlier records in the same output folder.")

if records:
    print(f"\nLAST {min(args.last, len(records))} EVENT(S)")
    for record in records[-args.last:]:
        centroid = record.get("centroid") or {}
        score = record.get("score") or {}
        track = record.get("track") or {}
        print(
            f"  {record.get('timestamp')} {record.get('event_type')} "
            f"sensor={record.get('sensor_label', 'n/a')} "
            f"peak={record.get('peak_intensity', 'n/a')} "
            f"cluster={record.get('cluster_size', 'n/a')} "
            f"centroid=({centroid.get('x', 'n/a')},{centroid.get('y', 'n/a')}) "
            f"confidence={score.get('confidence_class', record.get('confidence_class', 'n/a'))}"
        )
        if track:
            print(
                f"    track: azimuth={track.get('azimuth_degrees_from_true_north')} "
                f"elevation={track.get('elevation_degrees')}"
            )
