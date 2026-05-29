from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    crops = output_dir / "crops"
    snapshots = output_dir / "snapshots"
    crops.mkdir(exist_ok=True)
    snapshots.mkdir(exist_ok=True)
    return {"root": output_dir, "crops": crops, "snapshots": snapshots}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), separators=(",", ":")) + "\n")
        handle.flush()


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(record), indent=2) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
