#!/usr/bin/env python3
"""Convert MINT raw .npy frames to FITS files.

This intentionally preserves the original .npy files and writes .fits copies
next to them by default. It uses only NumPy and a small standards-compliant
primary-HDU FITS writer, so it does not require astropy/fitsio.

Examples:
  scripts/npy_to_fits.py orbit_ray_output/two_cam/verified_track_raws
  scripts/npy_to_fits.py frame.npy --output-dir fits
  scripts/npy_to_fits.py frame.npy --overwrite
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

FITS_BLOCK_SIZE = 2880


def _fits_card(keyword: str, value: object | None = None, comment: str | None = None) -> bytes:
    """Return one 80-byte FITS header card."""
    keyword = keyword.upper()[:8]
    if value is None:
        text = keyword
    else:
        if isinstance(value, bool):
            value_text = "T" if value else "F"
            value_field = f"{value_text:>20}"
        elif isinstance(value, int):
            value_field = f"{value:>20d}"
        elif isinstance(value, float):
            value_field = f"{value:>20.10G}"
        else:
            escaped = str(value).replace("'", "''")
            # FITS string values are single-quoted. Keep room for the comment.
            value_field = f"'{escaped[:68]}'"
        text = f"{keyword:<8}= {value_field}"
        if comment:
            text += f" / {comment}"
    return text[:80].ljust(80).encode("ascii", errors="replace")


def _pad_block(data: bytes, pad_byte: bytes = b" ") -> bytes:
    remainder = len(data) % FITS_BLOCK_SIZE
    if not remainder:
        return data
    return data + pad_byte * (FITS_BLOCK_SIZE - remainder)


def _prepare_fits_data(array: np.ndarray) -> tuple[np.ndarray, int]:
    """Return FITS-compatible data array and BITPIX."""
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D image array, got shape {array.shape}")

    # MINT raw frames are uint8. Keep those lossless and simple for PixInsight.
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array), 8

    # FITS requires big-endian byte order for multi-byte numeric data.
    if array.dtype == np.int16:
        return np.ascontiguousarray(array.astype(">i2", copy=False)), 16
    if array.dtype == np.int32:
        return np.ascontiguousarray(array.astype(">i4", copy=False)), 32
    if array.dtype == np.float32:
        return np.ascontiguousarray(array.astype(">f4", copy=False)), -32
    if array.dtype == np.float64:
        return np.ascontiguousarray(array.astype(">f8", copy=False)), -64

    # Conservative fallback: store unsupported integer/boolean types as float32.
    return np.ascontiguousarray(array.astype(">f4")), -32


def write_fits(npy_path: Path, fits_path: Path, *, overwrite: bool = False) -> None:
    if fits_path.exists() and not overwrite:
        raise FileExistsError(f"exists: {fits_path} (use --overwrite to replace)")

    array = np.load(npy_path)
    fits_array, bitpix = _prepare_fits_data(array)
    rows, cols = fits_array.shape

    cards = [
        _fits_card("SIMPLE", True, "conforms to FITS standard"),
        _fits_card("BITPIX", bitpix, "array data type"),
        _fits_card("NAXIS", 2, "number of array dimensions"),
        _fits_card("NAXIS1", cols, "columns"),
        _fits_card("NAXIS2", rows, "rows"),
        _fits_card("EXTEND", True),
        _fits_card("DATE", datetime.now(timezone.utc).isoformat(timespec="seconds"), "UTC created"),
        _fits_card("ORIGIN", "MINT"),
        _fits_card("BUNIT", "ADU"),
        _fits_card("SRCFILE", npy_path.name, "source NPY filename"),
        _fits_card("COMMENT", "Converted from MINT raw .npy frame; original file preserved."),
        b"END".ljust(80, b" "),
    ]
    header = _pad_block(b"".join(cards), b" ")
    data = _pad_block(fits_array.tobytes(order="C"), b"\0")

    fits_path.parent.mkdir(parents=True, exist_ok=True)
    with fits_path.open("wb") as f:
        f.write(header)
        f.write(data)


def iter_npy_inputs(paths: Iterable[Path], *, recursive: bool) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            pattern = "**/*.npy" if recursive else "*.npy"
            yield from sorted(path.glob(pattern))
        elif path.suffix.lower() == ".npy":
            yield path
        else:
            raise ValueError(f"not a .npy file or directory: {path}")


def output_path_for(npy_path: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return npy_path.with_suffix(".fits")
    return output_dir / f"{npy_path.stem}.fits"


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert MINT .npy raw frames to FITS copies.")
    parser.add_argument("paths", nargs="+", type=Path, help=".npy files or directories containing .npy files")
    parser.add_argument("--output-dir", type=Path, help="write FITS files here instead of next to each .npy")
    parser.add_argument("--recursive", action="store_true", help="recurse when an input path is a directory")
    parser.add_argument("--overwrite", action="store_true", help="replace existing .fits files")
    args = parser.parse_args()

    converted = 0
    for npy_path in iter_npy_inputs(args.paths, recursive=args.recursive):
        fits_path = output_path_for(npy_path, args.output_dir)
        write_fits(npy_path, fits_path, overwrite=args.overwrite)
        print(f"{npy_path} -> {fits_path}")
        converted += 1

    print(f"Converted {converted} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
