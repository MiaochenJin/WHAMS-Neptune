#!/usr/bin/env python3
"""
WHAMS Parquet to Mmap Converter

Converts WHAMS parquet files to memory-mapped format for fast training data loading.

Usage:
    python convert_to_mmap.py --input /path/to/WHAMSv1/BDT/21217 --output /path/to/output/whams_train

The converter produces two files:
    - <output>.idx: Event index with metadata
    - <output>.dat: Sensor data

Author: Claude Code
Date: 2026-01-29
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Import from whams_neptune package
from whams_neptune.dataloaders.mmap_format import (
    WHAMS_EVENT_RECORD_DTYPE,
    WHAMS_SENSOR_RECORD_DTYPE,
    RAW_MORPHOLOGY_MAP,
    STARTING_MORPHOLOGIES,
    MMAP_HEADER_SIZE,
    create_header,
    SENSOR_FEATURE_COLUMNS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Column Name Mappings
# =============================================================================

# MC truth columns in WHAMS parquet
MC_COLUMNS = {
    "energy": "mc_primary_energy",
    "dir_x": "mc_primary_dir_x",
    "dir_y": "mc_primary_dir_y",
    "dir_z": "mc_primary_dir_z",
    "morphology": "mc_event_morphology",
}

# Sensor-level columns (nested arrays in parquet)
# These map from our dtype field names to parquet column names
SENSOR_COLUMNS = {
    "x": "pulse_sensor_pos_x",
    "y": "pulse_sensor_pos_y",
    "z": "pulse_sensor_pos_z",
    "t_first": "pulse_summary_t_first",
    "t_last": "pulse_summary_t_last",
    "t20": "pulse_summary_t20",
    "t50": "pulse_summary_t50",
    "t_mean": "pulse_summary_t_mean",
    "t_std": "pulse_summary_t_std",
    "c_total": "pulse_summary_c_total",
    "c_500ns": "pulse_summary_c_500ns",
    "c_100ns": "pulse_summary_c_100ns",
}


# =============================================================================
# Parquet Reading Utilities
# =============================================================================


def get_parquet_files(
    input_paths: List[Path], file_range: Optional[Tuple[int, int]] = None
) -> List[Path]:
    """Get list of parquet files from input directories.

    Args:
        input_paths: List of directories or parquet files
        file_range: Optional (start, end) range of file indices to use

    Returns:
        List of parquet file paths
    """
    all_files = []

    for path in input_paths:
        if path.is_file() and path.suffix == ".parquet":
            all_files.append(path)
        elif path.is_dir():
            files = sorted(path.glob("*.parquet"))
            all_files.extend(files)
        else:
            logger.warning(f"Skipping invalid path: {path}")

    # Sort all files for reproducibility
    all_files = sorted(all_files)

    # Apply file range if specified
    if file_range is not None:
        start, end = file_range
        all_files = all_files[start:end]

    logger.info(f"Found {len(all_files)} parquet files")
    return all_files


def inspect_parquet_schema(file_path: Path) -> Dict[str, Any]:
    """Inspect parquet file schema to understand column structure.

    Args:
        file_path: Path to a parquet file

    Returns:
        Dict with schema information
    """
    schema = pq.read_schema(file_path)

    info = {
        "columns": {},
        "has_nested": False,
    }

    for field in schema:
        field_info = {
            "type": str(field.type),
            "is_list": pa.types.is_list(field.type),
        }
        if pa.types.is_list(field.type):
            info["has_nested"] = True
            field_info["value_type"] = str(field.type.value_type)

        info["columns"][field.name] = field_info

    return info


def read_parquet_batch(
    file_path: Path,
    mc_columns: List[str],
    sensor_columns: List[str],
) -> Tuple[pa.Table, int]:
    """Read a parquet file and extract relevant columns.

    Args:
        file_path: Path to parquet file
        mc_columns: List of MC truth column names to read
        sensor_columns: List of sensor-level column names to read

    Returns:
        Tuple of (PyArrow table, number of events)
    """
    # Read all needed columns
    all_columns = mc_columns + sensor_columns
    table = pq.read_table(file_path, columns=all_columns)
    return table, table.num_rows


# =============================================================================
# Conversion Functions
# =============================================================================


def convert_morphology(morphology_str: str) -> Tuple[int, int, bool]:
    """Convert morphology string to integer codes and starting flag.

    Args:
        morphology_str: Raw morphology string (e.g., "0T1C", "Background")

    Returns:
        Tuple of (simplified_morphology, raw_morphology, is_starting)
    """
    # Map raw morphology
    raw_morph = RAW_MORPHOLOGY_MAP.get(morphology_str, -1)

    # Determine if starting
    is_starting = morphology_str in STARTING_MORPHOLOGIES

    # Simplified morphology (Track=0, Cascade=1, Mixed=2)
    # Track: 0T0C, 1T0C (through-going tracks)
    # Cascade: 0T1C (single cascade)
    # Mixed: everything else with tracks and cascades
    if morphology_str in {"0T0C", "1T0C"}:
        simplified = 0  # Track
    elif morphology_str == "0T1C":
        simplified = 1  # Cascade
    elif morphology_str in {"0TnC", "1T1C", "1TnC", "2TnC"}:
        simplified = 2  # Mixed
    elif morphology_str in {"Background", "BackGround"}:
        simplified = 0  # Treat as track (atmospheric muon backgrounds)
    else:
        simplified = -1  # Unknown

    return simplified, raw_morph, is_starting


def extract_dataset_id(file_path: Path) -> int:
    """Extract dataset ID from file path.

    Assumes path structure like: .../WHAMSv1/BDT/21217/file.parquet

    Args:
        file_path: Path to parquet file

    Returns:
        Dataset ID as integer, or 0 if not found
    """
    parts = file_path.parts
    for part in parts:
        if part.isdigit() and len(part) == 5:  # Dataset IDs are 5 digits
            return int(part)
        # Handle extended datasets like "21316_extended"
        if "_" in part:
            base = part.split("_")[0]
            if base.isdigit() and len(base) == 5:
                return int(base)
    return 0


def process_parquet_file(
    file_path: Path,
    idx_file,
    dat_file,
    sensor_offset: int,
    dataset_id: int,
    starting_only: bool = False,
) -> Tuple[int, int]:
    """Process a single parquet file and write to mmap files.

    Args:
        file_path: Path to parquet file
        idx_file: Open file handle for .idx file
        dat_file: Open file handle for .dat file
        sensor_offset: Current offset into sensor data file
        dataset_id: Dataset ID for this file
        starting_only: If True, only include starting events (0T1C, 0TnC, 1T1C, 1TnC, 2TnC)

    Returns:
        Tuple of (number of events processed, total sensors written)
    """
    # Determine which columns to read
    mc_cols_to_read = list(MC_COLUMNS.values())
    sensor_cols_to_read = list(SENSOR_COLUMNS.values())

    # Check schema to see what columns exist
    schema = pq.read_schema(file_path)
    available_cols = set(schema.names)

    mc_cols_to_read = [c for c in mc_cols_to_read if c in available_cols]
    sensor_cols_to_read = [c for c in sensor_cols_to_read if c in available_cols]

    if not sensor_cols_to_read:
        logger.warning(f"No sensor columns found in {file_path}")
        return 0, 0

    # Read the file
    table = pq.read_table(file_path, columns=mc_cols_to_read + sensor_cols_to_read)
    n_events = table.num_rows

    if n_events == 0:
        return 0, 0

    # Extract MC columns as numpy arrays
    mc_energy = (
        np.array(table.column(MC_COLUMNS["energy"]).to_pylist(), dtype=np.float32)
        if MC_COLUMNS["energy"] in mc_cols_to_read
        else np.zeros(n_events, dtype=np.float32)
    )
    mc_dir_x = (
        np.array(table.column(MC_COLUMNS["dir_x"]).to_pylist(), dtype=np.float32)
        if MC_COLUMNS["dir_x"] in mc_cols_to_read
        else np.zeros(n_events, dtype=np.float32)
    )
    mc_dir_y = (
        np.array(table.column(MC_COLUMNS["dir_y"]).to_pylist(), dtype=np.float32)
        if MC_COLUMNS["dir_y"] in mc_cols_to_read
        else np.zeros(n_events, dtype=np.float32)
    )
    mc_dir_z = (
        np.array(table.column(MC_COLUMNS["dir_z"]).to_pylist(), dtype=np.float32)
        if MC_COLUMNS["dir_z"] in mc_cols_to_read
        else np.zeros(n_events, dtype=np.float32)
    )

    # Morphology (string column)
    if MC_COLUMNS["morphology"] in mc_cols_to_read:
        morphology_arr = table.column(MC_COLUMNS["morphology"])
        morphology_strs = morphology_arr.to_pylist()
    else:
        morphology_strs = ["Unknown"] * n_events

    # Extract sensor columns as list of arrays (one per event)
    sensor_data = {}
    for col_name in sensor_cols_to_read:
        col = table.column(col_name)
        # Convert to list of numpy arrays
        sensor_data[col_name] = [
            (
                np.array(row, dtype=np.float32)
                if row is not None
                else np.array([], dtype=np.float32)
            )
            for row in col.to_pylist()
        ]

    # Process each event
    total_sensors = 0
    event_records = []

    for i in range(n_events):
        # Get number of sensors for this event (from first sensor column)
        first_sensor_col = sensor_cols_to_read[0]
        n_sensors = len(sensor_data[first_sensor_col][i])

        if n_sensors == 0:
            # Skip events with no sensors
            continue

        # Check if we should filter for starting events only
        if starting_only:
            morph_str = morphology_strs[i]
            if morph_str not in STARTING_MORPHOLOGIES:
                # Skip non-starting events
                continue

        # Create sensor records for this event
        sensors = np.zeros(n_sensors, dtype=WHAMS_SENSOR_RECORD_DTYPE)

        for field_name, col_name in SENSOR_COLUMNS.items():
            if col_name in sensor_data:
                values = sensor_data[col_name][i]
                if len(values) == n_sensors:
                    sensors[field_name] = values
                else:
                    # Handle mismatched lengths by padding/truncating
                    min_len = min(len(values), n_sensors)
                    sensors[field_name][:min_len] = values[:min_len]

        # Write sensor data to .dat file
        dat_file.write(sensors.tobytes())

        # Compute event statistics
        total_charge = float(np.sum(sensors["c_total"]))

        # Convert morphology
        morph_simplified, morph_raw, is_starting = convert_morphology(morphology_strs[i])

        # Create event record
        event_rec = np.zeros(1, dtype=WHAMS_EVENT_RECORD_DTYPE)
        event_rec["sensor_start_idx"] = sensor_offset + total_sensors
        event_rec["sensor_end_idx"] = sensor_offset + total_sensors + n_sensors
        event_rec["num_sensors"] = n_sensors
        event_rec["total_charge"] = total_charge
        event_rec["mc_primary_energy"] = mc_energy[i]
        event_rec["mc_primary_dir_x"] = mc_dir_x[i]
        event_rec["mc_primary_dir_y"] = mc_dir_y[i]
        event_rec["mc_primary_dir_z"] = mc_dir_z[i]
        event_rec["morphology"] = morph_simplified
        event_rec["morphology_raw"] = morph_raw
        event_rec["starting"] = is_starting
        event_rec["dataset_id"] = dataset_id

        event_records.append(event_rec)
        total_sensors += n_sensors

    # Write all event records
    for rec in event_records:
        idx_file.write(rec.tobytes())

    return len(event_records), total_sensors


def convert_whams_to_mmap(
    input_paths: List[Path],
    output_path: Path,
    file_range: Optional[Tuple[int, int]] = None,
    show_progress: bool = True,
    starting_only: bool = False,
) -> Dict[str, int]:
    """Convert WHAMS parquet files to mmap format.

    Args:
        input_paths: List of input directories or parquet files
        output_path: Output path prefix (will create .idx and .dat files)
        file_range: Optional (start, end) range of file indices
        show_progress: Whether to show progress bar
        starting_only: If True, only include starting events (0T1C, 0TnC, 1T1C, 1TnC, 2TnC)

    Returns:
        Dict with conversion statistics
    """
    # Get all parquet files
    parquet_files = get_parquet_files(input_paths, file_range)

    if not parquet_files:
        raise ValueError("No parquet files found in input paths")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    idx_path = output_path.with_suffix(".idx")
    dat_path = output_path.with_suffix(".dat")

    logger.info(f"Output files: {idx_path}, {dat_path}")

    # First pass: count events and sensors to pre-allocate
    logger.info("First pass: counting events and sensors...")
    total_events = 0
    total_sensors = 0

    # We'll do a streaming approach - write as we go, then update header
    # This avoids needing to load everything into memory

    # Open output files
    with open(idx_path, "wb") as idx_file, open(dat_path, "wb") as dat_file:
        # Write placeholder header (will update later)
        idx_file.write(create_header(0, 0))

        # Process each file
        sensor_offset = 0
        events_written = 0

        iterator = (
            tqdm(parquet_files, desc="Converting") if show_progress else parquet_files
        )

        for file_path in iterator:
            dataset_id = extract_dataset_id(file_path)

            try:
                n_events, n_sensors = process_parquet_file(
                    file_path,
                    idx_file,
                    dat_file,
                    sensor_offset,
                    dataset_id,
                    starting_only=starting_only,
                )
                events_written += n_events
                sensor_offset += n_sensors
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        total_events = events_written
        total_sensors = sensor_offset

    # Update header with actual counts
    with open(idx_path, "r+b") as idx_file:
        idx_file.seek(0)
        idx_file.write(create_header(total_events, total_sensors))

    # Verify file sizes
    expected_idx_size = (
        MMAP_HEADER_SIZE + total_events * WHAMS_EVENT_RECORD_DTYPE.itemsize
    )
    expected_dat_size = total_sensors * WHAMS_SENSOR_RECORD_DTYPE.itemsize

    actual_idx_size = idx_path.stat().st_size
    actual_dat_size = dat_path.stat().st_size

    if actual_idx_size != expected_idx_size:
        logger.error(f"Index file size mismatch: {actual_idx_size} != {expected_idx_size}")
    if actual_dat_size != expected_dat_size:
        logger.error(f"Data file size mismatch: {actual_dat_size} != {expected_dat_size}")

    stats = {
        "total_events": total_events,
        "total_sensors": total_sensors,
        "num_files": len(parquet_files),
        "idx_size_mb": actual_idx_size / (1024 * 1024),
        "dat_size_mb": actual_dat_size / (1024 * 1024),
    }

    logger.info(f"Conversion complete:")
    logger.info(f"  Events: {total_events:,}")
    logger.info(f"  Sensors: {total_sensors:,}")
    logger.info(f"  Index file: {stats['idx_size_mb']:.2f} MB")
    logger.info(f"  Data file: {stats['dat_size_mb']:.2f} MB")

    return stats


# =============================================================================
# CLI
# =============================================================================


def parse_file_range(range_str: str) -> Tuple[int, int]:
    """Parse file range string like '0-100' into tuple."""
    if "-" not in range_str:
        raise ValueError(f"Invalid range format: {range_str}. Expected 'start-end'.")
    start, end = range_str.split("-")
    return int(start), int(end)


def main():
    parser = argparse.ArgumentParser(
        description="Convert WHAMS parquet files to mmap format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert a single dataset
    python convert_to_mmap.py --input /path/to/WHAMSv1/BDT/21217 --output /path/to/whams_train

    # Convert multiple datasets
    python convert_to_mmap.py --input /path/to/21217 /path/to/21218 --output /path/to/whams_train

    # Convert only first 100 files
    python convert_to_mmap.py --input /path/to/21217 --output /path/to/whams_test --file-range 0-100

    # Convert only starting events (for neutrino-only datasets)
    python convert_to_mmap.py --input /path/to/22663 --output /path/to/starting_events --starting-only
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        nargs="+",
        required=True,
        help="Input directories or parquet files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path prefix (creates .idx and .dat files)",
    )
    parser.add_argument(
        "--file-range",
        type=str,
        default=None,
        help="Range of file indices to convert (e.g., '0-100')",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress bar",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Just inspect schema of first file and exit",
    )
    parser.add_argument(
        "--starting-only",
        action="store_true",
        help="Only include starting events (0T1C, 0TnC, 1T1C, 1TnC, 2TnC)",
    )

    args = parser.parse_args()

    # Validate inputs
    for path in args.input:
        if not path.exists():
            logger.error(f"Input path does not exist: {path}")
            sys.exit(1)

    # Parse file range if provided
    file_range = None
    if args.file_range:
        try:
            file_range = parse_file_range(args.file_range)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

    # Inspect mode
    if args.inspect:
        files = get_parquet_files(args.input, file_range)
        if files:
            info = inspect_parquet_schema(files[0])
            print(f"\nSchema for: {files[0]}")
            print(f"Has nested columns: {info['has_nested']}")
            print("\nColumns:")
            for name, details in sorted(info["columns"].items()):
                print(f"  {name}: {details['type']}")
        return

    # Run conversion
    try:
        stats = convert_whams_to_mmap(
            args.input,
            args.output,
            file_range=file_range,
            show_progress=not args.quiet,
            starting_only=args.starting_only,
        )
        print(f"\nConversion successful!")
        print(f"  Events: {stats['total_events']:,}")
        print(f"  Sensors: {stats['total_sensors']:,}")
        print(f"  Output: {args.output}.idx, {args.output}.dat")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
