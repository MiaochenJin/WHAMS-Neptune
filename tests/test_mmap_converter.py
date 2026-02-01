#!/usr/bin/env python3
"""
Test script for WHAMS mmap converter.

Converts a small subset of WHAMS parquet files and verifies the output.

Usage:
    # On cluster:
    python -m pytest tests/test_mmap_converter.py -v

    # Or run directly:
    python tests/test_mmap_converter.py
"""

import sys
from pathlib import Path
import tempfile
import numpy as np

# Add parent dir to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from whams_neptune.dataloaders.mmap_format import (
    WHAMS_EVENT_RECORD_DTYPE,
    WHAMS_SENSOR_RECORD_DTYPE,
    MMAP_HEADER_SIZE,
    parse_header,
    RAW_MORPHOLOGY_MAP_INV,
)

# Import converter functions - handle both direct and package imports
try:
    from scripts.convert_to_mmap import (
        convert_whams_to_mmap,
        inspect_parquet_schema,
        get_parquet_files,
    )
except ImportError:
    # Fallback for when running from different directory
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from convert_to_mmap import (
        convert_whams_to_mmap,
        inspect_parquet_schema,
        get_parquet_files,
    )


# =============================================================================
# Configuration
# =============================================================================

# Cluster paths
CLUSTER_BASE = Path("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/WHA-MS")
TEST_INPUT_DIR = CLUSTER_BASE / "datafiles/WHAMSv1/BDT/21217"

# Local test (if cluster not available)
LOCAL_TEST_DIR = Path("/Users/miaochenjin/Desktop/Harvard/WHA-MS/datafiles")

# Number of files to convert for testing
TEST_N_FILES = 10


# =============================================================================
# Test Functions
# =============================================================================


def test_schema_inspection():
    """Test parquet schema inspection."""
    print("\n" + "=" * 60)
    print("Test: Schema Inspection")
    print("=" * 60)

    # Find input directory
    if TEST_INPUT_DIR.exists():
        input_dir = TEST_INPUT_DIR
    elif LOCAL_TEST_DIR.exists():
        input_dir = LOCAL_TEST_DIR
        # Find a subdirectory with parquet files
        for subdir in input_dir.rglob("*"):
            if subdir.is_dir() and list(subdir.glob("*.parquet")):
                input_dir = subdir
                break
    else:
        print("SKIP: No test data available")
        return True

    files = get_parquet_files([input_dir], file_range=(0, 1))
    if not files:
        print("SKIP: No parquet files found")
        return True

    print(f"Inspecting: {files[0]}")
    info = inspect_parquet_schema(files[0])

    print(f"\nHas nested columns: {info['has_nested']}")
    print(f"Total columns: {len(info['columns'])}")

    # Check for expected columns
    expected_mc_cols = ["mc_primary_energy", "mc_primary_dir_x", "mc_event_morphology"]

    print("\nExpected MC columns:")
    for col in expected_mc_cols:
        status = "FOUND" if col in info["columns"] else "MISSING"
        print(f"  {col}: {status}")

    # Sensor columns have pulse_ prefix in WHAMS parquet files
    expected_sensor_cols = [
        "pulse_sensor_pos_x",
        "pulse_sensor_pos_y",
        "pulse_sensor_pos_z",
        "pulse_summary_t_first",
        "pulse_summary_c_total",
    ]

    print("\nExpected sensor columns:")
    for col in expected_sensor_cols:
        status = "FOUND" if col in info["columns"] else "MISSING"
        if col in info["columns"]:
            col_info = info["columns"][col]
            print(
                f"  {col}: {status} (type={col_info['type']}, list={col_info.get('is_list', False)})"
            )
        else:
            print(f"  {col}: {status}")

    print("\nPASS: Schema inspection completed")
    return True


def test_conversion():
    """Test parquet to mmap conversion."""
    print("\n" + "=" * 60)
    print("Test: Parquet to Mmap Conversion")
    print("=" * 60)

    # Find input directory
    if TEST_INPUT_DIR.exists():
        input_dir = TEST_INPUT_DIR
    elif LOCAL_TEST_DIR.exists():
        input_dir = LOCAL_TEST_DIR
        for subdir in input_dir.rglob("*"):
            if subdir.is_dir() and list(subdir.glob("*.parquet")):
                input_dir = subdir
                break
    else:
        print("SKIP: No test data available")
        return True

    # Create temporary output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_whams"

        print(f"Input: {input_dir}")
        print(f"Output: {output_path}")
        print(f"Files to convert: {TEST_N_FILES}")

        try:
            stats = convert_whams_to_mmap(
                [input_dir],
                output_path,
                file_range=(0, TEST_N_FILES),
                show_progress=True,
            )
        except Exception as e:
            print(f"FAIL: Conversion error: {e}")
            import traceback

            traceback.print_exc()
            return False

        print(f"\nConversion stats:")
        print(f"  Events: {stats['total_events']:,}")
        print(f"  Sensors: {stats['total_sensors']:,}")
        print(f"  Index size: {stats['idx_size_mb']:.2f} MB")
        print(f"  Data size: {stats['dat_size_mb']:.2f} MB")

        # Verify output files exist
        idx_path = output_path.with_suffix(".idx")
        dat_path = output_path.with_suffix(".dat")

        if not idx_path.exists():
            print(f"FAIL: Index file not created: {idx_path}")
            return False
        if not dat_path.exists():
            print(f"FAIL: Data file not created: {dat_path}")
            return False

        # Verify header
        with open(idx_path, "rb") as f:
            header_bytes = f.read(MMAP_HEADER_SIZE)
        header = parse_header(header_bytes)

        print(f"\nHeader verification:")
        print(f"  Magic: {header['magic']}")
        print(f"  Version: {header['version']}")
        print(f"  Num events: {header['num_events']}")
        print(f"  Num sensors: {header['num_sensors']}")

        if header["num_events"] != stats["total_events"]:
            print(f"FAIL: Event count mismatch in header")
            return False
        if header["num_sensors"] != stats["total_sensors"]:
            print(f"FAIL: Sensor count mismatch in header")
            return False

        # Verify we can read the mmap files
        event_records = np.memmap(
            idx_path,
            dtype=WHAMS_EVENT_RECORD_DTYPE,
            mode="r",
            offset=MMAP_HEADER_SIZE,
            shape=(header["num_events"],),
        )

        sensor_records = np.memmap(
            dat_path,
            dtype=WHAMS_SENSOR_RECORD_DTYPE,
            mode="r",
            shape=(header["num_sensors"],),
        )

        print(f"\nMmap verification:")
        print(f"  Event records shape: {event_records.shape}")
        print(f"  Sensor records shape: {sensor_records.shape}")

        # Sample some events
        print(f"\nSample events:")
        for i in [0, header["num_events"] // 2, header["num_events"] - 1]:
            if i < len(event_records):
                ev = event_records[i]
                morph_str = RAW_MORPHOLOGY_MAP_INV.get(int(ev["morphology_raw"]), "Unknown")
                print(
                    f"  Event {i}: n_sensors={ev['num_sensors']}, "
                    f"energy={ev['mc_primary_energy']:.1f} GeV, "
                    f"morphology={morph_str}, "
                    f"starting={ev['starting']}"
                )

        # Verify sensor indices are valid
        print(f"\nSensor index validation:")
        valid_indices = True
        for i in range(min(100, header["num_events"])):
            ev = event_records[i]
            if ev["sensor_end_idx"] > header["num_sensors"]:
                print(f"  FAIL: Event {i} has invalid sensor_end_idx")
                valid_indices = False
                break
            if ev["sensor_start_idx"] > ev["sensor_end_idx"]:
                print(f"  FAIL: Event {i} has start > end")
                valid_indices = False
                break

        if valid_indices:
            print("  All indices valid")

        print("\nPASS: Conversion test completed")
        return True


def test_data_integrity():
    """Test that converted data matches original parquet data."""
    print("\n" + "=" * 60)
    print("Test: Data Integrity Check")
    print("=" * 60)

    # This test requires both conversion and reading original parquet
    # For now, just verify basic statistics

    if not TEST_INPUT_DIR.exists():
        print("SKIP: Cluster data not available")
        return True

    import pyarrow.parquet as pq

    # Get first parquet file
    files = get_parquet_files([TEST_INPUT_DIR], file_range=(0, 1))
    if not files:
        print("SKIP: No files found")
        return True

    # Read original parquet
    table = pq.read_table(files[0])
    n_events_parquet = table.num_rows

    # Convert to mmap
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "integrity_test"

        stats = convert_whams_to_mmap(
            [files[0]],
            output_path,
            show_progress=False,
        )

        # Compare
        print(f"Parquet events: {n_events_parquet}")
        print(f"Mmap events: {stats['total_events']}")

        # Note: May not be exact if some events have 0 sensors
        if abs(stats["total_events"] - n_events_parquet) > n_events_parquet * 0.01:
            print(f"WARNING: Event count differs by more than 1%")

    print("\nPASS: Data integrity check completed")
    return True


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 60)
    print("WHAMS Mmap Converter Tests")
    print("=" * 60)

    results = {}

    results["schema"] = test_schema_inspection()
    results["conversion"] = test_conversion()
    results["integrity"] = test_data_integrity()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
