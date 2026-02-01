"""WHAMS-specific mmap format definitions.

This module defines the binary format for memory-mapped WHAMS data files.
The format consists of two files:
- .idx file: Header + event index records
- .dat file: Sensor data records

The format is optimized for random access during training.
"""

import numpy as np
from typing import Dict

# Magic bytes to identify WHAMS mmap files
MMAP_MAGIC = b"WHAMS_MMAP"
MMAP_VERSION = 1

# Header size in bytes (fixed for easy seeking)
# Magic (10) + Version (4) + Num Events (8) + Num Sensors (8) + Reserved (34) = 64
MMAP_HEADER_SIZE = 64


# =============================================================================
# Morphology Mappings
# =============================================================================
#
# MMAP files store `morphology_raw` as an int8 index using the mapping below.
# At training time, you can map these indices to your desired class labels
# via the `morphology_mapping` parameter in MmapDataset/MmapDataModule.
#
# Example training-time mappings:
#
#   Binary starting classification:
#     {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0}
#     # 0T0C/1T0C/Background -> 0 (non-starting)
#     # 0T1C/0TnC/1T1C/1TnC/2TnC -> 1 (starting)
#
#   Multi-class morphology:
#     {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
#     # Each morphology gets its own class
#
# =============================================================================

# Simplified morphology mapping for model output (legacy, not used in mmap)
MORPHOLOGY_MAP: Dict[str, int] = {
    "Track": 0,
    "Cascade": 1,
    "Mixed": 2,
}

# Raw morphology mapping: string -> index stored in mmap files
# This is the DEFINITIVE mapping used during mmap conversion.
# The index is stored in the `morphology_raw` field (int8).
RAW_MORPHOLOGY_MAP: Dict[str, int] = {
    "0T0C": 0,        # Non-starting: 0 tracks, 0 cascades
    "0T1C": 1,        # Starting: 0 tracks, 1 cascade
    "0TnC": 2,        # Starting: 0 tracks, n cascades
    "1T0C": 3,        # Non-starting: 1 track, 0 cascades
    "1T1C": 4,        # Starting: 1 track, 1 cascade
    "1TnC": 5,        # Starting: 1 track, n cascades
    "2TnC": 6,        # Starting: 2 tracks, n cascades
    "Background": 7,  # Non-starting: atmospheric muon background
    "BackGround": 7,  # Handle case variation
}

# Inverse mapping: index -> string (for debugging/display)
RAW_MORPHOLOGY_MAP_INV: Dict[int, str] = {
    0: "0T0C",
    1: "0T1C",
    2: "0TnC",
    3: "1T0C",
    4: "1T1C",
    5: "1TnC",
    6: "2TnC",
    7: "Background",
}

# Starting morphologies (for quick lookup during conversion filtering)
STARTING_MORPHOLOGIES = {"0T1C", "0TnC", "1T1C", "1TnC", "2TnC"}

# Default training-time mapping: raw_index -> binary class (starting=1, non-starting=0)
DEFAULT_MORPHOLOGY_TO_LABEL: Dict[int, float] = {
    0: 0.0,  # 0T0C -> non-starting
    1: 1.0,  # 0T1C -> starting
    2: 1.0,  # 0TnC -> starting
    3: 0.0,  # 1T0C -> non-starting
    4: 1.0,  # 1T1C -> starting
    5: 1.0,  # 1TnC -> starting
    6: 1.0,  # 2TnC -> starting
    7: 0.0,  # Background -> non-starting
}


# =============================================================================
# Event Record Dtype
# =============================================================================

WHAMS_EVENT_RECORD_DTYPE = np.dtype(
    [
        # Sensor indices (into .dat file)
        ("sensor_start_idx", np.uint64),
        ("sensor_end_idx", np.uint64),
        # Hit statistics
        ("num_sensors", np.uint32),
        ("total_charge", np.float32),
        # MC Truth - Energy
        ("mc_primary_energy", np.float32),
        # MC Truth - Direction (unit vector components)
        ("mc_primary_dir_x", np.float32),
        ("mc_primary_dir_y", np.float32),
        ("mc_primary_dir_z", np.float32),
        # MC Truth - Morphology
        ("morphology", np.int8),  # Simplified (Track/Cascade/Mixed)
        ("morphology_raw", np.int8),  # Original (0T0C, 0T1C, etc.)
        # MC Truth - Starting flag
        ("starting", np.bool_),
        # Dataset ID (for tracking source)
        ("dataset_id", np.uint32),
    ]
)

# Record the actual size (no strict alignment requirement)
WHAMS_EVENT_RECORD_SIZE = WHAMS_EVENT_RECORD_DTYPE.itemsize


# =============================================================================
# Sensor Record Dtype
# =============================================================================

WHAMS_SENSOR_RECORD_DTYPE = np.dtype(
    [
        # Position (detector coordinates)
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        # Timing features
        ("t_first", np.float32),
        ("t_last", np.float32),
        ("t20", np.float32),
        ("t50", np.float32),
        ("t_mean", np.float32),
        ("t_std", np.float32),
        # Charge features
        ("c_total", np.float32),
        ("c_500ns", np.float32),
        ("c_100ns", np.float32),
    ]
)

# Record the actual size (12 float32s = 48 bytes)
WHAMS_SENSOR_RECORD_SIZE = WHAMS_SENSOR_RECORD_DTYPE.itemsize


# =============================================================================
# Header Structure
# =============================================================================


def create_header(num_events: int, num_sensors: int) -> bytes:
    """Create a mmap file header.

    Args:
        num_events: Total number of events in the file
        num_sensors: Total number of sensor records in the file

    Returns:
        64-byte header as bytes
    """
    header = bytearray(MMAP_HEADER_SIZE)

    # Magic bytes (10 bytes)
    header[0:10] = MMAP_MAGIC

    # Version (4 bytes, little-endian uint32)
    header[10:14] = np.array([MMAP_VERSION], dtype=np.uint32).tobytes()

    # Number of events (8 bytes, little-endian uint64)
    header[14:22] = np.array([num_events], dtype=np.uint64).tobytes()

    # Number of sensors (8 bytes, little-endian uint64)
    header[22:30] = np.array([num_sensors], dtype=np.uint64).tobytes()

    # Rest is reserved/padding (30-64)

    return bytes(header)


def parse_header(header_bytes: bytes) -> Dict:
    """Parse a mmap file header.

    Args:
        header_bytes: 64 bytes from start of file

    Returns:
        Dict with magic, version, num_events, num_sensors
    """
    if len(header_bytes) < MMAP_HEADER_SIZE:
        raise ValueError(f"Header too short: {len(header_bytes)} < {MMAP_HEADER_SIZE}")

    magic = header_bytes[0:10]
    if magic != MMAP_MAGIC:
        raise ValueError(f"Invalid magic bytes: {magic!r} != {MMAP_MAGIC!r}")

    version = np.frombuffer(header_bytes[10:14], dtype=np.uint32)[0]
    if version != MMAP_VERSION:
        raise ValueError(f"Unsupported version: {version} != {MMAP_VERSION}")

    num_events = np.frombuffer(header_bytes[14:22], dtype=np.uint64)[0]
    num_sensors = np.frombuffer(header_bytes[22:30], dtype=np.uint64)[0]

    return {
        "magic": magic,
        "version": version,
        "num_events": int(num_events),
        "num_sensors": int(num_sensors),
    }


# =============================================================================
# Feature Column Names (for reference)
# =============================================================================

# These are the sensor-level columns in WHAMS parquet files that map to our dtype
SENSOR_FEATURE_COLUMNS = [
    "x",
    "y",
    "z",
    "t_first",
    "t_last",
    "t20",
    "t50",
    "t_mean",
    "t_std",
    "c_total",
    "c_500ns",
    "c_100ns",
]

# MC truth columns in WHAMS parquet files
MC_TRUTH_COLUMNS = [
    "mc_primary_energy",
    "mc_primary_dir_x",
    "mc_primary_dir_y",
    "mc_primary_dir_z",
]

# Morphology column
MORPHOLOGY_COLUMN = "mc_event_morphology"
