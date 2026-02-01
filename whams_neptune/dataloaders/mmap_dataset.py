#!/usr/bin/env python3
"""
WHAMS Memory-Mapped Dataset

PyTorch Dataset implementation that loads WHAMS data from memory-mapped files
for fast, efficient training.

This module provides:
- MmapDataset: Main dataset class compatible with existing Neptune training
- MmapDataModule: PyTorch Lightning DataModule wrapper

The output format matches ParquetDataset for drop-in compatibility:
- coords: (N, 4) tensor of [x, y, z, t_first]
- features: (N, F) tensor of sensor features
- labels: (1, 6) tensor of [log_energy, dir_x, dir_y, dir_z, morphology_label, raw_morph_idx]
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .mmap_format import (
    WHAMS_EVENT_RECORD_DTYPE,
    WHAMS_SENSOR_RECORD_DTYPE,
    MMAP_HEADER_SIZE,
    parse_header,
    RAW_MORPHOLOGY_MAP_INV,
    DEFAULT_MORPHOLOGY_TO_LABEL,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Mmap Loading Utilities
# =============================================================================


def load_mmap_index(idx_path: Path) -> Tuple[Dict, np.ndarray]:
    """Load mmap index file.

    Args:
        idx_path: Path to .idx file

    Returns:
        Tuple of (header_dict, event_records_array)
    """
    if not idx_path.exists():
        raise FileNotFoundError(f"Index file not found: {idx_path}")

    # Read and parse header
    with open(idx_path, "rb") as f:
        header_bytes = f.read(MMAP_HEADER_SIZE)

    header = parse_header(header_bytes)

    # Memory-map the event records (skip header)
    event_records = np.memmap(
        idx_path,
        dtype=WHAMS_EVENT_RECORD_DTYPE,
        mode="r",
        offset=MMAP_HEADER_SIZE,
        shape=(header["num_events"],),
    )

    return header, event_records


def load_mmap_sensors(dat_path: Path, num_sensors: int) -> np.ndarray:
    """Load mmap sensor data file.

    Args:
        dat_path: Path to .dat file
        num_sensors: Expected number of sensors

    Returns:
        Memory-mapped sensor records array
    """
    if not dat_path.exists():
        raise FileNotFoundError(f"Data file not found: {dat_path}")

    sensor_records = np.memmap(
        dat_path,
        dtype=WHAMS_SENSOR_RECORD_DTYPE,
        mode="r",
        shape=(num_sensors,),
    )

    return sensor_records


# =============================================================================
# Dataset Class
# =============================================================================


class MmapDataset(Dataset):
    """Memory-mapped WHAMS dataset for fast training.

    This dataset reads pre-converted mmap files and returns data in the same
    format as the ParquetDataset, making it a drop-in replacement.

    Output format (matching ParquetDataset):
        - coords: (N, 4) tensor of sensor positions [x, y, z, t_first]
        - features: (N, F) tensor of sensor features
        - labels: (1, 6) tensor of [log_energy, dir_x, dir_y, dir_z, morphology_label, raw_morph_idx]

    Args:
        mmap_paths: Path prefix to mmap files (or list of prefixes)
                    Will look for <path>.idx and <path>.dat
        split: One of "full", "train", "val"
        val_split: Fraction of data for validation (default 0.2)
        split_seed: Random seed for train/val split (default 42)
        feature_columns: List of sensor feature columns to include
                        Default: all timing and charge features
        rescale: Whether to apply feature rescaling
        rescale_params: Dict of rescaling parameters per feature
        max_sensors: Maximum number of sensors per event (for padding)
                    If None, returns variable length (for use with collator)
        morphology_mapping: Dict mapping raw morphology index (0-7) to class label.
                           Default: binary starting classification (see mmap_format.py)
                           Example for binary: {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0}
                           Example for multi-class: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}

                           Raw morphology indices (stored in mmap files):
                             0: 0T0C (non-starting)
                             1: 0T1C (starting)
                             2: 0TnC (starting)
                             3: 1T0C (non-starting)
                             4: 1T1C (starting)
                             5: 1TnC (starting)
                             6: 2TnC (starting)
                             7: Background (non-starting)
    """

    # Default feature columns (matching ParquetDataset order for Neptune training)
    # CRITICAL: This order MUST match the order in parquet_dataset.py
    # The order is: charge features first, then timing features
    # See: whams_neptune/dataloaders/parquet_dataset.py
    #
    # WARNING: If you have existing mmap files trained with the old order
    # [t_first, c_total, c_500ns, c_100ns, ...], you need to either:
    # 1. Re-convert the mmap files with correct feature order, or
    # 2. Use the transpose evaluation script to reorder features at inference time
    DEFAULT_FEATURES = [
        "c_total",
        "c_500ns",
        "c_100ns",
        "t_first",
        "t_last",
        "t20",
        "t50",
        "t_mean",
        "t_std",
    ]

    def __init__(
        self,
        mmap_paths: Union[str, Path, List[Union[str, Path]]],
        split: str = "full",
        val_split: float = 0.2,
        split_seed: int = 42,
        feature_columns: Optional[List[str]] = None,
        rescale: bool = False,
        rescale_params: Optional[Dict] = None,
        max_sensors: Optional[int] = None,
        morphology_mapping: Optional[Dict[int, float]] = None,
    ):
        super().__init__()

        self.split = split
        self.val_split = val_split
        self.split_seed = split_seed
        self.rescale = rescale
        self.rescale_params = rescale_params or {}
        self.max_sensors = max_sensors

        # Morphology mapping: raw_morph_idx -> class label
        # Default is binary starting classification
        self.morphology_mapping = morphology_mapping or DEFAULT_MORPHOLOGY_TO_LABEL

        # Set feature columns
        self.feature_columns = feature_columns or self.DEFAULT_FEATURES

        # Load all mmap files
        if isinstance(mmap_paths, (str, Path)):
            mmap_paths = [mmap_paths]

        self.mmap_paths = [Path(p) for p in mmap_paths]

        # Load and concatenate all datasets
        self._load_mmap_files()

        # Create train/val split
        self._create_split()

        logger.info(
            f"MmapDataset initialized: {len(self)} events "
            f"(split={split}, features={len(self.feature_columns)})"
        )

    def _load_mmap_files(self):
        """Load all mmap files and create unified index."""
        self.event_records_list = []
        self.sensor_records_list = []
        self.headers = []
        self.file_offsets = []  # Track which file each event belongs to
        self.sensor_offsets = []  # Track sensor offset per file

        total_events = 0
        total_sensors = 0

        for mmap_path in self.mmap_paths:
            idx_path = mmap_path.with_suffix(".idx")
            dat_path = mmap_path.with_suffix(".dat")

            # Check if files exist (try both with and without suffix)
            if not idx_path.exists():
                idx_path = Path(str(mmap_path) + ".idx")
            if not dat_path.exists():
                dat_path = Path(str(mmap_path) + ".dat")

            if not idx_path.exists() or not dat_path.exists():
                logger.warning(f"Skipping missing mmap files: {mmap_path}")
                continue

            header, event_records = load_mmap_index(idx_path)
            sensor_records = load_mmap_sensors(dat_path, header["num_sensors"])

            self.headers.append(header)
            self.event_records_list.append(event_records)
            self.sensor_records_list.append(sensor_records)

            # Track offsets for unified indexing
            self.file_offsets.append(total_events)
            self.sensor_offsets.append(total_sensors)

            total_events += header["num_events"]
            total_sensors += header["num_sensors"]

        if total_events == 0:
            raise ValueError("No events loaded from mmap files")

        self.total_events = total_events
        self.total_sensors = total_sensors

    def _create_split(self):
        """Create train/val split indices."""
        # Create reproducible random split
        rng = np.random.default_rng(self.split_seed)
        indices = np.arange(self.total_events)
        rng.shuffle(indices)

        n_val = int(self.total_events * self.val_split)
        n_train = self.total_events - n_val

        if self.split == "train":
            self.indices = indices[:n_train]
        elif self.split == "val":
            self.indices = indices[n_train:]
        else:  # "full"
            self.indices = np.arange(self.total_events)

    def _get_file_and_local_idx(self, global_idx: int) -> Tuple[int, int]:
        """Convert global index to (file_idx, local_idx)."""
        for i, offset in enumerate(self.file_offsets):
            if i + 1 < len(self.file_offsets):
                if global_idx < self.file_offsets[i + 1]:
                    return i, global_idx - offset
            else:
                return i, global_idx - offset
        raise IndexError(f"Global index {global_idx} out of range")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single event.

        Returns:
            Tuple of (coords, features, labels):
                - coords: (N, 4) float tensor of [x, y, z, t_first] positions
                - features: (N, F) float tensor of sensor features
                - labels: (1, 6) tensor of [log_energy, dir_x, dir_y, dir_z, morphology_label, raw_morph_idx]
        """
        # Map to actual event index
        global_idx = self.indices[idx]

        # Find which file and local index
        file_idx, local_idx = self._get_file_and_local_idx(global_idx)

        # Get event record
        event = self.event_records_list[file_idx][local_idx]

        # Get sensor range (adjust for file's sensor offset)
        sensor_start = int(event["sensor_start_idx"]) - self.sensor_offsets[file_idx]
        sensor_end = int(event["sensor_end_idx"]) - self.sensor_offsets[file_idx]
        n_sensors = sensor_end - sensor_start

        # Get sensor data
        sensors = self.sensor_records_list[file_idx][sensor_start:sensor_end]

        # Extract coordinates (x, y, z, t_first) - matching ParquetDataset format
        coords = np.stack(
            [sensors["x"], sensors["y"], sensors["z"], sensors["t_first"]], axis=-1
        )

        # Extract features
        feature_arrays = []
        for col in self.feature_columns:
            feat = sensors[col].astype(np.float32)
            if self.rescale and col in self.rescale_params:
                params = self.rescale_params[col]
                feat = (feat - params.get("mean", 0)) / params.get("std", 1)
            feature_arrays.append(feat)

        features = (
            np.stack(feature_arrays, axis=-1)
            if feature_arrays
            else np.zeros((n_sensors, 0), dtype=np.float32)
        )

        # Extract labels in ParquetDataset format: (1, 6)
        # [log_energy, dir_x, dir_y, dir_z, morphology_label, raw_morph_idx]
        energy = float(event["mc_primary_energy"])
        log_energy = np.log10(energy) if energy > 0 else 0.0
        dir_x = float(event["mc_primary_dir_x"])
        dir_y = float(event["mc_primary_dir_y"])
        dir_z = float(event["mc_primary_dir_z"])
        raw_morph_idx = int(event["morphology_raw"])
        morphology_label = float(self.morphology_mapping.get(raw_morph_idx, 0.0))

        labels = np.array(
            [[log_energy, dir_x, dir_y, dir_z, morphology_label, float(raw_morph_idx)]],
            dtype=np.float32,
        )

        # Convert to tensors
        coords = torch.from_numpy(coords.astype(np.float32))
        features = torch.from_numpy(features.astype(np.float32))
        labels = torch.from_numpy(labels)

        # Optional padding to max_sensors
        if self.max_sensors is not None and n_sensors < self.max_sensors:
            pad_size = self.max_sensors - n_sensors
            coords = torch.nn.functional.pad(coords, (0, 0, 0, pad_size))
            features = torch.nn.functional.pad(features, (0, 0, 0, pad_size))

        return coords, features, labels

    def get_event_metadata(self, idx: int) -> Dict:
        """Get additional metadata for an event (for debugging/analysis)."""
        global_idx = self.indices[idx]
        file_idx, local_idx = self._get_file_and_local_idx(global_idx)
        event = self.event_records_list[file_idx][local_idx]

        return {
            "global_idx": global_idx,
            "file_idx": file_idx,
            "local_idx": local_idx,
            "num_sensors": int(event["num_sensors"]),
            "total_charge": float(event["total_charge"]),
            "energy": float(event["mc_primary_energy"]),
            "direction": [
                float(event["mc_primary_dir_x"]),
                float(event["mc_primary_dir_y"]),
                float(event["mc_primary_dir_z"]),
            ],
            "morphology_raw": RAW_MORPHOLOGY_MAP_INV.get(
                int(event["morphology_raw"]), "Unknown"
            ),
            "starting": bool(event["starting"]),
            "dataset_id": int(event["dataset_id"]),
        }


# =============================================================================
# DataModule (PyTorch Lightning)
# =============================================================================

try:
    import pytorch_lightning as pl

    class MmapDataModule(pl.LightningDataModule):
        """PyTorch Lightning DataModule for WHAMS mmap data.

        This is a drop-in replacement for existing ParquetDataModule.
        Uses IrregularDataCollator from utils.collators for variable-length batching.

        Args:
            mmap_paths: Path(s) to mmap files
            batch_size: Batch size for training
            val_split: Fraction for validation
            split_seed: Random seed for reproducible splits
            num_workers: DataLoader workers
            feature_columns: Which sensor features to use
            rescale: Whether to rescale features
            rescale_params: Rescaling parameters
            morphology_mapping: Dict mapping raw morphology index to class label.
                               See MmapDataset docstring for details.
        """

        def __init__(
            self,
            mmap_paths: Union[str, Path, List[Union[str, Path]]],
            batch_size: int = 32,
            val_split: float = 0.2,
            split_seed: int = 42,
            num_workers: int = 4,
            feature_columns: Optional[List[str]] = None,
            rescale: bool = False,
            rescale_params: Optional[Dict] = None,
            pin_memory: bool = True,
            morphology_mapping: Optional[Dict[int, float]] = None,
        ):
            super().__init__()
            self.mmap_paths = mmap_paths
            self.batch_size = batch_size
            self.val_split = val_split
            self.split_seed = split_seed
            self.num_workers = num_workers
            self.feature_columns = feature_columns
            self.rescale = rescale
            self.rescale_params = rescale_params
            self.pin_memory = pin_memory
            self.morphology_mapping = morphology_mapping

            self.train_dataset = None
            self.val_dataset = None

            # Import collator from utils.collators (standard location in WHAMS-Neptune)
            from whams_neptune.utils.collators import IrregularDataCollator
            self.collator = IrregularDataCollator()

        def setup(self, stage: Optional[str] = None):
            """Set up datasets."""
            common_kwargs = {
                "mmap_paths": self.mmap_paths,
                "val_split": self.val_split,
                "split_seed": self.split_seed,
                "feature_columns": self.feature_columns,
                "rescale": self.rescale,
                "rescale_params": self.rescale_params,
                "morphology_mapping": self.morphology_mapping,
            }

            if stage == "fit" or stage is None:
                self.train_dataset = MmapDataset(split="train", **common_kwargs)
                self.val_dataset = MmapDataset(split="val", **common_kwargs)

            if stage == "test" or stage is None:
                self.test_dataset = MmapDataset(split="full", **common_kwargs)

        def train_dataloader(self) -> DataLoader:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.collator,
                pin_memory=self.pin_memory,
            )

        def val_dataloader(self) -> DataLoader:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collator,
                pin_memory=self.pin_memory,
            )

        def test_dataloader(self) -> DataLoader:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collator,
                pin_memory=self.pin_memory,
            )

except ImportError:
    # PyTorch Lightning not available
    MmapDataModule = None
    logger.debug("PyTorch Lightning not available, MmapDataModule disabled")


# =============================================================================
# Fallback Collator
# =============================================================================


class _MmapCollator:
    """Fallback collator for variable-length events.

    This is used if IrregularDataCollator from utils.collators is not available.
    It handles variable-length sensor data by returning lists instead of padded tensors.
    """

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Collate a batch of events.

        Args:
            batch: List of (coords, features, labels) tuples

        Returns:
            Tuple of (coords_list, features_list, labels_batch):
                - coords_list: List of (N_i, 4) tensors
                - features_list: List of (N_i, F) tensors
                - labels_batch: (B, 1, 6) tensor of stacked labels
        """
        coords_list = [item[0] for item in batch]
        features_list = [item[1] for item in batch]
        labels_list = [item[2] for item in batch]

        # Stack labels into a single tensor
        labels_batch = torch.stack(labels_list, dim=0)

        return coords_list, features_list, labels_batch


# =============================================================================
# Utility Functions
# =============================================================================


def benchmark_dataloader(
    dataset: MmapDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    num_batches: int = 100,
) -> Dict:
    """Benchmark dataloader performance.

    Args:
        dataset: MmapDataset instance
        batch_size: Batch size to test
        num_workers: Number of workers
        num_batches: Number of batches to time

    Returns:
        Dict with timing statistics
    """
    import time

    collator = _MmapCollator()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # Warmup
    for i, batch in enumerate(loader):
        if i >= 5:
            break

    # Timing
    start = time.perf_counter()
    n_events = 0

    for i, (coords_list, features_list, labels) in enumerate(loader):
        n_events += len(coords_list)
        if i >= num_batches:
            break

    elapsed = time.perf_counter() - start

    return {
        "elapsed_seconds": elapsed,
        "batches": num_batches,
        "events": n_events,
        "events_per_second": n_events / elapsed,
        "batches_per_second": num_batches / elapsed,
    }
