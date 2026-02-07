#!/usr/bin/env python3
"""
WHAMS Memory-Mapped Dataset

PyTorch Dataset implementation that loads WHAMS data from memory-mapped files
for fast, efficient training.

This module provides:
- MmapDataset: Main dataset class compatible with existing Neptune training
- MmapDataModule: PyTorch Lightning DataModule wrapper
- WeightingConfig: Configuration for weighted sampling of high-E upgoing events

The output format matches ParquetDataset for drop-in compatibility:
- coords: (N, 4) tensor of [x, y, z, t_first]
- features: (N, F) tensor of sensor features
- labels: (1, 6) tensor of [log_energy, dir_x, dir_y, dir_z, morphology_label, raw_morph_idx]
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

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
# Weighting Configuration
# =============================================================================


@dataclass
class WeightingConfig:
    """Configuration for weighted sampling during training.

    This implements energy-dependent weighting for upgoing starting events
    to address the deficit of high-energy upgoing events in training data.

    Weight formula:
        For starting events (morphology_label == 1) with dir_z > 0 (upgoing):
            - energy >= e_very_high: w_very_high_upgoing
            - energy >= e_high:      w_high_upgoing
            - energy >= e_med:       w_med_upgoing
            - else:                  w_default
        For all other events: w_default

    Args:
        enabled: Whether to use weighted sampling (default False)
        e_very_high: Very high energy threshold in GeV (default 1e6 = 1 PeV)
        e_high: High energy threshold in GeV (default 1e5 = 100 TeV)
        e_med: Medium energy threshold in GeV (default 1e4 = 10 TeV)
        w_very_high_upgoing: Weight for very high energy upgoing starting (default 20.0)
        w_high_upgoing: Weight for high energy upgoing starting (default 10.0)
        w_med_upgoing: Weight for medium energy upgoing starting (default 3.0)
        w_default: Default weight for all other events (default 1.0)
    """
    enabled: bool = False
    e_very_high: float = 1e6   # 1 PeV in GeV
    e_high: float = 1e5        # 100 TeV in GeV
    e_med: float = 1e4         # 10 TeV in GeV
    w_very_high_upgoing: float = 20.0
    w_high_upgoing: float = 10.0
    w_med_upgoing: float = 3.0
    w_default: float = 1.0


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
        # Filter out events with 0 sensors (would cause batch size mismatch)
        valid_indices = []
        n_empty = 0
        for global_idx in range(self.total_events):
            file_idx, local_idx = self._get_file_and_local_idx(global_idx)
            event = self.event_records_list[file_idx][local_idx]
            n_sensors = int(event["num_sensors"])
            if n_sensors > 0:
                valid_indices.append(global_idx)
            else:
                n_empty += 1

        if n_empty > 0:
            logger.warning(f"Filtered out {n_empty} events with 0 sensors")

        # Create reproducible random split from valid indices only
        rng = np.random.default_rng(self.split_seed)
        indices = np.array(valid_indices)
        rng.shuffle(indices)

        n_valid = len(indices)
        n_val = int(n_valid * self.val_split)
        n_train = n_valid - n_val

        if self.split == "train":
            self.indices = indices[:n_train]
        elif self.split == "val":
            self.indices = indices[n_train:]
        else:  # "full" - use all valid indices (unshuffled for reproducibility)
            self.indices = np.array(valid_indices)

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

        # Get sensor range - indices are LOCAL to each file's .dat, no offset needed
        # (Previous bug: subtracting sensor_offsets caused wrong indices for non-first files)
        sensor_start = int(event["sensor_start_idx"])
        sensor_end = int(event["sensor_end_idx"])
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

    def compute_sample_weights(self, weighting_config: WeightingConfig) -> np.ndarray:
        """Compute sample weights for WeightedRandomSampler.

        This iterates through all events in the dataset (respecting the split)
        and computes weights based on energy, direction, and morphology.

        Args:
            weighting_config: WeightingConfig with thresholds and weights

        Returns:
            numpy array of weights with shape (len(self),)
        """
        logger.info("Computing sample weights for weighted sampling...")

        weights = np.ones(len(self), dtype=np.float64)

        # Counters for logging
        n_very_high_upgoing = 0
        n_high_upgoing = 0
        n_med_upgoing = 0
        n_starting_downgoing = 0
        n_background = 0

        for idx in range(len(self)):
            global_idx = self.indices[idx]
            file_idx, local_idx = self._get_file_and_local_idx(global_idx)
            event = self.event_records_list[file_idx][local_idx]

            # Get event properties
            energy = float(event["mc_primary_energy"])  # in GeV
            dir_z = float(event["mc_primary_dir_z"])
            raw_morph_idx = int(event["morphology_raw"])
            morphology_label = self.morphology_mapping.get(raw_morph_idx, 0.0)

            # Default weight
            weight = weighting_config.w_default

            # Only apply higher weights to starting events (morphology_label == 1)
            is_starting = (morphology_label == 1.0)
            is_upgoing = (dir_z > 0)

            if is_starting and is_upgoing:
                if energy >= weighting_config.e_very_high:
                    weight = weighting_config.w_very_high_upgoing
                    n_very_high_upgoing += 1
                elif energy >= weighting_config.e_high:
                    weight = weighting_config.w_high_upgoing
                    n_high_upgoing += 1
                elif energy >= weighting_config.e_med:
                    weight = weighting_config.w_med_upgoing
                    n_med_upgoing += 1
            elif is_starting:
                n_starting_downgoing += 1
            else:
                n_background += 1

            weights[idx] = weight

        # Cache the weights
        self._sample_weights = weights
        self._weighting_config = weighting_config

        # Log statistics
        self._log_weight_statistics(
            n_very_high_upgoing, n_high_upgoing, n_med_upgoing,
            n_starting_downgoing, n_background, weighting_config
        )

        return weights

    def _log_weight_statistics(
        self,
        n_very_high_upgoing: int,
        n_high_upgoing: int,
        n_med_upgoing: int,
        n_starting_downgoing: int,
        n_background: int,
        cfg: WeightingConfig,
    ):
        """Log weight statistics for debugging and verification."""
        total = len(self)

        # Compute effective sample sizes
        eff_very_high = n_very_high_upgoing * cfg.w_very_high_upgoing
        eff_high = n_high_upgoing * cfg.w_high_upgoing
        eff_med = n_med_upgoing * cfg.w_med_upgoing
        eff_other = (n_starting_downgoing + n_background) * cfg.w_default

        total_weight = eff_very_high + eff_high + eff_med + eff_other

        logger.info("=" * 60)
        logger.info("Sample Weight Statistics:")
        logger.info("=" * 60)
        logger.info(f"Total events: {total}")
        logger.info("")
        logger.info("Category counts and weights:")
        logger.info(f"  >1 PeV upgoing starting:   {n_very_high_upgoing:>8} x {cfg.w_very_high_upgoing:>5.1f} = {eff_very_high:>10.1f}")
        logger.info(f"  100TeV-1PeV upgoing start: {n_high_upgoing:>8} x {cfg.w_high_upgoing:>5.1f} = {eff_high:>10.1f}")
        logger.info(f"  10-100TeV upgoing start:   {n_med_upgoing:>8} x {cfg.w_med_upgoing:>5.1f} = {eff_med:>10.1f}")
        logger.info(f"  Starting downgoing:        {n_starting_downgoing:>8} x {cfg.w_default:>5.1f} = {n_starting_downgoing * cfg.w_default:>10.1f}")
        logger.info(f"  Background:                {n_background:>8} x {cfg.w_default:>5.1f} = {n_background * cfg.w_default:>10.1f}")
        logger.info("")
        logger.info(f"Total effective weight: {total_weight:.1f}")
        logger.info("")
        logger.info("Sampling probabilities per category:")
        if total_weight > 0:
            logger.info(f"  >1 PeV upgoing:            {100 * eff_very_high / total_weight:>6.2f}%")
            logger.info(f"  100TeV-1PeV upgoing:       {100 * eff_high / total_weight:>6.2f}%")
            logger.info(f"  10-100TeV upgoing:         {100 * eff_med / total_weight:>6.2f}%")
            logger.info(f"  Other events:              {100 * eff_other / total_weight:>6.2f}%")
        logger.info("=" * 60)

    def get_sample_weights(self) -> Optional[np.ndarray]:
        """Return cached sample weights, or None if not computed."""
        return getattr(self, "_sample_weights", None)


# =============================================================================
# DataModule (PyTorch Lightning)
# =============================================================================

try:
    import lightning.pytorch as pl

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
            weighting_config: Optional WeightingConfig for weighted sampling.
                             If provided and enabled, uses WeightedRandomSampler
                             to oversample high-energy upgoing starting events.
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
            weighting_config: Optional[WeightingConfig] = None,
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
            self.weighting_config = weighting_config

            self.train_dataset = None
            self.val_dataset = None
            self._train_sampler = None

            # Import collator from parquet_dataset (returns 4 values: coords, features, labels, batch_ids)
            # Note: utils.collators.IrregularDataCollator returns only 3 values, which is incompatible
            from whams_neptune.dataloaders.parquet_dataset import IrregularDataCollator
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

                # Compute sample weights if weighting is enabled
                if self.weighting_config and self.weighting_config.enabled:
                    weights = self.train_dataset.compute_sample_weights(self.weighting_config)
                    self._train_sampler = WeightedRandomSampler(
                        weights=weights,
                        num_samples=len(weights),
                        replacement=True,
                    )
                    logger.info("WeightedRandomSampler created for training")

            if stage == "test" or stage is None:
                self.test_dataset = MmapDataset(split="full", **common_kwargs)

        def train_dataloader(self) -> DataLoader:
            # Use weighted sampler if available, otherwise shuffle
            if self._train_sampler is not None:
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    sampler=self._train_sampler,
                    num_workers=self.num_workers,
                    collate_fn=self.collator,
                    pin_memory=self.pin_memory,
                )
            else:
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
