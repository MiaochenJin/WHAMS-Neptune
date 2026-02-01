#!/usr/bin/env python3
"""
Test script for WHAMS mmap dataloader.

Tests the MmapDataset class and verifies compatibility with Neptune training.

Usage:
    # On cluster:
    python -m pytest tests/test_mmap_dataloader.py -v

    # Or run directly:
    python tests/test_mmap_dataloader.py
"""

import sys
from pathlib import Path
import tempfile
import time
import numpy as np
import torch

# Add parent dir to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from whams_neptune.dataloaders.mmap_dataset import (
    MmapDataset,
    benchmark_dataloader,
    load_mmap_index,
)
from whams_neptune.dataloaders.mmap_format import RAW_MORPHOLOGY_MAP_INV

# Import converter - handle both direct and package imports
try:
    from scripts.convert_to_mmap import convert_whams_to_mmap
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from convert_to_mmap import convert_whams_to_mmap


# =============================================================================
# Configuration
# =============================================================================

# Cluster paths
CLUSTER_BASE = Path("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/WHA-MS")
TEST_INPUT_DIR = CLUSTER_BASE / "datafiles/WHAMSv1/BDT/21217"

# Test parameters
TEST_N_FILES = 20  # Number of parquet files to convert for testing
TEST_BATCH_SIZE = 32
TEST_NUM_WORKERS = 4


# =============================================================================
# Test Functions
# =============================================================================


def create_test_mmap(tmpdir: Path) -> Path:
    """Create test mmap files from parquet."""
    output_path = tmpdir / "test_data"

    if not TEST_INPUT_DIR.exists():
        raise FileNotFoundError(f"Test data not available: {TEST_INPUT_DIR}")

    print(f"Converting {TEST_N_FILES} files from {TEST_INPUT_DIR}...")
    stats = convert_whams_to_mmap(
        [TEST_INPUT_DIR],
        output_path,
        file_range=(0, TEST_N_FILES),
        show_progress=True,
    )

    print(
        f"Created test data: {stats['total_events']} events, {stats['total_sensors']} sensors"
    )
    return output_path


def test_dataset_basic():
    """Test basic dataset functionality."""
    print("\n" + "=" * 60)
    print("Test: Basic Dataset Functionality")
    print("=" * 60)

    if not TEST_INPUT_DIR.exists():
        print("SKIP: Test data not available")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mmap_path = create_test_mmap(tmpdir)

        # Create dataset
        dataset = MmapDataset(mmap_path, split="full")

        print(f"\nDataset info:")
        print(f"  Total events: {len(dataset)}")
        print(f"  Feature columns: {dataset.feature_columns}")

        # Test __getitem__
        print(f"\nTesting __getitem__:")
        for i in [0, len(dataset) // 2, len(dataset) - 1]:
            coords, features, labels = dataset[i]

            print(f"  Event {i}:")
            print(f"    coords shape: {coords.shape}")
            print(f"    features shape: {features.shape}")
            print(f"    labels shape: {labels.shape}")
            print(f"    labels: {labels.numpy()}")

            # Verify shapes match ParquetDataset format
            assert coords.dim() == 2, f"Expected 2D coords, got {coords.dim()}D"
            assert coords.shape[1] == 4, f"Expected 4 coord dims (x,y,z,t_first), got {coords.shape[1]}"
            assert features.dim() == 2, f"Expected 2D features, got {features.dim()}D"
            assert features.shape[0] == coords.shape[0], "Coords and features should have same length"
            assert labels.shape == (1, 6), f"Expected labels shape (1, 6), got {labels.shape}"

        # Test metadata
        print(f"\nTesting get_event_metadata:")
        meta = dataset.get_event_metadata(0)
        print(f"  First event metadata: {meta}")

        print("\nPASS: Basic dataset test completed")
        return True


def test_label_format():
    """Test that label format matches ParquetDataset."""
    print("\n" + "=" * 60)
    print("Test: Label Format Compatibility")
    print("=" * 60)

    if not TEST_INPUT_DIR.exists():
        print("SKIP: Test data not available")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mmap_path = create_test_mmap(tmpdir)

        dataset = MmapDataset(mmap_path, split="full")

        print("\nChecking label format for several events:")
        for i in range(min(5, len(dataset))):
            coords, features, labels = dataset[i]
            meta = dataset.get_event_metadata(i)

            # Labels should be (1, 6): [log_energy, dir_x, dir_y, dir_z, morphology_label, raw_morph_idx]
            log_energy = labels[0, 0].item()
            dir_x = labels[0, 1].item()
            dir_y = labels[0, 2].item()
            dir_z = labels[0, 3].item()
            morph_label = labels[0, 4].item()
            raw_morph = labels[0, 5].item()

            # Verify log_energy is computed correctly
            expected_log_energy = np.log10(meta["energy"]) if meta["energy"] > 0 else 0.0
            assert abs(log_energy - expected_log_energy) < 1e-5, f"log_energy mismatch: {log_energy} vs {expected_log_energy}"

            # Verify direction components
            assert abs(dir_x - meta["direction"][0]) < 1e-5, "dir_x mismatch"
            assert abs(dir_y - meta["direction"][1]) < 1e-5, "dir_y mismatch"
            assert abs(dir_z - meta["direction"][2]) < 1e-5, "dir_z mismatch"

            # Verify morphology_label matches starting status
            expected_morph_label = 1.0 if meta["starting"] else 0.0
            assert morph_label == expected_morph_label, f"morphology_label mismatch: {morph_label} vs {expected_morph_label}"

            print(
                f"  Event {i}: log_E={log_energy:.2f}, dir=({dir_x:.2f},{dir_y:.2f},{dir_z:.2f}), "
                f"morph_label={morph_label}, raw_morph={int(raw_morph)} ({meta['morphology_raw']})"
            )

        print("\nPASS: Label format test completed")
        return True


def test_train_val_split():
    """Test train/val splitting."""
    print("\n" + "=" * 60)
    print("Test: Train/Val Split")
    print("=" * 60)

    if not TEST_INPUT_DIR.exists():
        print("SKIP: Test data not available")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mmap_path = create_test_mmap(tmpdir)

        # Create full dataset to get total count
        full_dataset = MmapDataset(mmap_path, split="full")
        total_events = len(full_dataset)

        # Create train/val splits
        val_split = 0.2
        train_dataset = MmapDataset(mmap_path, split="train", val_split=val_split)
        val_dataset = MmapDataset(mmap_path, split="val", val_split=val_split)

        print(f"\nSplit verification:")
        print(f"  Total events: {total_events}")
        print(
            f"  Train events: {len(train_dataset)} ({100*len(train_dataset)/total_events:.1f}%)"
        )
        print(
            f"  Val events: {len(val_dataset)} ({100*len(val_dataset)/total_events:.1f}%)"
        )

        # Verify no overlap
        train_indices = set(train_dataset.indices)
        val_indices = set(val_dataset.indices)
        overlap = train_indices & val_indices

        if overlap:
            print(f"FAIL: {len(overlap)} overlapping indices between train and val")
            return False

        # Verify coverage
        combined = train_indices | val_indices
        if len(combined) != total_events:
            print(f"FAIL: Train + val = {len(combined)}, expected {total_events}")
            return False

        print(f"  No overlap between train and val: OK")
        print(f"  Combined coverage: {len(combined)} = {total_events}: OK")

        # Verify reproducibility
        train_dataset2 = MmapDataset(mmap_path, split="train", val_split=val_split)
        if not np.array_equal(train_dataset.indices, train_dataset2.indices):
            print("FAIL: Split not reproducible")
            return False
        print(f"  Split reproducibility: OK")

        print("\nPASS: Train/val split test completed")
        return True


def test_dataloader():
    """Test PyTorch DataLoader integration."""
    print("\n" + "=" * 60)
    print("Test: DataLoader Integration")
    print("=" * 60)

    if not TEST_INPUT_DIR.exists():
        print("SKIP: Test data not available")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mmap_path = create_test_mmap(tmpdir)

        dataset = MmapDataset(mmap_path, split="train")

        # Use the internal collator
        from whams_neptune.dataloaders.mmap_dataset import _MmapCollator

        collator = _MmapCollator()

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=TEST_BATCH_SIZE,
            shuffle=True,
            num_workers=TEST_NUM_WORKERS,
            collate_fn=collator,
        )

        print(f"\nDataLoader info:")
        print(f"  Batch size: {TEST_BATCH_SIZE}")
        print(f"  Num workers: {TEST_NUM_WORKERS}")
        print(f"  Num batches: {len(loader)}")

        # Iterate through a few batches
        print(f"\nIterating through batches:")
        for i, (coords_list, features_list, labels_batch) in enumerate(loader):
            if i >= 3:
                break

            print(f"  Batch {i}:")
            print(f"    Num events: {len(coords_list)}")
            print(f"    Labels shape: {labels_batch.shape}")

            # Check coords_list - now (N, 4) for each event
            n_sensors = [c.shape[0] for c in coords_list]
            coord_dims = [c.shape[1] for c in coords_list]
            print(
                f"    Sensors per event: min={min(n_sensors)}, max={max(n_sensors)}, mean={np.mean(n_sensors):.1f}"
            )
            print(f"    Coord dimensions: {set(coord_dims)} (should be {{4}})")

            # Verify types
            assert isinstance(coords_list, list), "coords should be list"
            assert isinstance(features_list, list), "features should be list"
            assert isinstance(labels_batch, torch.Tensor), "labels should be tensor"
            assert all(d == 4 for d in coord_dims), "All coords should have 4 dimensions"

        print("\nPASS: DataLoader test completed")
        return True


def test_benchmark():
    """Benchmark dataloader performance."""
    print("\n" + "=" * 60)
    print("Test: Dataloader Benchmark")
    print("=" * 60)

    if not TEST_INPUT_DIR.exists():
        print("SKIP: Test data not available")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mmap_path = create_test_mmap(tmpdir)

        dataset = MmapDataset(mmap_path, split="full")

        print(f"\nBenchmarking with {len(dataset)} events...")

        # Benchmark single-threaded
        print("\nSingle-threaded (num_workers=0):")
        stats_single = benchmark_dataloader(
            dataset,
            batch_size=TEST_BATCH_SIZE,
            num_workers=0,
            num_batches=50,
        )
        print(f"  Events/second: {stats_single['events_per_second']:.1f}")
        print(f"  Batches/second: {stats_single['batches_per_second']:.1f}")

        # Benchmark multi-threaded
        print(f"\nMulti-threaded (num_workers={TEST_NUM_WORKERS}):")
        stats_multi = benchmark_dataloader(
            dataset,
            batch_size=TEST_BATCH_SIZE,
            num_workers=TEST_NUM_WORKERS,
            num_batches=50,
        )
        print(f"  Events/second: {stats_multi['events_per_second']:.1f}")
        print(f"  Batches/second: {stats_multi['batches_per_second']:.1f}")

        speedup = stats_multi["events_per_second"] / stats_single["events_per_second"]
        print(f"\nSpeedup with {TEST_NUM_WORKERS} workers: {speedup:.2f}x")

        print("\nPASS: Benchmark completed")
        return True


def test_multiple_files():
    """Test loading multiple mmap files."""
    print("\n" + "=" * 60)
    print("Test: Multiple Mmap Files")
    print("=" * 60)

    if not TEST_INPUT_DIR.exists():
        print("SKIP: Test data not available")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create two separate mmap files
        mmap_path1 = tmpdir / "data1"
        mmap_path2 = tmpdir / "data2"

        print("Creating first mmap file (files 0-5)...")
        stats1 = convert_whams_to_mmap(
            [TEST_INPUT_DIR],
            mmap_path1,
            file_range=(0, 5),
            show_progress=False,
        )

        print("Creating second mmap file (files 5-10)...")
        stats2 = convert_whams_to_mmap(
            [TEST_INPUT_DIR],
            mmap_path2,
            file_range=(5, 10),
            show_progress=False,
        )

        total_expected = stats1["total_events"] + stats2["total_events"]

        # Load both files together
        dataset = MmapDataset([mmap_path1, mmap_path2], split="full")

        print(f"\nMulti-file dataset:")
        print(f"  File 1 events: {stats1['total_events']}")
        print(f"  File 2 events: {stats2['total_events']}")
        print(f"  Combined events: {len(dataset)}")
        print(f"  Expected: {total_expected}")

        if len(dataset) != total_expected:
            print(f"FAIL: Event count mismatch")
            return False

        # Test accessing events from both files
        print("\nAccessing events from both files:")
        for i in [0, stats1["total_events"] - 1, stats1["total_events"], len(dataset) - 1]:
            if i < len(dataset):
                coords, features, labels = dataset[i]
                meta = dataset.get_event_metadata(i)
                print(
                    f"  Event {i}: file_idx={meta['file_idx']}, local_idx={meta['local_idx']}, n_sensors={meta['num_sensors']}"
                )

        print("\nPASS: Multiple files test completed")
        return True


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 60)
    print("WHAMS Mmap Dataloader Tests")
    print("=" * 60)

    results = {}

    results["basic"] = test_dataset_basic()
    results["label_format"] = test_label_format()
    results["split"] = test_train_val_split()
    results["dataloader"] = test_dataloader()
    results["multiple_files"] = test_multiple_files()
    results["benchmark"] = test_benchmark()

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
