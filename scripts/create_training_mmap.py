#!/usr/bin/env python3
"""
Create mmap training data for balanced Neptune training.

Converts all sources to mmap format in datafiles/training/mmap-starting/

This script creates permanent mmap training data files using:
- NuGen 21217/18/19: All events (full training data)
- MuonGun 21315/17/18/21316_extended: 500 files each (all events, background)
- SIREN charms/inclusive: All files, starting events only
- NuGen 22663: All files, starting events only

Usage:
    # On the cluster
    salloc -c 32 --mem 100G -t 0-08:00 -p arguelles_delgado_gpu_a100 --gres=gpu:1
    cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/DBSearch/Neptune/.neptune-env/
    spack env activate .
    cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/WHAMS-Neptune
    python scripts/create_training_mmap.py

Author: Claude Code
Date: 2026-01-30
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Base paths
BASE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/WHA-MS"
NEPTUNE_BASE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/WHAMS-Neptune"
OUTPUT_DIR = f"{BASE}/datafiles/training/mmap-starting"

# Conversion configurations
CONVERSIONS = [
    # NuGen training (all events)
    {
        "name": "nugen_21217",
        "input": f"{BASE}/datafiles/WHAMSv1/BDT/21217",
        "starting_only": False,
        "description": "NuGen 21217 - low energy neutrinos",
    },
    {
        "name": "nugen_21218",
        "input": f"{BASE}/datafiles/WHAMSv1/BDT/21218",
        "starting_only": False,
        "description": "NuGen 21218 - medium energy neutrinos",
    },
    {
        "name": "nugen_21219",
        "input": f"{BASE}/datafiles/WHAMSv1/BDT/21219",
        "starting_only": False,
        "description": "NuGen 21219 - high energy neutrinos",
    },
    # MuonGun (500 files each, all events = background)
    {
        "name": "muongun_21315",
        "input": f"{BASE}/datafiles/WHAMSv1/BDT/21315",
        "file_range": "0-500",
        "starting_only": False,
        "description": "MuonGun 21315 - atmospheric muons",
    },
    {
        "name": "muongun_21317",
        "input": f"{BASE}/datafiles/WHAMSv1/BDT/21317",
        "file_range": "0-500",
        "starting_only": False,
        "description": "MuonGun 21317 - atmospheric muons",
    },
    {
        "name": "muongun_21318",
        "input": f"{BASE}/datafiles/WHAMSv1/BDT/21318",
        "file_range": "0-500",
        "starting_only": False,
        "description": "MuonGun 21318 - atmospheric muons",
    },
    {
        "name": "muongun_21316_extended",
        "input": f"{BASE}/datafiles/WHAMSv2/BDT/21316_extended",
        "file_range": "0-500",
        "starting_only": False,
        "description": "MuonGun 21316_extended - atmospheric muons (WHAMSv2)",
    },
    # SIREN (starting only)
    {
        "name": "siren_charms_starting",
        "input": f"{BASE}/datafiles/SIREN_charms_nue/BDT",
        "starting_only": True,
        "description": "SIREN charm neutrinos - starting events only",
    },
    {
        "name": "siren_inclusive_starting",
        "input": f"{BASE}/datafiles/SIREN_inclusive_nue/BDT",
        "starting_only": True,
        "description": "SIREN inclusive neutrinos - starting events only",
    },
    # 22663 (starting only)
    {
        "name": "nugen_22663_starting",
        "input": f"{BASE}/datafiles/WHAMSv2/BDT/22663",
        "starting_only": True,
        "description": "NuGen 22663 - high energy upgoing neutrinos (starting only)",
    },
]


def run_conversion(config: Dict) -> Optional[Dict]:
    """Run a single mmap conversion.

    Args:
        config: Conversion configuration dict with keys:
            - name: Output filename (without extension)
            - input: Input directory path
            - starting_only: Whether to filter to starting events
            - file_range: Optional "start-end" string
            - description: Human-readable description

    Returns:
        Dict with conversion statistics, or None if failed
    """
    name = config["name"]
    input_path = config["input"]
    starting_only = config.get("starting_only", False)
    file_range = config.get("file_range")
    description = config.get("description", "")

    output_path = f"{OUTPUT_DIR}/{name}"

    logger.info(f"Converting {name}...")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Starting only: {starting_only}")
    if file_range:
        logger.info(f"  File range: {file_range}")

    # Check if input exists
    if not Path(input_path).exists():
        logger.error(f"  Input path does not exist: {input_path}")
        return None

    # Build command
    cmd = [
        sys.executable,
        f"{NEPTUNE_BASE}/scripts/convert_to_mmap.py",
        "--input", input_path,
        "--output", output_path,
    ]

    if starting_only:
        cmd.append("--starting-only")

    if file_range:
        cmd.extend(["--file-range", file_range])

    # Run conversion (stream output for progress visibility)
    start_time = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            text=True,
            check=True,
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        stats = {
            "name": name,
            "input": input_path,
            "output": output_path,
            "starting_only": starting_only,
            "file_range": file_range,
            "description": description,
            "duration_seconds": duration,
            "success": True,
        }

        # Get file sizes and extract event count from index file
        idx_path = Path(f"{output_path}.idx")
        dat_path = Path(f"{output_path}.dat")
        if idx_path.exists():
            stats["idx_size_mb"] = idx_path.stat().st_size / (1024 * 1024)
            # Parse header to get event count (bytes 14-22 are num_events as uint64)
            with open(idx_path, "rb") as f:
                f.seek(14)
                import struct
                num_events = struct.unpack("<Q", f.read(8))[0]
                stats["total_events"] = num_events
        if dat_path.exists():
            stats["dat_size_mb"] = dat_path.stat().st_size / (1024 * 1024)

        logger.info(f"  Completed in {duration:.1f}s")
        if "total_events" in stats:
            logger.info(f"  Events: {stats['total_events']:,}")

        return stats

    except subprocess.CalledProcessError as e:
        logger.error(f"  Conversion failed: {e}")
        return {
            "name": name,
            "input": input_path,
            "success": False,
            "error": str(e),
        }


def main():
    """Run all conversions and create manifest."""
    logger.info("=" * 60)
    logger.info("WHAMS Training Data Conversion to Mmap Format")
    logger.info("=" * 60)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Number of datasets: {len(CONVERSIONS)}")
    logger.info("")

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run all conversions
    results = []
    for config in CONVERSIONS:
        result = run_conversion(config)
        if result:
            results.append(result)
        logger.info("")

    # Create manifest
    manifest = {
        "created": datetime.now().isoformat(),
        "output_directory": OUTPUT_DIR,
        "datasets": results,
        "summary": {
            "total_datasets": len(results),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", True)),
        },
    }

    # Calculate totals
    total_events = sum(r.get("total_events", 0) for r in results if r.get("success"))
    total_idx_mb = sum(r.get("idx_size_mb", 0) for r in results if r.get("success"))
    total_dat_mb = sum(r.get("dat_size_mb", 0) for r in results if r.get("success"))

    manifest["summary"]["total_events"] = total_events
    manifest["summary"]["total_idx_size_mb"] = total_idx_mb
    manifest["summary"]["total_dat_size_mb"] = total_dat_mb

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Successful: {manifest['summary']['successful']}/{manifest['summary']['total_datasets']}")
    logger.info(f"Total events: {total_events:,}")
    logger.info(f"Total size: {total_idx_mb + total_dat_mb:.1f} MB")
    logger.info(f"Manifest: {manifest_path}")

    # List failed conversions
    failed = [r for r in results if not r.get("success", True)]
    if failed:
        logger.warning("")
        logger.warning("FAILED CONVERSIONS:")
        for r in failed:
            logger.warning(f"  - {r['name']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
