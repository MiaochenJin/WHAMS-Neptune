#!/usr/bin/env python3
"""
Benchmark script to compare mmap vs parquet data loading performance.
Measures:
1. Dataset initialization time
2. DataLoader creation time
3. First batch loading time
4. Average iteration speed over N batches
"""

import argparse
import sys
import time
import yaml
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whams_neptune.dataloaders.parquet_dataset import ParquetDataModule
from whams_neptune.dataloaders.mmap_dataset import MmapDataModule


def benchmark_datamodule(dm, name, num_batches=100):
    """Benchmark a datamodule."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    # Time setup
    t0 = time.perf_counter()
    dm.setup(stage='fit')
    setup_time = time.perf_counter() - t0
    print(f"Setup time: {setup_time:.2f}s")
    
    # Time dataloader creation
    t0 = time.perf_counter()
    train_loader = dm.train_dataloader()
    loader_time = time.perf_counter() - t0
    print(f"DataLoader creation: {loader_time:.2f}s")
    
    # Get dataset size
    train_size = len(dm.train_dataset) if hasattr(dm, 'train_dataset') else 'unknown'
    print(f"Training samples: {train_size}")
    
    # Time first batch
    t0 = time.perf_counter()
    batch_iter = iter(train_loader)
    first_batch = next(batch_iter)
    first_batch_time = time.perf_counter() - t0
    print(f"First batch time: {first_batch_time:.2f}s")
    
    # Batch info
    coords, features, labels, batch_ids = first_batch
    print(f"Batch shape: coords={coords.shape}, features={features.shape}, labels={labels.shape}")
    
    # Benchmark N batches
    print(f"\nBenchmarking {num_batches} batches...")
    times = []
    for i in range(num_batches):
        t0 = time.perf_counter()
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_loader)
            batch = next(batch_iter)
        batch_time = time.perf_counter() - t0
        times.append(batch_time)
        
        if (i + 1) % 20 == 0:
            avg = sum(times[-20:]) / 20
            print(f"  Batch {i+1}: avg {1/avg:.1f} batches/s")
    
    avg_time = sum(times) / len(times)
    batches_per_sec = 1 / avg_time
    
    print(f"\nResults:")
    print(f"  Average batch time: {avg_time*1000:.1f}ms")
    print(f"  Throughput: {batches_per_sec:.1f} batches/s")
    print(f"  Total overhead (setup + first batch): {setup_time + first_batch_time:.2f}s")
    
    return {
        'name': name,
        'setup_time': setup_time,
        'loader_time': loader_time,
        'first_batch_time': first_batch_time,
        'avg_batch_time': avg_time,
        'batches_per_sec': batches_per_sec,
        'total_overhead': setup_time + first_batch_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mmap-cfg', help='Mmap config file')
    parser.add_argument('--parquet-cfg', help='Parquet config file')
    parser.add_argument('--num-batches', type=int, default=100, help='Number of batches to benchmark')
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('medium')
    
    results = []
    
    if args.mmap_cfg:
        with open(args.mmap_cfg) as f:
            cfg = yaml.safe_load(f)
        
        data_options = cfg.get('data_options', {})
        training_options = cfg.get('training_options', {})
        
        dm = MmapDataModule(
            mmap_paths=data_options.get('mmap_paths', []),
            batch_size=training_options.get('batch_size', 512),
            val_split=data_options.get('val_split', 0.1),
            split_seed=data_options.get('seed', 42),
            num_workers=training_options.get('num_workers', 16),
            rescale=data_options.get('rescale', False),
        )
        
        result = benchmark_datamodule(dm, 'MMAP', args.num_batches)
        results.append(result)
    
    if args.parquet_cfg:
        with open(args.parquet_cfg) as f:
            cfg = yaml.safe_load(f)
        
        dm = ParquetDataModule(cfg)
        result = benchmark_datamodule(dm, 'Parquet', args.num_batches)
        results.append(result)
    
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        mmap = results[0]
        parquet = results[1]
        
        print(f"Setup time:       MMAP {mmap['setup_time']:.2f}s vs Parquet {parquet['setup_time']:.2f}s ({parquet['setup_time']/mmap['setup_time']:.1f}x)")
        print(f"First batch:      MMAP {mmap['first_batch_time']:.2f}s vs Parquet {parquet['first_batch_time']:.2f}s")
        print(f"Total overhead:   MMAP {mmap['total_overhead']:.2f}s vs Parquet {parquet['total_overhead']:.2f}s")
        print(f"Throughput:       MMAP {mmap['batches_per_sec']:.1f}/s vs Parquet {parquet['batches_per_sec']:.1f}/s ({mmap['batches_per_sec']/parquet['batches_per_sec']:.1f}x)")


if __name__ == '__main__':
    main()
