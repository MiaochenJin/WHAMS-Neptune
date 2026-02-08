#!/usr/bin/env python3
"""
Neptune Training Script with Mmap Support

This is a modified version of run.py that supports both parquet and mmap data formats.
Use data_format: "mmap" in config to use memory-mapped training data.

Usage:
    python run_mmap.py -cfg configs/whams-starting-mmap.cfg
"""

import argparse
import sys
import os
import yaml
import torch
from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

# Ensure local neptune package is in path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from whams_neptune.lightning_model import NeptuneLightningModule
from whams_neptune.dataloaders.parquet_dataset import ParquetDataModule
from whams_neptune.dataloaders.mmap_dataset import MmapDataModule, WeightingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Neptune Training with PyTorch Lightning")
    parser.add_argument("-cfg", "--cfg_file", required=True, help="Path to YAML config")
    parser.add_argument("-ckpt", "--checkpoint", help="Path to checkpoint to resume/fine-tune from")
    parser.add_argument("--run-checklist", action="store_true", help="Run data and model checks before training.")
    return parser.parse_args()


def run_checklist(dm):
    print("--- Running Data Checklist (First 10 batches) ---")
    print("setup stage")
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 10:
            break

        # batch structure: coords, features, labels, batch_ids
        # labels is [N, 5], morphology is at index 4
        labels = batch[2]

        if labels.shape[1] > 4:
            morph_labels = labels[:, 4].long().cpu()
            counts = torch.bincount(morph_labels)
            print(f"Batch {batch_idx:02d} | Total Events: {len(morph_labels)} | Counts (Category 0, 1, ...): {counts.tolist()}")
        else:
            print(f"Batch {batch_idx:02d} | Labels tensor size {labels.shape} too small for morphology check.")

    print("--- Checklist Complete ---")


def parse_weighting_config(weighting_opts: dict) -> WeightingConfig:
    """Parse weighting options from config into WeightingConfig dataclass.

    Args:
        weighting_opts: Dict with weighting options from YAML config

    Returns:
        WeightingConfig instance
    """
    # Convert to float explicitly since YAML may parse scientific notation as strings
    return WeightingConfig(
        enabled=weighting_opts.get("enabled", False),
        e_very_high=float(weighting_opts.get("e_very_high", 1e6)),
        e_high=float(weighting_opts.get("e_high", 1e5)),
        e_med=float(weighting_opts.get("e_med", 1e4)),
        w_very_high_upgoing=float(weighting_opts.get("w_very_high_upgoing", 20.0)),
        w_high_upgoing=float(weighting_opts.get("w_high_upgoing", 10.0)),
        w_med_upgoing=float(weighting_opts.get("w_med_upgoing", 3.0)),
        w_default=float(weighting_opts.get("w_default", 1.0)),
    )


def create_datamodule(cfg):
    """Create the appropriate DataModule based on config.

    If data_format is "mmap", use MmapDataModule.
    Otherwise, use ParquetDataModule (default).
    """
    data_format = cfg.get("data_format", "parquet")
    data_options = cfg.get("data_options", {})
    training_options = cfg.get("training_options", {})

    if data_format == "mmap":
        # Mmap data loading
        mmap_paths = data_options.get("mmap_paths", [])
        if not mmap_paths:
            raise ValueError("data_options.mmap_paths required for mmap format")

        # Parse weighting config if present
        weighting_config = None
        weighting_opts = data_options.get("weighting", {})
        if weighting_opts.get("enabled", False):
            weighting_config = parse_weighting_config(weighting_opts)
            print(f"Weighted sampling enabled:")
            print(f"  e_very_high={weighting_config.e_very_high:.0e} GeV, w={weighting_config.w_very_high_upgoing}")
            print(f"  e_high={weighting_config.e_high:.0e} GeV, w={weighting_config.w_high_upgoing}")
            print(f"  e_med={weighting_config.e_med:.0e} GeV, w={weighting_config.w_med_upgoing}")

        # Parse morphology mapping if present (for multiclass training)
        morphology_mapping = None
        morph_opts = data_options.get("morphology_mapping", None)
        if morph_opts is not None:
            morphology_mapping = {int(k): float(v) for k, v in morph_opts.items()}
            print(f"Morphology mapping: {morphology_mapping}")

        dm = MmapDataModule(
            mmap_paths=mmap_paths,
            batch_size=training_options.get("batch_size", 32),
            val_split=data_options.get("val_split", 0.1),
            split_seed=data_options.get("seed", 42),
            num_workers=training_options.get("num_workers", 4),
            rescale=data_options.get("rescale", False),
            morphology_mapping=morphology_mapping,
            weighting_config=weighting_config,
        )
    else:
        # Parquet data loading (original behavior)
        dm = ParquetDataModule(cfg)

    return dm


def main():
    torch.set_float32_matmul_precision('medium')
    args = parse_args()

    with open(args.cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    # Create DataModule
    dm = create_datamodule(cfg)

    if args.run_checklist:
        run_checklist(dm)
        return

    # LightningModule
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = NeptuneLightningModule.load_from_checkpoint(
            args.checkpoint,
            model_options=cfg['model_options'],
            training_options=cfg['training_options'],
            strict=False
        )
    else:
        model = NeptuneLightningModule(
            model_options=cfg['model_options'],
            training_options=cfg['training_options']
        )

    # Logger
    run_name = cfg.get("project_name", "whams-neptune-run")
    save_dir = Path(cfg.get("project_save_dir", "checkpoints")) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Copy config file to the run directory
    import shutil
    shutil.copy(args.cfg_file, save_dir / "config.yaml")

    wandb_logger = WandbLogger(
        name=run_name,
        id=run_name,
        project="WHAMS",
        save_dir=str(save_dir),
        config=cfg,
        resume="allow"
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_dir),
        filename='best_checkpoint',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = pl.Trainer(
        accelerator=cfg.get("accelerator", "auto"),
        devices=cfg.get("num_devices", 1),
        max_epochs=cfg['training_options']['epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=cfg['training_options'].get("log_every_n_steps", 10)
    )

    # Start training
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
