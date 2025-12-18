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

def parse_args():
    parser = argparse.ArgumentParser(description="Neptune Training with PyTorch Lightning")
    parser.add_argument("-cfg", "--cfg_file", required=True, help="Path to YAML config")
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
            # Move to CPU for bincount if it's on GPU, though dataloader usually yields CPU
            morph_labels = labels[:, 4].long().cpu()
            counts = torch.bincount(morph_labels)
            print(f"Batch {batch_idx:02d} | Total Events: {len(morph_labels)} | Counts (Category 0, 1, ...): {counts.tolist()}")
        else:
            print(f"Batch {batch_idx:02d} | Labels tensor size {labels.shape} too small for morphology check.")

    print("--- Checklist Complete ---")


def main():
    torch.set_float32_matmul_precision('medium')
    args = parse_args()

    with open(args.cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    
    if args.run_checklist:
        dm = ParquetDataModule(cfg)
        run_checklist(dm)
        return # Exit after checklist is complete

    # DataModule
    dm = ParquetDataModule(cfg)

    # LightningModule
    model = NeptuneLightningModule(
        model_options=cfg['model_options'],
        training_options=cfg['training_options']
    )

    # Logger
    run_name = cfg.get("project_name", "whams-neptune-run")
    # Create a unique directory for this run
    save_dir = Path(cfg.get("project_save_dir", "checkpoints")) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Copy config file to the run directory
    import shutil
    shutil.copy(args.cfg_file, save_dir / "config.yaml")

    wandb_logger = WandbLogger(
        name=run_name,
        id=run_name,
        project="WHAMS",  # Use "WHAMS" as the main project name
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
