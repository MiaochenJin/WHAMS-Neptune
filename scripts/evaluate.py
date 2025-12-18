import argparse
import sys
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# Ensure local whams_neptune package is in path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from whams_neptune.lightning_model import NeptuneLightningModule
from whams_neptune.dataloaders.parquet_dataset import ParquetDataModule

def parse_args():
    parser = argparse.ArgumentParser(description="Neptune Model Evaluation")
    parser.add_argument("-cfg", "--cfg_file", required=True, help="Path to YAML config used for training.")
    parser.add_argument("-ckpt", "--checkpoint_path", required=True, help="Path to a Lightning checkpoint (.ckpt file).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on.")
    parser.add_argument("-o", "--output_dir", default="evaluation_results", help="Directory to save plots and metrics.")
    parser.add_argument("-n", "--n-events", type=int, default=1, help="Number of events to sample per file (default: 1000). Set to -1 for all.")
    return parser.parse_args()

def plot_confusion_matrix(cm, x_labels, y_labels, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=x_labels, yticklabels=y_labels)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Morphology")
    plt.title("Morphology Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confusion matrix to {output_path}")

def plot_roc_curve(y_true_one_hot, y_score, n_classes, class_labels, output_path):
    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Check if we actually have samples for each class to avoid errors
    for i in range(n_classes):
        if np.sum(y_true_one_hot[:, i]) == 0:
            print(f"Warning: No positive samples for class {i} ({class_labels[i]}). Skipping ROC.")
            continue
            
        y_true_class = y_true_one_hot[:, i]
        y_score_class = y_score[:, i]
        
        fpr[i], tpr[i], _ = roc_curve(y_true_class, y_score_class)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('viridis', n_classes)
    
    for i in range(n_classes):
        if i in fpr:
            plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
                     label=f'ROC {class_labels[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved ROC curve to {output_path}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Configuration
    print(f"Loading configuration from: {args.cfg_file}")
    with open(args.cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Extract Labels from Config
    data_opts = cfg.get("data_options", {})
    morph_map = data_opts.get("morphology_mapping", {})
    
    # Raw labels (Y-axis): sorted keys of morphology_mapping
    raw_labels = sorted(list(morph_map.keys()))
    
    # Model/Predicted labels (X-axis): from plotting_morphology or inferred
    # In whams-starting.cfg: plotting_morphology: {0: "Background", 1: "Starting"}
    plotting_morph = data_opts.get("plotting_morphology", {})
    if plotting_morph:
        # Sort by index to ensure correct order
        pred_labels = [plotting_morph[i] for i in sorted(plotting_morph.keys())]
    else:
        # Fallback
        num_classes = cfg['model_options'].get('num_classes', 2)
        pred_labels = [f"Class {i}" for i in range(num_classes)]
        
    print(f"True Morphology Labels (Y-axis): {raw_labels}")
    print(f"Predicted Labels (X-axis): {pred_labels}")

    # 3. Load DataModule
    # Inject limit_per_file if requested
    if args.n_events > 0:
        print(f"Limiting to {args.n_events} events per file.")
        if "data_options" not in cfg:
            cfg["data_options"] = {}
        cfg["data_options"]["limit_per_file"] = args.n_events

    dm = ParquetDataModule(cfg)
    print("Setting up DataModule...")
    dm.setup(stage="fit")
    val_loader = dm.val_dataloader()

    # 4. Load Model
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    # We pass strict=False because sometimes checkpoints might have extra/missing keys 
    # if the model code changed slightly, but mainly to be robust. 
    # Ideally, we init with params from cfg.
    model = NeptuneLightningModule.load_from_checkpoint(
        args.checkpoint_path,
        model_options=cfg['model_options'],
        training_options=cfg['training_options']
    )
    model.eval()
    model.freeze()
    
    device = torch.device(args.device)
    model.to(device)

    # 5. Inference
    all_logits = []
    all_raw_indices = []
    all_mapped_targets = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            coords, features, labels, batch_ids = [t.to(device) for t in batch]
            
            # Forward pass
            logits = model(coords, features, batch_ids)
            
            # labels structure: [log_energy, dir_x, dir_y, dir_z, morphology_label, raw_morph_idx]
            # morphology_label (index 4) is the training target (0/1)
            # raw_morph_idx (index 5) is the index into raw_labels
            
            targets_mapped = labels[:, 4].long().cpu().numpy()
            
            # Check if we have the 6th element (raw index)
            if labels.shape[1] > 5:
                raw_idx = labels[:, 5].long().cpu().numpy()
            else:
                # Fallback if dataset wasn't updated correctly or old cache?
                # Should not happen if we updated ParquetDataset
                raw_idx = np.full_like(targets_mapped, -1)
                
            all_logits.append(logits.cpu().numpy())
            all_mapped_targets.append(targets_mapped)
            all_raw_indices.append(raw_idx)

    all_logits = np.concatenate(all_logits, axis=0)
    all_mapped_targets = np.concatenate(all_mapped_targets, axis=0)
    all_raw_indices = np.concatenate(all_raw_indices, axis=0)

    # 6. Confusion Matrix (Raw Morphology vs Predicted Class)
    pred_classes = np.argmax(all_logits, axis=1)
    
    n_raw = len(raw_labels)
    n_pred = len(pred_labels)
    
    cm = np.zeros((n_raw, n_pred), dtype=int)
    
    for t, p in zip(all_raw_indices, pred_classes):
        if t >= 0 and t < n_raw and p >= 0 and p < n_pred:
            cm[t, p] += 1
            
    print("\n--- Confusion Matrix (Counts) ---")
    print(cm)
    
    # Filter rows with zero events
    row_sums = cm.sum(axis=1)
    valid_rows = row_sums > 0
    
    cm_filtered = cm[valid_rows]
    raw_labels_filtered = [label for i, label in enumerate(raw_labels) if valid_rows[i]]
    row_sums_filtered = row_sums[valid_rows]
    
    if len(cm_filtered) == 0:
        print("Warning: No events found for any morphology category.")
    else:
        # Normalize to percentage
        # Avoid division by zero (though filtered rows shouldn't have sum 0)
        cm_pct = (cm_filtered.astype('float') / row_sums_filtered[:, np.newaxis]) * 100
        
        print("\n--- Confusion Matrix (Percentage) ---")
        print(cm_pct)

        plot_confusion_matrix(
            cm_pct,
            x_labels=pred_labels,
            y_labels=raw_labels_filtered,
            output_path=os.path.join(args.output_dir, "confusion_matrix.png")
        )
    
    # 7. ROC Curves (using mapped training targets)
    # We compute ROC against the binary/coarse targets used for training
    y_true_one_hot = np.eye(len(pred_labels))[all_mapped_targets]
    
    # Softmax for probabilities
    probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
    
    plot_roc_curve(
        y_true_one_hot, 
        probs, 
        len(pred_labels), 
        pred_labels, 
        os.path.join(args.output_dir, "roc_curve.png")
    )

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
