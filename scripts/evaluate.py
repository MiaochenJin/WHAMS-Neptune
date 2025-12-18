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

def plot_angular_error_distribution(angular_errors_deg, output_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(angular_errors_deg, bins=50, kde=True)
    plt.xlabel("Angular Error (degrees)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Angular Reconstruction Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved angular error distribution to {output_path}")

def plot_zenith_distribution(true_cos_zenith, pred_cos_zenith, output_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(true_cos_zenith, bins=50, kde=True, color='blue', label='True Zenith', stat='density')
    sns.histplot(pred_cos_zenith, bins=50, kde=True, color='red', label='Predicted Zenith', stat='density')
    plt.xlabel("Cosine(Zenith Angle)")
    plt.ylabel("Density")
    plt.title("Distribution of True and Predicted Cosine(Zenith)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved zenith angle distribution to {output_path}")

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
    try:
        model = NeptuneLightningModule.load_from_checkpoint(
            args.checkpoint_path,
            model_options=cfg['model_options'],
            training_options=cfg['training_options']
        )
    except KeyError as e:
        if "pytorch-lightning_version" in str(e):
            print("Detected checkpoint without Lightning metadata. Loading weights directly.")
            model = NeptuneLightningModule(
                model_options=cfg['model_options'],
                training_options=cfg['training_options']
            )
            checkpoint = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
            # Adjust key if necessary, e.g., if weights are nested under 'state_dict'
            state_dict = checkpoint.get('state_dict', checkpoint)
            # The lightning module nests the actual model, so we need to adjust keys
            # from "layer.weight" to "model.layer.weight"
            model_state_dict = {"model." + k: v for k, v in state_dict.items()}
            model.load_state_dict(model_state_dict, strict=False)
        else:
            raise e
    model.eval()
    model.freeze()
    
    device = torch.device(args.device)
    model.to(device)

    # 5. Inference
    all_preds = []
    all_labels = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            coords, features, labels, batch_ids = [t.to(device) for t in batch]
            
            # Forward pass
            preds = model(coords, features, batch_ids)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 6. Evaluation based on task
    task = cfg['model_options'].get('downstream_task')

    if task == 'morphology_classification':
        # Existing morphology evaluation logic
        all_logits = all_preds.numpy()
        all_mapped_targets = all_labels[:, 4].long().numpy()
        all_raw_indices = all_labels[:, 5].long().numpy() if all_labels.shape[1] > 5 else np.full_like(all_mapped_targets, -1)

        pred_classes = np.argmax(all_logits, axis=1)
        
        n_raw = len(raw_labels)
        n_pred = len(pred_labels)
        
        cm = np.zeros((n_raw, n_pred), dtype=int)
        
        for t, p in zip(all_raw_indices, pred_classes):
            if t >= 0 and t < n_raw and p >= 0 and p < n_pred:
                cm[t, p] += 1
                
        print("\n--- Confusion Matrix (Counts) ---")
        print(cm)
        
        row_sums = cm.sum(axis=1)
        valid_rows = row_sums > 0
        cm_filtered = cm[valid_rows]
        raw_labels_filtered = [label for i, label in enumerate(raw_labels) if valid_rows[i]]
        row_sums_filtered = row_sums[valid_rows]
        
        if len(cm_filtered) > 0:
            cm_pct = (cm_filtered.astype('float') / row_sums_filtered[:, np.newaxis]) * 100
            print("\n--- Confusion Matrix (Percentage) ---")
            print(cm_pct)
            plot_confusion_matrix(
                cm_pct,
                x_labels=pred_labels,
                y_labels=raw_labels_filtered,
                output_path=os.path.join(args.output_dir, "confusion_matrix.png")
            )
        
        y_true_one_hot = np.eye(len(pred_labels))[all_mapped_targets]
        probs = torch.softmax(all_preds, dim=1).numpy()
        plot_roc_curve(
            y_true_one_hot,
            probs,
            len(pred_labels),
            pred_labels,
            os.path.join(args.output_dir, "roc_curve.png")
        )

    elif task == 'angular_reco':
        from whams_neptune.losses.vmf import angular_distance_loss
        
        true_dirs = all_labels[:, 1:4]
        # Normalize predictions
        pred_dirs = torch.nn.functional.normalize(all_preds, p=2, dim=1)
        
        angular_errors_rad = angular_distance_loss(pred_dirs, true_dirs, reduction='none')
        angular_errors_deg = np.rad2deg(angular_errors_rad.numpy())
        
        median_error = np.median(angular_errors_deg)
        mean_error = np.mean(angular_errors_deg)
        std_error = np.std(angular_errors_deg)
        
        print("\n--- Angular Reconstruction Metrics ---")
        print(f"Median Angular Error: {median_error:.4f} degrees")
        print(f"Mean Angular Error:   {mean_error:.4f} degrees")
        print(f"Std Dev of Error:   {std_error:.4f} degrees")

        plot_angular_error_distribution(
            angular_errors_deg,
            output_path=os.path.join(args.output_dir, "angular_error_distribution.png")
        )

        # Calculate cosine of zenith angles
        true_cos_zenith = true_dirs[:, 2].numpy()
        pred_cos_zenith = pred_dirs[:, 2].numpy()

        plot_zenith_distribution(
            true_cos_zenith,
            pred_cos_zenith,
            output_path=os.path.join(args.output_dir, "zenith_distribution.png")
        )

    else:
        print(f"Evaluation for task '{task}' is not implemented.")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
