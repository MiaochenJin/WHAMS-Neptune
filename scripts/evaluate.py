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
from whams_neptune.dataloaders.mmap_dataset import MmapDataModule

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

def plot_energy_dependent_angular_error(log_energies, angular_errors_deg, output_path):
    from scipy.ndimage import gaussian_filter1d
    
    plt.figure(figsize=(10, 8))
    
    # Define bins
    n_bins_x = 20
    n_bins_y = 50
    
    # Create 2D histogram
    # H[i, j] is the count in x_bin i and y_bin j
    H, xedges, yedges = np.histogram2d(log_energies, angular_errors_deg, bins=[n_bins_x, n_bins_y])
    
    # Column normalize: Normalize each row (energy bin) to sum to 1
    # This gives us P(error | energy)
    with np.errstate(divide='ignore', invalid='ignore'):
        H_norm = H / H.sum(axis=1, keepdims=True)
    H_norm = np.nan_to_num(H_norm) # Replace NaNs with 0
    
    # Plot heatmap
    # pcolormesh expects X, Y to define the corners of the quadrilaterals
    # H.T aligns with X (cols) and Y (rows)
    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, H_norm.T, cmap='Blues', shading='flat')
    cbar = plt.colorbar()
    cbar.set_label("Density (Column Normalized)")
    
    # Calculate Median and 50% range per energy bin
    bin_centers_x = (xedges[:-1] + xedges[1:]) / 2
    medians = []
    p25s = []
    p75s = []
    
    for i in range(n_bins_x):
        # Find data points in this bin
        mask = (log_energies >= xedges[i]) & (log_energies < xedges[i+1])
        vals = angular_errors_deg[mask]
        
        if len(vals) > 0:
            medians.append(np.median(vals))
            p25s.append(np.percentile(vals, 25))
            p75s.append(np.percentile(vals, 75))
        else:
            medians.append(np.nan)
            p25s.append(np.nan)
            p75s.append(np.nan)
            
    medians = np.array(medians)
    p25s = np.array(p25s)
    p75s = np.array(p75s)
    
    # Smooth the lines (sigma=1.5 is a gentle smoothing)
    # Helper to smooth and handle NaNs
    def smooth_safe(y, sigma=1.5):
        nans = np.isnan(y)
        if nans.all(): return y
        # Fill NaNs with interpolation for smoothing
        y_filled = y.copy()
        x_idxs = np.arange(len(y))
        y_filled[nans] = np.interp(x_idxs[nans], x_idxs[~nans], y[~nans])
        return gaussian_filter1d(y_filled, sigma=sigma)

    medians_smooth = smooth_safe(medians)
    p25s_smooth = smooth_safe(p25s)
    p75s_smooth = smooth_safe(p75s)
    
    # Plot Statistics
    plt.plot(bin_centers_x, medians_smooth, 'r-', linewidth=2, label='Median Angular Error')
    plt.fill_between(bin_centers_x, p25s_smooth, p75s_smooth, color='red', alpha=0.3, label='50% Range')
    
    plt.xlabel("log10(Primary Energy)")
    plt.ylabel("Angular Error (degrees)")
    plt.ylim(0, 40)
    plt.title("Angular Error vs. Primary Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved energy dependent angular error plot to {output_path}")

def plot_kappa_correlations(angular_errors, pred_cos_zenith, kappa, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Angular Error vs Kappa
    # X: Angular Error, Y: Kappa
    n_bins = 50
    H1, xedges1, yedges1 = np.histogram2d(angular_errors, kappa, bins=n_bins)
    
    # Normalize per column (fix x, distribution over y)
    with np.errstate(divide='ignore', invalid='ignore'):
        H1_norm = H1 / H1.sum(axis=1, keepdims=True)
    H1_norm = np.nan_to_num(H1_norm)
    
    X1, Y1 = np.meshgrid(xedges1, yedges1)
    # H.T because pcolormesh X corresponds to columns of matrix, Y to rows
    mesh1 = axes[0].pcolormesh(X1, Y1, H1_norm.T, cmap='Blues', shading='flat')
    fig.colorbar(mesh1, ax=axes[0], label='Density (Column Normalized)')
    axes[0].set_xlabel('Angular Error (degrees)')
    axes[0].set_ylabel('Kappa (Vector Magnitude)')
    axes[0].set_title('Kappa vs Angular Error')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Predicted Cosine Zenith vs Kappa
    # X: Pred Cos Zenith, Y: Kappa
    H2, xedges2, yedges2 = np.histogram2d(pred_cos_zenith, kappa, bins=n_bins)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        H2_norm = H2 / H2.sum(axis=1, keepdims=True)
    H2_norm = np.nan_to_num(H2_norm)
    
    X2, Y2 = np.meshgrid(xedges2, yedges2)
    mesh2 = axes[1].pcolormesh(X2, Y2, H2_norm.T, cmap='Blues', shading='flat')
    fig.colorbar(mesh2, ax=axes[1], label='Density (Column Normalized)')
    axes[1].set_xlabel('Predicted Cosine Zenith')
    axes[1].set_ylabel('Kappa (Vector Magnitude)')
    axes[1].set_title('Kappa vs Predicted Zenith')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved kappa correlation plots to {output_path}")

def plot_distance_error_distribution(distance_errors, output_path):
    plt.figure(figsize=(10, 6))
    # binwidth or binrange can be used to limit the histogram calculation
    sns.histplot(distance_errors, bins=50, kde=True, binrange=(0, 1000))
    plt.xlabel("Euclidean Distance Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Vertex Reconstruction Error")
    plt.xlim(0, 1000)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved distance error distribution to {output_path}")

def plot_pull_distribution(pulls, output_path):
    """
    Plots the pull distribution for x, y, z.
    pulls: (N, 3) array
    """
    plt.figure(figsize=(10, 6))
    
    # Flatten or plot individually. Here overlaid.
    labels = ['x', 'y', 'z']
    colors = ['r', 'g', 'b']
    
    for i in range(3):
        sns.histplot(pulls[:, i], bins=50, kde=True, 
                     element="step", fill=False,
                     stat="density", label=f"Pull {labels[i]}", color=colors[i])
    
    # Plot standard Gaussian for reference
    x = np.linspace(-5, 5, 100)
    plt.plot(x, 1./np.sqrt(2*np.pi) * np.exp(-x**2/2), 'k--', label='Standard Normal')
    
    plt.xlabel("Pull ((True - Pred) / Sigma)")
    plt.ylabel("Density")
    plt.title("Pull Distribution (Vertex Reco)")
    plt.legend()
    plt.grid(True)
    plt.xlim(-5, 5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved pull distribution to {output_path}")

def plot_energy_dependent_distance_error(log_energies, distance_errors, output_path):
    from scipy.ndimage import gaussian_filter1d
    
    plt.figure(figsize=(10, 8))
    
    n_bins_x = 20
    n_bins_y = 50
    
    # Set range for histograms to limit the bins
    range_lims = [[log_energies.min(), log_energies.max()], [0, 1000]]
    H, xedges, yedges = np.histogram2d(log_energies, distance_errors, bins=[n_bins_x, n_bins_y], range=range_lims)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        H_norm = H / H.sum(axis=1, keepdims=True)
    H_norm = np.nan_to_num(H_norm)
    
    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, H_norm.T, cmap='Blues', shading='flat')
    cbar = plt.colorbar()
    cbar.set_label("Density (Column Normalized)")
    
    bin_centers_x = (xedges[:-1] + xedges[1:]) / 2
    medians = []
    
    for i in range(n_bins_x):
        mask = (log_energies >= xedges[i]) & (log_energies < xedges[i+1])
        vals = distance_errors[mask]
        if len(vals) > 0:
            medians.append(np.median(vals))
        else:
            medians.append(np.nan)
            
    medians = np.array(medians)
    
    # Helper to smooth
    def smooth_safe(y, sigma=1.5):
        nans = np.isnan(y)
        if nans.all(): return y
        y_filled = y.copy()
        x_idxs = np.arange(len(y))
        y_filled[nans] = np.interp(x_idxs[nans], x_idxs[~nans], y[~nans])
        return gaussian_filter1d(y_filled, sigma=sigma)

    medians_smooth = smooth_safe(medians)
    
    plt.plot(bin_centers_x, medians_smooth, 'r-', linewidth=2, label='Median Error')
    
    plt.xlabel("log10(Primary Energy)")
    plt.ylabel("Distance Error")
    plt.title("Distance Error vs. Primary Energy")
    plt.ylim(0, 1000)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved energy dependent distance error plot to {output_path}")

def plot_sigma_vs_error(sigma, distance_errors, output_path):
    plt.figure(figsize=(10, 6))
    
    # 2D Histogram
    # Use density=True to normalize counts
    # Limit range of bins
    range_lims = [[0, 1000], [0, 1000]]
    h = plt.hist2d(sigma, distance_errors, bins=50, cmap='Blues', density=True, cmin=1e-5, range=range_lims)
    plt.colorbar(label='Density')
    
    # Plot y=x line (Error = Sigma) ideally, or some relation
    max_val = max(sigma.max(), distance_errors.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    
    plt.xlabel("Predicted Sigma")
    plt.ylabel("True Distance Error")
    plt.title("True Error vs Predicted Sigma")
    plt.ylim(0, 1000)
    plt.xlim(0, 1000)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved sigma vs error plot to {output_path}")

def plot_true_vs_predicted_energy(true_log_energy, pred_log_energy, output_path):
    """
    Plots a 2D histogram of true vs predicted energy (column-normalized).
    """
    plt.figure(figsize=(10, 8))

    n_bins = 50

    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(true_log_energy, pred_log_energy, bins=n_bins)

    # Column normalize: P(predicted | true)
    with np.errstate(divide='ignore', invalid='ignore'):
        H_norm = H / H.sum(axis=1, keepdims=True)
    H_norm = np.nan_to_num(H_norm)

    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, H_norm.T, cmap='Blues', shading='flat')
    cbar = plt.colorbar()
    cbar.set_label("Density (Column Normalized)")

    # Plot diagonal line (perfect reconstruction)
    min_val = min(true_log_energy.min(), pred_log_energy.min())
    max_val = max(true_log_energy.max(), pred_log_energy.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Reconstruction')

    plt.xlabel("log10(True Energy)")
    plt.ylabel("log10(Predicted Energy)")
    plt.title("True vs Predicted Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved true vs predicted energy plot to {output_path}")

def plot_energy_resolution_distribution(residuals, output_path):
    """
    Plots the distribution of energy reconstruction residuals.
    residuals: (pred - true) in log10(E) space
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.xlabel("Energy Residual (log10(E_pred) - log10(E_true))")
    plt.ylabel("Frequency")
    plt.title("Distribution of Energy Reconstruction Residuals")
    plt.axvline(0, color='r', linestyle='--', label='Perfect Reconstruction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved energy resolution distribution to {output_path}")

def plot_energy_pull_distribution(pulls, output_path):
    """
    Plots the pull distribution for energy: (true - pred) / sigma
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(pulls, bins=50, kde=True, stat="density", color='blue')

    # Plot standard Gaussian for reference
    x = np.linspace(-5, 5, 100)
    plt.plot(x, 1./np.sqrt(2*np.pi) * np.exp(-x**2/2), 'r--', linewidth=2, label='Standard Normal')

    plt.xlabel("Pull ((log10(E_true) - log10(E_pred)) / σ)")
    plt.ylabel("Density")
    plt.title("Energy Reconstruction Pull Distribution")
    plt.legend()
    plt.grid(True)
    plt.xlim(-5, 5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved energy pull distribution to {output_path}")

def plot_fractional_energy_error_vs_energy(true_log_energy, pred_log_energy, output_path):
    """
    Plots fractional energy error in linear space as a function of true energy.
    Fractional error = (E_pred - E_true) / E_true
    """
    from scipy.ndimage import gaussian_filter1d

    # Convert to linear energy
    true_energy = 10 ** true_log_energy
    pred_energy = 10 ** pred_log_energy

    # Calculate fractional error
    fractional_error = (pred_energy - true_energy) / true_energy

    plt.figure(figsize=(10, 8))

    n_bins_x = 20
    n_bins_y = 50

    # 2D histogram
    range_lims = [[true_log_energy.min(), true_log_energy.max()], [-2, 2]]
    H, xedges, yedges = np.histogram2d(true_log_energy, fractional_error, bins=[n_bins_x, n_bins_y], range=range_lims)

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        H_norm = H / H.sum(axis=1, keepdims=True)
    H_norm = np.nan_to_num(H_norm)

    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, H_norm.T, cmap='Blues', shading='flat')
    cbar = plt.colorbar()
    cbar.set_label("Density (Column Normalized)")

    # Calculate statistics per energy bin
    bin_centers_x = (xedges[:-1] + xedges[1:]) / 2
    medians = []
    rms_vals = []
    p25s = []
    p75s = []

    for i in range(n_bins_x):
        mask = (true_log_energy >= xedges[i]) & (true_log_energy < xedges[i+1])
        vals = fractional_error[mask]

        if len(vals) > 0:
            medians.append(np.median(vals))
            rms_vals.append(np.sqrt(np.mean(vals**2)))
            p25s.append(np.percentile(vals, 25))
            p75s.append(np.percentile(vals, 75))
        else:
            medians.append(np.nan)
            rms_vals.append(np.nan)
            p25s.append(np.nan)
            p75s.append(np.nan)

    medians = np.array(medians)
    rms_vals = np.array(rms_vals)
    p25s = np.array(p25s)
    p75s = np.array(p75s)

    # Smooth helper
    def smooth_safe(y, sigma=1.5):
        nans = np.isnan(y)
        if nans.all(): return y
        y_filled = y.copy()
        x_idxs = np.arange(len(y))
        y_filled[nans] = np.interp(x_idxs[nans], x_idxs[~nans], y[~nans])
        return gaussian_filter1d(y_filled, sigma=sigma)

    medians_smooth = smooth_safe(medians)
    rms_smooth = smooth_safe(rms_vals)
    p25s_smooth = smooth_safe(p25s)
    p75s_smooth = smooth_safe(p75s)

    # Plot statistics
    plt.plot(bin_centers_x, medians_smooth, 'r-', linewidth=2, label='Median Fractional Error')
    plt.plot(bin_centers_x, rms_smooth, 'g-', linewidth=2, label='RMS')
    plt.plot(bin_centers_x, -rms_smooth, 'g-', linewidth=2)
    plt.fill_between(bin_centers_x, p25s_smooth, p75s_smooth, color='red', alpha=0.2, label='50% Range')

    plt.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel("log10(True Energy)")
    plt.ylabel("Fractional Error (E_pred - E_true) / E_true")
    plt.ylim(-2, 2)
    plt.title("Fractional Energy Error vs. True Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved fractional energy error vs energy plot to {output_path}")

def plot_energy_resolution_vs_energy(true_energies, residuals, output_path):
    """
    Plots energy resolution (RMS of residuals) as a function of true energy.
    """
    from scipy.ndimage import gaussian_filter1d

    plt.figure(figsize=(10, 8))

    n_bins_x = 20
    n_bins_y = 50

    # 2D histogram
    range_lims = [[true_energies.min(), true_energies.max()], [-1.5, 1.5]]
    H, xedges, yedges = np.histogram2d(true_energies, residuals, bins=[n_bins_x, n_bins_y], range=range_lims)

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        H_norm = H / H.sum(axis=1, keepdims=True)
    H_norm = np.nan_to_num(H_norm)

    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, H_norm.T, cmap='Blues', shading='flat')
    cbar = plt.colorbar()
    cbar.set_label("Density (Column Normalized)")

    # Calculate statistics per energy bin
    bin_centers_x = (xedges[:-1] + xedges[1:]) / 2
    medians = []
    rms_vals = []
    p25s = []
    p75s = []

    for i in range(n_bins_x):
        mask = (true_energies >= xedges[i]) & (true_energies < xedges[i+1])
        vals = residuals[mask]

        if len(vals) > 0:
            medians.append(np.median(vals))
            rms_vals.append(np.sqrt(np.mean(vals**2)))
            p25s.append(np.percentile(vals, 25))
            p75s.append(np.percentile(vals, 75))
        else:
            medians.append(np.nan)
            rms_vals.append(np.nan)
            p25s.append(np.nan)
            p75s.append(np.nan)

    medians = np.array(medians)
    rms_vals = np.array(rms_vals)
    p25s = np.array(p25s)
    p75s = np.array(p75s)

    # Smooth helper
    def smooth_safe(y, sigma=1.5):
        nans = np.isnan(y)
        if nans.all(): return y
        y_filled = y.copy()
        x_idxs = np.arange(len(y))
        y_filled[nans] = np.interp(x_idxs[nans], x_idxs[~nans], y[~nans])
        return gaussian_filter1d(y_filled, sigma=sigma)

    medians_smooth = smooth_safe(medians)
    rms_smooth = smooth_safe(rms_vals)
    p25s_smooth = smooth_safe(p25s)
    p75s_smooth = smooth_safe(p75s)

    # Plot statistics
    plt.plot(bin_centers_x, medians_smooth, 'r-', linewidth=2, label='Median Residual')
    plt.plot(bin_centers_x, rms_smooth, 'g-', linewidth=2, label='RMS')
    plt.plot(bin_centers_x, -rms_smooth, 'g-', linewidth=2)
    plt.fill_between(bin_centers_x, p25s_smooth, p75s_smooth, color='red', alpha=0.2, label='50% Range')

    plt.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel("log10(True Energy)")
    plt.ylabel("Energy Residual (log10(E_pred) - log10(E_true))")
    plt.ylim(-1.5, 1.5)
    plt.title("Energy Resolution vs. True Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved energy resolution vs energy plot to {output_path}")

def plot_sigma_vs_residual_energy(sigma, abs_residuals, output_path):
    """
    Plots predicted sigma vs absolute residuals for calibration check.
    """
    plt.figure(figsize=(10, 6))

    # 2D Histogram with limited range
    range_lims = [[0, 2], [0, 2]]
    h = plt.hist2d(sigma, abs_residuals, bins=50, cmap='Blues', density=True, cmin=1e-5, range=range_lims)
    plt.colorbar(label='Density')

    # Plot y=x line (ideal calibration)
    max_val = 2
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Calibration (|residual| = σ)')

    plt.xlabel("Predicted σ (log10(E))")
    plt.ylabel("Absolute Residual |log10(E_pred) - log10(E_true)|")
    plt.title("Uncertainty Calibration: Predicted σ vs Absolute Residual")
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved sigma vs residual plot to {output_path}")

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

    # Standard morphology names by raw index
    RAW_MORPHOLOGY_NAMES = {
        0: '0T0C', 1: '0T1C', 2: '0TnC', 3: '1T0C',
        4: '1T1C', 5: '1TnC', 6: '2TnC', 7: 'Background',
    }

    # Raw labels (Y-axis): sorted keys of morphology_mapping, with names
    raw_labels_idx = sorted(list(morph_map.keys()))
    raw_labels = [RAW_MORPHOLOGY_NAMES.get(i, f"Morph {i}") for i in raw_labels_idx]

    # Model/Predicted labels (X-axis): from plotting_morphology or derived from mapping
    plotting_morph = data_opts.get("plotting_morphology", {})
    num_classes = cfg['model_options'].get('num_classes', 2)
    if plotting_morph:
        pred_labels = [plotting_morph[i] for i in sorted(plotting_morph.keys())]
    elif morph_map:
        # Derive class names from morphology_mapping
        # Group raw morphology names by their mapped class index
        class_to_raw = {}
        for raw_idx, class_idx in sorted(morph_map.items()):
            ci = int(class_idx)
            if ci < 0:
                continue  # skip ignored classes
            name = RAW_MORPHOLOGY_NAMES.get(int(raw_idx), f"Morph {raw_idx}")
            class_to_raw.setdefault(ci, []).append(name)
        pred_labels = []
        for ci in range(num_classes):
            names = class_to_raw.get(ci, [f"Class {ci}"])
            if len(names) == 1:
                pred_labels.append(names[0])
            else:
                pred_labels.append("+".join(names))
    else:
        pred_labels = [f"Class {i}" for i in range(num_classes)]

    print(f"True Morphology Labels (Y-axis): {raw_labels}")
    print(f"Predicted Labels (X-axis): {pred_labels}")

    # 3. Load DataModule
    data_format = cfg.get('data_format', 'parquet')
    print(f"Data format: {data_format}")

    if data_format == 'mmap':
        # Parse morphology_mapping for mmap
        morph_map_opts = data_opts.get('morphology_mapping', None)
        morphology_mapping = None
        if morph_map_opts:
            morphology_mapping = {int(k): float(v) for k, v in morph_map_opts.items()}
            print(f"Morphology mapping: {morphology_mapping}")

        mmap_paths = data_opts.get('mmap_paths', [])
        training_opts = cfg.get('training_options', {})
        dm = MmapDataModule(
            mmap_paths=mmap_paths,
            batch_size=training_opts.get('batch_size', 512),
            num_workers=training_opts.get('num_workers', 8),
            val_split=data_opts.get('val_split', 0.1),
            split_seed=data_opts.get('seed', 42),
            rescale=data_opts.get('rescale', True),
            morphology_mapping=morphology_mapping,
        )
    else:
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
        # Morphology evaluation (supports binary and multiclass, handles -1 labels)
        all_logits = all_preds.numpy()
        all_mapped_targets = all_labels[:, 4].long().numpy()
        all_raw_indices = all_labels[:, 5].long().numpy() if all_labels.shape[1] > 5 else np.full_like(all_mapped_targets, -1)

        # Filter out -1 labels (ignored classes, e.g. 2TnC with no data)
        valid_mask = all_mapped_targets >= 0
        n_filtered = (~valid_mask).sum()
        if n_filtered > 0:
            print(f"Filtering {n_filtered} events with mapped label=-1")
            all_logits = all_logits[valid_mask]
            all_mapped_targets = all_mapped_targets[valid_mask]
            all_raw_indices = all_raw_indices[valid_mask]

        pred_classes = np.argmax(all_logits, axis=1)

        # Overall accuracy
        overall_acc = (pred_classes == all_mapped_targets).mean()
        print(f"\nOverall Accuracy: {overall_acc:.4f} ({overall_acc*100:.1f}%)")

        n_raw = len(raw_labels)
        n_pred = len(pred_labels)

        # Raw morphology confusion matrix (raw_labels x pred_labels)
        cm = np.zeros((n_raw, n_pred), dtype=int)
        for t, p in zip(all_raw_indices, pred_classes):
            if t >= 0 and t < n_raw and p >= 0 and p < n_pred:
                cm[t, p] += 1

        print("\n--- Confusion Matrix (Raw Morphology, Counts) ---")
        print(cm)

        row_sums = cm.sum(axis=1)
        valid_rows = row_sums > 0
        cm_filtered = cm[valid_rows]
        raw_labels_filtered = [label for i, label in enumerate(raw_labels) if valid_rows[i]]
        row_sums_filtered = row_sums[valid_rows]

        if len(cm_filtered) > 0:
            cm_pct = (cm_filtered.astype('float') / row_sums_filtered[:, np.newaxis]) * 100
            print("\n--- Confusion Matrix (Raw Morphology, Percentage) ---")
            print(cm_pct)
            plot_confusion_matrix(
                cm_pct,
                x_labels=pred_labels,
                y_labels=raw_labels_filtered,
                output_path=os.path.join(args.output_dir, "confusion_matrix.png")
            )

        # Mapped-class confusion matrix (pred_labels x pred_labels)
        cm_mapped = np.zeros((n_pred, n_pred), dtype=int)
        for t, p in zip(all_mapped_targets, pred_classes):
            if 0 <= t < n_pred and 0 <= p < n_pred:
                cm_mapped[t, p] += 1

        row_sums_mapped = cm_mapped.sum(axis=1)
        valid_mapped = row_sums_mapped > 0
        if valid_mapped.any():
            cm_mapped_pct = np.zeros_like(cm_mapped, dtype=float)
            for i in range(n_pred):
                if row_sums_mapped[i] > 0:
                    cm_mapped_pct[i] = cm_mapped[i].astype(float) / row_sums_mapped[i] * 100
            print("\n--- Confusion Matrix (Mapped Classes, Percentage) ---")
            print(cm_mapped_pct)
            mapped_labels_filtered = [pred_labels[i] for i in range(n_pred) if valid_mapped[i]]
            plot_confusion_matrix(
                cm_mapped_pct[valid_mapped][:, :],
                x_labels=pred_labels,
                y_labels=mapped_labels_filtered,
                output_path=os.path.join(args.output_dir, "confusion_matrix_mapped.png")
            )

        # Per-class accuracy
        print("\n--- Per-Class Accuracy ---")
        for c in range(n_pred):
            mask = all_mapped_targets == c
            if mask.sum() > 0:
                acc = (pred_classes[mask] == c).mean()
                print(f"  {pred_labels[c]:>15s}: {acc:.4f} ({acc*100:.1f}%)  [N={mask.sum()}]")

        # ROC curves
        valid_for_roc = (all_mapped_targets >= 0) & (all_mapped_targets < n_pred)
        targets_roc = all_mapped_targets[valid_for_roc]
        logits_roc = all_logits[valid_for_roc]
        y_true_one_hot = np.eye(n_pred)[targets_roc]
        probs = torch.softmax(torch.from_numpy(logits_roc), dim=1).numpy()
        plot_roc_curve(
            y_true_one_hot,
            probs,
            n_pred,
            pred_labels,
            os.path.join(args.output_dir, "roc_curve.png")
        )

    elif task == 'angular_reco':
        from whams_neptune.losses.vmf import angular_distance_loss
        
        log_energies = all_labels[:, 0].numpy()
        true_dirs = all_labels[:, 1:4]
        # Calculate kappa (vector magnitude) before normalization
        kappa = torch.norm(all_preds, p=2, dim=1).numpy()

        # Normalize predictions
        pred_dirs = torch.nn.functional.normalize(all_preds, p=2, dim=1)

        # Check components are valid (<= 1)
        assert (torch.abs(pred_dirs) <= 1.0 + 1e-5).all(), "Normalized vector components must be <= 1"
        
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
        
        plot_energy_dependent_angular_error(
            log_energies,
            angular_errors_deg,
            output_path=os.path.join(args.output_dir, "energy_angular_error.png")
        )

        # Calculate cosine of zenith angles
        # True zenith must be calculated from the full vector
        true_zenith_rad = np.arccos(true_dirs[:, 2].numpy())
        true_cos_zenith = np.cos(true_zenith_rad)
        
        # Predicted zenith can be taken directly from the z-component of the normalized vector
        pred_cos_zenith = pred_dirs[:, 2].numpy()

        plot_zenith_distribution(
            true_cos_zenith,
            pred_cos_zenith,
            output_path=os.path.join(args.output_dir, "zenith_distribution.png")
        )

        plot_kappa_correlations(
            angular_errors_deg,
            pred_cos_zenith,
            kappa,
            output_path=os.path.join(args.output_dir, "kappa_correlations.png")
        )

    elif task == 'vertex_reco':
        # Indices 6, 7, 8 are vtx_x, vtx_y, vtx_z
        true_pos = all_labels[:, 6:9].numpy()
        pred_pos = all_preds[:, :3].numpy()
        pred_raw_sigma = all_preds[:, 3].numpy()
        
        # Apply softplus to get actual sigma
        # Use logaddexp for numerical stability with large inputs
        # softplus(x) = log(1 + exp(x)) = log(exp(0) + exp(x)) = logaddexp(0, x)
        pred_sigma = np.logaddexp(0, pred_raw_sigma) + 1e-6
        
        # Calculate Distance Error
        diff = true_pos - pred_pos
        dist_errors = np.linalg.norm(diff, axis=1)
        
        # Calculate Pulls: (True - Pred) / Sigma
        pulls = diff / pred_sigma[:, np.newaxis]
        
        log_energies = all_labels[:, 0].numpy()
        
        # Filter out NaNs or Infs for plotting
        valid_mask = np.isfinite(dist_errors) & np.isfinite(pred_sigma) & np.all(np.isfinite(pulls), axis=1)
        
        if not np.all(valid_mask):
            n_dropped = len(dist_errors) - np.sum(valid_mask)
            print(f"Warning: Dropping {n_dropped} events with NaN/Inf values.")
            dist_errors = dist_errors[valid_mask]
            pred_sigma = pred_sigma[valid_mask]
            pulls = pulls[valid_mask]
            log_energies = log_energies[valid_mask]

        print("\n--- Vertex Reconstruction Metrics ---")
        print(f"Median Distance Error: {np.median(dist_errors):.4f}")
        print(f"Mean Distance Error:   {np.mean(dist_errors):.4f}")
        print(f"Std Dev of Error:      {np.std(dist_errors):.4f}")
        
        # Plots
        plot_distance_error_distribution(
            dist_errors,
            output_path=os.path.join(args.output_dir, "distance_error_distribution.png")
        )
        
        plot_pull_distribution(
            pulls,
            output_path=os.path.join(args.output_dir, "pull_distribution.png")
        )
        
        plot_energy_dependent_distance_error(
            log_energies,
            dist_errors,
            output_path=os.path.join(args.output_dir, "energy_distance_error.png")
        )
        
        plot_sigma_vs_error(
            pred_sigma,
            dist_errors,
            output_path=os.path.join(args.output_dir, "sigma_vs_error.png")
        )

    elif task == 'energy_reco':
        # Index 0 is log10(energy)
        true_log_energy = all_labels[:, 0].numpy()
        pred_log_energy = all_preds[:, 0].numpy()

        # Check if Gaussian NLL (2D output: mean, sigma)
        if all_preds.shape[1] == 2:
            pred_raw_sigma = all_preds[:, 1].numpy()
            # Apply softplus to get actual sigma
            pred_sigma = np.logaddexp(0, pred_raw_sigma) + 1e-6

            # Calculate residuals
            residuals = pred_log_energy - true_log_energy

            # Calculate pulls: (true - pred) / sigma
            pulls = (true_log_energy - pred_log_energy) / pred_sigma

            # Filter out NaNs or Infs
            valid_mask = np.isfinite(residuals) & np.isfinite(pred_sigma) & np.isfinite(pulls)

            if not np.all(valid_mask):
                n_dropped = len(residuals) - np.sum(valid_mask)
                print(f"Warning: Dropping {n_dropped} events with NaN/Inf values.")
                residuals = residuals[valid_mask]
                pred_sigma = pred_sigma[valid_mask]
                pulls = pulls[valid_mask]
                true_log_energy = true_log_energy[valid_mask]
                pred_log_energy = pred_log_energy[valid_mask]

            print("\n--- Energy Reconstruction Metrics (Gaussian NLL) ---")
            print(f"Mean Absolute Error (log10(E)): {np.mean(np.abs(residuals)):.4f}")
            print(f"Median Absolute Error (log10(E)): {np.median(np.abs(residuals)):.4f}")
            print(f"RMS Error (log10(E)): {np.sqrt(np.mean(residuals**2)):.4f}")
            print(f"Bias (log10(E)): {np.mean(residuals):.4f}")
            print(f"Mean Predicted σ: {np.mean(pred_sigma):.4f}")
            print(f"Pull Mean: {np.mean(pulls):.4f} (should be ~0)")
            print(f"Pull Std: {np.std(pulls):.4f} (should be ~1)")

            # Plots
            plot_true_vs_predicted_energy(
                true_log_energy,
                pred_log_energy,
                output_path=os.path.join(args.output_dir, "true_vs_predicted_energy.png")
            )

            plot_energy_resolution_distribution(
                residuals,
                output_path=os.path.join(args.output_dir, "energy_residual_distribution.png")
            )

            plot_energy_pull_distribution(
                pulls,
                output_path=os.path.join(args.output_dir, "energy_pull_distribution.png")
            )

            plot_energy_resolution_vs_energy(
                true_log_energy,
                residuals,
                output_path=os.path.join(args.output_dir, "energy_resolution_vs_energy.png")
            )

            plot_fractional_energy_error_vs_energy(
                true_log_energy,
                pred_log_energy,
                output_path=os.path.join(args.output_dir, "fractional_energy_error_vs_energy.png")
            )

            plot_sigma_vs_residual_energy(
                pred_sigma,
                np.abs(residuals),
                output_path=os.path.join(args.output_dir, "sigma_vs_residual.png")
            )

        else:
            # MSE mode (1D output: just mean)
            residuals = pred_log_energy - true_log_energy

            # Filter out NaNs
            valid_mask = np.isfinite(residuals)
            if not np.all(valid_mask):
                n_dropped = len(residuals) - np.sum(valid_mask)
                print(f"Warning: Dropping {n_dropped} events with NaN/Inf values.")
                residuals = residuals[valid_mask]
                true_log_energy = true_log_energy[valid_mask]
                pred_log_energy = pred_log_energy[valid_mask]

            print("\n--- Energy Reconstruction Metrics (MSE) ---")
            print(f"Mean Absolute Error (log10(E)): {np.mean(np.abs(residuals)):.4f}")
            print(f"Median Absolute Error (log10(E)): {np.median(np.abs(residuals)):.4f}")
            print(f"RMS Error (log10(E)): {np.sqrt(np.mean(residuals**2)):.4f}")
            print(f"Bias (log10(E)): {np.mean(residuals):.4f}")

            # Plots
            plot_true_vs_predicted_energy(
                true_log_energy,
                pred_log_energy,
                output_path=os.path.join(args.output_dir, "true_vs_predicted_energy.png")
            )

            plot_energy_resolution_distribution(
                residuals,
                output_path=os.path.join(args.output_dir, "energy_residual_distribution.png")
            )

            plot_energy_resolution_vs_energy(
                true_log_energy,
                residuals,
                output_path=os.path.join(args.output_dir, "energy_resolution_vs_energy.png")
            )

            plot_fractional_energy_error_vs_energy(
                true_log_energy,
                pred_log_energy,
                output_path=os.path.join(args.output_dir, "fractional_energy_error_vs_energy.png")
            )

    else:
        print(f"Evaluation for task '{task}' is not implemented.")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
