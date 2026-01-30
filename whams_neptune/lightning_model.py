import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import numpy as np
from typing import List, Dict, Any

from .model import NeptuneModel
from .losses.vmf import von_mises_fisher_loss, angular_distance_loss
from .losses.gaussian import (
    gaussian_nll_1d,
    isotropic_gaussian_nll_3d,
    mean_euclidean_distance,
    mean_absolute_error_1d,
    mean_squared_error_1d,
    mean_predicted_uncertainty_1d
)

# A placeholder for a more complete loss function collection
def gaussian_nll_loss(pred_mean, pred_std, target):
    """Gaussian Negative Log Likelihood loss."""
    # Ensure std is positive
    pred_std = torch.nn.functional.softplus(pred_std) + 1e-6
    # Calculate log variance
    log_var = 2 * torch.log(pred_std)
    # Calculate NLL
    nll = 0.5 * (log_var + (target - pred_mean)**2 / torch.exp(log_var))
    return nll.mean()


class NeptuneLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_options: Dict[str, Any],
        training_options: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()

        # Determine output_dim based on task
        task = self.hparams.model_options['downstream_task']
        loss_fn = self.hparams.model_options['loss_fn']
        
        if task == 'angular_reco':
            output_dim = 3
        elif task in ['energy_reco', 'deposited_energy_reco']:
            output_dim = 2 if loss_fn == 'gaussian_nll' else 1
        elif task == 'vertex_reco':
            output_dim = 4 # x, y, z, sigma
        elif task == 'morphology_classification':
            output_dim = self.hparams.model_options.get('num_classes', 3)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        self.model = NeptuneModel(
            in_channels=self.hparams.model_options['in_channels'],
            num_patches=self.hparams.model_options['num_patches'],
            token_dim=self.hparams.model_options['token_dim'],
            num_layers=self.hparams.model_options['num_layers'],
            num_heads=self.hparams.model_options['num_heads'],
            hidden_dim=self.hparams.model_options['hidden_dim'],
            dropout=self.hparams.model_options['dropout'],
            output_dim=output_dim,
            k_neighbors=self.hparams.model_options['k_neighbors'],
        )
        
        self.validation_step_outputs = []

    def forward(self, coords, features, batch_ids):
        return self.model(coords, features, batch_ids)

    def _get_loss_function(self):
        task = self.hparams.model_options['downstream_task']
        loss_choice = self.hparams.model_options['loss_fn']
        
        if task == 'angular_reco':
            get_labels = lambda labels: labels[:, 1:4] # dir_x, dir_y, dir_z
            if loss_choice == 'vmf':
                return lambda preds, labels: von_mises_fisher_loss(preds, get_labels(labels))
            elif loss_choice == 'angular_distance':
                return lambda preds, labels: angular_distance_loss(preds, get_labels(labels))
        
        elif task == 'energy_reco':
            get_labels = lambda labels: labels[:, 0] # log_energy
            if loss_choice == 'gaussian_nll':
                return lambda preds, labels: gaussian_nll_1d(preds, get_labels(labels))
            else: # Default to MSE for energy
                return lambda preds, labels: F.mse_loss(preds.squeeze(), get_labels(labels))

        elif task == 'vertex_reco':
            # labels indices: 6, 7, 8 for vtx_x, vtx_y, vtx_z
            get_labels = lambda labels: labels[:, 6:9]
            if loss_choice == 'isotropic_gaussian_nll_3d':
                return lambda preds, labels: isotropic_gaussian_nll_3d(preds, get_labels(labels))
            else:
                 # Fallback to MSE on coordinates if specified
                return lambda preds, labels: F.mse_loss(preds[:, :3], get_labels(labels))

        elif task == 'morphology_classification':
            get_labels = lambda labels: labels[:, 4].long() # morphology_label
            return lambda preds, labels: F.cross_entropy(preds, get_labels(labels))
            
        raise ValueError(f"Unsupported task/loss combination: {task}/{loss_choice}")

    def step(self, batch):
        coords, features, labels, batch_ids = batch
        
        preds = self.model(coords, features, batch_ids)
        
        if self.hparams.model_options['downstream_task'] == 'angular_reco':
             # Normalize predictions for angular reconstruction
             preds = F.normalize(preds, p=2, dim=1)

        loss_func = self._get_loss_function()
        loss = loss_func(preds, labels)
        
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.training_options['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.training_options['batch_size'])
        self.validation_step_outputs.append({"preds": preds.detach(), "labels": labels.detach()})
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs], dim=0)
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs], dim=0)
        
        task = self.hparams.model_options['downstream_task']

        if task == 'angular_reco':
            true_dirs = all_labels[:, 1:4]
            angular_errors_rad = angular_distance_loss(all_preds, true_dirs, reduction='none')
            median_angular_error_deg = torch.rad2deg(torch.median(angular_errors_rad))
            self.log('val_median_angular_error_deg', median_angular_error_deg, prog_bar=True)

        elif task == 'energy_reco':
            true_energy = all_labels[:, 0]
            mae = mean_absolute_error_1d(all_preds, true_energy)
            mse = mean_squared_error_1d(all_preds, true_energy)
            self.log('val_mae', mae, prog_bar=True)
            self.log('val_mse', mse, prog_bar=True)

            # If using Gaussian NLL (2D output), also log mean predicted uncertainty
            if all_preds.shape[1] == 2:
                mean_uncertainty = mean_predicted_uncertainty_1d(all_preds)
                self.log('val_mean_sigma', mean_uncertainty, prog_bar=True)

        elif task == 'vertex_reco':
            true_pos = all_labels[:, 6:9]
            mean_dist = mean_euclidean_distance(all_preds, true_pos)
            self.log('val_mean_distance_error', mean_dist, prog_bar=True)

        elif task == 'morphology_classification':
            true_labels = all_labels[:, 4].long()
            predicted_labels = torch.argmax(all_preds, dim=1)
            
            # Compute accuracy
            correct = (predicted_labels == true_labels).sum().float()
            accuracy = correct / len(true_labels)
            
            self.log('val_accuracy', accuracy, prog_bar=True)
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=float(self.hparams.training_options['lr']), 
            weight_decay=float(self.hparams.training_options.get('weight_decay', 1e-5))
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.hparams.training_options['epochs'], 
            eta_min=float(self.hparams.training_options.get('eta_min', 1e-7))
        )
        return [optimizer], [scheduler]