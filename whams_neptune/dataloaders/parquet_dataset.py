import torch
import numpy as np
import awkward as ak
import pyarrow.parquet as pq
import glob
import os
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from typing import List, Union, Optional, Dict
import warnings

# --- Sampler and Collator Utilities (from old project) ---

class IrregularDataCollator:
    def __call__(self, samples):
        coords_list, features_list, labels_list = [], [], []
        batch_ids = []
        
        for i, (coords, features, labels) in enumerate(samples):
            coords_list.append(coords)
            features_list.append(features)
            labels_list.append(labels)
            batch_ids.extend([i] * len(coords))
            
        coords_cat = torch.cat(coords_list, dim=0)
        features_cat = torch.cat(features_list, dim=0)
        labels_cat = torch.cat(labels_list, dim=0)
        batch_ids_tensor = torch.tensor(batch_ids, dtype=torch.long)
        
        return coords_cat, features_cat, labels_cat, batch_ids_tensor

class ParquetFileSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, cumulative_lengths, shuffle_files=False):
        self.dataset = dataset
        self.cumulative_lengths = cumulative_lengths
        self.shuffle_files = shuffle_files

    def __iter__(self):
        file_indices = np.arange(len(self.cumulative_lengths) - 1)
        if self.shuffle_files:
            np.random.shuffle(file_indices)

        indices = []
        for file_idx in file_indices:
            start = self.cumulative_lengths[file_idx]
            end = self.cumulative_lengths[file_idx + 1]
            indices.extend(range(start, end))
        return iter(indices)

    def __len__(self):
        return len(self.dataset)

# --- Main Dataset and DataModule ---

class ParquetDataset(Dataset):
    REQUIRED_COLUMNS = [
        'pulse_sensor_pos_x', 'pulse_sensor_pos_y', 'pulse_sensor_pos_z',
        'pulse_summary_c_total', 'pulse_summary_c_500ns', 'pulse_summary_c_100ns',
        'pulse_summary_t_first', 'pulse_summary_t_last', 'pulse_summary_t20',
        'pulse_summary_t50', 'pulse_summary_t_mean', 'pulse_summary_t_std',
        'mc_primary_energy', 'mc_primary_dir_x', 'mc_primary_dir_y', 'mc_primary_dir_z',
        'mc_event_morphology'
    ]

    def __init__(self, files: List[str], morph_map: Optional[Dict[str, int]] = None, rescale: bool = False):
        self.files = files
        self.cumulative_lengths = []
        self._calculate_cumulative_lengths()
        
        self.current_file_index = -1
        self.current_data = None
        self.morphology_map = morph_map if morph_map is not None else {'Track': 0, 'Cascade': 1, 'Mixed': 2}
        self.rescale = rescale

    def _calculate_cumulative_lengths(self):
        num_events = [pq.ParquetFile(f).metadata.num_rows for f in self.files]
        self.cumulative_lengths = np.concatenate(([0], np.cumsum(num_events)))
        self.dataset_size = self.cumulative_lengths[-1]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.dataset_size:
            raise IndexError("Index out of range")
            
        file_index = np.searchsorted(self.cumulative_lengths, idx + 1) - 1
        
        if file_index != self.current_file_index:
            self.current_file_index = file_index
            # Only read required columns to avoid issues with extra columns like 'whams-bdt'
            self.current_data = ak.from_parquet(self.files[file_index], columns=self.REQUIRED_COLUMNS)
            
        local_idx = idx - self.cumulative_lengths[file_index]
        event = self.current_data[local_idx]

        # --- Feature Extraction ---
        pos_x = ak.to_numpy(event['pulse_sensor_pos_x']).astype(np.float32)
        pos_y = ak.to_numpy(event['pulse_sensor_pos_y']).astype(np.float32)
        pos_z = ak.to_numpy(event['pulse_sensor_pos_z']).astype(np.float32)
        t_first = ak.to_numpy(event['pulse_summary_t_first']).astype(np.float32)

        coords = torch.from_numpy(np.stack([pos_x, pos_y, pos_z, t_first], axis=1))

        # 9 summary stats as features
        features = torch.from_numpy(np.stack([
            ak.to_numpy(event['pulse_summary_c_total']).astype(np.float32),
            ak.to_numpy(event['pulse_summary_c_500ns']).astype(np.float32),
            ak.to_numpy(event['pulse_summary_c_100ns']).astype(np.float32),
            t_first,
            ak.to_numpy(event['pulse_summary_t_last']).astype(np.float32),
            ak.to_numpy(event['pulse_summary_t20']).astype(np.float32),
            ak.to_numpy(event['pulse_summary_t50']).astype(np.float32),
            ak.to_numpy(event['pulse_summary_t_mean']).astype(np.float32),
            ak.to_numpy(event['pulse_summary_t_std']).astype(np.float32),
        ], axis=1))

        if self.rescale:
             # Apply Solar-Nu style rescaling
             # Coords: x,y,z (m -> km), t (ns -> us)
             coords = coords * 1e-3
             # Features: log1p transformation
             features = torch.log1p(features)
        
        # --- Label Extraction ---
        log_energy = np.log10(max(float(event['mc_primary_energy']), 1e-6))
        dir_x = float(event['mc_primary_dir_x'])
        dir_y = float(event['mc_primary_dir_y'])
        dir_z = float(event['mc_primary_dir_z'])

        morphology_str = str(event['mc_event_morphology'])
        morphology_label = self.morphology_map.get(morphology_str, -1)
        if morphology_label == -1:
            warnings.warn(f"Unknown morphology '{morphology_str}' encountered. Defaulting to 0 (Background).")
            morphology_label = 0
            
        labels = torch.tensor([[
            log_energy, dir_x, dir_y, dir_z, morphology_label
        ]], dtype=torch.float32)

        return coords, features, labels

class ParquetDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.train_files = []
        self.val_files = []

    def setup(self, stage: Optional[str] = None):
        data_opts = self.cfg["data_options"]
        
        if "train_files" in data_opts and "valid_files" in data_opts:
            train_patterns = data_opts["train_files"]
            valid_patterns = data_opts["valid_files"]
        elif "data_paths" in data_opts:
            all_patterns = data_opts["data_paths"]
            val_split = data_opts.get("val_split", 0.1)
            
            all_files = sorted([f for pattern in all_patterns for f in glob.glob(pattern)])
            
            np.random.seed(data_opts.get("seed", 42))
            np.random.shuffle(all_files)
            
            split_idx = int(len(all_files) * (1 - val_split))
            self.train_files = all_files[:split_idx]
            self.val_files = all_files[split_idx:]
        else:
            raise ValueError("Config must provide 'train_files'/'valid_files' or 'data_paths'.")

        if not self.train_files and "train_files" in data_opts:
            self.train_files = sorted([f for pattern in train_patterns for f in glob.glob(pattern)])
        if not self.val_files and "valid_files" in data_opts:
            self.val_files = sorted([f for pattern in valid_patterns for f in glob.glob(pattern)])

        morph_map = data_opts.get("morphology_mapping")
        rescale = data_opts.get("rescale", False)
        if rescale: print("Rescaling of features is enabled.")
        self.train_dataset = ParquetDataset(self.train_files, morph_map=morph_map, rescale=rescale)
        self.val_dataset = ParquetDataset(self.val_files, morph_map=morph_map, rescale=rescale)

    def _create_dataloader(self, dataset, shuffle):
        sampler = ParquetFileSampler(dataset, dataset.cumulative_lengths, shuffle_files=shuffle)
        
        return DataLoader(
            dataset,
            batch_size=self.cfg['training_options']['batch_size'],
            sampler=sampler,
            num_workers=self.cfg['training_options'].get('num_workers', 0),
            collate_fn=IrregularDataCollator(),
            pin_memory=True,
            persistent_workers=bool(self.cfg['training_options'].get('num_workers', 0) > 0)
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False)