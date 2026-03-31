from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray



def stratified_event_split(events: np.ndarray, n: int, train_ratio: float, val_ratio: float, seed: int) -> SplitIndices:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    event_idx = indices[events == 0]
    censor_idx = indices[events == 1]
    rng.shuffle(event_idx)
    rng.shuffle(censor_idx)

    def split_group(group: np.ndarray):
        n_train = int(len(group) * train_ratio)
        n_val = int(len(group) * val_ratio)
        train = group[:n_train]
        val = group[n_train:n_train + n_val]
        test = group[n_train + n_val:]
        return train, val, test

    e_train, e_val, e_test = split_group(event_idx)
    c_train, c_val, c_test = split_group(censor_idx)
    train = np.concatenate([e_train, c_train])
    val = np.concatenate([e_val, c_val])
    test = np.concatenate([e_test, c_test])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return SplitIndices(train=train, val=val, test=test)


class ProcessedMultimodalDataset(Dataset):
    def __init__(self, data_path: str, indices: Sequence[int]):
        raw = np.load(data_path, allow_pickle=True)
        self.patient_ids = raw['patient_ids'].astype(str)
        self.times = raw['times'].astype(np.float32)
        self.events = raw['events'].astype(np.int64)
        self.bins = raw['bins'].astype(np.int64)
        self.gene_expr = raw['gene_expr'].astype(np.float32)
        self.gene_adj = raw['gene_adj'].astype(np.float32)
        self.wsi_features = raw['wsi_features']
        self.wsi_coords = raw['wsi_coords'] if 'wsi_coords' in raw.files else np.array([], dtype=object)
        self.modality_mask = raw['modality_mask'].astype(np.float32)
        self.indices = np.array(indices, dtype=np.int64)
        self.num_time_bins = int(raw['bin_edges'].shape[0] - 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        real_idx = int(self.indices[idx])
        coords = None
        if len(self.wsi_coords) > 0:
            coords = np.asarray(self.wsi_coords[real_idx], dtype=np.float32)
        return {
            'patient_id': self.patient_ids[real_idx],
            'time': self.times[real_idx],
            'event': self.events[real_idx],
            'bin': self.bins[real_idx],
            'wsi_features': np.asarray(self.wsi_features[real_idx], dtype=np.float32),
            'wsi_coords': coords,
            'gene_expr': self.gene_expr[real_idx],
            'gene_adj': self.gene_adj,
            'modality_mask': self.modality_mask[real_idx],
        }



def multimodal_collate(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    bsz = len(batch)
    max_nodes = max(item['wsi_features'].shape[0] for item in batch)
    feat_dim = batch[0]['wsi_features'].shape[1]
    has_coords = batch[0]['wsi_coords'] is not None
    wsi_features = np.zeros((bsz, max_nodes, feat_dim), dtype=np.float32)
    wsi_mask = np.zeros((bsz, max_nodes), dtype=np.float32)
    wsi_coords = np.zeros((bsz, max_nodes, 2), dtype=np.float32) if has_coords else None
    for i, item in enumerate(batch):
        n = item['wsi_features'].shape[0]
        wsi_features[i, :n] = item['wsi_features']
        wsi_mask[i, :n] = 1.0
        if has_coords and item['wsi_coords'] is not None:
            wsi_coords[i, :n] = item['wsi_coords']
    output = {
        'wsi_features': torch.from_numpy(wsi_features),
        'wsi_mask': torch.from_numpy(wsi_mask),
        'gene_expr': torch.from_numpy(np.stack([item['gene_expr'] for item in batch], axis=0)),
        'gene_adj': torch.from_numpy(batch[0]['gene_adj']),
        'times': torch.tensor([item['time'] for item in batch], dtype=torch.float32),
        'events': torch.tensor([item['event'] for item in batch], dtype=torch.long),
        'bins': torch.tensor([item['bin'] for item in batch], dtype=torch.long),
        'modality_mask': torch.from_numpy(np.stack([item['modality_mask'] for item in batch], axis=0).astype(np.float32)),
    }
    if has_coords:
        output['wsi_coords'] = torch.from_numpy(wsi_coords)
    return output
