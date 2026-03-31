from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .graphs import build_gene_coexpression_adjacency, select_top_genes_by_signal


@dataclass
class ProcessedDataSummary:
    num_samples: int
    num_time_bins: int
    pathology_input_dim: int
    genomics_input_dim: int
    selected_gene_count: int



def _validate_raw(raw: Dict[str, np.ndarray]) -> None:
    required = ["patient_ids", "times", "events", "gene_expr", "wsi_features"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f"Missing raw keys: {missing}")



def _discretize_times(times: np.ndarray, num_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(times, q=np.linspace(0.0, 1.0, num_bins + 1))
    edges[0] = min(edges[0], times.min())
    edges[-1] = max(edges[-1], times.max()) + 1e-6
    bins = np.digitize(times, edges[1:-1], right=False)
    return bins.astype(np.int64), edges.astype(np.float32)



def _normalize_gene_expr(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)



def _truncate_wsi_features(wsi_features: np.ndarray, wsi_coords: np.ndarray | None, min_patches: int, max_patches: int):
    truncated_features = []
    truncated_coords = [] if wsi_coords is not None else None
    for i, feat in enumerate(wsi_features):
        feat = np.asarray(feat, dtype=np.float32)
        if feat.shape[0] < min_patches:
            repeats = int(np.ceil(min_patches / feat.shape[0]))
            feat = np.concatenate([feat] * repeats, axis=0)[:min_patches]
            if wsi_coords is not None:
                coords = np.asarray(wsi_coords[i], dtype=np.float32)
                coords = np.concatenate([coords] * repeats, axis=0)[:min_patches]
        else:
            feat = feat[:max_patches]
            if wsi_coords is not None:
                coords = np.asarray(wsi_coords[i], dtype=np.float32)[:max_patches]
        truncated_features.append(feat)
        if wsi_coords is not None:
            truncated_coords.append(coords)
    return np.array(truncated_features, dtype=object), None if truncated_coords is None else np.array(truncated_coords, dtype=object)



def load_raw_npz(path: str | Path) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {key: raw[key] for key in raw.files}



def process_raw_dataset(
    raw_path: str | Path,
    output_path: str | Path,
    num_time_bins: int,
    genomics_top_k: int,
    genomics_edge_threshold: float,
    min_wsi_patches: int,
    max_wsi_patches: int,
) -> ProcessedDataSummary:
    raw = load_raw_npz(raw_path)
    _validate_raw(raw)
    patient_ids = raw["patient_ids"].astype(str)
    times = raw["times"].astype(np.float32)
    events = raw["events"].astype(np.int64)
    gene_expr = raw["gene_expr"].astype(np.float32)
    wsi_features = raw["wsi_features"]
    wsi_coords = raw["wsi_coords"] if "wsi_coords" in raw else None
    modality_mask = raw["modality_mask"].astype(np.float32) if "modality_mask" in raw else np.ones((len(patient_ids), 2), dtype=np.float32)
    gene_names = raw["gene_names"].astype(str) if "gene_names" in raw else np.array([f"gene_{i}" for i in range(gene_expr.shape[1])], dtype=object)

    bins, bin_edges = _discretize_times(times, num_time_bins)
    selected_gene_idx = select_top_genes_by_signal(gene_expr, events, top_k=genomics_top_k)
    gene_expr_selected = gene_expr[:, selected_gene_idx]
    gene_expr_norm, gene_mean, gene_std = _normalize_gene_expr(gene_expr_selected)
    gene_adj = build_gene_coexpression_adjacency(gene_expr_norm, threshold=genomics_edge_threshold)

    wsi_features, wsi_coords = _truncate_wsi_features(
        wsi_features=wsi_features,
        wsi_coords=wsi_coords,
        min_patches=min_wsi_patches,
        max_patches=max_wsi_patches,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        patient_ids=patient_ids,
        times=times,
        events=events,
        bins=bins,
        bin_edges=bin_edges,
        gene_expr=gene_expr_norm,
        gene_mean=gene_mean,
        gene_std=gene_std,
        gene_adj=gene_adj.astype(np.float32),
        gene_names=gene_names[selected_gene_idx],
        selected_gene_idx=selected_gene_idx.astype(np.int64),
        wsi_features=wsi_features,
        wsi_coords=np.array([], dtype=object) if wsi_coords is None else wsi_coords,
        modality_mask=modality_mask,
    )
    dim_p = int(np.asarray(wsi_features[0]).shape[1])
    return ProcessedDataSummary(
        num_samples=len(patient_ids),
        num_time_bins=num_time_bins,
        pathology_input_dim=dim_p,
        genomics_input_dim=gene_expr_norm.shape[1],
        selected_gene_count=len(selected_gene_idx),
    )
