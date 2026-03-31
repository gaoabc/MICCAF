from __future__ import annotations

from typing import Tuple

import numpy as np
import torch



def select_top_genes_by_signal(gene_expr: np.ndarray, events: np.ndarray, top_k: int) -> np.ndarray:
    if gene_expr.ndim != 2:
        raise ValueError("gene_expr must be 2D")
    observed = (events == 0)
    if observed.sum() == 0 or observed.sum() == len(events):
        scores = np.var(gene_expr, axis=0)
    else:
        mean_event = gene_expr[observed].mean(axis=0)
        mean_censor = gene_expr[~observed].mean(axis=0)
        var_event = gene_expr[observed].var(axis=0) + 1e-6
        var_censor = gene_expr[~observed].var(axis=0) + 1e-6
        scores = np.abs(mean_event - mean_censor) / np.sqrt(var_event + var_censor)
    order = np.argsort(scores)[::-1]
    return order[:top_k]



def build_gene_coexpression_adjacency(gene_expr: np.ndarray, threshold: float) -> np.ndarray:
    corr = np.corrcoef(gene_expr.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    adj = (np.abs(corr) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    return adj



def knn_adjacency_from_points(points: torch.Tensor, valid_mask: torch.Tensor, k: int) -> torch.Tensor:

    bsz, num_nodes, _ = points.shape
    adjacency = torch.zeros(bsz, num_nodes, num_nodes, device=points.device, dtype=points.dtype)
    large_value = 1e6
    for b in range(bsz):
        mask = valid_mask[b]
        idx = torch.where(mask > 0)[0]
        if idx.numel() == 0:
            continue
        sample = points[b, idx]
        dists = torch.cdist(sample, sample, p=2)
        dists.fill_diagonal_(large_value)
        neighbor_k = min(k, max(int(idx.numel()) - 1, 1))
        nn_idx = dists.topk(neighbor_k, largest=False, dim=-1).indices
        adjacency[b, idx[:, None], idx[nn_idx]] = 1.0
        adjacency[b, idx, idx] = 1.0
    return adjacency



def normalize_adjacency(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    degree = adj.sum(dim=-1, keepdim=True).clamp_min(eps)
    return adj / degree
