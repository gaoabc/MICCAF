from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn.functional as F

from .survival import discrete_time_nll



def gaussian_kernel(x: torch.Tensor, bandwidth: float) -> torch.Tensor:
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    sq_dist = x_norm + x_norm.T - 2.0 * x @ x.T
    sq_dist = torch.clamp(sq_dist, min=0.0)
    return torch.exp(-sq_dist / (2.0 * (bandwidth ** 2) + 1e-8))



def hsic(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> torch.Tensor:
    if x.size(0) < 2:
        return x.new_tensor(0.0)
    kx = gaussian_kernel(x, bandwidth)
    ky = gaussian_kernel(y, bandwidth)
    n = x.size(0)
    h = torch.eye(n, device=x.device, dtype=x.dtype) - torch.ones(n, n, device=x.device, dtype=x.dtype) / n
    return torch.trace(kx @ h @ ky @ h) / ((n - 1) ** 2)



def modality_ib_loss(raw_pooled: torch.Tensor, layer_outputs: List[torch.Tensor], labels: torch.Tensor, lambda_m: float, bandwidth: float) -> torch.Tensor:
    loss = raw_pooled.new_tensor(0.0)
    for layer in layer_outputs:
        loss = loss + hsic(raw_pooled, layer, bandwidth) - lambda_m * hsic(layer, labels, bandwidth)
    return loss



def multimodal_ib_loss(
    pathology_raw: torch.Tensor,
    pathology_layers: List[torch.Tensor],
    genomics_raw: torch.Tensor,
    genomics_layers: List[torch.Tensor],
    labels: torch.Tensor,
    lambda_m: float,
    lambda_ibg: float,
    bandwidth: float,
) -> torch.Tensor:
    loss_p = modality_ib_loss(pathology_raw, pathology_layers, labels, lambda_m=lambda_m, bandwidth=bandwidth)
    loss_g = modality_ib_loss(genomics_raw, genomics_layers, labels, lambda_m=lambda_m, bandwidth=bandwidth)
    return loss_p + lambda_ibg * loss_g



def imputation_consistency_loss(x_p: torch.Tensor, x_p_hat: torch.Tensor, x_g: torch.Tensor, x_g_hat: torch.Tensor, lambda_g: float) -> torch.Tensor:
    return F.mse_loss(x_p_hat, x_p) + lambda_g * F.mse_loss(x_g_hat, x_g)



def total_loss(
    fused_hazards: torch.Tensor,
    bins: torch.Tensor,
    events: torch.Tensor,
    ib_loss_value: torch.Tensor,
    imp_loss_value: torch.Tensor,
    lambda_ib: float,
    lambda_imp: float,
) -> dict:
    surv = discrete_time_nll(fused_hazards, bins=bins, censor_status=events)
    total = surv + lambda_ib * ib_loss_value + lambda_imp * imp_loss_value
    return {
        'loss_total': total,
        'loss_surv': surv,
        'loss_ib': ib_loss_value,
        'loss_imp': imp_loss_value,
    }
