from __future__ import annotations

import torch



def hazards_to_survival(hazards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    hazards = hazards.clamp(min=eps, max=1.0 - eps)
    return torch.cumprod(1.0 - hazards, dim=-1)



def discrete_time_nll(hazards: torch.Tensor, bins: torch.Tensor, censor_status: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    hazards = hazards.clamp(min=eps, max=1.0 - eps)
    surv = hazards_to_survival(hazards, eps=eps)
    batch_indices = torch.arange(hazards.size(0), device=hazards.device)
    k = bins.long().clamp(min=0, max=hazards.size(1) - 1)

    s_k = surv[batch_indices, k]
    s_prev = torch.ones_like(s_k)
    valid_prev = k > 0
    s_prev[valid_prev] = surv[batch_indices[valid_prev], k[valid_prev] - 1]
    h_k = hazards[batch_indices, k]

    censored = censor_status.float()
    observed = 1.0 - censored
    loss = -censored * torch.log(s_k + eps) - observed * torch.log(s_prev + eps) - observed * torch.log(h_k + eps)
    return loss.mean()



def risk_from_hazards(hazards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    surv = hazards_to_survival(hazards, eps=eps)
    final_surv = surv[:, -1]
    return -torch.log(final_surv + eps)



def confidence_from_hazards(hazards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    surv = hazards_to_survival(hazards, eps=eps)
    return (1.0 - surv[:, -1]).clamp(min=0.0, max=1.0)
