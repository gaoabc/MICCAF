from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import torch



def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved



def detach_metrics(metrics: Dict[str, torch.Tensor | float]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            output[key] = float(value.detach().cpu().item())
        else:
            output[key] = float(value)
    return output



def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def latest_checkpoint(directory: str | Path) -> Path | None:
    directory = Path(directory)
    checkpoints = sorted(directory.glob('*.pt'))
    return checkpoints[-1] if checkpoints else None
