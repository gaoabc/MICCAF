from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from miccaf.config import load_config
from miccaf.dataset import ProcessedMultimodalDataset, multimodal_collate
from miccaf.engine import load_checkpoint
from miccaf.io_utils import save_json
from miccaf.survival import hazards_to_survival, risk_from_hazards
from miccaf.seed import set_seed
from train import build_model



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sample-index', type=int, required=True)
    parser.add_argument('--output-json', type=str, default='')
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.seed))
    dataset = ProcessedMultimodalDataset(args.data_path, indices=[args.sample_index])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=multimodal_collate)
    model = build_model(config, dataset)
    load_checkpoint(args.checkpoint, model)
    device = torch.device(str(config.device))
    model.to(device)
    model.eval()
    batch = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(batch)
        hazards = outputs['fused_hazards']
        survival = hazards_to_survival(hazards)
        risk = risk_from_hazards(hazards)
    result = {
        'sample_index': args.sample_index,
        'risk': float(risk.item()),
        'hazards': hazards.squeeze(0).detach().cpu().tolist(),
        'survival': survival.squeeze(0).detach().cpu().tolist(),
        'confidence_p': float(outputs['confidence_p'].item()),
        'confidence_g': float(outputs['confidence_g'].item()),
    }
    print(result)
    if args.output_json:
        save_json(result, args.output_json)


if __name__ == '__main__':
    main()
