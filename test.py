from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from miccaf.config import load_config
from miccaf.dataset import ProcessedMultimodalDataset, multimodal_collate, stratified_event_split
from miccaf.engine import evaluate_model, load_checkpoint
from miccaf.io_utils import save_json
from miccaf.model import MICCAFModel
from miccaf.seed import set_seed
from train import build_model



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/mnt/data2/home/Public/config/config.yaml')
    parser.add_argument('--data-path', type=str, default='/mnt/data2/home/Public/datas/')
    parser.add_argument('--checkpoint', type=str, default='/mnt/data2/home/Public/results/best.pt')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output-json', type=str, default='/mnt/data2/home/Public/results/output_json/')
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.seed))
    dataset_ref = ProcessedMultimodalDataset(args.data_path, indices=[0])
    patient_count = len(dataset_ref.patient_ids)
    all_dataset = ProcessedMultimodalDataset(args.data_path, indices=list(range(patient_count)))
    splits = stratified_event_split(
        events=all_dataset.events,
        n=len(all_dataset.patient_ids),
        train_ratio=float(config.data.train_ratio),
        val_ratio=float(config.data.val_ratio),
        seed=int(config.data.split_seed),
    )
    split_indices = {'train': splits.train, 'val': splits.val, 'test': splits.test}[args.split]
    dataset = ProcessedMultimodalDataset(args.data_path, indices=split_indices)
    loader = DataLoader(dataset, batch_size=int(config.training.batch_size), shuffle=False, num_workers=int(config.training.num_workers), collate_fn=multimodal_collate)
    model = build_model(config, dataset)
    load_checkpoint(args.checkpoint, model)
    device = torch.device(str(config.device))
    model.to(device)
    result = evaluate_model(model, loader, device, config)
    print(result)
    if args.output_json:
        save_json(result, args.output_json)


if __name__ == '__main__':
    main()
