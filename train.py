from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from miccaf.config import load_config
from miccaf.dataset import ProcessedMultimodalDataset, multimodal_collate, stratified_event_split
from miccaf.engine import fit_model
from miccaf.io_utils import ensure_dir, save_json
from miccaf.model import MICCAFModel
from miccaf.seed import set_seed
from miccaf.utils import count_parameters



def build_model(config, dataset: ProcessedMultimodalDataset) -> MICCAFModel:
    sample = dataset[0]
    pathology_input_dim = int(sample['wsi_features'].shape[1])
    genomics_input_dim = int(sample['gene_expr'].shape[0])
    model = MICCAFModel(
        pathology_input_dim=pathology_input_dim,
        genomics_input_dim=genomics_input_dim,
        num_time_bins=dataset.num_time_bins,
        pathology_hidden_dim=int(config.model.pathology_hidden_dim),
        genomics_hidden_dim=int(config.model.genomics_hidden_dim),
        fusion_hidden_dim=int(config.model.fusion_hidden_dim),
        num_graph_layers=int(config.model.num_graph_layers),
        graphsage_neighbors=int(config.model.graphsage_neighbors),
        gat_heads=int(config.model.gat_heads),
        dropout=float(config.model.dropout),
        attention_pooling_dim=int(config.model.attention_pooling_dim),
        cra_depth=int(config.model.cra_depth),
        cra_hidden_dim=int(config.model.cra_hidden_dim),
        beta=float(config.model.beta),
        lambda_m=float(config.loss.lambda_m),
        lambda_k=float(config.loss.lambda_k),
        lambda_ibg=float(config.loss.lambda_ibg),
        lambda_g=float(config.loss.lambda_g),
        disable_iic=bool(config.aublations.disable_iic),
        disable_mmi=bool(config.aublations.disable_mmi),
        disable_iaf=bool(config.aublations.disable_iaf),
        dynamic_wsi_graph=bool(config.model.dynamic_wsi_graph),
        use_normalized_fusion=bool(config.model.use_normalized_fusion),
    )
    return model



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.seed))
    output_dir = ensure_dir(args.output_dir)
    patient_count = len(ProcessedMultimodalDataset(args.data_path, indices=[0]).patient_ids)
    dataset_all = ProcessedMultimodalDataset(args.data_path, indices=list(range(patient_count)))
    splits = stratified_event_split(
        events=dataset_all.events,
        n=len(dataset_all.patient_ids),
        train_ratio=float(config.data.train_ratio),
        val_ratio=float(config.data.val_ratio),
        seed=int(config.data.split_seed),
    )
    train_dataset = ProcessedMultimodalDataset(args.data_path, indices=splits.train)
    val_dataset = ProcessedMultimodalDataset(args.data_path, indices=splits.val)
    model = build_model(config, train_dataset)
    device = torch.device(str(config.device))
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=int(config.training.batch_size), shuffle=True, num_workers=int(config.training.num_workers), collate_fn=multimodal_collate)
    val_loader = DataLoader(val_dataset, batch_size=int(config.training.batch_size), shuffle=False, num_workers=int(config.training.num_workers), collate_fn=multimodal_collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training.lr), weight_decay=float(config.training.weight_decay))
    fit_model(model, train_loader, val_loader, optimizer, device, output_dir, config)
    save_json({'num_parameters': count_parameters(model)}, output_dir / 'model_info.json')
    print(f"Training finished. Best checkpoint saved to: {Path(output_dir) / 'best.pt'}")


if __name__ == '__main__':
    main()
