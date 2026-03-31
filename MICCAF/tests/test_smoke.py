from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from miccaf.config import load_config
from miccaf.dataset import ProcessedMultimodalDataset, multimodal_collate, stratified_event_split
from miccaf.model import MICCAFModel



def test_forward_smoke():
    root = Path(__file__).resolve().parent.parent
    data_path = root / 'toy_run' / 'processed' / 'processed_multimodal_toy.npz'
    if not data_path.exists():
        return
    cfg = load_config(root / 'configs' / 'default.yaml')
    ds = ProcessedMultimodalDataset(str(data_path), indices=[0, 1, 2, 3])
    loader = DataLoader(ds, batch_size=2, collate_fn=multimodal_collate)
    batch = next(iter(loader))
    sample = ds[0]
    model = MICCAFModel(
        pathology_input_dim=sample['wsi_features'].shape[1],
        genomics_input_dim=sample['gene_expr'].shape[0],
        num_time_bins=ds.num_time_bins,
        pathology_hidden_dim=int(cfg.model.pathology_hidden_dim),
        genomics_hidden_dim=int(cfg.model.genomics_hidden_dim),
        fusion_hidden_dim=int(cfg.model.fusion_hidden_dim),
        num_graph_layers=int(cfg.model.num_graph_layers),
        graphsage_neighbors=int(cfg.model.graphsage_neighbors),
        gat_heads=int(cfg.model.gat_heads),
        dropout=float(cfg.model.dropout),
        attention_pooling_dim=int(cfg.model.attention_pooling_dim),
        cra_depth=int(cfg.model.cra_depth),
        cra_hidden_dim=int(cfg.model.cra_hidden_dim),
        beta=float(cfg.model.beta),
        lambda_m=float(cfg.loss.lambda_m),
        lambda_k=float(cfg.loss.lambda_k),
        lambda_ibg=float(cfg.loss.lambda_ibg),
        lambda_g=float(cfg.loss.lambda_g),
        disable_iic=bool(cfg.aublations.disable_iic),
        disable_mmi=bool(cfg.aublations.disable_mmi),
        disable_iaf=bool(cfg.aublations.disable_iaf),
        dynamic_wsi_graph=bool(cfg.model.dynamic_wsi_graph),
        use_normalized_fusion=bool(cfg.model.use_normalized_fusion),
    )
    outputs = model(batch)
    assert outputs['fused_hazards'].shape[0] == 2
