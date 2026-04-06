from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .layers import CascadeResidualAutoencoder, DynamicWSIGraphEncoder, GenomicsGraphEncoder, MLP
from .losses import imputation_consistency_loss, multimodal_ib_loss
from .survival import confidence_from_hazards


@dataclass
class MICCAFOutputs:
    pathology_hazards: torch.Tensor
    genomics_hazards: torch.Tensor
    fused_hazards: torch.Tensor
    confidence_p: torch.Tensor
    confidence_g: torch.Tensor
    ib_loss: torch.Tensor
    imp_loss: torch.Tensor
    pathology_repr: torch.Tensor
    genomics_repr: torch.Tensor
    pathology_imputed: torch.Tensor
    genomics_imputed: torch.Tensor


class MICCAFModel(nn.Module):
    def __init__(
        self,
        pathology_input_dim: int,
        genomics_input_dim: int,
        num_time_bins: int,
        pathology_hidden_dim: int,
        genomics_hidden_dim: int,
        fusion_hidden_dim: int,
        num_graph_layers: int,
        graphsage_neighbors: int,
        gat_heads: int,
        dropout: float,
        attention_pooling_dim: int,
        cra_depth: int,
        cra_hidden_dim: int,
        beta: float,
        lambda_m: float,
        lambda_k: float,
        lambda_ibg: float,
        lambda_g: float,
        disable_iic: bool = False,
        disable_mmi: bool = False,
        disable_iaf: bool = False,
        dynamic_wsi_graph: bool = True,
        use_normalized_fusion: bool = True,
    ):
        super().__init__()
        self.pathology_encoder = DynamicWSIGraphEncoder(
            input_dim=pathology_input_dim,
            hidden_dim=pathology_hidden_dim,
            num_layers=num_graph_layers,
            neighbors=graphsage_neighbors,
            attn_pool_dim=attention_pooling_dim,
            dropout=dropout,
        )
        self.genomics_encoder = GenomicsGraphEncoder(
            hidden_dim=genomics_hidden_dim,
            num_layers=num_graph_layers,
            num_heads=gat_heads,
            attn_pool_dim=attention_pooling_dim,
            dropout=dropout,
        )
        self.pathology_proj = MLP([pathology_hidden_dim, fusion_hidden_dim, fusion_hidden_dim], dropout=dropout, activate_last=True)
        self.genomics_proj = MLP([genomics_hidden_dim, fusion_hidden_dim, fusion_hidden_dim], dropout=dropout, activate_last=True)
        self.cra_from_pathology = CascadeResidualAutoencoder(input_dim=fusion_hidden_dim * 2, hidden_dim=cra_hidden_dim, depth=cra_depth, dropout=dropout)
        self.cra_from_genomics = CascadeResidualAutoencoder(input_dim=fusion_hidden_dim * 2, hidden_dim=cra_hidden_dim, depth=cra_depth, dropout=dropout)
        self.pathology_head = nn.Sequential(nn.Linear(fusion_hidden_dim, fusion_hidden_dim), nn.ReLU(inplace=True), nn.Linear(fusion_hidden_dim, num_time_bins), nn.Sigmoid())
        self.genomics_head = nn.Sequential(nn.Linear(fusion_hidden_dim, fusion_hidden_dim), nn.ReLU(inplace=True), nn.Linear(fusion_hidden_dim, num_time_bins), nn.Sigmoid())
        self.beta = beta
        self.lambda_m = lambda_m
        self.lambda_k = lambda_k
        self.lambda_ibg = lambda_ibg
        self.lambda_g = lambda_g
        self.disable_iic = disable_iic
        self.disable_mmi = disable_mmi
        self.disable_iaf = disable_iaf
        self.dynamic_wsi_graph = dynamic_wsi_graph
        self.use_normalized_fusion = use_normalized_fusion
        self.fusion_hidden_dim = fusion_hidden_dim
        self.num_time_bins = num_time_bins

    def _make_label_features(self, bins: torch.Tensor, events: torch.Tensor, num_bins: int) -> torch.Tensor:
        norm_bins = bins.float().unsqueeze(-1) / max(float(num_bins - 1), 1.0)
        observed = (1.0 - events.float()).unsqueeze(-1)
        censored = events.float().unsqueeze(-1)
        return torch.cat([norm_bins, observed, censored], dim=-1)

    def _apply_modality_masks(self, pathology_repr: torch.Tensor, genomics_repr: torch.Tensor, modality_mask: torch.Tensor):
        pathology_repr = pathology_repr * modality_mask[:, 0:1]
        genomics_repr = genomics_repr * modality_mask[:, 1:2]
        return pathology_repr, genomics_repr

    def _mmi(self, pathology_repr: torch.Tensor, genomics_repr: torch.Tensor, modality_mask: torch.Tensor):
        placeholder_g = torch.zeros_like(genomics_repr)
        placeholder_p = torch.zeros_like(pathology_repr)
        v_p = torch.cat([pathology_repr, placeholder_g], dim=-1)
        v_g = torch.cat([placeholder_p, genomics_repr], dim=-1)
        v_p_completed, _ = self.cra_from_pathology(v_p)
        v_g_completed, _ = self.cra_from_genomics(v_g)
        x_p_hat = v_g_completed[:, :self.fusion_hidden_dim]
        x_g_hat = v_p_completed[:, self.fusion_hidden_dim:]

        pathology_final = torch.where(modality_mask[:, 0:1] > 0, pathology_repr, x_p_hat)
        genomics_final = torch.where(modality_mask[:, 1:2] > 0, genomics_repr, x_g_hat)
        imp_loss = imputation_consistency_loss(pathology_repr, x_p_hat, genomics_repr, x_g_hat, lambda_g=self.lambda_g)
        return pathology_final, genomics_final, x_p_hat, x_g_hat, imp_loss

    def _iaf(self, h_p: torch.Tensor, h_g: torch.Tensor):
        conf_p = confidence_from_hazards(h_p)
        conf_g = confidence_from_hazards(h_g)
        w_p = (1.0 - conf_g).pow(self.beta).unsqueeze(-1)
        w_g = (1.0 - conf_p).pow(self.beta).unsqueeze(-1)
        if self.use_normalized_fusion:
            denom = (w_p + w_g).clamp_min(1e-8)
            fused = (w_p / denom) * h_p + (w_g / denom) * h_g
        else:
            fused = w_p * h_p + w_g * h_g
        return fused, conf_p, conf_g

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pathology_outputs = self.pathology_encoder(
            x=batch['wsi_features'],
            mask=batch['wsi_mask'],
            coords=batch.get('wsi_coords'),
            dynamic_graph=self.dynamic_wsi_graph,
        )
        genomics_outputs = self.genomics_encoder(
            gene_expr=batch['gene_expr'],
            gene_adj=batch['gene_adj'],
        )
        pathology_repr = self.pathology_proj(pathology_outputs['final_pooled'])
        genomics_repr = self.genomics_proj(genomics_outputs['final_pooled'])
        pathology_repr, genomics_repr = self._apply_modality_masks(pathology_repr, genomics_repr, batch['modality_mask'])

        labels = self._make_label_features(batch['bins'], batch['events'], num_bins=self.num_time_bins)
        if self.disable_iic:
            ib_loss_value = pathology_repr.new_tensor(0.0)
        else:
            ib_loss_value = multimodal_ib_loss(
                pathology_raw=pathology_outputs['raw_pooled'],
                pathology_layers=pathology_outputs['layer_pooled_outputs'],
                genomics_raw=genomics_outputs['raw_pooled'],
                genomics_layers=genomics_outputs['layer_pooled_outputs'],
                labels=labels,
                lambda_m=self.lambda_m,
                lambda_ibg=self.lambda_ibg,
                bandwidth=self.lambda_k,
            )

        if self.disable_mmi:
            pathology_final = pathology_repr
            genomics_final = genomics_repr
            pathology_imputed = pathology_repr
            genomics_imputed = genomics_repr
            imp_loss_value = pathology_repr.new_tensor(0.0)
        else:
            pathology_final, genomics_final, pathology_imputed, genomics_imputed, imp_loss_value = self._mmi(
                pathology_repr=pathology_repr,
                genomics_repr=genomics_repr,
                modality_mask=batch['modality_mask'],
            )

        h_p = self.pathology_head(pathology_final)
        h_g = self.genomics_head(genomics_final)

        if self.disable_iaf:
            fused = 0.5 * (h_p + h_g)
            conf_p = confidence_from_hazards(h_p)
            conf_g = confidence_from_hazards(h_g)
        else:
            fused, conf_p, conf_g = self._iaf(h_p, h_g)

        return {
            'pathology_hazards': h_p,
            'genomics_hazards': h_g,
            'fused_hazards': fused,
            'confidence_p': conf_p,
            'confidence_g': conf_g,
            'ib_loss': ib_loss_value,
            'imp_loss': imp_loss_value,
            'pathology_repr': pathology_final,
            'genomics_repr': genomics_final,
            'pathology_imputed': pathology_imputed,
            'genomics_imputed': genomics_imputed,
        }
