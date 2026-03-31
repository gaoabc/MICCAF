from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graphs import knn_adjacency_from_points, normalize_adjacency


class MLP(nn.Module):
    def __init__(self, dims: List[int], dropout: float = 0.0, activate_last: bool = False):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or activate_last:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(mask <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        return pooled


class GraphSAGELayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(input_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_norm = normalize_adjacency(adj)
        neigh = torch.bmm(adj_norm, x)
        out = self.linear(torch.cat([x, neigh], dim=-1))
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        return out


class DenseGATLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim * num_heads, bias=False)
        self.attn_src = nn.Parameter(torch.randn(num_heads, output_dim))
        self.attn_dst = nn.Parameter(torch.randn(num_heads, output_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        bsz, nodes, _ = x.shape
        h = self.proj(x).view(bsz, nodes, self.num_heads, self.output_dim)
        src_logits = torch.einsum('bnhd,hd->bnh', h, self.attn_src)
        dst_logits = torch.einsum('bnhd,hd->bnh', h, self.attn_dst)
        logits = src_logits.unsqueeze(2) + dst_logits.unsqueeze(1)
        logits = F.leaky_relu(logits, negative_slope=0.2)
        mask = adj.unsqueeze(-1) > 0
        logits = logits.masked_fill(~mask, -1e9)
        attn = torch.softmax(logits, dim=2)
        attn = self.dropout(attn)
        out = torch.einsum('bijh,bjhd->bihd', attn, h).reshape(bsz, nodes, self.num_heads * self.output_dim)
        return out


class ResidualAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = MLP([input_dim, hidden_dim, hidden_dim], dropout=dropout, activate_last=True)
        self.decoder = MLP([hidden_dim, hidden_dim, input_dim], dropout=dropout, activate_last=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class CascadeResidualAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        current = x
        deltas = []
        for block in self.blocks:
            delta = block(current)
            current = current + delta
            deltas.append(delta)
        return current, deltas


class DynamicWSIGraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, neighbors: int, attn_pool_dim: int, dropout: float):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([
            GraphSAGELayer(dims[i], dims[i + 1], dropout=dropout)
            for i in range(num_layers)
        ])
        self.poolers = nn.ModuleList([
            AttentionPooling(hidden_dim, attn_pool_dim)
            for _ in range(num_layers)
        ])
        self.input_pool = AttentionPooling(input_dim, attn_pool_dim)
        self.neighbors = neighbors

    def forward(self, x: torch.Tensor, mask: torch.Tensor, coords: torch.Tensor | None = None, dynamic_graph: bool = True):
        points = coords if coords is not None else x
        raw_pooled = self.input_pool(x, mask)
        layer_node_outputs = []
        layer_pooled_outputs = []
        current = x
        base_adj = knn_adjacency_from_points(points, mask, self.neighbors)
        for layer, pool in zip(self.layers, self.poolers):
            adj = knn_adjacency_from_points(current if dynamic_graph else points, mask, self.neighbors) if dynamic_graph else base_adj
            current = layer(current, adj)
            layer_node_outputs.append(current)
            layer_pooled_outputs.append(pool(current, mask))
        return {
            'raw_pooled': raw_pooled,
            'layer_node_outputs': layer_node_outputs,
            'layer_pooled_outputs': layer_pooled_outputs,
            'final_pooled': layer_pooled_outputs[-1],
        }


class GenomicsGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, attn_pool_dim: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList()
        self.poolers = nn.ModuleList()
        in_dim = hidden_dim
        for _ in range(num_layers):
            layer = DenseGATLayer(input_dim=in_dim, output_dim=hidden_dim // num_heads, num_heads=num_heads, dropout=dropout)
            self.layers.append(layer)
            self.poolers.append(AttentionPooling(hidden_dim, attn_pool_dim))
            in_dim = hidden_dim
        self.input_pool = AttentionPooling(hidden_dim, attn_pool_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, gene_expr: torch.Tensor, gene_adj: torch.Tensor):
        x = gene_expr.unsqueeze(-1)
        x = self.input_proj(x)
        x = F.relu(x, inplace=True)
        raw_pooled = self.input_pool(x, torch.ones(x.shape[:2], device=x.device, dtype=x.dtype))
        layer_node_outputs = []
        layer_pooled_outputs = []
        if gene_adj.dim() == 2:
            gene_adj = gene_adj.unsqueeze(0).expand(x.size(0), -1, -1)
        current = x
        for layer, pool in zip(self.layers, self.poolers):
            current = layer(current, gene_adj)
            current = F.elu(current)
            current = self.dropout(current)
            layer_node_outputs.append(current)
            layer_pooled_outputs.append(pool(current, torch.ones(current.shape[:2], device=current.device, dtype=current.dtype)))
        return {
            'raw_pooled': raw_pooled,
            'layer_node_outputs': layer_node_outputs,
            'layer_pooled_outputs': layer_pooled_outputs,
            'final_pooled': layer_pooled_outputs[-1],
        }
