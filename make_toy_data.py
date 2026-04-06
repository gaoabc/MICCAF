from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np



def make_toy_dataset(output_dir: str, n_samples: int = 96, pathology_dim: int = 64, gene_dim: int = 256) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    patient_ids = np.array([f'patient_{i:04d}' for i in range(n_samples)], dtype=object)
    latent_risk = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    times = np.exp(3.0 - 0.8 * latent_risk + 0.1 * rng.normal(size=n_samples)).astype(np.float32)
    observed_prob = 1.0 / (1.0 + np.exp(-latent_risk))

    events = np.where(rng.uniform(size=n_samples) < observed_prob, 0, 1).astype(np.int64)

    gene_expr = []
    wsi_features = []
    wsi_coords = []
    modality_mask = np.ones((n_samples, 2), dtype=np.float32)
    for i in range(n_samples):
        num_patches = rng.integers(24, 96)
        pathology_signal = latent_risk[i] + 0.2 * rng.normal(size=(num_patches, 1))
        patch_feat = np.concatenate([
            pathology_signal,
            rng.normal(size=(num_patches, pathology_dim - 1))
        ], axis=1).astype(np.float32)
        coords = rng.uniform(low=0.0, high=1.0, size=(num_patches, 2)).astype(np.float32)
        gene_signal = latent_risk[i] + 0.15 * rng.normal(size=(gene_dim, 1))
        genes = np.concatenate([
            gene_signal,
            rng.normal(size=(gene_dim, 0))
        ], axis=1).squeeze(-1)
        genes = genes + rng.normal(size=gene_dim) * 0.25
        gene_expr.append(genes.astype(np.float32))
        wsi_features.append(patch_feat)
        wsi_coords.append(coords)
        if rng.uniform() < 0.2:
            modality_mask[i, 0] = 0.0
        if rng.uniform() < 0.2:
            modality_mask[i, 1] = 0.0
    gene_expr = np.stack(gene_expr, axis=0)
    gene_names = np.array([f'gene_{i:04d}' for i in range(gene_dim)], dtype=object)
    path = output_dir / 'raw_multimodal_toy.npz'
    np.savez_compressed(
        path,
        patient_ids=patient_ids,
        times=times,
        events=events,
        gene_expr=gene_expr,
        wsi_features=np.array(wsi_features, dtype=object),
        wsi_coords=np.array(wsi_coords, dtype=object),
        gene_names=gene_names,
        modality_mask=modality_mask,
    )
    return path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='/mnt/data2/home/Public/results/')
    parser.add_argument('--n-samples', type=int, default=96)
    parser.add_argument('--pathology-dim', type=int, default=64)
    parser.add_argument('--gene-dim', type=int, default=256)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    path = make_toy_dataset(args.output_dir, args.n_samples, args.pathology_dim, args.gene_dim)
    print(f'Wrote toy dataset to {path}')


if __name__ == '__main__':
    main()
