from __future__ import annotations

import argparse

from miccaf.config import load_config
from miccaf.preprocessing import process_raw_dataset



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--raw-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = process_raw_dataset(
        raw_path=args.raw_path,
        output_path=args.output_path,
        num_time_bins=int(config.data.num_time_bins),
        genomics_top_k=int(config.data.genomics_top_k),
        genomics_edge_threshold=float(config.data.genomics_edge_threshold),
        min_wsi_patches=int(config.data.min_wsi_patches),
        max_wsi_patches=int(config.data.max_wsi_patches),
    )
    print(summary)


if __name__ == '__main__':
    main()
