"""Wrapper: generate a raw AIA movie for one specific AARP quickly.

Usage:
    python -m src.torch.vit.raw_movie_single --aarp-id 377
    python -m src.torch.vit.raw_movie_single --aarp-id 377 --json-path solar_dataset_subset.json
    python -m src.torch.vit.raw_movie_single --aarp-id 377 --output-path plots/my_movie.mp4
"""

import argparse
import json

from src.torch.vit.create_raw_movie import make_raw_movie
from src.torch.vit.ig import single_aarp
from src.torch.vit.utils import dfs_from_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aarp-id",    type=int, required=True)
    parser.add_argument("--json-path",  default="solar_dataset.json")
    parser.add_argument("--output-path", default=None,
                        help="Defaults to plots/raw_movies/<aarp_id>.mp4")
    parser.add_argument("--fps",        type=int, default=5)
    args = parser.parse_args()

    output_path = args.output_path or f"plots/raw_movies/{args.aarp_id}.mp4"

    with open(args.json_path) as f:
        metadata = json.load(f)

    training_df, val_df, test_df = dfs_from_metadata(metadata)
    all_df = [training_df, val_df, test_df]

    aarp_df = None
    for df in all_df:
        if not df.empty and args.aarp_id in df.aarp_id.values:
            aarp_df = df.query(f"aarp_id == {args.aarp_id}")
            break

    if aarp_df is None:
        raise ValueError(f"AARP {args.aarp_id} not found in {args.json_path}")

    make_raw_movie(single_aarp(args.aarp_id, aarp_df), output_path, fps=args.fps)
