"""Generate per-timestep attribution contour grid movie for a single AARP.

For each timestep, renders all 7 AIA passbands as a grid with SDO/AIA colormaps
and per-passband attribution contours, then stitches frames into an MP4 via ffmpeg.

Usage:
    python -m src.torch.vit.create_contour_grid_movie \
        --json-path solar_dataset_subset.json \
        --aarp-id 377 \
        --output-path plots/subset/movie_377_contour_grid.mp4
"""

import os
import tempfile
import subprocess
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from src.torch.vit.train import TrainingConfig
from src.torch.vit.utils import get_data_model, dfs_from_metadata
from src.torch.vit.class_wise_distribution import run_pred_and_ig
from src.torch.vit.plot_contour_image_grid import (
    plot_aia_image_grid,
    CONTOUR_CONFIG,
    DEFAULT_VMAX_PERCENTILE,
)
from aarp_ml.dataset import all_wavelengths

PASSBANDS = list(all_wavelengths)
DEFAULT_FPS = 1


def make_contour_grid_movie(
    s_images: np.ndarray,
    attributions: list,
    output_path: str,
    fps: int = DEFAULT_FPS,
) -> None:
    """Render per-timestep contour grid frames and stitch into MP4.

    Args:
        s_images: Array of shape (T, 7, H, W) — raw AIA images.
        attributions: List of T tensors each of shape (7, H, W) — IG attributions.
        output_path: Destination MP4 file path.
        fps: Frames per second for the output video.
    """
    n_frames = s_images.shape[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        for t_idx in tqdm(range(n_frames), desc="Rendering frames"):
            ig_out = np.array(attributions[t_idx])  # shape (7, H, W)
            fig = plot_aia_image_grid(
                images=s_images[t_idx],          # shape (7, H, W)
                passbands=PASSBANDS,
                saliency=ig_out,
                contour_config=CONTOUR_CONFIG,
                vmax_percentile=DEFAULT_VMAX_PERCENTILE,
                show=False,
            )
            frame_path = os.path.join(tmpdir, f"frame_{t_idx:04d}.png")
            fig.savefig(frame_path, bbox_inches="tight", dpi=100)

        # Stitch frames with ffmpeg
        frame_pattern = os.path.join(tmpdir, "frame_%04d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        subprocess.run(cmd, check=True)

    print(f"Saved movie to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path",  default="solar_dataset.json")
    parser.add_argument("--model-path", default="outputs/glad-shape-197/trained_model.pth",
                        help="Path to trained ViT model checkpoint.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output movies. Defaults to "
                             "plots/contour_grid_movies/<run-id> derived from --model-path.")
    parser.add_argument("--fps",        type=int, default=DEFAULT_FPS)
    parser.add_argument("--splits",     nargs="+", default=["validation", "test"],
                        choices=["training", "validation", "test"])
    args = parser.parse_args()

    if args.output_dir is None:
        run_id = Path(args.model_path).parent.name
        args.output_dir = f"plots/contour_grid_movies/{run_id}"

    config = TrainingConfig(json_path=args.json_path, stats_file="stats.pkl")
    config.trained_model_path = args.model_path
    metadata, model, transform, device = get_data_model(config)
    training_df, val_df, test_df = dfs_from_metadata(metadata)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for df, split in [(training_df, "training"), (val_df, "validation"), (test_df, "test")]:
        if df.empty or split not in args.splits:
            continue
        for aarp_id in df.aarp_id.unique():
            aarp_id_df = df.query(f"aarp_id == {aarp_id}")
            s_images, attributions = run_pred_and_ig(aarp_id, aarp_id_df, transform, model, device)
            make_contour_grid_movie(
                s_images=s_images,
                attributions=attributions,
                output_path=os.path.join(args.output_dir, f"{aarp_id}.mp4"),
                fps=args.fps,
            )
            del s_images, attributions
            gc.collect()
            torch.cuda.empty_cache()
