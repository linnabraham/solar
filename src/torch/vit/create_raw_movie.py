"""Generate a raw AIA image movie for a single AARP (no attribution overlay).

Each frame shows all 7 AIA passbands in a grid with SDO/AIA colormaps and
the timestamp + AARP metadata as the figure title. Frames are stitched into
an MP4 via ffmpeg.

Usage:
    python -m src.torch.vit.create_raw_movie \
        --json-path solar_dataset_subset.json \
        --aarp-id 377 \
        --output-path plots/subset/movie_377_raw.mp4
"""

import json
import os
import subprocess
import tempfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from src.torch.vit.utils import dfs_from_metadata
from src.torch.vit.ig import single_aarp
from aarp_ml.dataset import all_wavelengths

PASSBANDS = list(all_wavelengths)
DEFAULT_FPS = 5
DEFAULT_VMAX_PERCENTILE = 99.9
DEFAULT_COLS = 4


def make_raw_movie(s_aarp, output_path, fps=DEFAULT_FPS, vmax_percentile=DEFAULT_VMAX_PERCENTILE):
    s_images = s_aarp.get_images()   # (T, 7, H, W)
    timestamps = list(s_aarp.timestamps)  # convert to list for positional indexing
    n_frames = s_images.shape[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        for t_idx in tqdm(range(n_frames), desc=f"Rendering aarp={s_aarp.aarp_id}"):
            frame = s_images[t_idx]   # (7, H, W)
            rows = int(np.ceil(len(PASSBANDS) / DEFAULT_COLS))
            fig, axes = plt.subplots(rows, DEFAULT_COLS, figsize=(DEFAULT_COLS * 3, rows * 3))
            axes = np.array(axes).flatten()

            for idx, (passband, image) in enumerate(zip(PASSBANDS, frame)):
                ax = axes[idx]
                vmax = np.percentile(image, vmax_percentile)
                ax.imshow(image, cmap=matplotlib.colormaps[f'sdoaia{passband}'], origin='lower', vmax=vmax)
                ax.set_title(f'{passband} Å', fontsize=8)
                ax.axis('off')

            for idx in range(len(PASSBANDS), len(axes)):
                axes[idx].axis('off')

            ts = timestamps[t_idx]
            ts_str = ts.strftime('%Y-%m-%dT%H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
            fig.suptitle(f"AARP {s_aarp.aarp_id} | label={s_aarp.label} | {ts_str}", fontsize=9)
            plt.tight_layout()
            fig.savefig(os.path.join(tmpdir, f"frame_{t_idx:04d}.png"), bbox_inches="tight", dpi=100)
            plt.close(fig)

        # pad ensures width/height are divisible by 2, required by libx264
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%04d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            output_path,
        ], check=True)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path",   default="solar_dataset.json")
    parser.add_argument("--output-dir",  default="plots/raw_movies")
    parser.add_argument("--fps",         type=int, default=DEFAULT_FPS)
    args = parser.parse_args()

    with open(args.json_path) as f:
        metadata = json.load(f)
    training_df, val_df, test_df = dfs_from_metadata(metadata)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for df, split in [(training_df, "training"), (val_df, "validation"), (test_df, "test")]:
        if df.empty:
            continue
        for aarp_id in df.aarp_id.unique():
            aarp_id_df = df.query(f"aarp_id == {aarp_id}")
            output_path = os.path.join(args.output_dir, f"{aarp_id}.mp4")
            make_raw_movie(single_aarp(aarp_id, aarp_id_df), output_path, fps=args.fps)
