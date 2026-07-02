"""Paper figure: AARP preprocessing stages.

Produces a 4-row × 9-column figure showing two representative flare and two
representative no-flare active regions at each step of the preprocessing pipeline:

    Raw (DN)  →  Log-transformed  →  Z-score normalised (model input)

Channels shown: 94 Å, 131 Å, 171 Å.

AARP selection (auto, unless overridden with --pos-ids / --neg-ids):
  Pool all AARPs across training, validation, and test splits.
  Use the mean 171 Å intensity of the middle frame as an AR size/brightness
  proxy.  Pick one from the bottom quartile (compact/quiet) and one from the
  top quartile (extended/bright) within each class.  This avoids cherry-
  picking while demonstrating intra-class morphological diversity.

Output
------
  <output-dir>/aarp_preprocessing.png

Usage
-----
    python -m src.torch.vit.paper_aarp_figure
    python -m src.torch.vit.paper_aarp_figure --output-dir plots/paper_figures
    python -m src.torch.vit.paper_aarp_figure --pos-ids 377 4920 --neg-ids 1275 3229
"""

import argparse
import collections
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sunpy.visualization.colormaps  # noqa: F401 — registers sdoaia* colormaps with matplotlib
from astropy.io import fits
from matplotlib.gridspec import GridSpec

from astro_utils.utils import read_fits_single

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNEL_INDICES = [0, 1, 2]          # 94 Å, 131 Å, 171 Å
PASSBANDS       = [94, 131, 171]
CHANNEL_171_IDX = 2                   # index of 171 Å in the 7-channel stack

STAGES          = ["raw", "log"]
STAGE_LABELS    = ["Raw (DN)", "Log-transformed"]

VMAX_PERCENTILE = 99.9

# Row colours (RGBA background patches)
FLARE_BG    = "#ddeeff"   # light blue
NOFLARE_BG  = "#fff3dd"   # light orange

DEFAULT_JSON_PATH  = "solar_dataset.json"
DEFAULT_STATS_FILE = "stats.pkl"
DEFAULT_OUTPUT_DIR = "plots/paper_figures"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_metadata(json_path: str) -> Dict[int, dict]:
    """Return {aarp_id: {'label': int, 'entries': [entry, ...], 'subset': str}}.

    Merges all three splits.  If an AARP appears in multiple splits (it
    shouldn't in this dataset), the first occurrence wins.
    """
    with open(json_path) as f:
        data = json.load(f)

    aarp_map: Dict[int, dict] = {}
    for subset in ("training", "validation", "test"):
        by_id: Dict[int, list] = collections.defaultdict(list)
        for entry in data.get(subset, []):
            by_id[entry["aarp_id"]].append(entry)
        for aarp_id, entries in by_id.items():
            if aarp_id not in aarp_map:
                aarp_map[aarp_id] = {
                    "label":   entries[0]["label"],
                    "entries": entries,
                    "subset":  subset,
                    "n_frames": len(entries),
                }
    return aarp_map


def load_stats(stats_file: str, n_channels: int = 7) -> Tuple[List[float], List[float]]:
    with open(stats_file, "rb") as f:
        stats_data = pickle.load(f)
    means = [stats_data["mean"][f"channel_{i}"] for i in range(n_channels)]
    stds  = [stats_data["std"][f"channel_{i}"]  for i in range(n_channels)]
    return means, stds


def middle_entry(entries: list) -> dict:
    return entries[len(entries) // 2]


def _parse_extracted_path(extracted_path: str) -> Tuple[str, str, str]:
    """Derive the compressed-cube path, T_REC, and frame timestamp from an
    extracted FITS path stored in solar_dataset.json.

    Extracted path pattern:
        data/E8/extracted/{pos|neg}/{aarp_id}_{pb}_{t_rec}_TAI_{frame_ts}.fits

    Compressed cube path pattern:
        data/E8/compressed/{pos|neg}/{t_rec}_7h@1h_AARP{aarp_id}_{pb}.fits

    Returns
    -------
    cube_path : str   — relative path to the 7-hour compressed cube
    t_rec_tai : str   — T_REC header value, e.g. "2011.02.11_15:48:00_TAI"
    frame_ts  : str   — frame timestamp, e.g. "2011-02-11T15:42:02Z"
    """
    p        = Path(extracted_path)
    cls_dir  = p.parent.name                           # "pos" or "neg"
    stem     = p.stem                                  # everything before ".fits"
    pre_tai, frame_ts = stem.split("_TAI_", 1)
    aarp_id_str, pb_str, t_rec = pre_tai.split("_", 2)
    cube_name = f"{t_rec}_7h@1h_AARP{aarp_id_str}_{pb_str}.fits"
    cube_path = str(Path("data/E8/compressed") / cls_dir / cube_name)
    return cube_path, f"{t_rec}_TAI", frame_ts


def _read_native_frame(cube_path: str, t_rec_tai: str, frame_ts: str) -> np.ndarray:
    """Return the unpadded native-resolution frame from a compressed 7h cube.

    Each cube contains 7 HDUs (one per hour of the 7h window), each with its
    own T_REC.  Scans all HDUs for the frame whose T_IMGxx matches *frame_ts*.
    *t_rec_tai* is retained as a parameter for error context only.
    """
    with fits.open(cube_path) as hdul:
        for hdu_idx in range(1, len(hdul)):
            hdu = hdul[hdu_idx]
            n_frames = hdu.data.shape[0]
            for i in range(n_frames):
                if hdu.header.get(f"T_IMG{i:02d}") == frame_ts:
                    return hdu.data[i].copy().astype(np.float32)
    raise ValueError(
        f"Frame {frame_ts!r} not found in any HDU of {cube_path}"
    )


def load_channels(entry: dict, channel_indices: List[int]) -> np.ndarray:
    """Load selected channels at native (unpadded) resolution from the
    compressed 7-hour cubes.  Returns (C, H, W) float32.

    The compressed cubes store images at the true AR cutout size before any
    padding to 512×512, so the returned array reflects the actual spatial
    extent of the active region.
    """
    imgs = []
    for idx in channel_indices:
        cube_path, t_rec_tai, frame_ts = _parse_extracted_path(entry[str(idx)])
        imgs.append(_read_native_frame(cube_path, t_rec_tai, frame_ts))
    return np.stack(imgs, axis=0)   # (C, H_native, W_native)


def has_compressed_cube(aarp_info: dict) -> bool:
    """Return True if the compressed 7h cube exists for the middle frame's first channel."""
    entry = middle_entry(aarp_info["entries"])
    cube_path, _, _ = _parse_extracted_path(entry["0"])
    return Path(cube_path).exists()


def mean_171_intensity(aarp_info: dict) -> float:
    """Mean pixel intensity in the 171 Å channel of the middle frame.

    Uses the padded extracted FITS (available for all AARPs) — sufficient
    for relative ranking during AARP selection.
    """
    entry = middle_entry(aarp_info["entries"])
    img   = read_fits_single(entry[str(CHANNEL_171_IDX)]).astype(np.float32)
    img   = np.where(img < 0, 0.0, img)
    return float(img.mean())


# ---------------------------------------------------------------------------
# AARP selection
# ---------------------------------------------------------------------------

def select_aarps(
    aarp_map: Dict[int, dict],
    label: int,
    require_cubes: bool = True,
) -> Tuple[int, int]:
    """Return (compact_aarp_id, extended_aarp_id) for the given class.

    Uses mean 171 Å intensity of the middle frame as a brightness proxy.
    Picks one AARP from the bottom quartile and one from the top quartile.

    If *require_cubes* is True (default), restricts candidates to AARPs
    whose compressed 7h cubes are present locally so native-resolution
    images can be read.
    """
    candidates = [(aid, info) for aid, info in aarp_map.items()
                  if info["label"] == label]

    if require_cubes:
        with_cubes    = [(aid, info) for aid, info in candidates if has_compressed_cube(info)]
        without_cubes = [aid for aid, info in candidates if not has_compressed_cube(info)]
        if without_cubes:
            print(f"  ⚠  Skipping {len(without_cubes)} label={label} AARPs "
                  f"(no local compressed cube): {without_cubes}")
        candidates = with_cubes

    if len(candidates) < 2:
        raise RuntimeError(
            f"Need ≥2 AARPs with compressed cubes for label={label}, "
            f"found {len(candidates)}. "
            f"Either download more cubes or use --pos-ids / --neg-ids to specify "
            f"AARPs manually (and set require_cubes=False)."
        )

    print(f"  Computing 171 Å intensities for {len(candidates)} label={label} AARPs …")
    scored = []
    for aid, info in candidates:
        intensity = mean_171_intensity(info)
        scored.append((aid, intensity))
        print(f"    AARP {aid:5d}  [{info['subset']:10s}]  171Å mean = {intensity:8.1f}")

    scored.sort(key=lambda x: x[1])
    q25_idx = max(0, int(len(scored) * 0.25) - 1)
    q75_idx = min(len(scored) - 1, int(len(scored) * 0.75))

    compact_id  = scored[q25_idx][0]
    extended_id = scored[q75_idx][0]
    print(f"  → Selected compact  AARP {compact_id}  (171Å={scored[q25_idx][1]:.1f})")
    print(f"  → Selected extended AARP {extended_id} (171Å={scored[q75_idx][1]:.1f})")
    return compact_id, extended_id


# ---------------------------------------------------------------------------
# Processing stages
# ---------------------------------------------------------------------------

def apply_log(raw: np.ndarray) -> np.ndarray:
    """Natural log with guard against ≤0 values."""
    x = raw.copy()
    x[x < 0]  = 0.0
    x[x == 0] = 1.0
    return np.log(x)


def apply_zscore(log_img: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (log_img - mean) / std


def build_stage_images(
    raw: np.ndarray,          # (C, H, W)  C=3 channels
    channel_indices: List[int],
    means: List[float],
    stds:  List[float],
) -> Dict[str, np.ndarray]:
    """Return {'raw': arr, 'log': arr}, both (C, H, W)."""
    log = np.stack([apply_log(raw[i]) for i in range(len(channel_indices))], axis=0)
    return {"raw": raw, "log": log}


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def _imshow(ax, img, cmap, vmax_p=VMAX_PERCENTILE):
    """Display one image panel scaled to the 99.9th percentile."""
    vmax = np.percentile(img, vmax_p)
    ax.imshow(img, cmap=cmap, origin="lower", vmin=0.0, vmax=vmax, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_class_figure(
    aarp_ids: List[int],
    class_label: str,
    bg_color: str,
    label_color: str,
    aarp_map: Dict[int, dict],
    means: List[float],
    stds:  List[float],
    output_path: str,
) -> None:
    """Assemble and save a 2-row × 6-column figure for one class.

    Columns: 3 channels × 2 stages (raw, log) = 6 panels per row.
    """
    n_rows, n_img_cols = 2, 6   # 2 AARPs, 3 channels × 2 stages

    # ── Load images ──────────────────────────────────────────────────────────
    stage_images = []
    row_labels   = []
    for aarp_id in aarp_ids:
        info  = aarp_map[aarp_id]
        entry = middle_entry(info["entries"])
        print(f"  AARP {aarp_id}  ts={entry['timestamp']}")
        raw = load_channels(entry, CHANNEL_INDICES)
        stage_images.append(build_stage_images(raw, CHANNEL_INDICES, means, stds))
        row_labels.append(f"AARP {aarp_id}\n({info['subset']}, n={info['n_frames']})")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(
        n_rows, n_img_cols,
        figure=fig,
        hspace=0.06,
        wspace=0.03,
        left=0.11,
        right=0.99,
        top=0.88,
        bottom=0.04,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_img_cols)]
            for r in range(n_rows)]

    # Background tint
    for r in range(n_rows):
        for c in range(n_img_cols):
            axes[r][c].set_facecolor(bg_color)

    # ── Fill panels ───────────────────────────────────────────────────────────
    # cols 0-2 = raw (94, 131, 171),  cols 3-5 = log (94, 131, 171)
    col_to_stage_ch = [(stage, ch)
                       for stage in STAGES
                       for ch in range(len(CHANNEL_INDICES))]

    for r, (aarp_id, stages_dict) in enumerate(zip(aarp_ids, stage_images)):
        for c, (stage, ch_idx) in enumerate(col_to_stage_ch):
            ax  = axes[r][c]
            img = stages_dict[stage][ch_idx]
            pb  = PASSBANDS[ch_idx]
            try:
                cmap = matplotlib.colormaps[f"sdoaia{pb}"]
            except KeyError:
                cmap = "gray"
            _imshow(ax, img, cmap)

            # Passband label inset top-left inside each panel (first row only)
            if r == 0:
                ax.text(
                    0.04, 0.97, f"{pb} Å",
                    transform=ax.transAxes,
                    fontsize=8, va="top", ha="left",
                    color="white", fontweight="bold",
                )

    # ── Row labels (AARP ID + meta) ───────────────────────────────────────────
    for r in range(n_rows):
        pos = axes[r][0].get_position()
        fig.text(
            0.005, pos.y0 + pos.height / 2,
            row_labels[r],
            va="center", ha="left", fontsize=8,
            color="#222222", linespacing=1.4,
        )

    # ── Stage group headers ───────────────────────────────────────────────────
    stage_col_starts = [0, 3]   # raw starts at col 0, log at col 3
    for start_col, stage_label in zip(stage_col_starts, STAGE_LABELS):
        left_pos  = axes[0][start_col].get_position()
        right_pos = axes[0][start_col + 2].get_position()
        x_mid = (left_pos.x0 + right_pos.x1) / 2
        y_top = left_pos.y1
        fig.text(
            x_mid, y_top + 0.02,
            stage_label,
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#111111",
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate AARP preprocessing stages figure for paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--json-path",   default=DEFAULT_JSON_PATH)
    p.add_argument("--stats-file",  default=DEFAULT_STATS_FILE)
    p.add_argument("--output-dir",  default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--pos-ids", nargs=2, type=int, default=None,
        metavar=("AARP_ID_1", "AARP_ID_2"),
        help="Manual override: two positive (flare) AARP IDs",
    )
    p.add_argument(
        "--neg-ids", nargs=2, type=int, default=None,
        metavar=("AARP_ID_1", "AARP_ID_2"),
        help="Manual override: two negative (no-flare) AARP IDs",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("Loading metadata …")
    aarp_map = load_metadata(args.json_path)
    means, stds = load_stats(args.stats_file)
    print(f"  {sum(1 for v in aarp_map.values() if v['label']==1)} positive AARPs, "
          f"{sum(1 for v in aarp_map.values() if v['label']==0)} negative AARPs "
          f"(across all splits)")

    # ── Select AARPs ──────────────────────────────────────────────────────────
    if args.pos_ids:
        selected_pos = tuple(args.pos_ids)
        print(f"Using manually specified positive AARPs: {selected_pos}")
    else:
        print("\nAuto-selecting positive AARPs …")
        selected_pos = select_aarps(aarp_map, label=1)

    if args.neg_ids:
        selected_neg = tuple(args.neg_ids)
        print(f"Using manually specified negative AARPs: {selected_neg}")
    else:
        print("\nAuto-selecting negative AARPs …")
        selected_neg = select_aarps(aarp_map, label=0)

    print(f"\nFinal selection:")
    print(f"  Positive (flare)   : AARP {selected_pos[0]}  &  AARP {selected_pos[1]}")
    print(f"  Negative (no-flare): AARP {selected_neg[0]}  &  AARP {selected_neg[1]}")

    # ── Build two figures (one per class) ─────────────────────────────────────
    print("\nLoading images for flare AARPs …")
    build_class_figure(
        aarp_ids    = list(selected_pos),
        class_label = "Flare Active Regions",
        bg_color    = FLARE_BG,
        label_color = "#1a5276",
        aarp_map    = aarp_map,
        means       = means,
        stds        = stds,
        output_path = str(Path(args.output_dir) / "aarp_preprocessing_flare.png"),
    )

    print("\nLoading images for no-flare AARPs …")
    build_class_figure(
        aarp_ids    = list(selected_neg),
        class_label = "Non-flare Active Regions",
        bg_color    = NOFLARE_BG,
        label_color = "#784212",
        aarp_map    = aarp_map,
        means       = means,
        stds        = stds,
        output_path = str(Path(args.output_dir) / "aarp_preprocessing_noflare.png"),
    )


if __name__ == "__main__":
    main()
