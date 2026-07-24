"""Reconstruct the active-region (AR) bounding box inside a padded 512x512 AARP frame.

Every extracted AARP frame is produced by `aarp_ml/data_prep.py::pad_with_quiet()` +
`pad_and_scale()` (invoked from `src/data_single.py --extract`): the native-resolution
image is first centered-padded into a global `dim x dim` square (`dim` = the max native
height/width over the whole dataset), then that square is resized to the final training
shape (512x512, or 224x224 downstream for `vit_l_16`). The padding is not black -- it's
random samples from the frame's own pixel distribution -- so recovering the AR box
requires reversing this known geometric transform, not thresholding on intensity.

Per-frame native size lives in `data/shape_limited.csv` (and `data/combined_clean.csv`),
keyed by `fits_fullpath`, which exactly string-matches the channel path values in
`solar_dataset.json` / `cv_folds/fold_*.json` entries.
"""

import warnings
from functools import lru_cache

import pandas as pd
import numpy as np

DEFAULT_SHAPE_CSV = "data/shape_limited.csv"
DEFAULT_GROUPED_DF_CSV = "data/grouped_df.csv"


def load_shape_lookup(csv_path: str = DEFAULT_SHAPE_CSV) -> dict:
    """Return {fits_fullpath: (img_height, img_width)}, the native pre-pad size per frame."""
    df = pd.read_csv(csv_path, usecols=["fits_fullpath", "img_height", "img_width"])
    return {
        row.fits_fullpath: (row.img_height, row.img_width)
        for row in df.itertuples(index=False)
    }


def compute_pad_dim(grouped_df_path: str = DEFAULT_GROUPED_DF_CSV) -> int:
    """Global square pad target: max(max_height.max(), max_width.max()) over the dataset."""
    gdf = pd.read_csv(grouped_df_path, usecols=["max_height", "max_width"])
    return int(max(gdf.max_height.max(), gdf.max_width.max()))


def bbox_in_frame(img_height: int, img_width: int, dim: int, final_size: int = 512) -> tuple:
    """AR content box (y0, y1, x0, x1) after centered pad-to-(dim, dim) then resize-to-(final_size, final_size).

    Pass final_size=224 to get the box directly in vit_l_16's resized input space --
    composing the dim->512 pad-resize with the 512->224 eval-time resize is itself just
    one overall linear scale factor, so this works without an intermediate 512-space box.
    """
    if img_height > dim or img_width > dim:
        raise ValueError(f"native size ({img_height}, {img_width}) exceeds pad target dim={dim}")

    scale = final_size / dim
    pad_top = (dim - img_height) // 2
    pad_left = (dim - img_width) // 2

    y0 = pad_top * scale
    y1 = (pad_top + img_height) * scale
    x0 = pad_left * scale
    x1 = (pad_left + img_width) * scale
    return (y0, y1, x0, x1)


def ar_bbox_for_entry(entry: dict, shape_lookup: dict, dim: int, final_size: int = 512,
                       n_channels: int = 7) -> tuple:
    """AR box for a solar_dataset.json-style entry (channel paths '0'..str(n_channels-1)).

    Looks up native size per channel and returns the shared box if all channels agree.
    If channels disagree (not observed in this dataset during validation, but not assumed
    away here), warns and returns the union (enclosing) box across channels instead of
    silently picking one -- a union box is the conservative choice for masking, since it
    guarantees no real AR pixel in any channel is excluded.
    """
    boxes = []
    for i in range(n_channels):
        path = entry[str(i)]
        if path not in shape_lookup:
            raise KeyError(f"{path!r} not found in shape lookup -- rebuild it from the CSV that matches this JSON")
        img_height, img_width = shape_lookup[path]
        boxes.append(bbox_in_frame(img_height, img_width, dim, final_size=final_size))

    boxes = np.array(boxes)  # (n_channels, 4) -> y0, y1, x0, x1
    if not np.allclose(boxes, boxes[0], atol=1e-6):
        warnings.warn(
            f"AARP {entry.get('aarp_id')}: channels disagree on native size -- "
            f"using the union box across channels instead of a single shared box."
        )
        y0 = boxes[:, 0].min()
        y1 = boxes[:, 1].max()
        x0 = boxes[:, 2].min()
        x1 = boxes[:, 3].max()
        return (y0, y1, x0, x1)

    return tuple(boxes[0])


def ar_mask(bbox: tuple, shape: tuple = (512, 512)) -> np.ndarray:
    """Boolean mask, True inside the AR box, for elementwise multiplication against
    an (H, W) or (C, H, W) image/attribution array (broadcasts over a leading channel dim)."""
    y0, y1, x0, x1 = bbox
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    y0i, y1i = max(0, int(round(y0))), min(h, int(round(y1)))
    x0i, x1i = max(0, int(round(x0))), min(w, int(round(x1)))
    mask[y0i:y1i, x0i:x1i] = True
    return mask
