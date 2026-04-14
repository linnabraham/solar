#!/usr/bin/env python3
"""
AARP Subset Pipeline

Downloads a subset of AARP sequences from the NASA public server, extracts and
preprocesses them identically to the full training pipeline, then generates a
solar_dataset_subset.json that is a drop-in for solar_dataset.json.

Preprocessing is intentionally identical to data_single.py (same aarp_ml.data_prep
functions, same biggest_shape from grouped_df.csv) so that model inputs match the
training distribution exactly.

Phases:
  --select    Validate target AARPs exist in URL CSVs; print file counts
  --download  Download compressed 7h FITS from NASA public server
  --process   Extract timesteps + pad/resize to 512x512
  --json      Generate solar_dataset_subset.json
  --all       Run all phases sequentially (default when no flag given)

Usage:
    python -m src.aarp_subset_pipeline --all
    python -m src.aarp_subset_pipeline --process --json   # if already downloaded
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from aarp_ml.data_prep import pad_and_resize_in_parallel, pad_with_quiet
from src.fits_parallel_download import download_urls_in_parallel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Subset selection — both classes, all splits represented, prefer larger max_dim (less padding).
#
# test  pos:  377  (max_dim=~1200, ~231 frames), 1449 (max_dim=~500,  ~55 frames)
# test  neg: 4296  (max_dim=1166, ~616 frames)
# val   pos: 3563  (max_dim=1975, ~231 frames)
# val   neg:  185  (max_dim=851,  ~528 frames)
# train pos: 1807  (max_dim=1019, ~231 frames)  — good shape, manageable download
# train neg:  903  (max_dim=581,   ~55 frames)  — original small anchor
TARGET_AARP_IDS = [377, 1449, 4296, 3563, 185, 1807, 903]

# Local directories — mirror the layout expected by solar_dataset.json
COMPRESSED_POS = Path("data/E8/compressed/pos")
COMPRESSED_NEG = Path("data/E8/compressed/neg")
EXTRACTED_POS  = Path("data/E8/extracted/pos")
EXTRACTED_NEG  = Path("data/E8/extracted/neg")

# Input data files (already in repo)
POS_URLS_CSV   = Path("data/pos_urls_selected_downloaded.csv")
NEG_URLS_CSV   = Path("data/neg_urls_selected_downloaded.csv")
GROUPED_DF_CSV = Path("data/grouped_df.csv")
FULL_JSON      = Path("solar_dataset.json")

OUTPUT_JSON = Path("solar_dataset_subset.json")

# Padding dimensions taken from the full-dataset grouped_df
# (max_width=2131, max_height=1043 → dim = max = 2131)
BIGGEST_DIM  = 2131
BIGGEST_SHAPE = (BIGGEST_DIM, BIGGEST_DIM)
TARGET_SHAPE  = (512, 512)

DOWNLOAD_WORKERS = 8


# ---------------------------------------------------------------------------
# Phase 1 — Select
# ---------------------------------------------------------------------------

def phase_select() -> dict[int, dict]:
    """
    Look up each target AARP in the pre-filtered URL CSVs and return:
        {aarp_id: {'urls': [...], 'label': 0|1}}

    URLs are intersected with grouped_df.csv so we only download the files
    that actually passed shape/off-limb selection in the original pipeline
    (some AARPs have more entries in the URL CSVs than ended up in the dataset).
    Raises if any AARP is missing.
    """
    _banner("PHASE 1 — SELECT")

    pos_df = pd.read_csv(POS_URLS_CSV)
    neg_df = pd.read_csv(NEG_URLS_CSV)

    # Build set of filenames that made it through selection (from grouped_df)
    grouped_df = pd.read_csv(GROUPED_DF_CSV)
    selected_filenames = set(
        Path(p).name for p in grouped_df.compressed_fits_fullpath
        if grouped_df.AARP[grouped_df.compressed_fits_fullpath == p].isin(TARGET_AARP_IDS).any()
    )
    # Simpler: just get all filenames in grouped_df for target AARPs
    target_rows = grouped_df[grouped_df.AARP.isin(TARGET_AARP_IDS)]
    selected_filenames = set(Path(p).name for p in target_rows.compressed_fits_fullpath)

    info: dict[int, dict] = {}
    missing = []

    for aid in TARGET_AARP_IDS:
        pos_rows = pos_df[pos_df.AARP == aid]
        neg_rows = neg_df[neg_df.AARP == aid]

        if len(pos_rows):
            raw_urls = pos_rows.urls.tolist()
            label = 1
        elif len(neg_rows):
            raw_urls = neg_rows.urls.tolist()
            label = 0
        else:
            print(f"  AARP {aid:5d}  *** NOT FOUND in URL CSVs ***")
            missing.append(aid)
            continue

        # Keep only URLs whose filename is in grouped_df (passed selection)
        filtered_urls = [u for u in raw_urls if Path(u).name in selected_filenames]
        skipped = len(raw_urls) - len(filtered_urls)

        label_str = "pos" if label == 1 else "neg"
        print(f"  AARP {aid:5d}  {label_str}  {len(filtered_urls):3d} files to download"
              + (f"  ({skipped} skipped — not in grouped_df)" if skipped else ""))

        info[aid] = {"urls": filtered_urls, "label": label}

    if missing:
        raise ValueError(
            f"AARPs not found in {POS_URLS_CSV} or {NEG_URLS_CSV}: {missing}"
        )

    total = sum(len(v["urls"]) for v in info.values())
    print(f"\n  Total compressed files to download: {total}")
    return info


# ---------------------------------------------------------------------------
# Phase 2 — Download
# ---------------------------------------------------------------------------

def phase_download(aarp_info: dict[int, dict]) -> None:
    """Download compressed 7h FITS to data/E8/compressed/{pos,neg}/."""
    _banner("PHASE 2 — DOWNLOAD")

    for aid, meta in aarp_info.items():
        dest = COMPRESSED_POS if meta["label"] == 1 else COMPRESSED_NEG
        dest.mkdir(parents=True, exist_ok=True)
        label_str = "pos" if meta["label"] == 1 else "neg"
        n = len(meta["urls"])
        print(f"\n  AARP {aid} ({label_str})  {n} files  →  {dest}")
        download_urls_in_parallel(meta["urls"], dest, max_workers=DOWNLOAD_WORKERS)


# ---------------------------------------------------------------------------
# Phase 3 — Process
# ---------------------------------------------------------------------------

def phase_process() -> None:
    """
    Extract individual timestep frames from compressed 7h FITS and pad/resize
    to TARGET_SHAPE.  Uses the same aarp_ml.data_prep functions as data_single.py
    so outputs are bit-identical to the training data.

    Compressed file list is taken from grouped_df.csv (one row per compressed
    file after shape-limiting in the original pipeline) — server-side absolute
    paths are remapped to local paths by matching on filename.
    """
    _banner("PHASE 3 — EXTRACT + PAD/RESIZE")
    print(f"  biggest_shape={BIGGEST_SHAPE}  target_shape={TARGET_SHAPE}")

    grouped_df = pd.read_csv(GROUPED_DF_CSV)

    for label, comp_dir, ext_dir in [
        (1, COMPRESSED_POS, EXTRACTED_POS),
        (0, COMPRESSED_NEG, EXTRACTED_NEG),
    ]:
        target_ids = [a for a in TARGET_AARP_IDS if _label_of(a) == label]
        if not target_ids:
            continue

        label_str = "pos" if label == 1 else "neg"
        print(f"\n  [{label_str}] AARPs: {target_ids}")

        # Rows in grouped_df for these AARPs
        rows = grouped_df[
            grouped_df.AARP.isin(target_ids) & (grouped_df.Label == label)
        ]
        if rows.empty:
            print(f"  WARNING: no rows in grouped_df for {target_ids} — skipping")
            continue

        # Remap server absolute path → local path (filename is identical)
        local_paths = rows.compressed_fits_fullpath.apply(
            lambda p: str(comp_dir / Path(p).name)
        )

        missing = [p for p in local_paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"  {len(missing)} compressed file(s) not found locally.\n"
                f"  Run --download first.  First missing:\n    {missing[0]}"
            )

        ext_dir.mkdir(parents=True, exist_ok=True)

        # Determine which AARPs need (re-)processing by checking:
        #   1. extracted file count matches expected count from full JSON
        #   2. no zero-byte files (truncated writes)
        with open(FULL_JSON) as _f:
            _full = json.load(_f)
        expected_counts = {}
        for split in ("training", "validation", "test"):
            for entry in _full.get(split, []):
                if entry["aarp_id"] in target_ids:
                    expected_counts[entry["aarp_id"]] = \
                        expected_counts.get(entry["aarp_id"], 0) + 1

        ids_to_process = []
        for aid in target_ids:
            if _label_of(aid) != label:
                continue
            existing = list(ext_dir.glob(f"{aid}_*.fits"))
            expected = expected_counts.get(aid, 0)
            zero_byte = [f for f in existing if f.stat().st_size == 0]
            if zero_byte:
                print(f"  AARP {aid}: {len(zero_byte)} zero-byte file(s) — will re-process")
                ids_to_process.append(aid)
            elif len(existing) == expected and expected > 0:
                print(f"  AARP {aid}: {len(existing)}/{expected} frames found — skipping")
            else:
                print(f"  AARP {aid}: {len(existing)}/{expected} frames found — will process")
                ids_to_process.append(aid)

        if not ids_to_process:
            continue

        to_process = [
            p for p in local_paths
            if rows[rows.compressed_fits_fullpath.apply(
                lambda x: Path(x).name) == Path(p).name].AARP.values[0]
            in ids_to_process
        ]

        print(f"  Extracting {len(to_process)} compressed file(s) for AARPs {ids_to_process} → {ext_dir}")

        pad_and_resize_in_parallel(
            to_process,
            padding_func=pad_with_quiet,
            dest=str(ext_dir),
            biggest_shape=BIGGEST_SHAPE,
            targ_shape=TARGET_SHAPE,
        )

        n_out = len(list(ext_dir.glob("*.fits")))
        print(f"  {ext_dir}: {n_out} frames total")


def _label_of(aarp_id: int) -> int:
    """Return 1 (pos) or 0 (neg) for aarp_id from the URL CSVs."""
    if not hasattr(_label_of, "_cache"):
        pos_df = pd.read_csv(POS_URLS_CSV)
        neg_df = pd.read_csv(NEG_URLS_CSV)
        cache: dict[int, int] = {}
        for a in pos_df.AARP.unique():
            cache[int(a)] = 1
        for a in neg_df.AARP.unique():
            if int(a) not in cache:
                cache[int(a)] = 0
        _label_of._cache = cache
    return _label_of._cache[aarp_id]


# ---------------------------------------------------------------------------
# Phase 4 — JSON
# ---------------------------------------------------------------------------

def phase_json() -> None:
    """
    Filter solar_dataset.json entries for TARGET_AARP_IDS and write
    solar_dataset_subset.json, preserving the original split assignments
    (training / validation / test) and relative file paths.

    The extracted filenames are determined by FITS headers (T_START, T_IMGxx),
    which are immutable — so they match the original JSON entries exactly.
    stats.pkl is reused unchanged (computed on the full training set; valid
    for inference and analysis).
    """
    _banner("PHASE 4 — GENERATE JSON")

    with open(FULL_JSON) as f:
        full = json.load(f)

    subset: dict = {
        "name": full["name"],
        "description": full["description"],
        "channels": full["channels"],
        "training": [],
        "validation": [],
        "test": [],
    }

    aarp_summary: dict[int, dict] = {}
    for split in ("training", "validation", "test"):
        for entry in full.get(split, []):
            aid = entry["aarp_id"]
            if aid in TARGET_AARP_IDS:
                subset[split].append(entry)
                if aid not in aarp_summary:
                    aarp_summary[aid] = {"split": split, "label": entry["label"], "n": 0}
                aarp_summary[aid]["n"] += 1

    print(f"\n  AARP summary:")
    for aid, info in sorted(aarp_summary.items()):
        lbl = "pos" if info["label"] == 1 else "neg"
        print(f"    AARP {aid:5d}  {info['split']:10s}  {lbl}  {info['n']} frames")

    print(f"\n  Split totals:")
    for split in ("training", "validation", "test"):
        print(f"    {split}: {len(subset[split])} entries")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(subset, f, indent=4)

    print(f"\n  Written: {OUTPUT_JSON}")
    print(f"\n  Next steps:")
    print(f"    python -m src.check_json --json {OUTPUT_JSON}")
    print(f"    mv solar_dataset.json solar_dataset_full.json")
    print(f"    ln -s {OUTPUT_JSON} solar_dataset.json")
    print(f"    dvc repro confusion-matrix-val")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and preprocess an AARP subset for per-AARP analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--select",   action="store_true", help="Phase 1: validate URLs")
    parser.add_argument("--download", action="store_true", help="Phase 2: download FITS")
    parser.add_argument("--process",  action="store_true", help="Phase 3: extract + resize")
    parser.add_argument("--json",     action="store_true", help="Phase 4: write subset JSON")
    parser.add_argument("--all",      action="store_true", help="Run all phases")
    args = parser.parse_args()

    if not any([args.select, args.download, args.process, args.json, args.all]):
        args.all = True

    print(f"\n{'#'*60}")
    print(f"  AARP Subset Pipeline")
    print(f"  Target AARPs: {TARGET_AARP_IDS}")
    print(f"{'#'*60}")
    print(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")

    try:
        aarp_info = None

        if args.select or args.download or args.all:
            aarp_info = phase_select()

        if args.download or args.all:
            if aarp_info is None:
                aarp_info = phase_select()
            phase_download(aarp_info)

        if args.process or args.all:
            phase_process()

        if args.json or args.all:
            phase_json()

        print(f"\n  Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
