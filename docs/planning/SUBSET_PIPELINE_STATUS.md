# Subset Pipeline — Status & Plan

**Branch**: `feat/per-aarp-analysis`  
**Goal**: Run existing DVC analysis stages on a small locally-downloaded subset of AARPs,
without depending on the central data server.

---

## Target AARPs (initial 3-AARP validation set)

| AARP | Split    | Label | Compressed files |
|------|----------|-------|-----------------|
| 1449 | test     | pos   | 7               |
| 377  | test     | pos   | 28              |
| 903  | training | neg   | 24              |

Note: test split has only positive AARPs — affects distribution/kshap stages (see below).

---

## Data Pipeline Script: `src/aarp_subset_pipeline.py`

Phases:
- `--select`   : Validate URLs, intersect with `grouped_df.csv` (skip shape/off-limb filtered files)
- `--download` : Fetch compressed FITS from NASA public server (run manually once, not in DVC)
- `--process`  : Extract + pad/resize to 512×512 using `aarp_ml.data_prep.pad_and_resize_in_parallel`
- `--json`     : Filter `solar_dataset.json` for target AARPs → `solar_dataset_subset.json`

Key design decisions:
- Same preprocessing functions as `data_single.py` — model inputs match training distribution exactly
- `solar_dataset_subset.json` is a drop-in subset, no file swapping needed
- URL selection cross-references `grouped_df.csv` (reduced AARP 903 from 98 → 24 files)

---

## Completed Stages

| DVC stage                       | Script                            | Status  | Notes |
|---------------------------------|-----------------------------------|---------|-------|
| `subset-data`                   | `aarp_subset_pipeline.py`         | ✅ done  | |
| `subset-confusion-matrix`       | `test.py`                         | ✅ done  | Added mkdir before savefig |
| `subset-prediction-plots`       | `predictions_analyze.py`          | ✅ done  | Added `--json-path`, `--output-dir`; guarded empty df loops |
| `subset-contour-grid`           | `plot_contour_image_grid.py`      | ✅ done  | Added `--json-path`, `--aarp-id`, `--output-path`; cross-split AARP lookup |
| `subset-create-distribution`    | `class_wise_distribution.py`      | ✅ done  | Added `--json-path`, `--output-neg`, `--output-pos`; tqdm on IG loop |
| `subset-plot-class-dist`        | `plot_class_wise_distribution.py` | ✅ done* | Added `--json-path`, `--attributions-*`, `--output-dir`; guarded empty neg list |
| `subset-attribution-analysis`   | `attributions_analyze.py`         | ✅ done  | Added `--json-path`, `--output-dir`; guarded empty val/test loops |
| `subset-create-movie-377`       | `create_movies.py`                | ✅ done  | Added full argparse; cross-split AARP lookup |
| `subset-create-contour-movie-377` | `create_contour_grid_movie.py`  | ✅ done  | New standalone script; all 7 passbands + IG contours per frame |

\* `subset-plot-class-dist`: seaborn was missing from env — installed + added to `torch-tf-312.environment.yml`.
Empty `attributions_list_neg` guard added to `get_intensities_using_attributions` (test split has no neg AARPs).

---

## Skipped Stages

| Stage              | Reason |
|--------------------|--------|
| `subset-kshap`     | kshap runs on training data only. Subset training split = 1 neg AARP (903). Single-class, single-AARP input gives statistically meaningless channel importance scores. Use full-dataset `kshap-stats` stage instead. |
| `subset-kshap-analyze` | Depends on `shap_stats.json` from full dataset — no subset version needed. |

---

## Script Modifications Summary

All changes preserve backwards compatibility — argparse defaults match old hardcoded values exactly,
so existing full-dataset DVC stages are unaffected.

| Script | Changes |
|--------|---------|
| `src/torch/vit/test.py` | Added `mkdir` before `savefig` |
| `src/torch/vit/utils.py` | Guarded timestamp conversion with `if not df.empty` |
| `src/torch/vit/predictions_analyze.py` | Added `--json-path`, `--output-dir`; guarded empty df loops |
| `src/torch/vit/plot_contour_image_grid.py` | Added `--json-path`, `--aarp-id`, `--output-path`; cross-split AARP lookup |
| `src/torch/vit/class_wise_distribution.py` | Added `--json-path`, `--output-neg/pos`; tqdm on IG loop; empty-list guard |
| `src/torch/vit/plot_class_wise_distribution.py` | Added `--json-path`, `--attributions-*`, `--output-dir` |
| `src/torch/vit/attributions_analyze.py` | Added `--json-path`, `--output-dir`; guarded empty loops |
| `src/torch/vit/create_movies.py` | Added full argparse; cross-split AARP lookup; fixed model path typo (`output/` → `outputs/`) |
| `torch-tf-312.environment.yml` | Added `scienceplots`, `seaborn` |

New script:
- `src/torch/vit/create_contour_grid_movie.py` — standalone; all 7-passband contour grid movie via ffmpeg

---

## Known Issues

See `docs/planning/KNOWN_ISSUES.md` for `vit_pytorch==1.10` pin rationale.

---

## Plan: Next Steps

- Run and validate `subset-plot-class-dist` (seaborn fix + empty-neg guard applied, needs rerun)
- Run and validate `subset-attribution-analysis`
- Run and validate `subset-create-movie-377`
- Run and validate `subset-create-contour-movie-377`
- Consider writing `create_raw_movie.py` — raw AIA animation (no attribution) using `single_aarp`
  loader, lifting logic from `aarp_ml/aarp_sequence_module.py:create_aarp_movie`

---

## Expanding the Subset

To add more AARPs: edit `TARGET_AARP_IDS` in `src/aarp_subset_pipeline.py`, re-run `--download --process --json`.

Candidate additions (small file counts):
- AARP 4781 (training, pos, ~10 compressed) — adds pos training sample, enables kshap
- AARP 2026 (test, neg, ~15 compressed) — adds neg test sample for balanced distribution plots
