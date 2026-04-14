# Subset Pipeline — Status & Plan

**Branch**: `feat/per-aarp-analysis`  
**Goal**: Run existing DVC analysis stages on a small locally-downloaded subset of AARPs,
without depending on the central data server.

---

## Target AARPs (current 7-AARP selection)

| AARP | Split      | Class | Timesteps | max_dim |
|------|------------|-------|-----------|---------|
|  903 | training   | neg   |        55 |     581 |
| 1807 | training   | pos   |       231 |    1019 |
|  185 | validation | neg   |       528 |     851 |
| 3563 | validation | pos   |       231 |    1975 |
|  377 | test       | pos   |       231 |     696 |
| 1449 | test       | pos   |        55 |     783 |
| 4296 | test       | neg   |       616 |    1166 |

Both classes represented in all three splits. Selection prioritises larger `max_dim`
(less black padding after resize to 512×512) within each split/class combination.

Original 3-AARP set (377, 1449, 903) was a quick-start with minimal downloads;
expanded to 7 AARPs to cover all splits and both classes.

---

## Data Pipeline Script: `src/aarp_subset_pipeline.py`

Phases:
- `--select`   : Validate URLs, intersect with `grouped_df.csv`
- `--download` : Fetch compressed FITS from NASA public server (manual, not in DVC)
- `--process`  : Extract + pad/resize to 512×512 using `aarp_ml.data_prep`
- `--json`     : Filter `solar_dataset.json` for target AARPs → `solar_dataset_subset.json`

Run all phases in one shot (download skips existing files, process skips complete AARPs):
```
python -m src.aarp_subset_pipeline --all
```

Skip logic in `--process`:
- Checks extracted file count against expected count from `solar_dataset.json`
- Checks for zero-byte files (truncated writes)
- Only re-processes AARPs that fail either check

`--download` is intentionally excluded from the DVC stage — external network fetches
are not reproducible. DVC stage runs `--process --json` only.

---

## Completed Stages

| DVC stage                         | Script                            | Status   | Notes |
|-----------------------------------|-----------------------------------|----------|-------|
| `subset-data`                     | `aarp_subset_pipeline.py`         | ✅ done   | |
| `subset-confusion-matrix`         | `test.py`                         | ✅ done   | Added mkdir before savefig |
| `subset-prediction-plots`         | `predictions_analyze.py`          | ✅ done   | Added `--json-path`, `--output-dir`; guarded empty df loops |
| `subset-contour-grid`             | `plot_contour_image_grid.py`      | ✅ done   | Added `--json-path`, `--aarp-id`, `--output-path`; cross-split AARP lookup |
| `subset-create-distribution`      | `class_wise_distribution.py`      | ✅ done   | Added `--json-path`, `--output-neg/pos`; tqdm on IG loop; empty-list guard |
| `subset-plot-class-dist`          | `plot_class_wise_distribution.py` | ✅ done*  | Added `--json-path`, `--attributions-*`, `--output-dir`; guarded empty neg list |
| `subset-attribution-analysis`     | `attributions_analyze.py`         | ✅ done   | Added `--json-path`, `--output-dir`; guarded empty loops |
| `subset-raw-movies`               | `create_raw_movie.py`             | ✅ done   | New script; loops all AARPs; directory output |
| `subset-attribution-movies`       | `create_movies.py`                | 🔲 ready  | Updated to loop all AARPs; directory output |
| `subset-contour-grid-movies`      | `create_contour_grid_movie.py`    | 🔲 ready  | New script; loops all AARPs; directory output |

\* `subset-plot-class-dist`: seaborn missing from env (installed + added to yml);
empty `attributions_list_neg` guard added (test split had no neg AARPs in original 3-AARP set).

---

## Skipped Stages

| Stage               | Reason |
|---------------------|--------|
| `subset-kshap`      | kshap runs on training data only. Meaningful results require multiple training AARPs of both classes with enough samples for KernelSHAP perturbations. Current subset too small. Use full-dataset `kshap-stats` stage instead. |
| `subset-kshap-analyze` | Depends on `shap_stats.json` from full dataset. |

---

## Script Modifications Summary

All changes preserve backwards compatibility — argparse defaults match old hardcoded values,
so existing full-dataset DVC stages are unaffected.

| Script | Changes |
|--------|---------|
| `src/torch/vit/test.py` | Added mkdir before savefig |
| `src/torch/vit/utils.py` | Guarded timestamp conversion with `if not df.empty` |
| `src/torch/vit/predictions_analyze.py` | Added `--json-path`, `--output-dir`; guarded empty df loops |
| `src/torch/vit/plot_contour_image_grid.py` | Added `--json-path`, `--aarp-id`, `--output-path`; cross-split AARP lookup |
| `src/torch/vit/class_wise_distribution.py` | Added `--json-path`, `--output-neg/pos`; tqdm on IG loop; empty-list guard |
| `src/torch/vit/plot_class_wise_distribution.py` | Added `--json-path`, `--attributions-*`, `--output-dir` |
| `src/torch/vit/attributions_analyze.py` | Added `--json-path`, `--output-dir`; guarded empty loops |
| `src/torch/vit/create_movies.py` | Loops all AARPs; `--output-dir`; fixed model path typo (`output/` → `outputs/`) |
| `src/torch/vit/create_raw_movie.py` | New standalone script; all-AARP loop; no model needed |
| `src/torch/vit/create_contour_grid_movie.py` | New standalone script; all-AARP loop; all 7 passbands + IG contours per frame |
| `src/aarp_subset_pipeline.py` | Expanded TARGET_AARP_IDS to 7; added skip logic with count + zero-byte checks |
| `torch-tf-312.environment.yml` | Added `scienceplots`, `seaborn` |

---

## DVC Stage Design Notes

- All subset stages use `--json-path solar_dataset_subset.json` — no file swapping
- Movie stages output to directories (`cache: false`) — adding AARPs just adds new MP4s
- `solar_dataset_subset.json` is a dep of all subset stages — changing the subset
  (adding AARPs) automatically invalidates all downstream stages
- `--download` excluded from DVC by design — external fetches are not reproducible

---

## Known Issues

See `docs/planning/KNOWN_ISSUES.md` for `vit_pytorch==1.10` pin rationale.

---

## Expanding the Subset

To add more AARPs: edit `TARGET_AARP_IDS` in `src/aarp_subset_pipeline.py`,
run `python -m src.aarp_subset_pipeline --all`, then `dvc commit subset-data`.

Notable unselected candidates:
- AARP 4920 (test, pos, max_dim=2131) — largest in dataset, zero padding
- AARP 1126 (train, neg, max_dim=1959) — large train neg
- AARP 833  (val, pos,  max_dim=882)  — alternative val pos
