# Subset Pipeline — Status & Plan

**Branch**: `feat/per-aarp-analysis`  
**Goal**: Run existing DVC analysis stages on a small locally-downloaded subset of AARPs,
without depending on the central data server.

---

## What Was Done (April 10, 2026)

### Data pipeline script: `src/aarp_subset_pipeline.py`

New script (separate from the earlier `src/subset_downloader.py` which handled only AARP 4920).

**Target AARPs** (initial 3-AARP validation set):

| AARP | Split    | Label | Compressed files |
|------|----------|-------|-----------------|
| 1449 | test     | pos   | 7               |
| 377  | test     | pos   | 28              |
| 903  | training | neg   | 24              |

**Phases:**
- `--select`   : Validate URLs in pre-filtered CSVs, intersect with `grouped_df.csv` (skip
                 files that didn't pass shape/off-limb selection in the original pipeline)
- `--download` : Download compressed 7h FITS from NASA public server → `data/E8/compressed/{pos,neg}/`
- `--process`  : Extract timesteps + pad/resize to 512×512 using identical `aarp_ml.data_prep`
                 functions as `data_single.py` — ensures bit-identical output to training data
- `--json`     : Filter `solar_dataset.json` entries for target AARPs →
                 `solar_dataset_subset.json` (relative paths, correct split assignments)

**Key design decisions:**
- Uses `aarp_ml.data_prep.pad_and_resize_in_parallel` + `pad_with_quiet` — same as full pipeline,
  not reimplemented, so model inputs match training distribution exactly
- URL selection cross-references `grouped_df.csv` to avoid downloading files filtered out by
  shape/off-limb selection (reduced AARP 903 from 98 → 24 files)
- `solar_dataset_subset.json` is a drop-in subset of `solar_dataset.json` — no file swapping

### DVC stages added

Two new stages appended to `dvc.yaml`:

**`subset-data`** — runs `aarp_subset_pipeline.py --process --json`
- Intentionally excludes `--download` (one-time external fetch, not DVC-reproducible)
- Output: `solar_dataset_subset.json`

**`subset-confusion-matrix`** — runs `test.py` with subset JSON directly
- No symlink swap needed — uses `--json-path solar_dataset_subset.json`
- Output: `plots/subset/cm.png` (separate from full-dataset `plots/cm.png`)

### Known issue fixed

`vit_pytorch` version mismatch with `outputs/glad-shape-197/trained_model.pth`:
- Checkpoint was trained with older vit_pytorch where `cls_token` shape was `[1, 1, 1024]`
- Newer vit_pytorch uses `[num_cls_tokens, dim]` → shape mismatch on load
- **Fix**: Pinned `vit_pytorch==1.10` (committed by user)
- Documented in `docs/planning/KNOWN_ISSUES.md`

---

## Current State

- [x] `src/aarp_subset_pipeline.py` written and validated (select + json phases tested)
- [x] Data downloaded: `data/E8/compressed/{pos,neg}/` populated
- [x] `solar_dataset_subset.json` generated
- [x] `vit_pytorch==1.10` pinned, `confusion-matrix-val` reproduced successfully
- [x] `subset-data` and `subset-confusion-matrix` stages added to `dvc.yaml`
- [ ] `dvc repro subset-confusion-matrix` — next to run

---

## Plan: Remaining Stages

Stages will be added incrementally. Scripts that hardcode `"solar_dataset.json"` need a
`--json-path` argument added before they can be wired into a subset stage.

| DVC stage (subset)            | Script                      | `--json-path` ready? | Priority |
|-------------------------------|-----------------------------|----------------------|----------|
| `subset-confusion-matrix`     | `test.py`                   | ✅ yes               | done     |
| `subset-prediction-plots`     | `predictions_analyze.py`    | ❌ hardcoded         | next     |
| `subset-attribution-analysis` | `attributions_analyze.py`   | ❌ hardcoded         | next     |
| `subset-create-distribution`  | `class_wise_distribution.py`| ❌ hardcoded         | later    |
| `subset-plot-class-dist`      | `plot_class_wise_distribution.py` | ❌ hardcoded   | later    |
| `subset-contour-grid`         | `plot_contour_image_grid.py`| ❌ hardcoded         | later    |
| `subset-kshap`                | `kshap.py`                  | ✅ yes               | later    |
| `subset-kshap-analyze`        | `analyze_shap.py`           | n/a (uses shap_stats.json) | later |
| `subset-movie`                | `create_movies.py`          | ❌ hardcoded         | later    |

**Pattern for adding `--json-path` to hardcoded scripts:**
Each script needs ~3 lines: add `argparse` arg with `default="solar_dataset.json"`, thread it
through to wherever `TrainingConfig` or `open('solar_dataset.json')` is called.

---

## AARP Selection — Expanding the Subset

Current subset is 3 AARPs (2 pos/test + 1 neg/train). To add more, edit `TARGET_AARP_IDS`
in `src/aarp_subset_pipeline.py` and re-run `--download --process --json`.

Candidate additions (small file counts):
- AARP 4781 (training, pos, ~10 compressed) — adds pos training sample
- AARP 2026 (test, neg, 15 compressed after groupby filter) — adds neg test sample for balanced CM
