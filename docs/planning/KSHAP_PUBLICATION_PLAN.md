# KernelSHAP stage — publication-grade rework plan

Scope: bring `src/torch/vit/kshap.py` and `src/torch/vit/analyze_shap.py` (DVC stages `kshap-stats` and `kshap-analyze`) to a state suitable for journal publication of per-passband SHAP importance results for the ViT solar-flare model.

This plan covers items **#1, #3, #4, #5** from the methodological review. Item **#6 (baseline choice)** is intentionally out of scope for now and will be revisited.

## Background — why the current figure is misleading

`shap_stats.json` is generated with `target=true_label`, so for label=1 samples AIA_131 attributions are strongly negative (~−13) and for label=0 samples strongly positive (~+14). The current `analyze_shap.py` boxplots **mix both classes**, so the median SHAP for AIA_131 lands near zero with enormous spread — making the most important channel look uninformative. A one-sample t-test against zero on this mixed distribution will fail to reject. This is a presentation problem, not a model problem.

Three other issues compound it:

- Augmentation (random flips) is enabled on the SHAP data loader, so each attribution is for an arbitrarily flipped frame.
- `WeightedRandomSampler(replacement=True)` lets the same image appear multiple times; the printed `N=50` overstates effective sample size.
- No filter on prediction correctness, so misclassified samples (with reversed-direction attributions) contaminate the aggregation.

## Issue list and fix plan

### #3 — Drop augmentation during attribution

**Why:** Explainability inputs must be deterministic. Random flips during SHAP make each per-image attribution conditional on an arbitrary geometric transform.

**Where:** `src/torch/vit/kshap.py`, `get_data_loader()` (currently builds a `v2.Compose([AIALogTransform, RandomHorizontalFlip, RandomVerticalFlip])`).

**Change:** Pass a bare `AIALogTransform(means, stds)` as the dataset transform.

**Risk:** None. The flip probability constant (`DEFAULT_FLIP_PROBABILITY`) becomes unused; leave it defined but stop referencing it (preserve-existing-files convention).

### #4 — Replace WeightedRandomSampler with deterministic stratified sampler

**Why:** With `replacement=True` and `max_samples=50`, the same AARP can be drawn multiple times; statistical claims (CIs, t-tests) assume i.i.d. samples. Stratified-without-replacement also guarantees balanced class counts independent of the dataset's class ratio.

**Where:** `kshap.py` — `get_weighted_sampler()` is replaced (left in place but no longer called).

**Change:**
- Add `seed: int = 42` to `KShapConfig`.
- New helper `get_stratified_indices(dataset, max_samples, seed)`:
  - Group dataset indices by `dataset.labels`.
  - With `np.random.default_rng(seed)`, draw `max_samples // 2` indices from each class without replacement.
  - Raise a clear error if a class has fewer than `max_samples // 2` available.
- `get_data_loader` uses `SubsetRandomSampler(indices)` over those indices.
- `main()` loop's `if i >= config.max_samples: break` becomes a redundant safety guard (the loader is now exact-size).

**Risk:** Low. Backward-incompatible only insofar as the chosen samples will differ — but the prior selection was non-reproducible anyway.

### #5 — Save model prediction per record (filter at analysis time)

**Why:** Misclassified samples produce backwards-direction attributions and should not be aggregated with correct ones in the headline figure. Filtering at compute time would couple the (expensive) JSON to one analysis policy; saving the prediction lets `analyze_shap.py` filter freely.

**Where:** `kshap.py` `main()` loop; record schema; `analyze_shap.py` `load_and_process` and figure code.

**Schema change:** Each record in `shap_stats.json` gains a top-level `"prediction": int` field.

```json
{
  "round": 0,
  "label": 1,
  "prediction": 1,
  "importance": {...},
  "std_error": {...}
}
```

**Change in `kshap.py`:**
- Before calling `do_kernel_shap`, run a forward pass:
  ```python
  with torch.no_grad():
      pred = int(model(image).argmax(dim=1).item())
  ```
- Add `prediction=pred` to the saved record.

**Change in `analyze_shap.py`:**
- Add `correct_only: bool = True` to `SHAPAnalysisConfig`.
- In `load_and_process`, if `prediction` exists in the records and `correct_only` is True, drop rows where `prediction != label`. Print kept / total counts.
- If `prediction` is absent (legacy JSON), warn and skip the filter.

**Risk:** One extra forward pass per sample (cheap relative to KernelShap). Old `shap_stats.json` becomes a "legacy" record; DVC will rerun the stage anyway because the script changed.

### #1 — Replace mixed-class boxplot with two paper-ready panels

**Why:** This is the single biggest publication risk. The current figure visually undersells the model's most important channels.

**Where:** `src/torch/vit/analyze_shap.py` — full rewrite of `run_statistics` and `plot_results`.

**New outputs (to `plots/kshap/results/`):**

1. **`global_mean_abs_shap.png` — global feature importance.**
   - Horizontal or vertical bar chart of `mean(|SHAP|)` per channel.
   - Bootstrap 95% CI as error bars (n_boot = 1000, percentile method).
   - Channels sorted left-to-right by AIA wavelength (94 → 335).
   - Categorical color per channel (one bar per channel, no continuous palette).
   - Caption / annotation: total N after correctness filter, kept-vs-total count.

2. **`class_stratified_shap.png` — direction by class.**
   - Two-panel facet on `label` (panel 0 = no-flare, panel 1 = flare-positive).
   - Boxplot + jittered stripplot of *signed* SHAP per channel, shared y-axis.
   - Reference line at zero per panel.
   - Caption: "SHAP attributions are computed with respect to the true class. Positive values push the model toward the true class."

3. **`shap_summary.csv` — numeric summary for the paper text.**
   - Per-channel-per-class: N, mean signed SHAP, mean |SHAP|, bootstrap 95% CI, Wilcoxon signed-rank statistic and p-value.
   - Bonferroni-corrected p-value across 7 channels per class.

**Removed:**
- The single mixed-class boxplot (`significance_plot.png`).
- The `ttest_1samp` against zero in `run_statistics` — replaced by Wilcoxon signed-rank (non-parametric, no normality assumption) plus bootstrap CIs.

**Channel ordering:** The current data ordering happens to be wavelength-sorted (94, 131, 171, 193, 211, 304, 335) — preserve it explicitly in the figure code rather than relying on dict-iteration order.

**DPI / fonts:** Build the figure with `dpi=config.dpi` from the start; do not mix `figure(dpi=150)` with `savefig(dpi=300)`. Save additionally as PDF for paper inclusion (vector text, no rasterization).

**Risk:** Cosmetic / iterative. Easy to revise based on co-author feedback.

## Order of operations

Independent fixes, but logical sequence to minimize wasted runs:

1. **#3** (drop augmentation) — single line, no schema impact.
2. **#4** (stratified sampler) — affects which samples are picked; smoke-test with `max_samples=4` before a full run.
3. **#5** (save prediction) — schema change; existing `shap_stats.json` becomes legacy.
4. Rerun `dvc repro kshap-stats` once. ~50 KernelShap evaluations, GPU.
5. **#1** (figure rewrite) — purely in `analyze_shap.py`. Iterate freely; no GPU needed.
6. Rerun `dvc repro kshap-analyze`.

## DVC stage updates

`dvc.yaml` `kshap-stats` and `kshap-analyze` stages keep the same command lines and dep lists. Outputs grow:

```yaml
kshap-analyze:
  cmd: python -m src.torch.vit.analyze_shap
  deps:
    - src/torch/vit/analyze_shap.py
    - shap_stats.json
  outs:
    - plots/kshap/results
```

`plots/kshap/results/` is the directory output, so the new files (PNG + PDF + CSV) are picked up automatically without listing them.

## Out of scope (this round)

- **#6 — baseline choice.** Current zero-image baseline retained for now. Switching to a mean-image (or per-class-mean) baseline is a separate change with its own validation; revisit after the methodological fixes above are in place.
- **Long-format Parquet schema.** Discussed in the structural review; deferred.
- **`passbands.py` shared module.** Deferred until other consumers of channel constants emerge.
- **`params.yaml` integration for `n_samples` / `max_samples`.** Deferred.

## Acceptance criteria

- `kshap-stats` runs cleanly with `max_samples=50` and produces a JSON where every record has a `prediction` field.
- `kshap-analyze` produces three artifacts in `plots/kshap/results/`: `global_mean_abs_shap.png`, `class_stratified_shap.png`, `shap_summary.csv`.
- Both stages reproducible end-to-end via `dvc repro kshap-stats kshap-analyze`.
- Stratified sampler produces exactly `max_samples` unique samples with `max_samples // 2` per class.
- Random flips no longer appear in the SHAP loader's transform.
