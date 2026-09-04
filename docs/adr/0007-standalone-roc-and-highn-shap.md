# ADR-0007: Standalone ROC and higher-N SHAP ranking variants

**Status:** Accepted — 2026-09-04

## Context

Two gaps noticed while reviewing dropout03's Performance and SHAP tiers:

1. **No ROC curve without a baseline comparison.** The only ROC stage
   (`exp-roc`) goes through `compare_models.py`, which required both
   `--vit-path` and `--xgb-path` and always plotted both curves together.
   There was no way to get a model's own ROC curve without also pairing it
   against the fixed XGBoost baseline.
2. **SHAP ranking's statistical power is capped by sample count.**
   `exp-kshap-stats` runs `kshap.py` at its default `--max-samples 50` (25
   per class after the balanced split) — few enough that the per-channel
   Wilcoxon significance tests `analyze_shap.py` already computes are
   underpowered. Checked: no registered experiment has ever run with a
   higher sample count; this is a genuinely new addition, not a rerun of an
   existing option.

## Decision

**Standalone ROC:** made `compare_models.py`'s `--xgb-path` optional rather
than writing a new script. `plot_roc_comparison()`, `plot_cm_comparison()`,
and `build_metrics_table()` already iterated generically over a `models:
List[Dict]` — no rewrite needed, just conditionally build a 1-model list and
skip XGBoost inference when `--xgb-path` is omitted (`plot_cm_comparison`
already had `n==1` handling). New `exp-roc-solo` DVC stage (foreach
`tier_core`), writing to a separate `roc_solo_test/` dir — does not touch or
replace the existing XGBoost-paired `roc_test/` output. Verified two ways
before wiring into DVC: standalone mode reproduces the exact same ViT AUC as
the paired run (0.865), and re-running the *existing* paired stage after the
patch gave identical results to before it (0.865/0.256) — confirms the
change is additive, not a regression.

**Higher-N SHAP:** new `exp-kshap-stats-highn`/`exp-kshap-analyze-highn`
stages (+ `-validation` counterparts), `--max-samples 200` instead of the
default 50. Cost is linear in sample count for this script (~8s/sample
regardless of N, unlike the IG stages) — ~27 min per subset instead of ~7,
nowhere near IG-attributions territory. Separate output (`kshap_highn/`),
does not replace the default-N `kshap/` results.

**Scope for both:** registered only for `pretrained-vit-dropout03-epoch01-log`
and `-epoch78-log`, not retroactively for the other 5 registry experiments —
a scope decision, not a technical limitation of either change.

## Consequences

- `paper/battery.yaml` gains two entries: `roc_solo` and `shap_ranking_highn`,
  both `status: staged`.
- `exp-roc-solo` inherits the same known gap as `exp-roc`: `compare_models.py`
  has no `--channels` support, so it would fail if pointed at a channel-subset
  experiment.
- Both variants follow the same test-only-for-epoch78/both-splits-for-epoch01
  discipline established in [ADR-0006](0006-two-checkpoint-transparency-dropout03.md),
  for consistency rather than a new cost constraint (`exp-roc`/`exp-roc-solo`
  were already test-only stages by construction, no validation variant exists
  for either).
