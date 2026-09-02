# ADR-0004: Experiment registry and report battery

**Status:** Accepted — originally decided 2026-07-17, extended 2026-09-02

## Context

The project needed a way to compare many (checkpoint × pipeline-state)
combinations for the eventual paper without duplicating evaluation code per
model or losing track of what's already been computed.

## Decision

Established prior to this session (recorded here for reference, since this
session actively extended it):

- **`paper/experiments.yaml`** is the registry: one spec per experiment
  (checkpoint path, stats file, dataset JSON, plus free-form extra-CLI-args
  fields like `channels_args`), grouped into `tier_core`/`tier_explain`/`tier_shap`
  membership dicts. An *experiment* = (model checkpoint × pipeline state);
  the same checkpoint under different stats is a distinct, valid experiment.
- **`dvc.yaml`**'s `exp-*` stages `foreach` over those tier dicts, generating
  one parametrized DVC stage instance per experiment per battery item, writing
  to `artifacts/<exp-id>/` and `metrics/<exp-id>/`.
- **`paper/battery.yaml`** is the contract listing every possible report
  section — caption, output path template, which DVC stage produces it, tier,
  and `status` (`staged` / `needs-stage` / `needs-extraction` / `manual`).
- **`notebooks/experiment_report.ipynb`** is a read-only, papermill-parametrized
  viewer, rendered per experiment by `src/paper/render_reports.py` into
  `reports/<exp-id>.html`. It computes nothing itself — only reads from
  `artifacts/`/`metrics/`.

### This session's extension

- Added a `model_type_args` field (following the existing `channels_args`
  pattern) to **every** spec in the registry, and wired
  `${item.model_type_args}` into the `exp-eval-test`/`exp-eval-validation`
  stage command templates in `dvc.yaml` — this is what lets an experiment
  specify `--model-type vit-pretrained`.
- Registered **`pretrained-vit-dropout03-epoch01-log`** (see
  [ADR-0001](0001-checkpoint-selection-dropout03.md)) as the first
  pretrained-ViT entry in the registry — `tier_core` only so far.
- Ran `exp-eval-test@pretrained-vit-dropout03-epoch01-log` and
  `exp-eval-validation@...` for real via `dvc repro` — the first genuine
  (non-smoke-test) DVC-tracked output for this checkpoint. Results matched
  every number already established earlier in the session exactly (test TSS
  0.4585, val precision pinned at 1.0000) — a strong cross-check that the
  registry wiring is correct.

## Consequences

- Every *other* `exp-*` stage (IG attributions, contour grids, kSHAP, etc.)
  still needs the same `${item.model_type_args}` wiring added to its own
  `dvc.yaml` template before this experiment can move into `tier_explain`/
  `tier_shap` — not done yet, tier by tier as the corresponding scripts get
  used at scale.
- Running `dvc repro` on this experiment surfaced that **`stats.pkl.dvc`'s
  tracked hash was stale**, still pointing at the pre-normalization-fix file
  (`fe8e2d59...`, the exact hash flagged as an open follow-up in
  `docs/postmortems/normalization-bug.md`). `dvc repro` corrected the local
  `.dvc` pointer to the true current (correct, log-space) hash
  (`054da108...`) as a side effect. This closes a year-old-plus documented
  gap, but **only locally** — it has not been pushed to the DVC remote.
