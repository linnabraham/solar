# ADR-0002: Deferring k-fold CV as the fix for noisy single-split estimates

**Status:** Deferred — 2026-09-02 (revisit if a comparison hinges on it)

## Context

The project's dominant source of instability is region scarcity: only 17
positive AARPs exist in the whole dataset, and any single train/val/test split
gives a high-variance estimate of a model's true skill — a point already
established repeatedly across this project's history (e.g. an earlier 2-channel
CV run showed balanced_accuracy swinging 0.624–0.719 across folds 0–2 on an
*identical* recipe, purely from which AARPs landed in which split).

k-fold CV is the structurally correct fix: rotating which AARPs are held out
across `cv_folds/fold_0.json`–`fold_4.json` (already generated) gives both a
mean estimate and a real, direct measurement of region-composition variance —
something no amount of clever handling of a single split (windowed selection,
bootstrap, etc.) can produce, since those only characterize noise *within* one
fixed partition.

Cost: full CV requires **retraining**, not just re-evaluating, per fold — 5x
the GPU-hours of a single run for 5 folds (10x to compare two recipes). Folds
1–4 also don't yet have their own leakage-free stats files
(`cv_folds/fold_N_stats.pkl`) computed.

## Decision

**Not pursuing CV right now.** The collaborators driving this work are not
asking for it, and the cost is real. Continuing to work from single-split
point estimates (e.g. [ADR-0001](0001-checkpoint-selection-dropout03.md)),
with explicit uncertainty caveats rather than false precision.

## Consequences

- Any recipe-level comparison (e.g. dropout 0.0 vs. 0.3) made without CV should
  be treated as **unproven** if its effect size is smaller than the
  ~0.05–0.1 balanced-accuracy noise floor already demonstrated from
  region-composition alone.
- The fold infrastructure (`make_cv_folds.py`, `run_cv.py`, `cv_folds/fold_*.json`)
  and cost breakdown are already documented — a cheaper middle ground (3 folds
  instead of 5) exists if this gets revisited.
- Should a specific comparison later become high-stakes enough to need a real
  answer, this is the decision to reopen — not a new investigation from
  scratch.
