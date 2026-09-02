# ADR-0005: Git workflow for the publication-wrapup phase

**Status:** Accepted — 2026-09-02

## Context

`feat/two-channel-training` had drifted from its name — 14 commits covering
the normalization fix, CV-fold infrastructure, the pretrained-ViT refactor,
and more, far beyond "two-channel training" — and had never been pushed to
any remote. Separately, `.gitignore` was missing entries for `outputs/`
(384GB), `output/` (223GB, a stray singular-named duplicate), `notebooks/`
(206MB), `reports/`, `metrics/`, `artifacts/` — a stray `git add -A` would
have tried to stage over 600GB.

## Decision

- **Squashed** the 14 commits into 8 thematically-grouped commits (one per
  major capability: normalization fix, channel-subset support, sampler+seed
  fix, pretrained-ViT infra, eval/DVC-restructure, CV-fold infra, AR-geometry
  utility, docs), verifying the final tree was byte-identical to the original
  before replacing history (`git diff <old> <new>` empty). Original history
  preserved at `feat/two-channel-training-backup`.
- **Renamed** to `feat/publication-wrapup`, branched from the squashed tip —
  an accurate name for the actual current phase of work (checkpoint
  methodology, multi-architecture analysis support, experiment registry),
  rather than continuing to rename-by-implication under a stale branch name.
  Renaming was confirmed inconsequential to any future merge, since neither
  branch has ever had a remote upstream.
- **Fixed `.gitignore`**: `notebooks/`, `outputs/`, `output/`, `reports/`,
  `metrics/`, `artifacts/` now ignored outright; `cv_folds/` gets a partial
  ignore (`cv_folds/*` + explicit `!cv_folds/fold_{0..4}.json` un-ignores) so
  only the lightweight, reproducibility-relevant split definitions are
  tracked, not the generated `.pkl` stats or result JSONs.
- **New work is committed in small, single-purpose commits** — one script's
  patch per commit, each preceded by a smoke test on a real checkpoint before
  committing, rather than accumulating a large uncommitted diff.

## Consequences

- Future sessions should keep committing to `feat/publication-wrapup` with
  the same discipline: validate before committing, one concern per commit.
- `feat/two-channel-training-backup` should eventually be deleted once
  confidence in the squashed history is fully established — not done yet,
  kept intentionally as a safety net.
- Whether/when to merge stable pieces of this branch back to `dev` is still
  an open, separate decision — not addressed by this ADR.
