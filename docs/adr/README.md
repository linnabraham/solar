# Architecture & Process Decision Records

Short records of consequential decisions made during the publication-wrapup phase
(`feat/publication-wrapup`, branched off the squashed `feat/two-channel-training`
history) — what was decided, why, and what it implies for work that follows. Not
a full history; see git log and `docs/postmortems/` for that. Written so a
collaborator or a future session can pick up the repo without re-deriving context
that's already settled.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-checkpoint-selection-dropout03.md) | Checkpoint selection for `pretrained-vit-solar-dataset-dropout03` | Accepted |
| [0002](0002-defer-kfold-cv.md) | Deferring k-fold CV as the fix for noisy single-split estimates | Deferred |
| [0003](0003-multi-architecture-analysis-support.md) | Multi-architecture support pattern for the analysis toolchain | Accepted, in progress |
| [0004](0004-experiment-registry-and-report-battery.md) | Experiment registry and report battery | Accepted (predates this phase; extended here) |
| [0005](0005-git-workflow-publication-wrapup.md) | Git workflow for this phase | Accepted |
