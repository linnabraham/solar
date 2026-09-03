# ADR-0006: Two-checkpoint transparency for dropout03 Performance/Explainability reporting

**Status:** Accepted, in progress — 2026-09-03. Refines [ADR-0001](0001-checkpoint-selection-dropout03.md).

## Context

ADR-0001 established using `epoch_01` (chosen via a 100-epoch test-curve sweep) as
the working checkpoint for `pretrained-vit-solar-dataset-dropout03`, with an
explicit caveat that its test metrics are optimistic — the epoch was chosen
*because* it scored best on test, so its own test numbers can't be read as an
independent measurement.

That caveat lived only in ADR-0001's text. It was not visible in the actual
generated artifacts (`cm_test.png`, `scores_test.json`, `roc_test/*`) — read in
isolation, `pretrained-vit-dropout03-epoch01-log`'s Performance-tier output
looks exactly like every other registered experiment's, with no signal that
this one's checkpoint was selected differently and more circularly than the
rest.

**Options considered and rejected:**
- A "peek-free" epoch (e.g. the final epoch, 99) as the sole reported
  checkpoint — removes the selection bias, but still hands the reader one
  clean number that could be over-trusted just the same.
- A full duplicated battery (validation *and* test, for both checkpoints) —
  too expensive (would double the ~8h Explainability cost per checkpoint),
  and conflates two different sources of variation (which split, which
  checkpoint) in a single comparison instead of isolating one.

**A real mistake made and corrected mid-investigation:** the first candidate
proposed for "the val-based comparison checkpoint" was `epoch_48` — this run's
true minimum-val_loss epoch. Checking properly (deriving val TSS/specificity/
balanced_accuracy from precision+recall+accuracy, since this run only logs
those four val fields) showed epoch_48 is one of the 96-of-100 epochs with val
precision pinned at exactly 1.0 — the boundary-pinning degenerate-classifier
artifact this project has repeatedly flagged as untrustworthy — and it loses
to `epoch_78` (the collaborator's own original manual pick, from the email that
started this whole investigation) on every discriminative val metric except
raw loss:

| | epoch_78 | epoch_48 |
|---|---:|---:|
| val_loss | 3.339 | **1.577** (lower, but not the right criterion) |
| val_accuracy | **0.764** | 0.703 |
| val_precision | 0.980 (not pinned) | 1.000 (pinned) |
| val balanced_accuracy (derived) | **0.748** | 0.682 |
| val TSS (derived) | **0.496** | 0.365 |

This project already established, in an earlier session, that val
balanced-accuracy/TSS should be prioritized over raw val_loss for exactly this
kind of case (loss can be dominated by confident-but-uninformative behavior
without reflecting real discriminative quality). Applying that consistently,
`epoch_78` — not `epoch_48` — is the defensible val-based checkpoint.

## Decision

- Register a second experiment, **`pretrained-vit-dropout03-epoch78-log`**
  (same training run, `epoch_78.pth`), chosen by the corrected val criteria
  above: non-degenerate, best val balanced-accuracy/TSS checked.
- Run it **test-data-only** — deliberately skip its own validation-subset
  evaluation through the pipeline (we already have those numbers precisely,
  derived by hand above). This isolates the checkpoint-choice variable while
  holding the evaluation data fixed, rather than conflating "different
  checkpoint" and "different split" in one comparison.
- Build the same battery tiers for `epoch_78` as for `epoch_01`, in the same
  order, so they can be read side by side. **The goal is transparency, not
  debiasing** — this doesn't produce an unbiased number for either checkpoint;
  it makes the instability of any single number visible by showing two real,
  differently (and defensibly) selected checkpoints disagree on identical
  data.

## Consequences (Performance tier already complete for both)

| metric | epoch_01 (test-selected) | epoch_78 (val-selected) |
|---|---:|---:|
| image-level TSS (threshold 0.5) | 0.459 | 0.262 |
| ROC-AUC (threshold-free) | 0.865 | 0.828 |
| AR-level TSS (8 AARPs, mean-pooled) | 0.500 | 0.500 — **identical** |

This surfaced a sharper point than originally intended: it's not just *which
epoch* you pick that swings the answer — it's also *which metric and
aggregation level* you report it at. The AR-level view shows literally no gap
between the two checkpoints; image-level TSS shows a large one; AUC shows a
small one. All four numbers come from the exact same two checkpoints
evaluated on the exact same test data.

`epoch_01`'s artifacts also currently include leftover validation-side
Performance output (`cm_validation.png`, `scores_validation.json`) generated
before this two-checkpoint plan was adopted — not removed, but should be read
as historical, not part of the matched test-only comparison.

**Sequencing from here** (per direct instruction, not yet complete): build
Explainability tier for `epoch_78` next (contour grids + prediction plots,
cheap; `exp-ig-attributions-test` only, ~4.5h — test-only, consistent with the
rest of this ADR), then return to `epoch_01` and finish what's still
outstanding there — its IG-attributions need regenerating regardless of this
ADR, since the pre-fix run's output was invalidated by the upsample-back
resolution bug (fixed in commit `a5c80fd`, never re-run). `epoch_01` was
originally registered with both validation and test wiring, so its
regeneration covers both splits (~8h), unlike `epoch_78`'s test-only scope.
