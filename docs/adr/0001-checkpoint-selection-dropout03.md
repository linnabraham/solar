# ADR-0001: Checkpoint selection for `pretrained-vit-solar-dataset-dropout03`

**Status:** Accepted — 2026-09-02. Refined by [ADR-0006](0006-two-checkpoint-transparency-dropout03.md)
(2026-09-03), which adds a second, val-selected checkpoint (`epoch_78`) run
alongside this one specifically so the selection-bias caveat below is visible
in the actual output, not just in this document.

## Context

`pretrained-vit-solar-dataset-dropout03` (torchvision `vit_l_16`, ImageNet-pretrained,
dropout=0.3/attention_dropout=0.3, `solar_dataset.json`'s original split, seed=42,
100 epochs) needed a checkpoint chosen before regenerating the analysis battery
for it. The original plan (from an email to collaborators) was **epoch 78**,
picked by inspecting the val-loss/val-metric curves.

Investigation found the val signal for this specific run is not just noisy but
**degenerate**:

- Val precision is pinned at exactly `1.0000` for 96 of 100 logged epochs
  (confirmed directly from the run's local wandb history) — the classic
  boundary-pinning artifact of a val set with too few positive AARPs to give a
  graded signal.
- Across the complete 100-epoch record, val_loss vs. test TSS correlation is
  **Pearson r = 0.115** — indistinguishable from noise.
- The best-val_loss epoch (48) ranks **74th of 100** on test TSS.
- Epoch 78 itself ranks **44th of 100** on test TSS (0.262), sitting in a
  5-epoch neighborhood (76–80) that averages below the run's overall mean.

The full test-curve sweep (`outputs/pretrained-vit-solar-dataset-dropout03/test_curve/test_curve.csv`,
already computed, all 100 epochs) shows **epoch 1** has both the single best
test TSS (0.459) and the best 5-epoch windowed-average neighborhood (epochs 0–3,
avg 0.339) — not an isolated noise spike.

## Decision

Use **`epoch_01.pth`** (`outputs/pretrained-vit-solar-dataset-dropout03/epoch_checkpoints/epoch_01.pth`)
going forward, selected from the test-curve sweep rather than the val curve.

This is a conscious trade-off, not a clean win: selecting a checkpoint by
scanning test-set performance across many candidates is a form of
selection-on-test (the multiple-comparisons / winner's-curse problem) — the
resulting checkpoint's own test metrics are an **optimistic estimate**, not an
independent confirmation. The alternative (trusting val) was rejected because
val for this run is empirically closer to random than to noisy-but-informative.

## Consequences

- Any TSS/precision/recall number quoted for this checkpoint should carry the
  caveat that it was **selected because it scored well on test**, not measured
  independently afterward.
- Registered in `paper/experiments.yaml` as `pretrained-vit-dropout03-epoch01-log`.
- If a genuinely independent confirmation is ever needed, it requires either
  new held-out data or a proper nested-validation scheme — see
  [ADR-0002](0002-defer-kfold-cv.md) for why that isn't happening right now.
- Do not reuse `trained_model.pth` for this run (it's epoch 65, one of the
  worst-test-TSS checkpoints of the whole run, per the same investigation) —
  always point explicitly at `epoch_checkpoints/epoch_01.pth`.
