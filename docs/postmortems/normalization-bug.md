# Postmortem: `AIALogTransform` z-scored with raw-space statistics

**Status:** Fixed in code (`f4adc6e`, 2026-07-11). `stats.pkl` itself is not yet re-tracked in DVC —
see [Open follow-ups](#open-follow-ups).

## Summary

`AIALogTransform` log-transforms every pixel before z-scoring it, but the `mean`/`std` values it
z-scored against (`stats.pkl`) were computed on **raw, linear-space** pixel values, not on the
log-transformed values. The mismatch was introduced when z-score normalization was added to the ViT
data pipeline in April 2025, went unexamined through two later on/off toggles of that same line, and
was not caught until July 2026 — over a year later — while investigating why Integrated Gradients
attribution maps looked qualitatively different across model checkpoints.

Every ViT model trained before the fix (`glad-shape-197` among them) was trained on this
mis-normalized input.

## Timeline

| Date | Commit | Event |
|---|---|---|
| 2025-01-21 | `27a3fd8` | PyTorch ViT pipeline added (`aia_euv`, `CustomTransform`). Transform only does `log(x)`; `means`/`stds` are accepted in the constructor but unused. |
| 2025-04-06 | `89c46d7` | *"Add channel wise z-score normalization to ViT model"* — `x = (x - self.means) / self.stds` added after the log. This is the moment the transform started requiring log-space stats. No accompanying change to how `means`/`stds` are computed. |
| 2025-04-10 | `49956dd` | *"Remove passband wise z-score normalization"* — z-score removed again, 4 days later. No explanation in the commit message. |
| 2025-06-05 | `8cfea17` | *"Add back z-score normalization"* — re-added ~2 months later. Again a single-line toggle; the source of `means`/`stds` is not discussed in either toggle. |
| 2025-06-20 | `ba154e5` | The `stats.pkl` that ends up DVC-tracked (md5 `fe8e2d59c30489c29ec6e44ff8afb493`) is added — generated via the raw-space path described below. This is the file every pre-fix model, including `glad-shape-197`, was trained on. |
| 2025-07-08 | `9517dde` | `data_single.py` wired to call `aarp_ml.model.training.compute_mean_and_std` for its `--stats` flag, formalizing the raw-space path as the official pipeline entry point. |
| 2025-07-17 | `b01d71e` | A second, independent stats script, `src/torch/compute_mean_std.py`, is written as a faster PyTorch rewrite. It imports `AIALogTransform` but never applies it before averaging — reproducing the same bug independently. Never wired into `data_single.py`; a parallel confirmation of the blind spot, not a cause. |
| 2026-07-11 | `f4adc6e` / `43d162c` | Fix: `compute_log_mean_and_std()` added, log-transforms before accumulating sum / sum-of-squares; `data_single.py` switched over to it; `AIALogTransform` fields renamed `means`/`stds` → `log_means`/`log_stds` to make the contract explicit. |

## Root cause

The codebase had two normalization philosophies living side by side, and one stats artifact got
shared between them without anything enforcing the contract:

- **TensorFlow / AlexNet path** (`aarp_ml/model/training.py`, older than the ViT pipeline):
  normalizes with a Keras layer *inside* the model (`add_custom_layers`) and never log-transforms
  its input anywhere in the data pipeline. Its stats function, `compute_mean_and_std`, correctly
  computes raw-space mean/std for that design.
- **PyTorch / ViT path** (`aarp_ml/torch/dataset.py`): log-transforms every pixel in
  `AIALogTransform.__call__`, so it needs mean/std of the *log-transformed* values to produce a
  correct z-score.

When z-score normalization was bolted onto `AIALogTransform` in April 2025, the only stats-computing
utility that existed in the repo was the TensorFlow one, built for the other model. It was reused
as a convenient source of `mean`/`std` for the new transform. Nothing about the two toggles that
followed (April 10, June 5) engaged with whether those stats were the right *kind* of stats for what
the log-transforming transform needed — the question was never asked, so the mismatch rode along
silently for a year, surfaced only by an unrelated interpretability investigation.

## Evidence

**The scale mismatch, quantified.** Old (raw-space, `fe8e2d59...`) vs. new (log-space) per-channel
std, and the resulting amplification factor once you divide by std in the normalization step:

| Passband | Old std (raw) | New std (log) | Ratio |
|---|---:|---:|---:|
| 94 Å  |   5.621 | 0.5785 |    9.7× |
| 131 Å |  21.142 | 0.4680 |   45.2× |
| 171 Å | 291.663 | 0.5678 |  513.7× |
| 193 Å | 376.336 | 0.4045 |  930.4× |
| 211 Å | 243.732 | 0.4466 |  545.7× |
| 304 Å | 120.732 | 0.3975 |  303.7× |
| 335 Å |  35.054 | 0.5791 |   60.5× |

Because normalization divides by `std`, and gradients propagate through that division by the chain
rule, this scale difference alone predicts a large jump in the magnitude of any gradient-based
attribution (Integrated Gradients, saliency, etc.) computed on a post-fix model relative to a
pre-fix one — independent of anything the model actually learned.

**What this looked like downstream.** While comparing Integrated Gradients mean-baseline attribution
maps between `glad-shape-197` (pre-fix) and a same-architecture, 7-channel model trained after the
fix (`cv-fold0-…-normfix-l1`), on the **same AARP, same frame, same baseline definition**:

- Attribution magnitude increased by roughly 4–5 orders of magnitude, consistent with the std-ratio
  table above.
- Attribution texture changed from dense/continuous to sparse and aligned with the ViT's 16×16 patch
  grid — a handful of "winner" patches carrying nearly all the attribution mass instead of it
  spreading smoothly. A controlled check ruled out channel count, IG baseline choice, and run-to-run
  nondeterminism as explanations; the change tracks the normalization fix and nothing else that was
  varied.

This does not mean the fixed normalization is somehow worse — it means gradient-based attribution
maps computed on the two normalization regimes are not visually or numerically comparable, and the
smoother pre-fix maps should not be preferred just because they look nicer; they were smoother partly
because zero/near-baseline background pixels were picking up spurious attribution under the
mismatched scale.

## The fix

`f4adc6e` (2026-07-11) adds `compute_log_mean_and_std()` to `aarp_ml/torch/dataset.py`:

```python
def compute_log_mean_and_std(json_path, batch_size=32, num_workers=4):
    """Compute per-channel mean and std of log-transformed pixel values on the training set."""
    dataset = aia_euv(json_path=json_path, subset='training', transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    ...
    for features, _ in tqdm(loader, desc="Computing log-space stats"):
        log_f = torch.log(features.clamp(min=1))
        sum_log    += log_f.sum(dim=[0, 2, 3])
        sum_log_sq += (log_f ** 2).sum(dim=[0, 2, 3])
        ...
    mean = sum_log / count
    std  = torch.sqrt(sum_log_sq / count - mean ** 2)
    return mean, std
```

`AIALogTransform`'s fields were renamed `means`/`stds` → `log_means`/`log_stds`, and its docstring
now states the log-space contract explicitly, so a future caller can't as easily repeat this mistake
by accident. `data_single.py --stats` now calls this function instead of the TensorFlow one.

## Verification

Confirmed the fix produces internally-consistent stats by checking that a flat baseline image built
from `exp(log_mean)` per channel (the geometric mean, i.e. the z-score-zero point) transforms back to
a z-score of ~0 for every channel — see `notebooks/export-aia-attribution-cube-cvfold0-7ch-meanbaseline.ipynb`.
Re-derived interpretability results for models trained under the corrected normalization
(`young-snowball-224`, `cv-fold0-…-normfix-l1`) separately from any pre-fix model, and confirmed via
a same-AARP, same-frame comparison (`notebooks/compare-ig-maps-old-vs-normfix.ipynb`) that the
texture/magnitude change is attributable to normalization and not to a confound (channel count,
baseline choice, checkpoint nondeterminism).

## Open follow-ups

- **`stats.pkl` is not yet re-tracked in DVC.** `stats.pkl.dvc` still points at the pre-fix hash
  (`fe8e2d59c30489c29ec6e44ff8afb493`, committed 2025-06-20); the live file on disk is the corrected
  log-space version but has no committed provenance of its own. Run `dvc add stats.pkl` and commit
  once the current training run using it is done, so future work can pin a real hash the way the old
  file could be pinned, instead of relying on a manually-preserved backup (`stats_raw.pkl`) the way
  this investigation had to.
- **Two independent stats-computation implementations exist**
  (`aarp_ml/model/training.py::compute_mean_and_std` for TensorFlow,
  `src/torch/compute_mean_std.py` as an unused PyTorch rewrite, plus the now-canonical
  `aarp_ml/torch/dataset.py::compute_log_mean_and_std`). Worth deleting or clearly marking the dead
  ones so a future refactor doesn't reuse the wrong one again.
- **Any interpretability/attribution result computed on a pre-fix checkpoint** (`glad-shape-197` and
  anything trained before 2026-07-11) should be treated as reflecting the old, mismatched
  normalization. It is not a fair baseline for comparison against post-fix models without explicitly
  accounting for this — see `notebooks/compare-ig-maps-old-vs-normfix.ipynb` for the same-AARP
  comparison methodology used to make that comparison fair.
- Consider a lightweight sanity check (e.g. asserting `stats.pkl`'s per-channel std falls in a
  plausible log-space range, or a unit test that round-trips a flat baseline through the transform
  and checks it lands near z-score 0) to catch a class-of-bug repeat earlier than a year out.
