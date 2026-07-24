# Session summary: IG attribution exports, mean-baseline investigation, normalization postmortem

**Branch:** `feat/two-channel-training`. **Session span:** roughly 2026-07-03 through 2026-07-24.

This covers a long session that started as a simple export task and turned into a real investigation
of a normalization bug affecting model interpretability. Read this before picking the thread back up.

## What was asked for, in order

1. Export an AIA data cube + IG attribution cube for a collaborator (zero baseline).
2. Same, but with a mean baseline instead of zero.
3. Same, but with `multiply_by_inputs=False` as a second variant.
4. Same, but for a new 2-channel (94, 131 Å) model.
5. Explain why the mean-baseline maps looked "patchy" compared to earlier "continuous" ones.
6. Discuss how to interpret this for the paper being written.
7. Check the same thing on the newest CV experiments (discovered mid-session — see below).
8. Point out that the comparison used different AARPs for different models — is that fair?
9. Turn the ad hoc comparison script into a real, rerunnable notebook.
10. Investigate, via git archaeology, how the normalization bug was introduced.
11. Write that up as `docs/postmortems/normalization-bug.md` (done).
12. This file.

## Notebooks produced (all in `notebooks/`)

| Notebook | Model | AARP | Baseline | Notes |
|---|---|---|---|---|
| `export-aia-attribution-cube.ipynb` | `glad-shape-197` (7ch, old norm) | 377 | zero | Reuses precomputed `data/intermediate-outs/subset_attributions_pos.pt` — no GPU/recompute. |
| `export-aia-attribution-cube-meanbaseline.ipynb` | `glad-shape-197` | 3563 (was 377, briefly 2026) | mean (flat, `stats_raw.pkl` arithmetic mean) | Recomputes IG. Has `MULTIPLY_BY_INPUTS` toggle (both True/False variants exported for AARP 377). AARP later switched to 3563 for a fair cross-model comparison — see below. Uses an inline "legacy" `AIALogTransform` + a DVC-cache copy of the old `stats.pkl`, because the live `stats.pkl` was mid-overwrite by a concurrent training run for most of the session. |
| `export-aia-attribution-cube-2ch-meanbaseline.ipynb` | `young-snowball-224` (2ch, new norm) | 377 | mean (flat, geometric mean via `AIALogTransform` directly — this model postdates the fix) | First model checked after the normalization fix; first sighting of "blocky" attribution texture. |
| `export-aia-attribution-cube-cvfold0-7ch-meanbaseline.ipynb` | `cv-fold0-…-normfix-l1-allepochs`, checkpoint `epoch_02.pth` (7ch, new norm + L1) | 3563 (config now also has 2026 drafted but not run) | mean (flat, geometric mean) | Deliberately uses `epoch_02`, **not** the best-val-loss checkpoint — per a parallel investigation (different session) into whether best val loss is the right checkpoint-selection criterion. This run was still training live during the session (GPU-sharing safety pattern below applies). |
| `compare-ig-maps-old-vs-normfix.ipynb` | loads exports from the above, no GPU | 3563 for both panels (default); `PRIOR_PANELS` var holds the earlier 377-vs-3563 mismatched-AARP version for reference | mean | **The fair comparison.** Confirms the smooth-vs-blocky texture difference holds even on a literal same-AARP, same-frame, same-channel-count comparison — so it isolates to normalization, not channel count or which AARP was used. |

All exports (`exports/*.npz`) are self-describing: each carries a `metadata_json` array field with
checkpoint path, stats source + md5, baseline definition, stride, `multiply_by_inputs`, split used,
and (where relevant) the model's prediction confidence on the plotted frame. Trust that metadata over
filenames.

## The central finding

Attribution maps computed on models trained **before** the log-normalization fix (`glad-shape-197`)
look smooth/continuous. Maps from models trained **after** the fix (`young-snowball-224`,
`cv-fold0-…-normfix-l1`) look sparse and blocky, aligned with the ViT's 16×16 patch grid, and are
roughly 4–5 orders of magnitude larger in raw value.

Ruled out as explanations: temporal stride, IG run-to-run nondeterminism (verified bit-identical
across repeated/separate-process runs), colormap/rendering differences, and channel count (a
same-channel-count-as-glad-shape-197 model, `cv-fold0` with 7 channels, is *also* blocky). Confirmed
cause: the normalization fix itself — the corrected `stats.pkl` has 9.7×–930× smaller per-channel
std than the old one (varies a lot by passband), and since normalization divides by std, gradients
get amplified by roughly that factor through the chain rule. Larger/steeper gradients are consistent
with IG's 100-step Riemann approximation collapsing onto a few "winner" patches instead of spreading
smoothly. Full root-cause git archaeology (which commit introduced the mismatch, why, over what
timeline) is in `docs/postmortems/normalization-bug.md` — don't duplicate that investigation, just
read it.

**Implication for the paper:** blocky mean-baseline maps are the expected, correct-pipeline texture,
not a defect to fix by preferring the old (buggy) normalization because its output looks nicer.
Attribution comparisons across the normalization boundary need same-AARP, same-frame methodology
(see below) to be meaningful; comparisons within one normalization regime are fine as-is.

## Fair-AARP-comparison methodology (for reuse)

A model's own test split isn't automatically comparable to another model's test split if they were
trained on different data splits (original `solar_dataset.json` vs. `cv_folds/fold_0.json` here).
Established two validity levels:

- **Strict:** AARP in the *test* split of both models. Only candidate found so far: **AARP 2026**
  (non-flare, 132 timesteps). Config drafted in the relevant notebooks but **not yet run/exported**
  for either model — a real gap if the strict version is ever needed for the paper.
- **Relaxed (used):** AARP not in the *training* split of either model — validation is acceptable
  because no gradient updates happen on it, only checkpoint selection. This is much less restrictive
  and is what unlocked **AARP 3563** (flare-positive, 231 timesteps): validation for `glad-shape-197`,
  test for `cv-fold0`. Also flagged as available under this criterion but unused: AARP 833
  (val for `glad-shape-197` / test for `cv-fold0`) and AARP 1449 (test for `glad-shape-197` / val for
  `cv-fold0`).

To extend this to a flare-positive *strict* comparison, a 7-channel normfix model would need to be
trained on fold_2 (AARP 377 is fair there) or fold_3 (AARP 1449 or 4920) — none exists yet, only
fold_0.

## GPU-sharing pattern (reuse this for future recomputes)

Multiple times this session, a live training job (30GB+) was competing for the GPU. Established
pattern, present in every recompute notebook: read `torch.cuda.mem_get_info()` live, cap this
process via `torch.cuda.set_per_process_memory_fraction()` leaving 2–3GB headroom, use `ib_size=1`
for `do_ig`, wrap the per-timestep IG call in `try/except RuntimeError` so an OOM degrades to a
shorter export instead of crashing, and print a timing-based ETA after the first few iterations. CUDA
memory is per-process — worst case this notebook's own process OOMs, it doesn't evict or crash the
other job. Verified live training jobs were undisturbed (GPU memory unchanged, same PID still running)
after every recompute this session.

## Loose ends / good next steps

- **AARP 2026 strict comparison** not yet built — would need one recompute pass per model
  (`glad-shape-197` and `cv-fold0`), same pattern as everything else here.
- **Checkpoint selection for `cv-fold0-…-normfix-l1-allepochs`** is an open question being
  investigated in a different session (best val loss ≠ best test performance, apparently). Once that
  session lands on a criterion, re-point `export-aia-attribution-cube-cvfold0-7ch-meanbaseline.ipynb`
  and `compare-ig-maps-old-vs-normfix.ipynb` at whichever checkpoint that investigation settles on —
  `epoch_02` was a deliberate placeholder, not a conclusion. Check whether that training run has
  finished (it was at epoch 5 of 20 when last checked in this session).
- **`stats.pkl` still isn't re-tracked in DVC** (flagged in the postmortem's follow-ups). The live
  file is the correct post-fix version but has no committed hash of its own; `stats_raw.pkl` at repo
  root is a manually-preserved copy of the old pre-fix stats, useful for any future comparison against
  pre-fix checkpoints — don't delete it.
- **Two dead/duplicate stats-computation implementations** still exist in the codebase
  (`aarp_ml/model/training.py::compute_mean_and_std` for the old TF path,
  `src/torch/compute_mean_std.py`, an unused PyTorch rewrite that independently reproduced the same
  bug). Worth deleting or clearly marking, per the postmortem.
- **Unrecognized files in `exports/` and `outputs/`** (`aarp_377_aia_attribution_cube_balmy_energy_223.npz`,
  `aarp_3563_aia_attribution_cube_cv_fold0.npz`, `outputs/balmy-energy-223/`) — not created in this
  session, most likely from the user's other concurrent session. Left untouched; flagged in
  conversation at the time, not investigated further.
- **Diagnostic memo Artifact** (zero-vs-mean baseline A/B test, then updated with the 3-way
  normalization finding): `https://claude.ai/code/artifact/7976c84a-7dfd-4ffc-b3db-ef2014523579` —
  covers the same material as `compare-ig-maps-old-vs-normfix.ipynb`'s first (mismatched-AARP) pass;
  not updated with the later fair-AARP-3563 result. Could be refreshed if it's going to be shared
  externally.

## Key files to know about

- `docs/postmortems/normalization-bug.md` — full root-cause writeup, timeline, evidence, fix, open
  follow-ups. The source of truth for "why did this happen."
- `notebooks/compare-ig-maps-old-vs-normfix.ipynb` — the fair comparison, rerunnable, no GPU needed.
- `stats_raw.pkl` (repo root) — manually preserved pre-fix stats, needed for any pre-fix-model work.
- `cv_folds/fold_0.json`, `cv_folds/fold_0_stats.pkl` — fold-0 data split and its (correct, log-space)
  normalization stats, used by the `cv-fold0` notebooks.
