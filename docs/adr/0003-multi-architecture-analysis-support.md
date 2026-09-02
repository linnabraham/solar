# ADR-0003: Multi-architecture support pattern for the analysis toolchain

**Status:** Accepted, in progress — 2026-09-02

## Context

Two ViT architectures now need to run through the same analysis toolchain:
`DeepFlare_ViT` (from-scratch, `vit_pytorch`, 512×512 native input) and
torchvision's pretrained `vit_l_16` (224×224 native input, entirely different
internal module tree). Most analysis scripts hardcoded `DeepFlare_ViT`
construction, either via the shared `utils.py::get_model_and_transform()` or
via their own duplicate copy of the same loading logic, with no resize step
anywhere in the image pipeline.

## Decision

**Central dispatch, not duplicate scripts, not a parallel toolchain:**

- `utils.py::get_model_and_transform()` (and `get_data_model()`, which wraps
  it) dispatches on `config.model_type` (`"deepflare_vit"` default vs.
  `"vit_pretrained"`), still returning the same `(model, transform, device)`
  3-tuple — fully backward compatible; the ~14 existing callers needed zero
  changes for the default case.
- A `needs_resize(config)` helper reports whether (and to what size) a script
  must resize inputs before the forward pass. Each script applies this itself
  in its own data pipeline — there's no single central place to do it, since
  how images get built into tensors differs per script.
- Where a script overlays attribution directly on a raw display image
  (`plot_contour_image_grid.py`), `get_attribution_for_image()` resizes down
  for the forward/backward pass then **upsamples the attribution back to
  native resolution** before returning, so spatial alignment with the raw
  image is preserved. Where a script only reduces attribution to scalar
  summaries (the IG distribution/timeseries scripts), no upsample-back is
  needed — the two resolutions are never compared against each other.
- Every patched script gets a `--model-type {vit, vit-pretrained}` CLI flag,
  matching the convention `evaluate_aarp.py`/`compare_models.py`/`test_curve.py`
  already used.
- `model_randomization_test.py` needed a genuinely **separate** cascade
  definition (`get_cascade_groups(model, model_type=...)`), not just a resize
  fix — the two architectures have completely different module trees
  (`mlp_head`/`transformer.layers`/`to_patch_embedding` vs.
  `heads.head`/`encoder.layers`/`conv_proj`) and different depths (4 blocks vs.
  24), so the pretrained cascade has 27 levels instead of 7.

## Consequences

Patched and smoke-tested this session (each against `epoch_01.pth` before
committing): `predictions_analyze.py`, `class_wise_distribution.py`,
`plot_contour_image_grid.py`, `prediction_timeseries.py`, `intensity_boxplot.py`,
`kshap.py`, `shap_stability_test.py`, `shap_faithfulness_test.py`,
`model_randomization_test.py`, `test.py`.

Two real pre-existing bugs surfaced and were fixed along the way, not
introduced by this pattern:
- `get_model_and_transform()` never called `model.to(device)` — every prior
  caller either worked around it themselves or (in `predictions_analyze.py`)
  relied on an unrelated duplicate reload block that happened to do it.
- `intensity_boxplot.py` had no `--stride`, unlike its sibling
  `class_wise_distribution.py` — caught the hard way, via an accidental
  38-minute unstrided smoke test on a 451-frame AARP.

**Going forward:** any new analysis script should route model construction
through `get_data_model()`/`get_model_and_transform()`, call `needs_resize(config)`,
and add `--model-type` to its CLI — not hand-roll a new DeepFlare_ViT-only
loading path.
