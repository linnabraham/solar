# Test / Confusion Matrix Validation

::: src.torch.vit.test
    handler: python
    options:
      show_source: true
      docstring_style: google

## Related Stages

- [confusion-matrix-val](../stages/confusion-matrix-val.md) — full dataset (validation split)
- [subset-confusion-matrix](../stages/subset-confusion-matrix.md) — 7-AARP subset (test split)

Runs model inference on a split and saves a confusion matrix PNG.
