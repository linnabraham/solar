# AARP Subset Pipeline

::: src.aarp_subset_pipeline
    handler: python
    options:
      show_source: true
      docstring_style: google

## Related Stage

**Pipeline Stage**: [subset-data](../stages/subset-data.md)

This script is executed by the `subset-data` DVC stage to download, extract, and preprocess a representative subset of AARPs from the full dataset, then generate `solar_dataset_subset.json`.
