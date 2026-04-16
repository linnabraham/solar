# `subset-plot-class-dist`

**Implementation**: [View code documentation](../code/plot_class_wise_distribution.md)


**Command**
```bash
python -m src.torch.vit.plot_class_wise_distribution --json-path solar_dataset_subset.json --attributions-neg data/intermediate-outs/subset_attributions_neg.pt --attributions-pos data/intermediate-outs/subset_attributions_pos.pt --output-dir plots/subset/distribution
```

**Dependencies**
- `src/torch/vit/plot_class_wise_distribution.py`
- `src/torch/vit/class_wise_distribution.py`
- `src/torch/vit/ig.py`
- `src/torch/vit/utils.py`
- `solar_dataset_subset.json`
- `data/intermediate-outs/subset_attributions_neg.pt`
- `data/intermediate-outs/subset_attributions_pos.pt`

**Outputs**
- `plots/subset/distribution`

## Notes

_Add your notes about this stage here._
