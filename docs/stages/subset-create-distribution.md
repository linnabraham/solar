# `subset-create-distribution`

**Implementation**: [View code documentation](../code/class_wise_distribution.md)


**Command**
```bash
python -m src.torch.vit.class_wise_distribution --json-path solar_dataset_subset.json --output-neg data/intermediate-outs/subset_attributions_neg.pt --output-pos data/intermediate-outs/subset_attributions_pos.pt
```

**Dependencies**
- `src/torch/vit/class_wise_distribution.py`
- `src/torch/vit/ig.py`
- `src/torch/vit/utils.py`
- `solar_dataset_subset.json`
- `outputs/glad-shape-197/trained_model.pth`
- `stats.pkl`

**Outputs**
- `data/intermediate-outs/subset_attributions_neg.pt` _(not cached)_
- `data/intermediate-outs/subset_attributions_pos.pt` _(not cached)_

## Notes

_Add your notes about this stage here._
