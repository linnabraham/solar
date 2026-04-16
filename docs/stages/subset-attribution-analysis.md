# `subset-attribution-analysis`

**Implementation**: [View code documentation](../code/attributions_analyze.md)


**Command**
```bash
python -m src.torch.vit.attributions_analyze --json-path solar_dataset_subset.json --output-dir plots/subset/predictions
```

**Dependencies**
- `src/torch/vit/attributions_analyze.py`
- `src/torch/vit/class_wise_distribution.py`
- `src/torch/vit/ig.py`
- `src/torch/vit/predictions_analyze.py`
- `src/torch/vit/train.py`
- `src/torch/vit/utils.py`
- `solar_dataset_subset.json`
- `outputs/glad-shape-197/trained_model.pth`
- `stats.pkl`

**Outputs**
- `plots/subset/predictions/377/attributions_percentiles_99.png`
- `plots/subset/predictions/377/filtered_image_intensities.png`
- `plots/subset/predictions/1449/attributions_percentiles_99.png`
- `plots/subset/predictions/1449/filtered_image_intensities.png`
- `plots/subset/predictions/4296/attributions_percentiles_99.png`
- `plots/subset/predictions/4296/filtered_image_intensities.png`
- `plots/subset/predictions/3563/attributions_percentiles_99.png`
- `plots/subset/predictions/3563/filtered_image_intensities.png`
- `plots/subset/predictions/185/attributions_percentiles_99.png`
- `plots/subset/predictions/185/filtered_image_intensities.png`
- `plots/subset/predictions/1807/attributions_percentiles_99.png`
- `plots/subset/predictions/1807/filtered_image_intensities.png`
- `plots/subset/predictions/903/attributions_percentiles_99.png`
- `plots/subset/predictions/903/filtered_image_intensities.png`

## Notes

_Add your notes about this stage here._
