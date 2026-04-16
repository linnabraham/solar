# `attribution_analyzis`

**Implementation**: [View code documentation](../code/attributions_analyze.md)


**Command**
```bash
python -m src.torch.vit.attributions_analyze
```

**Dependencies**
- `src/torch/vit/attributions_analyze.py`
- `src/torch/vit/class_wise_distribution.py`
- `src/torch/vit/ig.py`
- `src/torch/vit/predictions_analyze.py`
- `src/torch/vit/train.py`
- `src/torch/vit/utils.py`
- `solar_dataset.json`
- `outputs/glad-shape-197/trained_model.pth`
- `stats.pkl`

**Outputs**
- `plots/predictions/1275/attributions_percentiles_99.png`
- `plots/predictions/1449/attributions_percentiles_99.png`
- `plots/predictions/185/attributions_percentiles_99.png`
- `plots/predictions/2026/attributions_percentiles_99.png`
- `plots/predictions/3153/attributions_percentiles_99.png`
- `plots/predictions/3229/attributions_percentiles_99.png`
- `plots/predictions/3364/attributions_percentiles_99.png`
- `plots/predictions/3563/attributions_percentiles_99.png`
- `plots/predictions/377/attributions_percentiles_99.png`
- `plots/predictions/401/attributions_percentiles_99.png`
- `plots/predictions/4296/attributions_percentiles_99.png`
- `plots/predictions/4920/attributions_percentiles_99.png`
- `plots/predictions/5894/attributions_percentiles_99.png`
- `plots/predictions/833/attributions_percentiles_99.png`
- `plots/predictions/1275/filtered_image_intensities.png`
- `plots/predictions/1449/filtered_image_intensities.png`
- `plots/predictions/185/filtered_image_intensities.png`
- `plots/predictions/2026/filtered_image_intensities.png`
- `plots/predictions/3153/filtered_image_intensities.png`
- `plots/predictions/3229/filtered_image_intensities.png`
- `plots/predictions/3364/filtered_image_intensities.png`
- `plots/predictions/3563/filtered_image_intensities.png`
- `plots/predictions/377/filtered_image_intensities.png`
- `plots/predictions/401/filtered_image_intensities.png`
- `plots/predictions/4296/filtered_image_intensities.png`
- `plots/predictions/4920/filtered_image_intensities.png`
- `plots/predictions/5894/filtered_image_intensities.png`
- `plots/predictions/833/filtered_image_intensities.png`

## Output

### Attributions Percentiles
![attributions_percentiles_99](../assets/plots/attributions_percentiles_99.png)

### Filtered Intensities
![filtered_image_intensities](../assets/plots/filtered_image_intensities.png)


## Notes

_Add your notes about this stage here._
