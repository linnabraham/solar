# `subset-contour-grid-movies`

**Implementation**: [View code documentation](../code/create_contour_grid_movie.md)


**Command**
```bash
python -m src.torch.vit.create_contour_grid_movie --json-path solar_dataset_subset.json --output-dir plots/subset/movies/contour_grid --splits training validation test
```

**Dependencies**
- `src/torch/vit/create_contour_grid_movie.py`
- `src/torch/vit/plot_contour_image_grid.py`
- `src/torch/vit/class_wise_distribution.py`
- `src/torch/vit/ig.py`
- `src/torch/vit/utils.py`
- `solar_dataset_subset.json`
- `outputs/glad-shape-197/trained_model.pth`
- `stats.pkl`

**Outputs**
- `plots/subset/movies/contour_grid` _(not cached)_

## Notes

_Add your notes about this stage here._
