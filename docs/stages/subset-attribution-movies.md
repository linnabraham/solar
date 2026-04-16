# `subset-attribution-movies`

**Implementation**: [View code documentation](../code/create_movies.md)


**Command**
```bash
python -m src.torch.vit.create_movies --json-path solar_dataset_subset.json --output-dir plots/subset/movies/attribution --splits training validation test
```

**Dependencies**
- `src/torch/vit/create_movies.py`
- `src/torch/vit/class_wise_distribution.py`
- `src/torch/vit/ig.py`
- `src/torch/vit/utils.py`
- `solar_dataset_subset.json`
- `outputs/glad-shape-197/trained_model.pth`
- `stats.pkl`

**Outputs**
- `plots/subset/movies/attribution` _(not cached)_

## Notes

_Add your notes about this stage here._
