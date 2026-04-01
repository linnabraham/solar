# Documentation Plan: Scripts & Dependencies for Plots and Outputs

**Status**: Planning Phase  
**Last Updated**: April 1, 2026  
**Owner**: User

## Overview

This plan documents a phased approach to adding docstrings to plot/output generation scripts defined in `dvc.yaml`. The goal is to capture function purposes, dependencies, and workflows while maintaining full user control over the documentation process.

## Discovery

### Plot/Output Stages in DVC Pipeline

Based on `dvc.yaml`, 6 stages generate plots and movie outputs:

| Stage | Module | Dependencies Count | Complexity | Purpose |
|-------|--------|-------------------|-----------|---------|
| `xgb-feat-importance` | `src/torch/xgb_feat_importance.py` | 2 | Low | Visualize XGBoost feature importance (gain vs. weight) |
| `prediction-plots` | `src/torch/vit/predictions_analyze.py` | 5 | Medium | Generate prediction overlays on GOES images |
| `plot-class-dist` | `src/torch/vit/plot_class_wise_distribution.py` | 7 | Medium | Distribution plots from attribution data |
| `plot_contour_image_grid` | `src/torch/vit/plot_contour_image_grid.py` | 7 | Medium-High | Contour grid visualization of attributions |
| `create_movie` | `src/torch/vit/create_movies.py` | 7 | High | Generate MP4 movie from sequence data |
| `attribution_analyzis` | `src/torch/vit/attributions_analyze.py` | 7 | High | Multi-output attribution analysis and filtering |

### Shared Dependencies

Multiple scripts depend on foundational modules:
- `src/torch/vit/ig.py` — Integrated Gradients wrapper & utilities
- `src/torch/vit/utils.py` — Shared data loading & visualization utilities
- `src/torch/vit/class_wise_distribution.py` — Attribution distribution calculations

## Approach

### Docstring Style
**Google-style docstrings** for clarity and auto-documentation support:

```python
def function_name(param1, param2):
    """One-line summary of function purpose.
    
    Longer description explaining the approach, key side effects,
    or algorithm if non-obvious.
    
    Args:
        param1 (type): Description and context.
        param2 (type): Description and context.
        
    Returns:
        type: Description of return value.
        
    Raises:
        ExceptionType: Condition when raised.
    """
```

### Control & Review Checkpoints

**Workflow per script/module**:

1. **Plan Review** — Show scope, function list, and template before any changes
2. **Draft** — Write docstrings in session notes (not touching files yet)
3. **Review** — User reviews, approves, requests rewrites, or skips
4. **Save** — Once approved, update the Python file

**User maintains full control**: approve before each save, skip phases, or request alternative approaches.

### Execution Phases

#### Phase 0: Code Refactoring Analysis & Improvements
- **CRITICAL**: Do this BEFORE documentation—cleaner code makes docs way easier
- **Targets**: All 6 plot scripts + 3 shared dependencies
- **Outcome**: Identified refactorings with rationale, user approval, then apply changes
- **Effort**: ~2-3 hours (analysis + review cycles)
- **Details**: See "[Phase 0: Refactoring Analysis](#phase-0-code-refactoring-analysis-detailed)" below

#### Phase 1: Pattern Establishment
- **Target**: `src/torch/xgb_feat_importance.py`
- **Rationale**: Simplest script (2 functions, clear workflow) + now with improved code
- **Outcome**: Establish docstring template and review workflow
- **Effort**: ~15-30 min (draft + review cycle)

#### Phase 2: Shared Dependencies (Parallel)
- **Targets**: `ig.py`, `utils.py`, `class_wise_distribution.py`
- **Rationale**: Multiple scripts depend on these; document once
- **Outcome**: Foundation modules are well-documented before complex scripts
- **Effort**: ~1-2 hours (3 modules × review cycles)

#### Phase 3: Multi-Dependency Scripts (Sequential)
- **Order of complexity**: 
  1. `prediction_plots` (5 deps, clearer purpose)
  2. `plot-class-dist` (7 deps, medium complexity)
  3. `plot_contour_image_grid` (7 deps, visualization-heavy)
  4. `create_movie` (7 deps, high complexity)
  5. `attribution_analyzis` (7 deps, most complex, multiple outputs)
- **Outcome**: Full documentation of all plot generation pipelines
- **Effort**: ~3-4 hours total (varies by stage)

## Decisions Made

1. **Documentation format**: Docstrings in Python files (not external markdown)
2. **Scope priority**: Start with code refactoring FIRST, then document (cleaner code = easier docs)
3. **Control mechanism**: Hybrid—propose refactorings, get approval before any file changes
4. **Docstring style**: Google (readable, auto-doc-friendly)
5. **Parallelization**: Phase 0 & 1 can overlap; Phase 2 ideally before Phase 3

## What's Included vs. Excluded

### Included (Updated Scope)
- **Phase 0**: Code refactoring analysis & improvements
- Function-level docstrings with clear purpose, parameters, returns
- Extract magic numbers/strings to named constants
- Consolidate duplicated visualization code
- Edge cases and error conditions (Raises section)

### Excluded (Out of Scope)
- External markdown documentation (existing `docs/stages/` separate from code)
- Extensive inline comments within function bodies (docstrings only)
- Unit test writing
- Major algorithm rewrites or logic changes

## Milestones & Checkpoints

| Phase | Checkpoint | Status |
|-------|-----------|--------|
| Phase 0 | Refactoring analysis & proposals ready | In progress |
| Phase 1 | ✓ Template & review workflow established | Pending refactoring |
| Phase 2 | Foundation modules documented | Not started |
| Phase 3 | All plot scripts documented | Not started |
| **Complete** | All improvements & docstrings merged to main | Not started |

## Next Steps

1. **Phase 0**: Review refactoring proposals below, approve/reject/modify
2. **Phase 0 Implementation**: Apply approved changes to all 9 modules
3. **Phase 1 Draft**: Resume with quality baseline established

---

# Phase 0: Code Refactoring Analysis (Detailed)

## Summary of Refactoring Opportunities

Analyzed all 9 modules (6 plot scripts + 3 shared dependencies). Found common patterns ripe for refactoring:
- **Hardcoded magic numbers & dictionaries** (percentiles, ranges, contour levels, figure sizes)
- **Duplicated visualization code** (GOES plotting functions appear twice)
- **Inconsistent parameter passing** (configs embedded in `__main__` blocks)
- **Missing error handling** (assertions instead of proper validation)
- **Implicit module dependencies** (wavelength indices, hardcoded feature names)

---

## Refactoring Proposals by Module

### 1. `src/torch/xgb_feat_importance.py`

**Current State**
```python
figsize=(10, 12)  # magic number
importance_type='weight'  # hardcoded default
```

**Issues**
- Hardcoded figure dimensions and defaults
- No docstrings
- Model loading path hardcoded in `__main__`

**Proposed Changes**
| # | Change | Rationale |
|---|--------|-----------|
| 1 | Extract `FIGSIZE_DEFAULT = (10, 12)` to module constant | Reusable, easier to test |
| 2 | Extract `IMPORTANCE_TYPES = ['weight', 'gain', 'cover']` + validate | Type safety, clarity |
| 3 | Add Google-style docstring to `plot_xgb_feature_importance()` | Documentation |
| 4 | Add comment block to `__main__` explaining the pipeline | Clarity without code changes |

**Effort**: ~10 minutes

---

### 2. `src/torch/vit/predictions_analyze.py` ⚠️ **HIGHEST PRIORITY**

**Current State**
- 200+ lines of visualization code
- **Two near-identical GOES plotting functions**: `plot_goes()` and `plot_custom_goes_with_aarp_sampling()` with duplicated logic
- Hardcoded figure parameters scattered throughout
- Multiple visualization function variants with overlapping code
- No docstrings

**Issues**
- **Code duplication**: `plot_goes()` → `plot_custom_goes_with_aarp_sampling()` copy-paste with minor variations
- **Magic numbers**: `figsize=(10,6)`, `dpi=150`, `rotation=45`, line widths, text offsets all scattered
- **Inconsistent parameter handling**: Some functions take `figsize`, others don't
- **Poor separation of concerns**: Data loading, visualization, and prediction mixed together

**Proposed Changes**
| # | Change | Rationale |
|---|--------|-----------|
| 1 | Create `GOESPlotConfig` dataclass for shared parameters | DRY principle, eliminates magic numbers |
| 2 | Refactor `plot_goes()` to be the single source of truth, extract into utility | Eliminate duplication |
| 3 | Simplify `plot_custom_goes_with_aarp_sampling()` to call `plot_goes()` + overlay logic | Composition over duplication |
| 4 | Consolidate `vizualize_goes_ts_predictions()` and `viz_predictions_2()` into one flexible function | Reduce variants |
| 5 | Add docstrings to all 4 visualization functions | Documentation |
| 6 | Extract `get_aarp_seq_dataset()` to a utility module (or mark it for shared use) | Reusable helper |

**Code Example - GOESPlotConfig**
```python
from dataclasses import dataclass

@dataclass
class GOESPlotConfig:
    figsize: tuple = (10, 6)
    dpi: int = 150
    columns: list = None  # ["xrsa", "xrsb"]
    xlimits: tuple = None  # (start, end)
    
    def __post_init__(self):
        if self.columns is None:
            self.columns = ["xrsa", "xrsb"]
```

**Effort**: ~45 minutes (main refactoring) + docstrings

---

### 3. `src/torch/vit/plot_class_wise_distribution.py`

**Current State**
```python
default_percentiles = {94: [80, 90, 99, 99.9], 131: [...], ...}
default_x_ranges = {94: (0, 8), 131: (0, 6), ...}
```

**Issues**
- Two large hardcoded dictionaries as function defaults—difficult to maintain and extend
- No docstrings
- Hardcoded magic strings like "no-latex" for plots

**Proposed Changes**
| # | Change | Rationale |
|---|--------|-----------|
| 1 | Create `PASSBAND_CONFIG` constant dict at module level | Consolidate config, easier to modify |
| 2 | Move plot style setup (`plt.style.use`, `plt.rcParams`) to a `configure_plot_style()` function | Reusable, testable |
| 3 | Add docstring to `plot_distribution_for_passband()` and `plot_all_passbands()` | Documentation |
| 4 | Add parameter validation for passband & percentile_levels | Error handling |

**Code Example**
```python
# At module level
PASSBAND_CONFIG = {
    94: {'percentiles': [80, 90, 99, 99.9], 'x_range': (0, 8)},
    131: {'percentiles': [80, 90, 99, 99.9], 'x_range': (0, 6)},
    # ... rest
}

def configure_plot_style():
    """Apply consistent plot styling for distribution plots."""
    plt.style.use(['no-latex'])
    plt.rcParams.update({'font.size': 24})
    sns.set_theme()
```

**Effort**: ~20 minutes

---

### 4. `src/torch/vit/plot_contour_image_grid.py`

**Current State**
```python
contour_config={
    94: {"num_levels": 35, "color": "red", ...},
    131: ...
}
```

**Issues**
- Hardcoded contour configuration in `__main__` block—not reusable
- Uses `assert` instead of proper validation
- No docstrings

**Proposed Changes**
| # | Change | Rationale |
|---|--------|-----------|
| 1 | Replace `assert` with proper `ValueError` checks | Better error messages, more robust |
| 2 | Extract contour config to module-level constant `CONTOUR_CONFIG` | Reusability & maintainability |
| 3 | Add docstrings to `plot_aia_image_grid()` | Documentation (already has good structure) |
| 4 | Add type hints for saliency and contour_config parameters | Type clarity |

**Code Example**
```python
CONTOUR_CONFIG = {
    94: {"num_levels": 35, "color": "red", "line_width": 1.5, "use_last_n": 2},
    131: {"num_levels": 10, "color": "red", "line_width": 1.5, "use_last_n": 2},
    # ... rest
}

def plot_aia_image_grid(images, passbands, cols=4, ...):
    """Plot grid of AIA images with optional contours.
    ...
    Raises:
        ValueError: If images and passbands length mismatch.
    """
    if len(images) != len(passbands):
        raise ValueError(f"Mismatch: {len(images)} images vs {len(passbands)} passbands")
```

**Effort**: ~15 minutes

---

### 5. `src/torch/vit/create_movies.py`

**Current State**
- Relatively clean, but hardcoded parameters in `__main__`
- No docstrings
- Hardcoded model path and dataset filenames

**Issues**
- Hardcoded paths: `"solar_dataset.json"`, `"stats.pkl"`, model paths
- Magic numbers: `fps=5`, `interval=50`
- No reusable config object

**Proposed Changes**
| # | Change | Rationale |
|---|--------|-----------|
| 1 | Add docstring to `make_attribution_movie()` | Documentation |
| 2 | Extract `MOVIE_CONFIG` defaults (fps, interval, contour_levels) | Easier parameter tuning |
| 3 | Extract hardcoded paths to config or CLI args (defer to Phase 2 if CLI refactor needed) | Reusability |

**Code Example**
```python
MOVIE_CONFIG = {
    'fps': 5,
    'interval': 50,  # milliseconds
    'contour_levels': 5,
    'vmax_percentile': 99.9,
}
```

**Effort**: ~10 minutes

---

### 6. `src/torch/vit/attributions_analyze.py`

**Current State**
- Hardcoded paths and batch size throughout
- Repetitive model loading code (same as other scripts)
- No docstrings
- Inconsistent error handling

**Issues**
- **Hardcoded paths**: `"solar_dataset.json"`, model paths, output directories
- **Model loading duplication**: Same checkpoint loading pattern in `attributions_analyze.py`, `create_movies.py`, `predictions_analyze.py` (extract to utility)
- **Magic numbers**: `percentile_level=99`, `num_levels=5`, batch sizes
- **No docstrings**

**Proposed Changes**
| # | Change | Rationale |
|---|--------|-----------|
| 1 | Extract `ATTRIBUTION_CONFIG` constant (percentile_level, num_levels, etc.) | DRY |
| 2 | Move model loading logic to utility function (see `ig.py` & `utils.py` improvements) | Eliminate duplication across 3 files |
| 3 | Add docstring to `create_plots()` | Documentation |
| 4 | Add docstring to `main()` outlining pipeline | Documentation |
| 5 | Extract hardcoded paths to function parameters or config object | Testability |

**Code Example**
```python
ATTRIBUTION_CONFIG = {
    'percentile_level': 99,
    'num_levels': 5,
    'channel': 1,  # passband 131
    'filter_threshold': 'max_contour',  # uses topmost contour level
}
```

**Effort**: ~20 minutes + shared utility refactoring

---

### 7. `src/torch/vit/ig.py` (Shared Dependency)

**Current State**
- Core class `single_aarp` is good
- Multiple utility functions lack docstrings
- Some functions have implicit dependencies on dataset structure

**Issues**
- No docstrings on `do_ig()`, `make_predictions()`, `run_ig()`, `ig_on_aarp_seq()`
- `do_ig()` function signature unclear (feature and baseline positional args, but baseline is called differently in different places)
- `__main__` block is untested code and should be removed or moved

**Proposed Changes**
| # | Change | Rationale |
|---|--------|-----------|
| 1 | Add comprehensive docstrings to all functions | Documentation |
| 2 | Clarify and standardize `do_ig()` baseline parameter (currently inconsistent usage) | Consistency |
| 3 | Remove or test `__main__` block | Code cleanliness |
| 4 | Add type hints for torch tensors | Type clarity |

**Effort**: ~15 minutes

---

### 8. `src/torch/vit/class_wise_distribution.py` (Shared Dependency)

**Current State**
- Generally well-structured
- Hardcoded wavelength index `all_wavelengths.index(passband)` 
- `__main__` block is incomplete/untested

**Issues**
- Implicit assumption that passband is in `all_wavelengths` (can fail silently)
- No docstrings on public functions
- `__main__` is skeleton code

**Proposed Changes**
| # | Change | Rationale |
|---|--------|-----------|
| 1 | Add docstring to `run_pred_and_ig()`, `get_intensities_using_attributions()`, `plot_intensity_distribution()` | Documentation |
| 2 | Add validation for passband → check if in `all_wavelengths` before indexing | Error handling |
| 3 | Remove or complete `__main__` block | Code cleanliness |
| 4 | Add type hints | Type clarity |

**Effort**: ~12 minutes

---

### 9. `src/torch/vit/utils.py` (Shared Dependency)

**Current State**
- Already well-documented in some places
- Some repetitive parameter handling
- Long utility function

**Issues**
- Some docstrings are inline; not all functions have formal docstrings
- `save_multi_channel_tensor_as_figure()` is very long and does many things (read 150+ lines but truncated)
- `get_model_and_transform()` and `get_data_model()` have some duplication

**Proposed Changes**
| # | Change | Rationale |
|---|--------|-----------|
| 1 | Formalize all function docstrings (use Google style consistently) | Documentation consistency |
| 2 | Consider splitting `save_multi_channel_tensor_as_figure()` into smaller functions if > 200 lines | Code clarity |
| 3 | Extract common model loading logic into a dedicated function | DRY |
| 4 | Add type hints to all functions | Type clarity |

**Effort**: ~20 minutes

---

## Refactoring Summary Table

| Module | # Issues | Complexity | Estimated Effort |
|--------|----------|-----------|-----------------|
| `xgb_feat_importance.py` | 3 | Low | 10 min |
| `predictions_analyze.py` | 5 | **Critical** | 45 min |
| `plot_class_wise_distribution.py` | 3 | Medium | 20 min |
| `plot_contour_image_grid.py` | 3 | Medium | 15 min |
| `create_movies.py` | 2 | Low | 10 min |
| `attributions_analyze.py` | 4 | Medium | 20 min |
| `ig.py` | 2 | Low | 15 min |
| `class_wise_distribution.py` | 2 | Low | 12 min |
| `utils.py` | 2 | Medium | 20 min |
| **TOTAL** | **26** | **Medium** | **~2.5 hours** |

---

## Implementation Strategy

**Order of Implementation** (respects dependencies):
1. **ig.py** & **class_wise_distribution.py** & **utils.py** — Core utilities (they don't depend on plot scripts)
2. **xgb_feat_importance.py** — Simplest plot script
3. **create_movies.py** — Short, independent
4. **plot_contour_image_grid.py** — Uses shared utilities
5. **plot_class_wise_distribution.py** — Uses shared utilities
6. **predictions_analyze.py** — Most complex; benefits from all utilities being clean
7. **attributions_analyze.py** — Also high complexity; benefits from preliminary work

**For Each Module**:
1. Show this analysis + proposed changes
2. Get user approval (approve all, skip some, request modifications)
3. Apply approved changes
4. Move to next

---

## Review Checkpoint

**Ready for Phase 0 implementation?**

Please review the proposals above and let me know:
- ✅ **Approve all** — Proceed with refactoring throughout
- 🔍 **Review specific module** — Show detailed code examples for one or more modules
- ⏭️ **Skip refactoring** — Jump directly to Phase 1 documentation (not recommended, but your call)
- 📝 **Modify proposal** — Which changes should be added/removed?
