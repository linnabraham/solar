# MkDocs Integration Plan: Documentation Architecture

**Status**: Planning Phase (Awaiting User Approval)  
**Date**: April 7, 2026  
**Purpose**: Elegant, maintainable documentation structure integrating existing pipeline docs with code documentation

---

## Current State Assessment

### What You Have ✅
- **mkdocs.yml**: Configured with Material theme (good foundation)
- **docs/stages/**: 11 well-structured stage documentation files
- **Existing docs**: index.md, pipeline.md, pipeline_details.md
- **DOCUMENTATION_PLAN.md**: Detailed refactoring and docstring strategy
- **dvc.yaml**: Clear description of all pipeline stages

### What's Missing 🔄
- **Navigation structure**: mkdocs.yml only has 3 nav items; stages aren't linked
- **Code documentation**: Python docstrings not yet comprehensively added
- **Inter-linking**: Stage docs don't reference code modules; no unified flow
- **Visual hierarchy**: No clear distinction between pipeline overview vs. implementation details
- **Search/discoverability**: Material theme search not optimized for code + pipeline

---

## Proposed MkDocs Structure (Simple & Focussed)

### Navigation Hierarchy

```
Home
├── Pipeline Stages
│   ├── data-single
│   ├── create-distribution
│   ├── confusion-matrix-val
│   ├── prediction-plots
│   ├── plot-class-dist
│   ├── plot_contour_image_grid
│   ├── attribution_analyzis
│   ├── create_movie
│   ├── xgb-feat-importance
│   ├── kshap-stats
│   └── kshap-analyze
├── Code Documentation
│   ├── predictions_analyze
│   ├── plot_class_wise_distribution
│   ├── plot_contour_image_grid
│   ├── create_movies
│   ├── attributions_analyze
│   ├── ig
│   ├── utils
│   ├── class_wise_distribution
│   ├── xgb_feat_importance
│   ├── kshap
│   └── analyze_shap
└── Pipeline Details (existing docs)
```

### File Structure (docs/)

```
docs/
├── index.md                          # Home page
├── stages/                           # [KEEP as is]
│   ├── data-single.md
│   ├── create-distribution.md
│   ├── confusion-matrix-val.md
│   ├── prediction-plots.md
│   ├── plot-class-dist.md
│   ├── plot_contour_image_grid.md
│   ├── attribution_analyzis.md
│   ├── create_movie.md
│   ├── xgb-feat-importance.md
│   ├── kshap-stats.md
│   └── kshap-analyze.md
├── code/                             # NEW - Generated from docstrings
│   ├── predictions_analyze.md
│   ├── plot_class_wise_distribution.md
│   ├── plot_contour_image_grid.md
│   ├── create_movies.md
│   ├── attributions_analyze.md
│   ├── ig.md
│   ├── utils.md
│   ├── class_wise_distribution.md
│   ├── xgb_feat_importance.md
│   ├── kshap.md
│   └── analyze_shap.md
├── pipeline.md                       # [existing]
├── pipeline_details.md               # [existing]
└── assets/
    └── plots/                        # [existing image assets]
```

---

## Implementation Strategy

### Phase 1: Setup MkDocs (Minimal Configuration)
**Goal**: Get docs site structure ready  
**Time**: ~20 minutes  
**Steps**:

1. **Update `mkdocs.yml`**: Replace with simplified version (shown above)
2. **Install mkdocstrings**:
   ```bash
   pip install mkdocstrings mkdocstrings-python
   ```
3. **Create `docs/code/` folder** and add 11 simple markdown files
4. **Each code doc file** contains just 3-5 lines:
   ```markdown
   # Module Name
   
   ::: src.torch.vit.module_name
       handler: python
       options:
         show_source: true
   ```

### Phase 2: Improve Docstrings (Gradual)
**Goal**: Add/improve docstrings in Python files as you refactor  
**Time**: ~4-6 hours (tied to your DOCUMENTATION_PLAN.md refactoring)  
**Approach**:
- Follow your existing refactoring plan
- Add Google-style docstrings to functions/classes
- Docs auto-update when you save
- **No rush**: Do this gradually, module by module

That's it. Simple.

---

## Timeline & Effort Estimate

| Phase | Task | Effort |
|-------|------|--------|
| **1** | Update mkdocs.yml, install mkdocstrings, create code/ folder with 11 markdown stubs | 20 min |
| **2** | Improve Python docstrings (incremental, as you refactor) | 4-6 hrs |
| **Total** | Minimal setup now, then gradual docstring improvements | **~30 min launch + incremental** |

---

## Proposed mkdocs.yml Configuration (Simple)

```yaml
site_name: Solar Flare Documentation
theme:
  name: material
  palette:
    scheme: default
    primary: blue
    accent: orange
  features:
    - navigation.instant
    - navigation.expand

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true

nav:
  - Home: index.md
  
  - Pipeline Stages:
    - data-single: stages/data-single.md
    - create-distribution: stages/create-distribution.md
    - confusion-matrix-val: stages/confusion-matrix-val.md
    - prediction-plots: stages/prediction-plots.md
    - plot-class-dist: stages/plot-class-dist.md
    - plot_contour_image_grid: stages/plot_contour_image_grid.md
    - attribution_analyzis: stages/attribution_analyzis.md
    - create_movie: stages/create_movie.md
    - xgb-feat-importance: stages/xgb-feat-importance.md
    - kshap-stats: stages/kshap-stats.md
    - kshap-analyze: stages/kshap-analyze.md
  
  - Code Documentation:
    - predictions_analyze: code/predictions_analyze.md
    - plot_class_wise_distribution: code/plot_class_wise_distribution.md
    - plot_contour_image_grid: code/plot_contour_image_grid.md
    - create_movies: code/create_movies.md
    - attributions_analyze: code/attributions_analyze.md
    - ig: code/ig.md
    - utils: code/utils.md
    - class_wise_distribution: code/class_wise_distribution.md
    - xgb_feat_importance: code/xgb_feat_importance.md
    - kshap: code/kshap.md
    - analyze_shap: code/analyze_shap.md
  
  - Pipeline Details: pipeline_details.md
```

---

## Code Documentation Generation: Concrete Example

### How It Works

You write docstrings in your Python files (using Google style). Then in markdown, you just add one line:

```markdown
::: src.torch.xgb_feat_importance
    handler: python
    options:
      show_source: true
      docstring_style: google
```

The `mkdocstrings` plugin automatically generates the full documentation from your docstrings.

### Example: What Users See

**File**: `docs/code/xgb_feat_importance.md`

```markdown
# XGBoost Feature Importance

::: src.torch.xgb_feat_importance
    handler: python
    options:
      show_source: true
```

**Renders as** (in the browser):

```
XGBoost Feature Importance Module

FeatureImportanceConfig
    Configuration for feature importance visualization.
    
    Attributes:
        model_path (str): Path to the trained XGBoost model JSON file.
        output_path (str): Path where the output PNG will be saved.
        importance_type (str): Type of importance metric ('weight', 'gain', or 'cover').
            Defaults to 'weight'.
        figsize (tuple): Figure size as (width, height) in inches. Defaults to (10, 12).
        dpi (int): Resolution in dots per inch. Defaults to 150.

plot_xgb_feature_importance(config: FeatureImportanceConfig, **kwargs) → bool
    Generate feature importance plots for XGBoost model.
    
    Creates two PNG files: one for 'gain' and one for 'weight' importance metrics.
    
    Args:
        config (FeatureImportanceConfig): Configuration object with model path and output settings.
        **kwargs: Additional arguments passed to plot formatting functions.
    
    Returns:
        bool: True if plot saved successfully, False otherwise.
    
    Raises:
        FileNotFoundError: If model file does not exist.
        ValueError: If importance_type is invalid.
    
    Example:
        >>> config = FeatureImportanceConfig(
        ...     model_path='models/xgb.json',
        ...     output_path='plots/',
        ...     importance_type='gain'
        ... )
        >>> plot_xgb_feature_importance(config)
```

That's it. **Automatic**. As you improve docstrings in your code, the docs get better.

### What Requires Docstrings?

Modern docstrings needed:
- **Functions**: Parameter descriptions, return types, what it does
- **Classes**: What the class does, constructor parameters
- **Modules**: Brief description at top of file

Your `xgb_feat_importance.py` already has good docstrings. Other files will need them as you refactor (aligned with your DOCUMENTATION_PLAN.md).

---

## Benefits of This Simple Approach

| Aspect | Benefit |
|--------|---------|
| **Single Source** | Docstrings in code = one place to update |
| **Auto-Updated** | Better docstrings → docs automatically improve |
| **Low Effort** | 20 min setup, then incremental docstring work |
| **Clean Navigation** | Stages + Code docs in flat, simple structure |
| **Discoverable** | Users can search both DVC pipeline stages and code |
| **Organic Growth** | You learn your codebase through docs, organize later |

---

## Implementation Status

### ✅ Phase 1 Complete (April 7, 2026)

**What was done:**

1. **Updated mkdocs.yml**:
   - Added mkdocstrings plugin for auto-generating code docs from docstrings
   - Organized Code Documentation section with Stage Scripts and Shared Utilities

2. **Created code documentation files** (13 total):
   
   **Stage Scripts** (11):
   - predictions_analyze.py
   - plot_class_wise_distribution.py
   - plot_contour_image_grid.py
   - create_movies.py
   - attributions_analyze.py
   - test.py
   - class_wise_distribution.py
   - kshap.py
   - analyze_shap.py
   - data_single.py
   - xgb_feat_importance.py
   
   **Shared Utilities** (3):
   - ig.py (Integrated Gradients)
   - utils.py
   - train.py

3. **Auto-generation approach**:
   - Each code doc file uses mkdocstrings syntax: `::: src.module.path`
   - Automatically extracts docstrings from Python files
   - Shows function signatures, parameters, returns, raises sections
   - Includes source code display
   - Updates automatically when docstrings improve

### ✅ Discovery Process

Analyzed dvc.yaml to identify:
- **11 main stage scripts** (referenced in pipeline stages)
- **3 shared utility modules** (used as dependencies across stages)
- **Dependency graph**: Which scripts depend on which utilities

### 📋 Next Steps

**Phase 2: Improve Python Docstrings** (Tied to DOCUMENTATION_PLAN.md)
- Add/improve Google-style docstrings to all 14 modules
- Follow refactoring strategy from DOCUMENTATION_PLAN.md
- Docs auto-update as you improve docstrings
- Effort: 4-6 hours (gradual, as you refactor)

---

## How It Works

1. **You write/improve docstrings** in Python files (Google style):
   ```python
   def my_function(param1, param2):
       """Short description.
       
       Longer description if needed.
       
       Args:
           param1 (type): Description.
           param2 (type): Description.
       
       Returns:
           type: Description.
       """
   ```

2. **mkdocstrings auto-generates** markdown documentation from these docstrings
3. **No manual markdown** needed—docs stay in sync with code automatically
4. **Search works** across both pipeline stages and code documentation
