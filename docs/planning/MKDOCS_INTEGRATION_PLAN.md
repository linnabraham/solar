# MkDocs Integration Plan: Documentation Architecture

**Status**: ✅ **COMPLETE - Implementation Finished (April 8, 2026)**  
**Date**: April 7-8, 2026  
**Purpose**: Elegant, maintainable documentation structure integrating pipeline stages with code documentation

---

## ✅ What Has Been Implemented

### Phase 1: MkDocs Setup (COMPLETE)
- ✅ mkdocs.yml configured with Material theme
- ✅ Plugins installed: search, mkdocstrings
- ✅ Navigation structure organized with Pipeline Stages + Code Documentation sections
- ✅ Planning docs excluded from public site (`exclude_docs: planning/`)

### Phase 2: Code Documentation Framework (COMPLETE)
- ✅ 13 code documentation markdown files created in `docs/code/`:
  - 11 stage scripts (predictions_analyze, plot_class_wise_distribution, etc.)
  - 3 shared utilities (ig, utils, train)
- ✅ Each code doc auto-generates from Python docstrings via mkdocstrings
- ✅ mkdocstrings configured for Google-style docstrings

### Phase 3: Bi-Directional Linking (COMPLETE)
- ✅ All 11 stage pages (`docs/stages/`) auto-linked to code docs via `generate_pipeline_docs.py`
  - Stage pages include: `**Implementation**: [View code documentation](../code/module_name.md)`
- ✅ All 11 code pages linked back to stage pages
  - Code pages include: `## Related Stage` section with link

### Phase 4: Validation & Automation (COMPLETE)
- ✅ `generate_pipeline_docs.py` enhanced with:
  - Stage→Script mapping discovery
  - Code doc existence validation
  - Non-destructive link insertion
  - Completeness reporting
- ✅ Validation status: **All 11 stages documented, 0 missing**

---

## 📊 Current Documentation Structure

```
docs/
├── index.md                         # Home page
├── mkdocs.yml                       # Site configuration
│
├── stages/                          # 11 stage documentation pages
│   ├── prediction-plots.md          # + Implementation link → code/predictions_analyze.md
│   ├── xgb-feat-importance.md       # + Implementation link → code/xgb_feat_importance.md
│   ├── plot-class-dist.md
│   ├── plot_contour_image_grid.md
│   ├── create_movie.md
│   ├── attribution_analyzis.md
│   ├── confusion-matrix-val.md
│   ├── data-single.md
│   ├── create-distribution.md
│   ├── kshap-stats.md
│   └── kshap-analyze.md
│
├── code/                            # 13 code documentation pages (auto-generated from docstrings)
│   ├── predictions_analyze.md       # + Related Stage link ← stages/prediction-plots.md
│   ├── plot_class_wise_distribution.md
│   ├── plot_contour_image_grid.md
│   ├── create_movies.md
│   ├── attributions_analyze.md
│   ├── test.md
│   ├── data_single.md
│   ├── class_wise_distribution.md
│   ├── kshap.md
│   ├── analyze_shap.md
│   ├── xgb_feat_importance.md
│   ├── ig.md
│   ├── utils.md
│   └── train.md
│
├── planning/                        # Documentation planning (excluded from public site)
│   ├── README.md                    # Overview of planning docs
│   ├── MKDOCS_INTEGRATION_PLAN.md   # This file - setup & current status
│   ├── DOCUMENTATION_PLAN.md        # Code refactoring strategy (from earlier)
│   └── DOCUMENTATION_STRATEGY.md    # Integration approach & best practices
│
├── pipeline.md                      # Auto-generated summary (from dvc.yaml)
├── pipeline_details.md              # Auto-generated detailed table (from dvc.yaml)
└── assets/
    └── plots/                       # Image assets for documentation
```

---

## 🔗 Bi-Directional Linking Map

| Stage | → | Code Doc | ← |
|-------|---|----------|---|
| prediction-plots | → | predictions_analyze | ← |
| plot-class-dist | → | plot_class_wise_distribution | ← |
| xgb-feat-importance | → | xgb_feat_importance | ← |
| plot_contour_image_grid | → | plot_contour_image_grid | ← |
| create_movie | → | create_movies | ← |
| attribution_analyzis | → | attributions_analyze | ← |
| confusion-matrix-val | → | test | ← |
| data-single | → | data_single | ← |
| create-distribution | → | class_wise_distribution | ← |
| kshap-stats | → | kshap | ← |
| kshap-analyze | → | analyze_shap | ← |

---

## 🛠️ Key Scripts & Tools

### `generate_pipeline_docs.py`
**Purpose**: Auto-generate and maintain documentation from dvc.yaml

**Functionality**:
- Discovers stage→script mapping
- Creates stage overview pages (pipeline.md, pipeline_details.md)
- Creates individual stage pages (if not exist)
- Auto-inserts implementation links to stage pages
- Validates code doc completeness
- Reports missing documentation

**Usage**:
```bash
python generate_pipeline_docs.py
```

**Output**: Validation report showing all 11 stages documented

---

## 📖 How Documentation Works

### 1. **View Live Documentation**
```bash
mkdocs serve --livereload --dev-addr localhost:8001
```
Then visit: `http://localhost:8001`

### 2. **Code Documentation Auto-Updates**
- Docstrings in Python files are extracted automatically
- When you improve a docstring, docs update on save
- No manual markdown needed for function/parameter docs

### 3. **Navigation**
- **Home** → **Pipeline Stages** → **Code Documentation**
- Stages have Implementation links
- Code docs have Related Stage links
- Full bidirectional navigation

---

## 🚀 How to Add a New Stage

When you add a new stage to `dvc.yaml`:

1. **Run the script**:
   ```bash
   python generate_pipeline_docs.py
   ```

2. **Script automatically**:
   - Discovers the new stage
   - Creates stage page in `docs/stages/new-stage-name.md`
   - Validates if code doc exists
   - Reports: ✗ "code doc missing" or ✓ "documented"

3. **If code doc doesn't exist, create it**:
   - Create: `docs/code/module_name.md`
   - Add: `::: src.path.to.module` (mkdocstrings directive)
   - Add: `## Related Stage` section

4. **Run script again** to add implementation link

---

## 🎯 Optional Future Enhancements

### Enhancement 1: Auto-Create Code Doc Stubs
*Difficulty: Easy | Benefit: Reduces manual setup*

Modify `generate_pipeline_docs.py` to auto-create missing code doc markdown files:
```python
# Check if code doc exists for stage's script
# If not, create stub with mkdocstrings template
# Auto-add Related Stage link
```

### Enhancement 2: Auto-Add Reciprocal Links  
*Difficulty: Easy | Benefit: Zero manual linking*

Modify `generate_pipeline_docs.py` to:
- Read code doc files
- Auto-insert "Related Stage" links if missing
- Non-destructive (like stage page linking)

### Enhancement 3: Output Examples Gallery
*Difficulty: Medium | Benefit: Visual artifact browsing*

Add separate `docs/gallery/` with:
- Organized by output type (plots, plots, distributions, etc.)
- Example outputs with descriptions
- Links to generating scripts

### Enhancement 4: Dependency Graph Visualization
*Difficulty: Hard | Benefit: Understand script relationships*

Generate Mermaid or graphviz diagrams showing:
- Stage → Stage dependencies
- Script → Script utilities used
- Auto-render in docs

---

## 📝 Session Resumption Checklist

**To resume development in next session:**

1. ✅ Documentation setup is complete
2. ✅ Links are bidirectional and working
3. ✅ Validation script runs successfully
4. ✅ Live preview works: `mkdocs serve --livereload --dev-addr localhost:8001`
5. 📋 Next steps depend on priorities:
   - Improve Python docstrings (from DOCUMENTATION_PLAN.md refactoring)
   - Add output example images to code docs
   - Implement enhancement #1, #2, #3, or #4
   - Or deploy live documentation site

---

## ✅ Verification Checklist

Run these commands to verify everything works:

```bash
# Run documentation generation & validation
python generate_pipeline_docs.py

# Expected output:
# ✓ All 11 stages documented
# ✗ Missing documentation: 0
# 11 code documentation link(s) added/updated

# View live docs
mkdocs serve --livereload --dev-addr localhost:8001

# Check one link works (in browser):
# 1. Go to Code Documentation → xgb_feat_importance
# 2. See "Related Stage" link → click it
# 3. Goes to Pipeline Stage xgb-feat-importance page
# 4. Page has "Implementation" link → click it
# 5. Back to code page (bidirectional!)
```

---

## 📚 Documentation Files

All planning/meta documentation:
- `docs/planning/README.md` - Planning overview
- `docs/planning/MKDOCS_INTEGRATION_PLAN.md` - This file
- `docs/planning/DOCUMENTATION_PLAN.md` - Code refactoring roadmap
- `docs/planning/DOCUMENTATION_STRATEGY.md` - Integration strategy & best practices

These are **excluded from public site** but available for reference.

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
