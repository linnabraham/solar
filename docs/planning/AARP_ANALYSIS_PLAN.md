# Per-AARP Level Analysis Plan

**Objective**: Build incremental per-AARP analysis capability by reusing existing code (which already iterates per-AARP), starting with 3-5 diverse AARPs, and establishing pause points for insight extraction before scaling.

**Data Scope**: Test set + sample from training split  
**Model**: Single existing VIT model (`outputs/glad-shape-197/trained_model.pth`)  
**Analysis Type**: Recreate predictions + attributions visualizations at per-AARP level  
**Philosophy**: Reuse 90% of existing code; minimal new orchestration logic; pause frequently for human insights

---

## PHASE 0: Setup & AARP Selection

**Goal**: Understand dataset characteristics; identify 3-5 diverse AARPs to work with.

### Steps

1. **Create `src/torch/vit/aarp_explorer.py`** — lightweight utility to:
   - Load test + sample train splits
   - List all unique AARP IDs with metadata:
     - `aarp_id`: identifier
     - `timestep_count`: number of observations for this AARP
     - `label_distribution`: counts of flare (1) vs. non-flare (0) samples
     - `date_range`: earliest to latest timestamp
     - `data_volume`: total images (timesteps × 7 channels)
   - Rank AARPs by "interestingness" score (mix of labels, temporal span, data count)
   - Reuse: `single_aarp` class from [src/torch/vit/ig.py](src/torch/vit/ig.py) for quick lookups

2. **Run `aarp_explorer.py`** to:
   - Generate summary table of all candidate AARPs
   - Score and rank by diversity
   - Select 3-5 AARPs with diverse characteristics:
     - **Type A**: High flare count (mostly label=1)
     - **Type B**: High non-flare count (mostly label=0)
     - **Type C**: Mixed labels
     - **Type D**: Few timesteps (sparse)
     - **Type E**: Many timesteps (dense)
   - Save selection summary to `docs/planning/selected_aarps.txt`

3. **⏸ PAUSE POINT A** — Review selected AARPs
   - Questions: Do they represent the diversity you want to analyze?
   - Log any observations to `docs/planning/AARP_ANALYSIS_MASTER.md` under "Phase 0 Findings"
   - Decision: Proceed to Phase 1 or adjust selection?

---

## PHASE 1: Single AARP Baseline

**Goal**: Generate all existing plots for ONE AARP to establish the per-AARP workflow pattern.

### Steps

4. **Create `src/torch/vit/per_aarp_analysis.py`** — main orchestrator script:
   - Accept `aarp_id` as command-line argument
   - Load model, stats, metadata (reuse existing loaders from config)
   - Call existing analysis functions in sequence (minimal wrapper):
     - `run_pred_and_ig()` from [src/torch/vit/class_wise_distribution.py](src/torch/vit/class_wise_distribution.py#L32) → get predictions + integrated gradients
     - `make_prediction_plot()` from [src/torch/vit/predictions_analyze.py](src/torch/vit/predictions_analyze.py#L284) → GOES timeseries + prediction overlay
     - `create_plots()` from [src/torch/vit/attributions_analyze.py](src/torch/vit/attributions_analyze.py#L52) → attribution details (percentiles, filtered intensities)
   - Organize outputs under: `plots/per_aarp_analysis/{aarp_id}/`
   - Add logging to track execution time and status

5. **Create output directory structure**:
   ```
   plots/per_aarp_analysis/{aarp_id}/
     ├── predictions/
     │   ├── goes_with_predictions.png
     │   └── flare_probability.txt  (e.g., "Flare: 87% confidence")
     ├── attributions/
     │   ├── attributions_percentiles_99.png
     │   └── filtered_image_intensities.png
     └── summary.txt  (AARP metadata + quick stats)
   ```

6. **Run on first AARP** (pick one with at least one flare event):
   ```bash
   python src/torch/vit/per_aarp_analysis.py --aarp-id <id1>
   ```

7. **⏸ PAUSE POINT B** — Inspect and document findings
   - Visual inspection of all 3 plots: are they sensible?
   - Questions to explore:
     - Do predictions overlay predictions correctly on GOES flares?
     - Are attributions concentrated on specific wavelengths (e.g., 94Å) or distributed randomly?
     - Are filtered intensities showing expected patterns?
     - Any obvious errors or crashes?
   - Log observations to `docs/planning/AARP_ANALYSIS_MASTER.md` under "Phase 1 Findings"
   - Create `docs/planning/AARP_ANALYSIS_LOG.md` with detailed observations for each AARP
   - Decision: Is the workflow correct? Proceed to Phase 2 or debug?

---

## PHASE 2: Generalization & Comparison

**Goal**: Verify the workflow works for diverse AARPs; identify patterns across different AARP types.

### Steps

8. **Run on remaining 2-4 AARPs**:
   ```bash
   for aarp_id in <id2> <id3> <id4> <id5>; do
       python src/torch/vit/per_aarp_analysis.py --aarp-id $aarp_id
   done
   ```
   - Monitor for crashes or unexpected behaviors
   - Log runtime for each AARP

9. **Create `src/torch/vit/aarp_compare.py`** — comparison utility:
   - Load all 3-5 AARP summaries
   - Generate comparison table with columns:
     - AARP ID
     - Timesteps
     - Label Distribution (e.g., "10 flares, 5 non-flares")
     - Top 3 Attribution Wavelengths (most frequently highest-attribution channels)
     - Max Prediction Confidence (across all timesteps)
     - Min Prediction Confidence
     - Key Observations (anomalies, patterns)
   - Save to `docs/planning/aarp_comparison_table.md`
   
10. **Add optional visual comparison** in `aarp_compare.py`:
    - Grid plot combining all prediction plots (one per AARP, side-by-side)
    - Save to `plots/per_aarp_analysis/comparison_grid.png`
    - Helps visually spot patterns and outliers

11. **⏸ PAUSE POINT C** — Deep analysis for insights
    - Questions to explore:
      - Do prediction confidence patterns differ by AARP type?
      - Are top attribution wavelengths consistently the same across AARPs, or AARP-specific?
      - Which AARP has the most/least confident predictions? Why?
      - Are there unexpected attributions or failures?
      - Do flare-heavy AARPs show different patterns than non-flare-heavy AARPs?
      - Temporal patterns: are confidence levels stable or do they change with time?
    - Create `docs/planning/PHASE2_INSIGHTS.md` with structured findings:
      - Key patterns observed
      - Anomalies and explanations
      - Hypotheses for next phases
    - Log to `docs/planning/AARP_ANALYSIS_MASTER.md` under "Phase 2 Findings"
    - Decision: Are patterns stable enough to scale? Proceed to Phase 3?

---

## PHASE 3: Full Batch Processing

**Goal**: Scale from 3-5 AARPs to all test + sample train AARPs; establish monitoring and quality checks.

### Steps

12. **Enhance `per_aarp_analysis.py`** with batch capabilities:
    - Add `--batch-mode` flag to process all AARPs in selected splits
    - Add `--subset` argument to specify which splits: test, train_sample, or both
    - Add `--dry-run` flag to report what would be processed without running
    - Add `--resume` flag to skip already-processed AARPs (checkpointing)
    - Add progress logging: write to `src/torch/vit/batch_log.txt`:
      - Timestamp, AARP ID, status (success/failed), execution time
      - Running count of successes/failures

13. **Run batch processing**:
    ```bash
    python src/torch/vit/per_aarp_analysis.py --batch-mode --subset test,train_sample --resume
    ```
    - Expected output: One subdirectory per AARP under `plots/per_aarp_analysis/`
    - Estimated output count: all test AARPs + training sample AARPs

14. **Create `src/torch/vit/aarp_batch_summary.py`** — aggregate analysis:
    - Count total AARPs processed, successes, failures
    - Compute aggregate statistics across all AARPs:
      - Mean/median prediction confidence (overall, by label)
      - Top N wavelengths (overall attribution distribution)
      - Confidence distribution (histogram)
      - Average execution time per AARP
    - Identify outliers:
      - AARPs with very high/low confidence (potential issues or special cases)
      - AARPs with unexpected attribution patterns
      - Failed AARPs (if any)
    - Save results to:
      - `docs/planning/BATCH_SUMMARY.md` (human-readable report)
      - `plots/per_aarp_analysis/batch_stats.json` (machine-readable stats)

15. **⏸ PAUSE POINT D** — Review batch results and scale readiness
    - Questions to explore:
      - Total AARP count? Is it what you expected?
      - Label and timestep distribution across all AARPs stable?
      - Any systematic errors (crashes for certain AARP types)?
      - Are aggregate attribution patterns stable or noisy?
      - Confidence distributions sensible (not all near 0 or all near 1)?
      - Execution time acceptable? Any scalability concerns?
      - Which outlier AARPs are most interesting?
    - Create `docs/planning/PHASE3_INSIGHTS.md` with:
      - Scale readiness assessment
      - Data quality findings
      - Candidate outliers for closer inspection
    - Log to `docs/planning/AARP_ANALYSIS_MASTER.md` under "Phase 3 Findings"
    - Decision: Ready for Phase 4 (optimization + extensions) or need refinements?

---

## PHASE 4: Scaling & Future Extensions

**Goal**: Optimize for production use; establish patterns for per-AARP analysis on additional stages.

### Steps

16. **Performance optimization** (if Phase 3 timing is slow):
    - Profile `per_aarp_analysis.py` to identify bottlenecks:
      - Model inference time? Data loading? Visualization rendering?
    - Consider batch-mode model predictions (load multiple AARPs at once if memory allows)
    - Add checkpoint/resume logic to skip completed AARPs (avoid re-computation)
    - Optional: Implement parallel processing via `multiprocessing.Pool` (only if no GPU memory conflicts)
    - Benchmark: measure speedup vs. Phase 3

17. **Capture ideas for Phase 4+ analysis enhancements** (defer implementation):
    - Per-AARP temporal heatmaps: show how attribution intensity evolves over time
    - Per-AARP feature importance ranking: which wavelengths matter most for this AARP's predictions?
    - Per-AARP anomaly flagging: flag low-confidence predictions, unexpected attributions, contradictions
    - SHAP-per-AARP: integrate `kshap.py` output for single AARP analysis
    - Per-AARP database: store results in queryable format (JSON, SQLite, Parquet) for analytical queries
    - Comparison across AARPs: cross-AARP patterns, clustering by behavior

18. **Document Phase 4 execution** (if completed):
    - Update `docs/planning/AARP_ANALYSIS_MASTER.md` with completion status
    - Save optimization notes to `docs/planning/PHASE4_NOTES.md`

---

## Files to Create

### **New Python Scripts**
| File | Purpose | ~LOC |
|------|---------|------|
| `src/torch/vit/aarp_explorer.py` | AARP discovery & ranking utility | 100 |
| `src/torch/vit/per_aarp_analysis.py` | Main orchestrator (wrapper + CLI) | 150 |
| `src/torch/vit/aarp_compare.py` | Comparison across multiple AARPs | 120 |
| `src/torch/vit/aarp_batch_summary.py` | Batch processing summaries & stats | 100 |

### **Documentation Files**
| File | Purpose | When Created |
|------|---------|--------------|
| `docs/planning/AARP_ANALYSIS_MASTER.md` | Master tracking file (session resume) | Now |
| `docs/planning/selected_aarps.txt` | AARP selection summary | Phase 0, Step 2 |
| `docs/planning/AARP_ANALYSIS_LOG.md` | Detailed per-AARP observations | Phase 1, Step 7 |
| `docs/planning/aarp_comparison_table.md` | Phase 2 comparison results | Phase 2, Step 9 |
| `docs/planning/PHASE2_INSIGHTS.md` | Phase 2 findings & hypotheses | Phase 2, Step 11 |
| `docs/planning/BATCH_SUMMARY.md` | Phase 3 aggregate statistics | Phase 3, Step 14 |
| `docs/planning/PHASE3_INSIGHTS.md` | Phase 3 findings & readiness | Phase 3, Step 15 |
| `docs/planning/PHASE4_NOTES.md` | Phase 4 optimization results | Phase 4, Step 18 |

### **Output Structure**
```
plots/per_aarp_analysis/
  ├── {aarp_id_1}/
  │   ├── predictions/
  │   │   ├── goes_with_predictions.png
  │   │   └── flare_probability.txt
  │   ├── attributions/
  │   │   ├── attributions_percentiles_99.png
  │   │   └── filtered_image_intensities.png
  │   └── summary.txt
  ├── {aarp_id_2}/
  │   └── [same structure]
  ├── {aarp_id_3}/
  │   └── [same structure]
  ├── comparison_grid.png (Phase 2+)
  └── batch_stats.json (Phase 3+)

src/torch/vit/
  ├── batch_log.txt (Phase 3, updated as batch processes)
```

---

## Reusable Existing Code (~90% of implementation)

| Component | File | Usage |
|-----------|------|-------|
| Model loading & device setup | [src/torch/vit/config.py](src/torch/vit/config.py) | Initialize model, stats |
| Single AARP wrapper | [src/torch/vit/ig.py](src/torch/vit/ig.py) | `single_aarp(aarp_id, df)` class |
| Predictions + IG computation | [src/torch/vit/class_wise_distribution.py](src/torch/vit/class_wise_distribution.py) | `run_pred_and_ig()` function |
| Prediction plot generation | [src/torch/vit/predictions_analyze.py](src/torch/vit/predictions_analyze.py) | `make_prediction_plot()` function |
| Attribution plot generation | [src/torch/vit/attributions_analyze.py](src/torch/vit/attributions_analyze.py) | `create_plots()` function |
| Data loading utilities | [src/torch/vit/ig.py](src/torch/vit/ig.py), [aarp_ml/dataset.py](aarp_ml/dataset.py) | Image/metadata loading |

---

## Verification Checklist

### Phase 0
- [ ] `aarp_explorer.py` runs without errors
- [ ] 3-5 AARPs selected and listed in `selected_aarps.txt`
- [ ] AARP diversity confirmed (mix of flare/non-flare, short/long timelines)
- [ ] Pause Point A observations logged

### Phase 1
- [ ] `per_aarp_analysis.py` accepts `--aarp-id` argument
- [ ] Script completes for first AARP without crashes
- [ ] All output files exist: `goes_with_predictions.png`, `attributions_percentiles_99.png`, `filtered_image_intensities.png`
- [ ] Plots visually sensible (predictions align with GOES, attributions not all noise)
- [ ] Pause Point B observations logged in `AARP_ANALYSIS_LOG.md`

### Phase 2
- [ ] Loop completes for all 3-5 AARPs
- [ ] `aarp_compare.py` generates comparison table and grid
- [ ] Comparison plots show different AARP signatures
- [ ] Pause Point C insights documented in `PHASE2_INSIGHTS.md`

### Phase 3
- [ ] `per_aarp_analysis.py --batch-mode` completes all AARPs
- [ ] `batch_log.txt` shows success/failure counts
- [ ] `batch_stats.json` computes sensible aggregate stats
- [ ] No systematic errors for any AARP type
- [ ] Pause Point D assessment documented in `PHASE3_INSIGHTS.md`

### Phase 4 (Optional)
- [ ] Performance profiling completed (if optimization attempted)
- [ ] Parallel processing working (if implemented)
- [ ] Checkpoint/resume logic functional (if implemented)
- [ ] Phase 4 completion notes in `PHASE4_NOTES.md`

---

## Key Decisions & Constraints

| Decision | Value | Rationale |
|----------|-------|-----------|
| AARP Count (Phase 1-2) | 3-5 | Balance between diversity and speed in early phases |
| Data Subset | Test + training sample | Enough data for patterns, fast iteration |
| Model | Single existing (`glad-shape-197`) | No model selection needed; focus on analysis |
| Analysis Scope | Predictions + attributions only | Excludes SHAP, heatmaps (Phase 4+ features) |
| Code Reuse Strategy | Wrap existing functions | Minimal new code; maximize validation of existing pipeline |
| Pause Strategy | After each major milestone | Enables human insight extraction before scaling |

---

## Potential Insights to Look For

**Phase 1-2 (Single & Few AARPs)**:
- Do all AARPs exhibit similar prediction patterns?
- Are there AARP-specific characteristics (e.g., always low confidence, always high confidence)?
- Do attributions vary by AARP, or do they always focus on same wavelengths?

**Phase 3 (Batch)**:
- What % of predictions are truly confident (>80% for flares)?
- Are there subpopulations: some always confident, some always uncertain?
- Do outlier AARPs correspond to unusual FITS images or edge cases?

**Phase 4+ (Optimization & Extensions)**:
- Can per-AARP patterns predict model failures before they occur?
- Can you identify AARP types and tune analysis depth accordingly?
- Do temporal trends within an AARP correlate with flare events?

---

## Next Steps After Phase 4

Once per-AARP analysis is robust, consider:
1. Integrate SHAP rankings per AARP (extend from [src/torch/vit/kshap.py](src/torch/vit/kshap.py))
2. Temporal heatmaps showing attribution evolution within each AARP
3. Comparison framework: cluster AARPs by behavior, identify anomalies
4. Database schema for queryable per-AARP results
5. Thesis write-up: findings about per-AARP prediction variability & patterns

---

**Created**: April 9, 2026  
**Status**: Ready for Phase 0 execution  
**Next Action**: Run `aarp_explorer.py` to select 3-5 diverse AARPs
