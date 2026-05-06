"""Publication-grade analysis of KernelSHAP attributions for the ViT solar-flare model.

Consumes ``shap_stats.json`` (output of ``src.torch.vit.kshap``) and produces:

1. ``global_mean_abs_shap.png`` / ``.pdf`` — bar chart of mean(|SHAP|) per AIA
   passband with bootstrap 95% CI. The headline global-importance figure.

2. ``class_stratified_shap.png`` / ``.pdf`` — two-panel boxplot of *signed* SHAP
   per channel, faceted by true class. Shows direction of contribution per class
   and avoids the sign-cancellation issue of mixing both classes in one plot.

3. ``shap_summary.csv`` — per-channel-per-class numeric summary (N, signed mean,
   |SHAP| mean, bootstrap 95% CI, Wilcoxon signed-rank p-value, Bonferroni-
   corrected p-value). Cite from the paper text.

Filtering: by default only correctly-classified samples (``prediction == label``)
are kept. Pass ``correct_only=False`` in the config to retain all samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ==================== Module Constants ====================
DEFAULT_INPUT_JSON = "shap_stats.json"
DEFAULT_OUTPUT_DIR = "plots/kshap/results"
DEFAULT_FIGSIZE_GLOBAL = (8, 5)
DEFAULT_FIGSIZE_STRATIFIED = (12, 5)
DEFAULT_DPI = 300
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_CI_LEVEL = 0.95
DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.4
DEFAULT_POINT_SIZE = 3
SIGNIFICANCE_THRESHOLD = 0.05
AIA_CHANNEL_PREFIX = "AIA_"
# Wavelength order used to sort channels left-to-right on every figure.
AIA_WAVELENGTHS = [94, 131, 171, 193, 211, 304, 335]

# Class label → human-readable name used in panel titles.
CLASS_NAMES = {0: "No-flare (label=0)", 1: "Flare-positive (label=1)"}


# ==================== Configuration ====================
@dataclass
class SHAPAnalysisConfig:
    """Configuration for SHAP statistical analysis and figure generation.

    Attributes:
        input_json: Path to JSON produced by ``src.torch.vit.kshap``.
        output_dir: Directory for figures and CSV summary.
        figsize_global: Figure size for the global mean(|SHAP|) bar chart.
        figsize_stratified: Figure size for the class-stratified boxplot.
        dpi: Output DPI (used for both creation and savefig — keep consistent).
        bootstrap_resamples: Number of bootstrap resamples for CI estimation.
        ci_level: Bootstrap CI level (e.g. 0.95 for 95% CI).
        seed: RNG seed for bootstrap reproducibility.
        alpha: Stripplot point transparency.
        point_size: Stripplot point size.
        significance_threshold: Per-test alpha used to flag "Significant" rows.
        correct_only: If True, drop records where ``prediction != label``.
        save_pdf: If True, also export PDF (vector) versions for publication.
    """
    input_json: str = DEFAULT_INPUT_JSON
    output_dir: str = DEFAULT_OUTPUT_DIR
    figsize_global: Tuple[float, float] = DEFAULT_FIGSIZE_GLOBAL
    figsize_stratified: Tuple[float, float] = DEFAULT_FIGSIZE_STRATIFIED
    dpi: int = DEFAULT_DPI
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    ci_level: float = DEFAULT_CI_LEVEL
    seed: int = DEFAULT_SEED
    alpha: float = DEFAULT_ALPHA
    point_size: int = DEFAULT_POINT_SIZE
    significance_threshold: float = SIGNIFICANCE_THRESHOLD
    correct_only: bool = True
    save_pdf: bool = True

    def __post_init__(self) -> None:
        if not Path(self.input_json).exists():
            raise FileNotFoundError(f"Input JSON not found: {self.input_json}")
        if self.dpi <= 0:
            raise ValueError(f"dpi must be positive, got {self.dpi}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.point_size <= 0:
            raise ValueError(f"point_size must be positive, got {self.point_size}")
        if not (0.0 < self.ci_level < 1.0):
            raise ValueError(f"ci_level must be in (0, 1), got {self.ci_level}")
        if self.bootstrap_resamples < 100:
            raise ValueError(
                f"bootstrap_resamples must be >= 100, got {self.bootstrap_resamples}"
            )


# ==================== Loading & filtering ====================
def channel_columns() -> List[str]:
    """Return AIA channel column names in canonical wavelength order."""
    return [f"{AIA_CHANNEL_PREFIX}{w}" for w in AIA_WAVELENGTHS]


def load_and_process(config: SHAPAnalysisConfig) -> pd.DataFrame:
    """Load ``shap_stats.json``, flatten to wide DataFrame, optionally filter.

    Schema of the returned DataFrame:
        round, label, prediction (if present), AIA_94, AIA_131, ... AIA_335

    If ``config.correct_only`` is True and the JSON contains a ``prediction``
    field, rows where ``prediction != label`` are dropped. Legacy JSONs without
    a ``prediction`` field are accepted with a warning; no filter is applied.

    Returns:
        pandas.DataFrame with one row per (kept) SHAP record.
    """
    with open(config.input_json, "r") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Input JSON must be a non-empty list of SHAP records.")

    rows: List[Dict] = []
    has_prediction_field = all("prediction" in entry for entry in data)
    for entry in data:
        if "importance" not in entry:
            raise KeyError(f"Record missing 'importance' field: {entry}")
        row: Dict = {"round": entry["round"], "label": entry["label"]}
        if has_prediction_field:
            row["prediction"] = entry["prediction"]
        row.update(entry["importance"])
        rows.append(row)

    df = pd.DataFrame(rows)

    total = len(df)
    if config.correct_only:
        if not has_prediction_field:
            print(
                "⚠ Legacy JSON without 'prediction' field — correct-only filter skipped. "
                "Re-run kshap.py to regenerate."
            )
        else:
            df = df[df["prediction"] == df["label"]].reset_index(drop=True)
    print(
        f"Loaded {total} records; kept {len(df)} after filter "
        f"(correct_only={config.correct_only})."
    )
    if len(df) == 0:
        raise ValueError("No records remain after filtering.")
    return df


# ==================== Statistics ====================
def bootstrap_ci(
    values: np.ndarray,
    statistic,
    n_resamples: int,
    ci_level: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for an arbitrary scalar statistic.

    Args:
        values: 1-D array of observed values.
        statistic: Callable ``np.ndarray -> float``.
        n_resamples: Number of bootstrap resamples.
        ci_level: e.g. 0.95.
        rng: Seeded ``numpy.random.Generator``.

    Returns:
        (low, high) percentile-method CI.
    """
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[i] = statistic(values[idx])
    alpha = (1.0 - ci_level) / 2.0
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1.0 - alpha))


def compute_summary(df: pd.DataFrame, config: SHAPAnalysisConfig) -> pd.DataFrame:
    """Per-channel × per-class summary statistics.

    Columns:
        Channel, Class, N, Mean_Signed_SHAP, Mean_Abs_SHAP,
        Abs_CI_Low, Abs_CI_High, Wilcoxon_Stat, Wilcoxon_P,
        Wilcoxon_P_Bonferroni, Significant.

    Wilcoxon signed-rank p-values are computed against a zero null per channel
    within each class, and Bonferroni-corrected across the 7 channels per class.
    """
    rng = np.random.default_rng(config.seed)
    channels = channel_columns()
    rows: List[Dict] = []

    for cls in sorted(df["label"].unique()):
        sub = df[df["label"] == cls]
        n_cls = len(sub)
        per_class_pvals: List[float] = []

        for ch in channels:
            vals = sub[ch].to_numpy(dtype=float)
            mean_signed = float(np.mean(vals)) if n_cls else np.nan
            abs_vals = np.abs(vals)
            mean_abs = float(np.mean(abs_vals)) if n_cls else np.nan
            abs_low, abs_high = bootstrap_ci(
                abs_vals, np.mean, config.bootstrap_resamples, config.ci_level, rng
            )

            # Wilcoxon needs >= 1 nonzero pair
            try:
                w_stat, w_p = stats.wilcoxon(vals, zero_method="wilcox")
                w_stat = float(w_stat)
                w_p = float(w_p)
            except ValueError:
                w_stat, w_p = np.nan, np.nan

            per_class_pvals.append(w_p)
            rows.append(
                {
                    "Channel": ch,
                    "Class": int(cls),
                    "N": n_cls,
                    "Mean_Signed_SHAP": mean_signed,
                    "Mean_Abs_SHAP": mean_abs,
                    "Abs_CI_Low": abs_low,
                    "Abs_CI_High": abs_high,
                    "Wilcoxon_Stat": w_stat,
                    "Wilcoxon_P": w_p,
                }
            )

        # Bonferroni correction across the channels tested for this class.
        n_tests = len(channels)
        for j in range(n_tests):
            p = per_class_pvals[j]
            corrected = min(p * n_tests, 1.0) if not np.isnan(p) else np.nan
            row_idx = len(rows) - n_tests + j
            rows[row_idx]["Wilcoxon_P_Bonferroni"] = corrected
            rows[row_idx]["Significant"] = (
                False
                if np.isnan(corrected)
                else corrected < config.significance_threshold
            )

    return pd.DataFrame(rows)


def compute_global_abs_summary(
    df: pd.DataFrame, config: SHAPAnalysisConfig
) -> pd.DataFrame:
    """Per-channel mean(|SHAP|) with bootstrap CI, pooled across both classes.

    Columns: Channel, N, Mean_Abs_SHAP, CI_Low, CI_High.
    Channels are returned in canonical wavelength order.
    """
    rng = np.random.default_rng(config.seed)
    rows: List[Dict] = []
    n = len(df)
    for ch in channel_columns():
        abs_vals = np.abs(df[ch].to_numpy(dtype=float))
        mean_abs = float(np.mean(abs_vals)) if n else np.nan
        low, high = bootstrap_ci(
            abs_vals, np.mean, config.bootstrap_resamples, config.ci_level, rng
        )
        rows.append(
            {
                "Channel": ch,
                "N": n,
                "Mean_Abs_SHAP": mean_abs,
                "CI_Low": low,
                "CI_High": high,
            }
        )
    return pd.DataFrame(rows)


# ==================== Plotting ====================
def _save_fig(fig: plt.Figure, output_path: Path, save_pdf: bool, dpi: int) -> None:
    """Save figure as PNG and optionally PDF (vector) for publication."""
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(str(output_path.with_suffix(".pdf")), bbox_inches="tight")


def plot_global_importance(
    abs_summary: pd.DataFrame,
    output_dir: Path,
    config: SHAPAnalysisConfig,
    n_total: int,
) -> None:
    """Bar chart of mean(|SHAP|) per channel with bootstrap 95% CI."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(
        figsize=config.figsize_global, dpi=config.dpi, layout="constrained"
    )

    channels = abs_summary["Channel"].tolist()
    means = abs_summary["Mean_Abs_SHAP"].to_numpy()
    err_low = means - abs_summary["CI_Low"].to_numpy()
    err_high = abs_summary["CI_High"].to_numpy() - means
    yerr = np.vstack([err_low, err_high])

    palette = sns.color_palette("colorblind", n_colors=len(channels))
    ax.bar(
        channels,
        means,
        yerr=yerr,
        color=palette,
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )
    ax.set_xlabel("AIA passband")
    ax.set_ylabel(r"Mean $|\mathrm{SHAP}|$ (per channel)")
    ax.set_title(
        f"Global passband importance — mean(|SHAP|) ± {int(config.ci_level * 100)}% bootstrap CI "
        f"(N={n_total})"
    )
    ax.tick_params(axis="x", rotation=0)

    output_path = output_dir / "global_mean_abs_shap.png"
    _save_fig(fig, output_path, config.save_pdf, config.dpi)
    plt.close(fig)
    print(f"✓ Saved {output_path}")


def plot_class_stratified(
    df: pd.DataFrame,
    output_dir: Path,
    config: SHAPAnalysisConfig,
) -> None:
    """Two-panel signed-SHAP boxplot, faceted by true class."""
    sns.set_style("whitegrid")
    classes = sorted(df["label"].unique())
    fig, axes = plt.subplots(
        1,
        len(classes),
        figsize=config.figsize_stratified,
        dpi=config.dpi,
        sharey=True,
        layout="constrained",
    )
    if len(classes) == 1:
        axes = [axes]

    channels = channel_columns()
    palette = sns.color_palette("colorblind", n_colors=len(channels))

    for ax, cls in zip(axes, classes):
        sub = df[df["label"] == cls]
        melted = sub.melt(
            id_vars=["round", "label"],
            value_vars=channels,
            var_name="Passband",
            value_name="SHAP",
        )
        melted["Passband"] = pd.Categorical(
            melted["Passband"], categories=channels, ordered=True
        )

        sns.boxplot(
            data=melted,
            x="Passband",
            y="SHAP",
            hue="Passband",
            palette=palette,
            showfliers=False,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=melted,
            x="Passband",
            y="SHAP",
            color="black",
            alpha=config.alpha,
            size=config.point_size,
            jitter=True,
            ax=ax,
        )
        ax.axhline(0, color="red", linestyle="--", linewidth=1.0)
        ax.set_title(f"{CLASS_NAMES.get(cls, f'Class {cls}')} (N={len(sub)})")
        ax.set_xlabel("AIA passband")
        ax.set_ylabel("Signed SHAP attribution")

    fig.suptitle(
        "Signed SHAP attributions by true class — positive values push toward the true class",
        fontsize=12,
    )
    output_path = output_dir / "class_stratified_shap.png"
    _save_fig(fig, output_path, config.save_pdf, config.dpi)
    plt.close(fig)
    print(f"✓ Saved {output_path}")


# ==================== Entrypoint ====================
def main() -> None:
    config = SHAPAnalysisConfig()
    print(f"Loading SHAP statistics from {config.input_json}...")

    df = load_and_process(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-channel × per-class summary table for paper text.
    summary = compute_summary(df, config)
    summary_path = output_dir / "shap_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6g")
    print(f"✓ Saved {summary_path}")
    print(summary.to_string(index=False))

    # Global mean(|SHAP|) bar chart.
    abs_summary = compute_global_abs_summary(df, config)
    plot_global_importance(abs_summary, output_dir, config, n_total=len(df))

    # Class-stratified signed-SHAP boxplot.
    plot_class_stratified(df, output_dir, config)

    print("✓ SHAP analysis complete")


if __name__ == "__main__":
    main()
