import argparse
import xgboost as xgb
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ==================== Module Constants ====================
DEFAULT_FIGSIZE = (12, 5)
DEFAULT_DPI = 150
VALID_IMPORTANCE_TYPES = {'weight', 'gain', 'cover'}

# ==================== Configuration Class ====================
@dataclass
class FeatureImportanceConfig:
    model_path: str
    output_path: str
    importance_type: str = 'weight'
    figsize: tuple = DEFAULT_FIGSIZE
    dpi: int = DEFAULT_DPI
    top_n: Optional[int] = None  # None = show all features

    def __post_init__(self):
        if self.importance_type not in VALID_IMPORTANCE_TYPES:
            raise ValueError(
                f"importance_type must be one of {VALID_IMPORTANCE_TYPES}, "
                f"got '{self.importance_type}'"
            )
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at '{self.model_path}'")

# ==================== Main Function ====================
def plot_xgb_feature_importance(config: FeatureImportanceConfig) -> None:
    """Generate and save XGBoost feature importance visualization.

    Feature names are read directly from the model (embedded at training time).
    Raises ValueError if the model was trained without feature_names in DMatrix.
    """
    # Load model — feature names are embedded in the saved JSON since training
    model = xgb.Booster()
    model.load_model(config.model_path)
    if not model.feature_names:
        raise ValueError(
            "Model has no feature names embedded. "
            "Retrain with feature_names passed to xgb.DMatrix."
        )

    # Get importance, sort descending, apply top_n limit
    importance_dict = model.get_score(importance_type=config.importance_type)
    if not importance_dict:
        raise ValueError("Model returned no feature importances — model may be untrained or trivial.")
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    if config.top_n is not None:
        sorted_features = sorted_features[:config.top_n]

    plot_names = [f for f, _ in sorted_features]
    plot_values = [v for _, v in sorted_features]

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')

    max_val = max(plot_values)
    norm_vals = [v / max_val for v in plot_values]
    colors = plt.cm.viridis_r(norm_vals)
    bars = ax.bar(range(len(plot_names)), plot_values, color=colors, width=0.6, edgecolor='white', linewidth=0.5)

    # Value labels above bars — weight is an integer count, gain/cover are floats
    is_count = config.importance_type == 'weight'
    for bar, val in zip(bars, plot_values):
        label = str(int(val)) if is_count else f'{val:.1f}'
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val * 0.01,
            label, ha='center', va='bottom', fontsize=7, color='#444444'
        )

    ax.set_xticks(range(len(plot_names)))
    ax.set_xticklabels(plot_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(config.importance_type.capitalize(), fontsize=11)
    ax.set_ylim(0, max_val * 1.12)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='white')
    ax.set_axisbelow(True)

    title = f"XGBoost Feature Importance ({config.importance_type.capitalize()})"
    if config.top_n is not None:
        title += f" — Top {len(plot_names)}"  # actual count, not requested top_n
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    
    # Save figure
    output_dir = Path(config.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.output_path, bbox_inches="tight", dpi=config.dpi)
    plt.close(fig)
    print(f"✓ Saved {config.importance_type} importance plot to {config.output_path}")

# ==================== Main Block ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to trained XGBoost model JSON")
    args = parser.parse_args()

    model_path = args.model_path
    run_name = Path(model_path).parent.name
    plot_dir = Path("plots") / "xgb" / run_name

    jobs = [
        FeatureImportanceConfig(
            model_path=model_path,
            output_path=str(plot_dir / "feat_importance_gain.png"),
            importance_type="gain",
        ),
        FeatureImportanceConfig(
            model_path=model_path,
            output_path=str(plot_dir / "feat_importance_weight.png"),
            importance_type="weight",
        ),
        FeatureImportanceConfig(
            model_path=model_path,
            output_path=str(plot_dir / "feat_importance_gain_top20.png"),
            importance_type="gain",
            top_n=20,
        ),
    ]

    for job_config in jobs:
        try:
            plot_xgb_feature_importance(job_config)
        except (FileNotFoundError, ValueError) as e:
            print(f"✗ Error processing {job_config.output_path}: {e}")
