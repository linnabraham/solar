import xgboost as xgb
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from src.torch.xgb_train import get_feature_names, TrainingConfig

# ==================== Module Constants ====================
DEFAULT_FIGSIZE = (12, 5)
DEFAULT_DPI = 150
VALID_IMPORTANCE_TYPES = {'weight', 'gain', 'cover'}
FEATURE_NAME_PREFIX = 'f'  # XGBoost internal naming convention for features

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
def plot_xgb_feature_importance(
    config: FeatureImportanceConfig,
    training_config: TrainingConfig
) -> None:
    """Generate and save XGBoost feature importance visualization.
    
    Loads a trained XGBoost model and creates a horizontal bar plot showing
    feature importance scores. Features are ranked by importance and labeled
    with human-readable names from the training config. Supports multiple
    importance types (e.g., 'weight' for frequency, 'gain' for performance).
    
    Args:
        config (FeatureImportanceConfig): Visualization configuration including
            model path, output path, and importance metric type.
        training_config (TrainingConfig): Training configuration containing
            dataset metadata and feature definitions.
    
    Returns:
        None. Saves PNG file to config.output_path.
        
    Raises:
        FileNotFoundError: If model or feature names file does not exist.
        ValueError: If importance_type is not valid or config validation fails.
        
    Side effects:
        - Creates output directory if needed (os.makedirs)
        - Saves matplotlib figure to disk
        - Closes matplotlib figure to release memory
        - Prints success message to stdout
        
    Example:
        >>> config = FeatureImportanceConfig(
        ...     model_path="model.json",
        ...     output_path="plots/importance.png",
        ...     importance_type="gain"
        ... )
        >>> plot_xgb_feature_importance(config, training_config)
    """
    # Load feature names and model
    feature_names = get_feature_names(training_config)
    model = xgb.Booster()
    model.load_model(config.model_path)

    # Get importance, sort descending, apply top_n limit
    importance_dict = model.get_score(importance_type=config.importance_type)
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    if config.top_n is not None:
        sorted_features = sorted_features[:config.top_n]

    # Map XGBoost internal feature names (f0, f1, ...) to human-readable names
    feature_map = {f'{FEATURE_NAME_PREFIX}{i}': name for i, name in enumerate(feature_names)}
    plot_names = [feature_map.get(f, f) for f, _ in sorted_features]
    plot_values = [v for _, v in sorted_features]

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')

    norm_vals = [v / max(plot_values) for v in plot_values]
    colors = plt.cm.viridis_r(norm_vals)
    bars = ax.bar(range(len(plot_names)), plot_values, color=colors, width=0.6, edgecolor='white', linewidth=0.5)

    # Value labels above bars
    for bar, val in zip(bars, plot_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(plot_values) * 0.01,
            f'{val:.1f}' if isinstance(val, float) else str(val),
            ha='center', va='bottom', fontsize=7, color='#444444'
        )

    ax.set_xticks(range(len(plot_names)))
    ax.set_xticklabels(plot_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(config.importance_type.capitalize(), fontsize=11)
    ax.set_ylim(0, max(plot_values) * 1.12)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='white')
    ax.set_axisbelow(True)

    title = f"XGBoost Feature Importance ({config.importance_type.capitalize()})"
    if config.top_n is not None:
        title += f" — Top {config.top_n}"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(config.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.output_path, bbox_inches="tight", dpi=config.dpi)
    plt.close(fig)
    print(f"✓ Saved {config.importance_type} importance plot to {config.output_path}")

# ==================== Main Block ====================
if __name__ == "__main__":
    # Define jobs as configuration objects
    jobs = [
        FeatureImportanceConfig(
            model_path="outputs/deep-spaceship-10/best_xgboost_model.json",
            output_path="plots/xgb_feat_importance_gain.png",
            importance_type="gain",
        ),
        FeatureImportanceConfig(
            model_path="outputs/deep-spaceship-10/best_xgboost_model.json",
            output_path="plots/xgb_feat_importance_weight.png",
            importance_type="weight",
        ),
        # Publication-friendly: top 20 features only
        FeatureImportanceConfig(
            model_path="outputs/deep-spaceship-10/best_xgboost_model.json",
            output_path="plots/xgb_feat_importance_gain_top20.png",
            importance_type="gain",
            top_n=20,
        ),
    ]
    
    # Create training config once
    training_config = TrainingConfig(
        json_path="solar_dataset.json",
        stats_file="stats.pkl"
    )
    
    # Process each job
    for job_config in jobs:
        try:
            plot_xgb_feature_importance(job_config, training_config)
        except (FileNotFoundError, ValueError) as e:
            print(f"✗ Error processing {job_config.output_path}: {e}")
