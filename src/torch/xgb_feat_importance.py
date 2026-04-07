import xgboost as xgb
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from src.torch.xgb_train import get_feature_names, TrainingConfig

# ==================== Module Constants ====================
DEFAULT_FIGSIZE = (10, 12)
DEFAULT_DPI = 150
VALID_IMPORTANCE_TYPES = {'weight', 'gain', 'cover'}
FEATURE_NAME_PREFIX = 'f'  # XGBoost internal naming convention for features

# ==================== Configuration Class ====================
@dataclass
class FeatureImportanceConfig:
    """Configuration for feature importance visualization.
    
    Attributes:
        model_path (str): Path to the trained XGBoost model JSON file.
        output_path (str): Path where the output PNG will be saved.
        importance_type (str): Type of importance metric ('weight', 'gain', or 'cover').
            Defaults to 'weight'.
        figsize (tuple): Figure size as (width, height) in inches. Defaults to (10, 12).
        dpi (int): Resolution in dots per inch. Defaults to 150.
    """
    model_path: str
    output_path: str
    importance_type: str = 'weight'
    figsize: tuple = DEFAULT_FIGSIZE
    dpi: int = DEFAULT_DPI
    
    def __post_init__(self):
        """Validate configuration after initialization.
        
        Raises:
            ValueError: If importance_type is not valid.
            FileNotFoundError: If model file does not exist.
        """
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
    
    # Get importance and map to human-readable names
    importance_dict = model.get_score(importance_type=config.importance_type)
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Map XGBoost internal feature names (f0, f1, ...) to human-readable names
    feature_map = {
        f'{FEATURE_NAME_PREFIX}{i}': name 
        for i, name in enumerate(feature_names)
    }
    sorted_feature_names = [feature_map.get(f, f) for f, _ in sorted_features]
    
    # Create and customize plot
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    xgb.plot_importance(model, importance_type=config.importance_type, ax=ax)
    ax.set_yticklabels(sorted_feature_names)
    
    # Dynamic title based on importance type
    title = f"XGBoost Feature Importance ({config.importance_type.capitalize()})"
    plt.title(title, fontsize=14)
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
