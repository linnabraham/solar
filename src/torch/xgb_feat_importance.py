import xgboost as xgb
import matplotlib.pyplot as plt
import os
from src.torch.xgb_train import get_feature_names, TrainingConfig

def plot_xgb_feature_importance(model_path, output_path, config, importance_type='weight', figsize=(10, 12)):
    feature_names = get_feature_names(config)

    # Load model
    model = xgb.Booster()
    model.load_model(model_path)

    # Get importance and map to names
    importance_dict = model.get_score(importance_type=importance_type)
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
    sorted_feature_names = [feature_map.get(f, f) for f, _ in sorted_features]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    xgb.plot_importance(model, importance_type=importance_type, ax=ax)
    ax.set_yticklabels(sorted_feature_names)
    plt.title('Feature Importance')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    jobs = [
        {
            "model_path": "output/deep-spaceship-10/best_xgboost_model.json",
            "output_path": "plots/xgb_feat_importance_gain.png",
            "importance_type": "gain"
        },
        {
            "model_path": "output/deep-spaceship-10/best_xgboost_model.json",
            "output_path": "plots/xgb_feat_importance_weight.png",
            "importance_type": "weight"
        },
    ]
    config = TrainingConfig(json_path="solar_dataset.json", stats_file="stats.pkl")
    for job in jobs:
        plot_xgb_feature_importance(
            model_path=job["model_path"],
            output_path=job["output_path"],
            config=config,
            importance_type=job.get("importance_type", "weight")
        )
