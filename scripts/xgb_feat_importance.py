import xgboost as xgb
import matplotlib.pyplot as plt
import os

def get_feature_names(feature_type, passbands):
    if feature_type == "simple":
        stats = ("min", "max", "mean")
        return [f'pb{pb}_{stat}' for pb in passbands for stat in stats]
    elif feature_type == "percentile":
        percentiles = [60, 80, 90, 95, 98, 99]
        return [f'pb{pb}_p{p}' for pb in passbands for p in percentiles]
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

def plot_xgb_feature_importance(model_path, feature_type, output_path, importance_type='weight', figsize=(10, 12)):
    passbands = ['94', '131', '171', '193', '211', '304', '335']
    feature_names = get_feature_names(feature_type, passbands)

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
            "model_path": "output/sage-shape-6/best_xgboost_model.json",
            "feature_type": "simple",
            "output_path":  'plots/xgb_feat_importance_simple.png',
            "importance_type": "weight"
        },
        {
            "model_path": "output/fine-thunder-7/best_xgboost_model.json",
            "feature_type": "percentile",
            "output_path": "plots/xgb_feat_importance_percentile.png",
            "importance_type": "weight"
        },
        {
            "model_path": "output/fine-thunder-7/best_xgboost_model.json",
            "feature_type": "percentile",
            "output_path": "plots/xgb_feat_importance_percentile_gain.png",
            "importance_type": "gain"
        },
    ]

    for job in jobs:
        plot_xgb_feature_importance(
            model_path=job["model_path"],
            feature_type=job["feature_type"],
            output_path=job["output_path"],
            importance_type=job.get("importance_type", "weight")
        )
