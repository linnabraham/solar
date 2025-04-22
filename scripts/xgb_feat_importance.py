import xgboost as xgb
import matplotlib.pyplot as plt

# Load the model
model = xgb.Booster()
model.load_model("best_xgboost_model.json")
# model.load_model("best_xgboost_model_b2.json")

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 12))  # Increase height

passbands = ['94', '131', '171', '193', '211', '304', '335']
percentiles = [60, 80, 90, 95, 98, 99]
feature_names = [f'pb{pb}_p{p}' for pb in passbands for p in percentiles]
importance_dict = model.get_score(importance_type='weight')
# Sort by importance
sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
# Map f0, f1, ... to actual names
feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
# Re-label sorted feature names
sorted_feature_names = [feature_map.get(f, f) for f, _ in sorted_features]

xgb.plot_importance(model, importance_type='weight', ax=ax)  # You can also use 'weight', 'cover', etc.
ax.set_yticklabels(sorted_feature_names)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('xgb_feat_importance_new.png', bbox_inches="tight")
# plt.savefig('xgb_feat_importance_b2_new.png', bbox_inches="tight")
