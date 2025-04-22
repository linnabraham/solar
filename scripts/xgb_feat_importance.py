import xgboost as xgb
import matplotlib.pyplot as plt

# Load the model
model = xgb.Booster()
# model.load_model("best_xgboost_model.json")
model.load_model("best_xgboost_model_b2.json")

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 12))  # Increase height
xgb.plot_importance(model, importance_type='weight', ax=ax)  # You can also use 'weight', 'cover', etc.
plt.title('Feature Importance')
plt.tight_layout()
# plt.savefig('xgb_feat_importance.png', bbox_inches="tight")
plt.savefig('xgb_feat_importance_b2.png', bbox_inches="tight")
