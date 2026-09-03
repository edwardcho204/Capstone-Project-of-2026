import pandas as pd

# Load combined model results
df = pd.read_csv("all_models_combined_results2.csv")

# Metrics to average
metrics = [
    "Train R^2",
    "Train MAE",
    "Train RMSE",
    "Train MAPE",
    "Test R^2",
    "Test MAE",
    "Test RMSE",
    "Test MAPE",
]

# Group by model and calculate mean
summary = (
    df.groupby("model")[metrics]
    .mean()
    .reset_index()
)

# Rename columns to match your table
summary = summary.rename(columns={
    "model": "Model",
    "Train R^2": "Mean Train R^2",
    "Train MAE": "Mean Train MAE",
    "Train RMSE": "Mean Train RMSE",
    "Train MAPE": "Mean Train MAPE",
    "Test R^2": "Mean Test R^2",
    "Test MAE": "Mean Test MAE",
    "Test RMSE": "Mean Test RMSE",
    "Test MAPE": "Mean Test MAPE",
})

# Optional: order models manually
model_order = [
    "Baseline",
    "Linear",
    "LinearNoSCA",
    "Ridge",
    "Lasso",
    "RandomForest",
    "LightGBM",
]

summary["Model"] = pd.Categorical(
    summary["Model"],
    categories=model_order,
    ordered=True
)

summary = summary.sort_values("Model")

# Round values
summary = summary.round(4)

# Save
summary.to_csv("model_mean_results_summary.csv", index=False)

print(summary.to_string(index=False))
print("\nSaved: model_mean_results_summary.csv")