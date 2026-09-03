import pandas as pd

# ============================================================
# FILE PATHS (EDIT THESE)
# ============================================================
files = {
    "Baseline": "all_station_current_baseline_results.csv",
    "Linear": "all_station_linear_model_results.csv",
    "Ridge": "all_station_ridge_model_results.csv",
    "Lasso": "all_station_lasso_results.csv",
    "RandomForest": "all_station_random_forest_results.csv",
    "LightGBM": "all_station_lightgbm_model_results.csv",
    "LinearNoSCA": "all_station_linear_noSCA_results.csv",
}

# ============================================================
# LOAD AND STANDARDIZE
# ============================================================
dfs = []

for model_name, path in files.items():
    print(f"Loading {model_name}...")

    df = pd.read_csv(path)

    # Standardize station column name
    if "station" in df.columns:
        df = df.rename(columns={"station": "station_complex_id"})

    # Add model column
    df["model"] = model_name

    dfs.append(df)

# ============================================================
# COMBINE
# ============================================================
combined_df = pd.concat(dfs, ignore_index=True)

# ============================================================
# CLEAN / ORDER COLUMNS
# ============================================================
cols_order = [
    "station_complex_id",
    "model",
    "n_train",
    "n_test",
    "n_features",

    "Train RMSE",
    "Train R^2",
    "Train MAE",
    "Train MAPE",

    "Test RMSE",
    "Test R^2",
    "Test MAE",
    "Test MAPE",
]

# Keep only existing columns (some models may not have all)
cols_order = [c for c in cols_order if c in combined_df.columns]

combined_df = combined_df[cols_order]

# ============================================================
# SAVE
# ============================================================
OUTPUT_PATH = "all_models_combined_results2.csv"
combined_df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved combined dataset to:")
print(OUTPUT_PATH)

print("\nPreview:")
print(combined_df.head(20).to_string(index=False))