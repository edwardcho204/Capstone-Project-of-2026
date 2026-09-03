# all libraries used
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Note: This file is condensed (combined together into one .py file) and are originally separate (.py) files, they will be designated by "-- [Name] --"

# -- Boxplot using R^2 (R-squared) [Comparsion of models used, represented by Boxplot]
# libraries used for Boxplot using R^2
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

# Load combined dataset
df = pd.read_csv("all_models_combined_results2.csv")

# Remove baseline
df = df[df["model"] != "Baseline"]

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="model",
    y="Test MAE"
)

plt.xticks(rotation=45)
plt.title("R^2 Comparison Across Models (Excluding Baseline)")
plt.ylabel("Test R^2")
plt.xlabel("Model")

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -- Boxplot using MAE (Comparsion of models used, represented by Boxplot) --
# libraries used for Boxplot using MAE
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

# Load combined dataset
df = pd.read_csv("all_models_combined_results2.csv")

# Remove baseline
df = df[df["model"] != "Baseline"]

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="model",
    y="Test MAE"
)

plt.xticks(rotation=45)
plt.title("MAE Comparison Across Models (Excluding Baseline)")
plt.ylabel("Test MAE")
plt.xlabel("Model")

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -- Combination of results from machine learning models into a singular dataset --
# libraries used for Combining separate datasets into one complete dataset from 2020 to 2025
#import pandas as pd
# Some of 2021 and 2022 are missing whihc will be addressed later via code


# Designation of separate csv files
files = {
    "Baseline": "all_station_current_baseline_results.csv",
    "Linear": "all_station_linear_model_results.csv",
    "Ridge": "all_station_ridge_model_results.csv",
    "Lasso": "all_station_lasso_results.csv",
    "RandomForest": "all_station_random_forest_results.csv",
    "LightGBM": "all_station_lightgbm_model_results.csv",
    "LinearNoSCA": "all_station_linear_noSCA_results.csv",
}

## Create separate datasets labeled by model name and station name
dfs = []

for model_name, path in files.items():
    print(f"Loading {model_name}...")

    df = pd.read_csv(path)

    # Standardize station column name
    # It was simpler to label it by station complex id as some stations can be one big station
    if "station" in df.columns:
        df = df.rename(columns={"station": "station_complex_id"})

    # Add model column
    df["model"] = model_name

    dfs.append(df)

# Combine the both model name and station name into one dataset
combined_df = pd.concat(dfs, ignore_index=True)

## Create more columns to compare performance of (machine learning) models by station complex id
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

## Confirmation and code to save into a singular large dataset
OUTPUT_PATH = "all_models_combined_results2.csv"
combined_df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved combined dataset to:")
print(OUTPUT_PATH)

print("\nPreview:")
print(combined_df.head(20).to_string(index=False))

# -- Data Visualizations used for Pseudo Research Paper --
# libraries used for data visualizations that would be used in the pseudo research paper
#import pandas as pd
#import matplotlib.pyplot as plt

# Load large final dataset (it has model name, station complex id, n_train, n_test, n_features, and metrics used)
df = pd.read_csv("all_models_combined_results2.csv")

df = df[df["model"] != "Baseline"].copy()

# Optional: print exact model names
print("Models found:")
print(df["model"].unique())

# Optional: If the CSV uses different labels.
model_order = [
    "Linear",
    "LinearNoSCA",
    "Ridge",
    "Lasso",
    "RandomForest",
    "LightGBM",
]

# Keep only models that actually exist in your data
model_order = [m for m in model_order if m in df["model"].unique()]

# If some models are not in model_order, add them at the end
remaining_models = [m for m in df["model"].unique() if m not in model_order]
model_order = model_order + remaining_models

df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df = df.sort_values("model")

# Plot by metric
def dot_plot_by_model(data, metric_col, output_file):
    plt.figure(figsize=(10, 6))

    # Convert model names to numeric y positions
    model_to_y = {model: i for i, model in enumerate(model_order)}

    for model in model_order:
        subset = data[data["model"] == model]

        y_vals = [model_to_y[model]] * len(subset)

        plt.scatter(
            subset[metric_col],
            y_vals,
            alpha=0.45,
            s=18
        )

    plt.yticks(
        range(len(model_order)),
        model_order
    )

    plt.xlabel(metric_col)
    plt.ylabel("Model")
    plt.title(f"{metric_col} Distribution Across Stations by Model")
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

    print(f"Saved: {output_file}")


# Call plot function by labeled Test Accuracy Metric
dot_plot_by_model(df, "Test R^2", "dotplot_test_r2_by_model.png")
dot_plot_by_model(df, "Test MAE", "dotplot_test_mae_by_model.png")
dot_plot_by_model(df, "Test RMSE", "dotplot_test_rmse_by_model.png")
dot_plot_by_model(df, "Test MAPE", "dotplot_test_mape_by_model.png")

# -- Linear Regression Model --
# libraries used for data visualization on Linear Regression with and without Station Connected Awareness (SCA) by R^2
#import pandas as pd
#import matplotlib.pyplot as plt

# Load datasets
with_sca = pd.read_csv("all_station_linear_model_results.csv")
without_sca = pd.read_csv("all_station_linear_noSCA_results.csv")

# Merge datasets with and without SCA
merged = with_sca.merge(
    without_sca,
    on="station_complex_id",
    suffixes=("_with_sca", "_without_sca")
)

# Create figure
plt.figure(figsize=(6,6))

plt.scatter(
    merged["Test R^2_without_sca"],
    merged["Test R^2_with_sca"],
    alpha=0.6
)

# Diagonal line graph (data visualization on how well Linear Regression did)
min_val = min(merged["Test R^2_with_sca"].min(), merged["Test R^2_without_sca"].min())
max_val = max(merged["Test R^2_with_sca"].max(), merged["Test R^2_without_sca"].max())

plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.xlabel("Linear (No SCA) Test R²")
plt.ylabel("Linear (With SCA) Test R²")
plt.title("Effect of SCA Features on Linear Regression Test R^2")

plt.grid(True)
plt.tight_layout()
plt.show()

# -- Display Mean/Average Values --
# libraries used
#import pandas as pd

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

# Round values to the fourth decimal place
summary = summary.round(4)

# Save and output that it saved
summary.to_csv("model_mean_results_summary.csv", index=False)

print(summary.to_string(index=False))
print("\nSaved: model_mean_results_summary.csv")

# -- Model Analysis --
# libraries used
#import pandas as pd
#import matplotlib.pyplot as plt

# Find and call (final) combined dataset
df = pd.read_csv("all_models_combined_results2.csv")
df = df[df["model"] != "Baseline"]

# Sort station IDs and assign each existing station a continuous x-position
station_order = sorted(df["station_complex_id"].unique())
station_to_x = {station_id: i for i, station_id in enumerate(station_order)}

df["station_x"] = df["station_complex_id"].map(station_to_x)

plt.figure(figsize=(16, 7))

for model in df["model"].unique():
    subset = df[df["model"] == model]

    plt.scatter(
        subset["station_x"],
        subset["Test R^2"],
        label=model,
        alpha=0.7,
        s=18
    )

plt.xlabel("Station ID")
plt.ylabel("Test R²")
plt.title("Model Performance Across Stations (Test R²)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

# Show only some station labels so x-axis does not get crowded
tick_every = 25
tick_positions = list(range(0, len(station_order), tick_every))
tick_labels = [station_order[i] for i in tick_positions]

# Plot/display the performance of models used 
plt.xticks(tick_positions, tick_labels, rotation=45)

plt.tight_layout()
plt.show()

# -- Null Analysis (How many null values are there in the original datasets) --
# libraries used
#import pandas as pd

# Read the CSV file
df = pd.read_csv("your_file.csv")

# Total number of observations (rows)
total_observations = len(df)

# Number of observations where ridership is null
null_ridership_count = df['ridership'].isnull().sum()

# Find which station has the most null ridership values
nulls_by_station = (
    df[df['ridership'].isnull()]
    .groupby('station_id')
    .size()
)

# Station with the most nulls
station_most_nulls = nulls_by_station.idxmax()
max_nulls = nulls_by_station.max()

# Print results
print("Total observations:", total_observations)
print("Observations with null ridership:", null_ridership_count)
print("Station with most null ridership:", station_most_nulls)
print("Number of nulls at that station:", max_nulls)
