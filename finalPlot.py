import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv("all_models_combined_results2.csv")

df = df[df["model"] != "Baseline"].copy()

# Optional: print exact model names
print("Models found:")
print(df["model"].unique())

# ============================================================
# OPTIONAL: ORDER MODELS
# Change these names if your CSV uses different labels
# ============================================================
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

# ============================================================
# PLOT FUNCTION
# ============================================================
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


# ============================================================
# CREATE FOUR PLOTS
# ============================================================
dot_plot_by_model(df, "Test R^2", "dotplot_test_r2_by_model.png")
dot_plot_by_model(df, "Test MAE", "dotplot_test_mae_by_model.png")
dot_plot_by_model(df, "Test RMSE", "dotplot_test_rmse_by_model.png")
dot_plot_by_model(df, "Test MAPE", "dotplot_test_mape_by_model.png")