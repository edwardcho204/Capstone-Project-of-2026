import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
with_sca = pd.read_csv("all_station_linear_model_results.csv")
without_sca = pd.read_csv("all_station_linear_noSCA_results.csv")

merged = with_sca.merge(
    without_sca,
    on="station_complex_id",
    suffixes=("_with_sca", "_without_sca")
)

plt.figure(figsize=(6,6))

plt.scatter(
    merged["Test R^2_without_sca"],
    merged["Test R^2_with_sca"],
    alpha=0.6
)

# diagonal line
min_val = min(merged["Test R^2_with_sca"].min(), merged["Test R^2_without_sca"].min())
max_val = max(merged["Test R^2_with_sca"].max(), merged["Test R^2_without_sca"].max())

plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.xlabel("Linear (No SCA) Test R²")
plt.ylabel("Linear (With SCA) Test R²")
plt.title("Effect of SCA Features on Linear Regression Test R^2")

plt.grid(True)
plt.tight_layout()
plt.show()