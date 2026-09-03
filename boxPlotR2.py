import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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