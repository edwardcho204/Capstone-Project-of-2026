import pandas as pd
import matplotlib.pyplot as plt

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

plt.xticks(tick_positions, tick_labels, rotation=45)

plt.tight_layout()
plt.show()