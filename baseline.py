# NOTE: Code by Junwen (Jerry) Zeng
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================
# CONFIG
# ============================================================
TRAIN_PATH = "train_data_featured_clean.csv"
TEST_PATH = "test_data_featured_clean.csv"
OUT_PATH = "all_station_current_baseline_results.csv"

RIDERSHIP_COL = "total_complex_ridership"
STATION_COL = "station_complex_id"
TIMESTAMP_COL = "transit_timestamp"
TARGET_COL = "target_3h_ahead"

# ============================================================
# METRICS
# ============================================================
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    mask = y_true != 0
    mape = np.nan if mask.sum() == 0 else np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100

    return rmse, r2, mae, mape


# ============================================================
# LOAD DATA
# ============================================================
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df[TIMESTAMP_COL] = pd.to_datetime(train_df[TIMESTAMP_COL], errors="coerce")
test_df[TIMESTAMP_COL] = pd.to_datetime(test_df[TIMESTAMP_COL], errors="coerce")

train_df["dataset_split"] = "train"
test_df["dataset_split"] = "test"

# ============================================================
# COMBINE FIRST, THEN CREATE TARGET
# ============================================================
# Create target separately so last 3 rows of TRAIN and TEST are removed independently
train_df = train_df.sort_values([STATION_COL, TIMESTAMP_COL]).reset_index(drop=True)
test_df = test_df.sort_values([STATION_COL, TIMESTAMP_COL]).reset_index(drop=True)

train_df[TARGET_COL] = (
    train_df.groupby(STATION_COL)[RIDERSHIP_COL]
    .shift(-3)
)

test_df[TARGET_COL] = (
    test_df.groupby(STATION_COL)[RIDERSHIP_COL]
    .shift(-3)
)

full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df = full_df.sort_values([STATION_COL, TIMESTAMP_COL]).reset_index(drop=True)

# ============================================================
# BASELINE: PREDICT t+3 USING CURRENT RIDERSHIP AT t
# ============================================================
results = []

stations = sorted(full_df[STATION_COL].unique())

for i, station_id in enumerate(stations, 1):
    print(f"\n[{i}/{len(stations)}] Evaluating current-ridership baseline for station {station_id}")

    station_df = full_df[full_df[STATION_COL] == station_id].copy()

    station_train = station_df[station_df["dataset_split"] == "train"].copy()
    station_test = station_df[station_df["dataset_split"] == "test"].copy()

    station_train = station_train.dropna(subset=[TARGET_COL, RIDERSHIP_COL]).copy()
    station_test = station_test.dropna(subset=[TARGET_COL, RIDERSHIP_COL]).copy()

    if len(station_train) == 0 or len(station_test) == 0:
        print("Skipping: empty train or test after dropping missing values.")
        continue

    y_train = station_train[TARGET_COL].values
    train_preds = station_train[RIDERSHIP_COL].values

    y_test = station_test[TARGET_COL].values
    test_preds = station_test[RIDERSHIP_COL].values

    train_rmse, train_r2, train_mae, train_mape = evaluate(y_train, train_preds)
    test_rmse, test_r2, test_mae, test_mape = evaluate(y_test, test_preds)

    print(f"Train RMSE: {train_rmse:.4f} | R²: {train_r2:.4f} | MAE: {train_mae:.4f} | MAPE: {train_mape:.2f}%")
    print(f"Test  RMSE: {test_rmse:.4f} | R²: {test_r2:.4f} | MAE: {test_mae:.4f} | MAPE: {test_mape:.2f}%")

    results.append({
        "station_complex_id": station_id,
        "n_train": len(station_train),
        "n_test": len(station_test),
        "n_features": 1,

        "Train RMSE": train_rmse,
        "Train R^2": train_r2,
        "Train MAE": train_mae,
        "Train MAPE": train_mape,

        "Test RMSE": test_rmse,
        "Test R^2": test_r2,
        "Test MAE": test_mae,
        "Test MAPE": test_mape,
    })

# ============================================================
# SAVE RESULTS
# ============================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Test R^2", ascending=False)

results_df.to_csv(OUT_PATH, index=False)

print(f"\nSaved baseline results to: {OUT_PATH}")
print(results_df.head(20).to_string(index=False))
