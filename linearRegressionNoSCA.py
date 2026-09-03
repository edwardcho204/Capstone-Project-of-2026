# NOTE: Code by Junwen (Jerry) Zeng
import pandas as pd
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================
# CONFIG
# ============================================================
TRAIN_PATH = "train_data_featured_clean.csv"
TEST_PATH = "test_data_featured_clean.csv"

OUT_PATH = "all_station_linear_noSCA_results.csv"

RIDERSHIP_COL = "total_complex_ridership"
STATION_COL = "station_complex_id"
TIMESTAMP_COL = "transit_timestamp"
TARGET_COL = "target_3h_ahead"

# ============================================================
# METRIC FUNCTION
# ============================================================
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

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

# Sort separately so last 3 rows of train/test are removed independently
train_df = train_df.sort_values([STATION_COL, TIMESTAMP_COL]).reset_index(drop=True)
test_df = test_df.sort_values([STATION_COL, TIMESTAMP_COL]).reset_index(drop=True)

# Create 3-hour-ahead target separately for train and test
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
# STATIONS
# ============================================================
train_stations = set(full_df.loc[full_df["dataset_split"] == "train", STATION_COL].unique())
test_stations = set(full_df.loc[full_df["dataset_split"] == "test", STATION_COL].unique())

stations = sorted(list(train_stations & test_stations))

print(f"Total stations to train: {len(stations)}")

# ============================================================
# MODEL LOOP
# ============================================================
results = []

for i, station_id in enumerate(stations, 1):
    print(f"\n{'=' * 80}")
    print(f"[{i}/{len(stations)}] Training Linear No-SCA model for station {station_id}")
    print(f"{'=' * 80}")

    start_time = time.time()

    station_df = full_df[full_df[STATION_COL] == station_id].copy()

    # --------------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------------
    drop_feature_cols = [
        TIMESTAMP_COL,
        STATION_COL,
        RIDERSHIP_COL,
        TARGET_COL,
        "dataset_split",

        # Remove year features because test year is unseen during training
        "is_year_2023",
        "is_year_2024",
        "is_year_2025",

        # Remove holiday because it was not confirmed accurate
        "isHoliday",
    ]

    feature_cols = [
        c for c in station_df.columns
        if c not in drop_feature_cols
    ]

    # This is the no-SCA model, so remove all SCA features
    feature_cols = [
        c for c in feature_cols
        if not c.startswith("sca_lag_")
    ]

    # Keep only numeric columns
    feature_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(station_df[c])
    ]

    # Remove duplicate names just in case
    feature_cols = list(dict.fromkeys(feature_cols))

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------
    station_df = station_df.dropna(subset=[TARGET_COL]).copy()

    station_train = station_df[station_df["dataset_split"] == "train"].copy()
    station_test = station_df[station_df["dataset_split"] == "test"].copy()

    if len(station_train) == 0 or len(station_test) == 0:
        print(f"Skipping station {station_id}: empty train/test.")
        continue

    # Final safety check: drop rows with missing feature values
    # Ideally this should remove none if preprocessing worked correctly.
    before_train = len(station_train)
    before_test = len(station_test)

    station_train = station_train.dropna(subset=feature_cols).copy()
    station_test = station_test.dropna(subset=feature_cols).copy()

    if len(station_train) < before_train or len(station_test) < before_test:
        print(
            f"Warning: dropped rows with missing features. "
            f"Train dropped: {before_train - len(station_train)}, "
            f"Test dropped: {before_test - len(station_test)}"
        )

    if len(station_train) == 0 or len(station_test) == 0:
        print(f"Skipping station {station_id}: empty after feature null removal.")
        continue

    X_train = station_train[feature_cols]
    y_train = station_train[TARGET_COL]

    X_test = station_test[feature_cols]
    y_test = station_test[TARGET_COL]

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} | Features: {len(feature_cols)}")

    # --------------------------------------------------------
    # MODEL TRAINING
    # --------------------------------------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_preds = np.clip(model.predict(X_train), 0, None)
    test_preds = np.clip(model.predict(X_test), 0, None)

    train_rmse, train_r2, train_mae, train_mape = evaluate(y_train.values, train_preds)
    test_rmse, test_r2, test_mae, test_mape = evaluate(y_test.values, test_preds)

    print("\n--- RESULTS ---")
    print(f"Train RMSE: {train_rmse:.4f} | R²: {train_r2:.4f} | MAE: {train_mae:.4f} | MAPE: {train_mape:.2f}%")
    print(f"Test  RMSE: {test_rmse:.4f} | R²: {test_r2:.4f} | MAE: {test_mae:.4f} | MAPE: {test_mape:.2f}%")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    results.append({
        "station_complex_id": station_id,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),

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

if len(results_df) > 0:
    results_df = results_df.sort_values("Test R^2", ascending=False)

results_df.to_csv(OUT_PATH, index=False)

print("\nSaved results to:")
print(OUT_PATH)

print("\nPreview:")
print(results_df.head(20).to_string(index=False))
