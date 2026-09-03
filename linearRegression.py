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
ADJ_PATH = "SC_adjacency_matrix.csv"

OUT_PATH = "all_station_linear_model_results.csv"

RIDERSHIP_COL = "total_complex_ridership"
STATION_COL = "station_complex_id"
TIMESTAMP_COL = "transit_timestamp"
TARGET_COL = "target_3h_ahead"

SCA_LAGS = [1, 3, 6, 12, 24]

# ============================================================
# METRIC FUNCTION
# ============================================================
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Avoid division by zero for MAPE
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
# LOAD ADJACENCY MATRIX
# ============================================================
adj = pd.read_csv(ADJ_PATH, index_col=0)
adj.index = pd.to_numeric(adj.index, errors="coerce")
adj.columns = pd.to_numeric(adj.columns, errors="coerce")

adj = adj.loc[adj.index.notna(), adj.columns.notna()].copy()
adj.index = adj.index.astype(int)
adj.columns = adj.columns.astype(int)

# ============================================================
# PRECOMPUTE LAG PIVOTS FOR SCA FEATURES
# ============================================================
lag_pivots = {}

for lag in SCA_LAGS:
    lag_col = f"lag_{lag}"

    if lag_col not in full_df.columns:
        raise ValueError(f"Missing required lag column: {lag_col}")

    lag_pivots[lag] = full_df.pivot_table(
        index=TIMESTAMP_COL,
        columns=STATION_COL,
        values=lag_col
    )

print("Finished creating lag pivot tables.")

# ============================================================
# ALL STATIONS
# ============================================================
train_stations = set(full_df.loc[full_df["dataset_split"] == "train", STATION_COL].unique())
test_stations = set(full_df.loc[full_df["dataset_split"] == "test", STATION_COL].unique())
adj_stations = set(adj.index)

stations = sorted(list(train_stations & test_stations & adj_stations))

print(f"Total stations to train: {len(stations)}")

# ============================================================
# MODEL LOOP
# ============================================================
results = []

for i, station_id in enumerate(stations, 1):
    print(f"\n{'='*80}")
    print(f"[{i}/{len(stations)}] Training station {station_id}")
    print(f"{'='*80}")
    start_time = time.time()

    station_df = full_df[full_df[STATION_COL] == station_id].copy()

    neighbors = adj.columns[adj.loc[station_id] == 1].tolist()

    # Add neighbor-specific SCA features
    for lag in SCA_LAGS:
        pivot = lag_pivots[lag]

        existing_neighbors = [n for n in neighbors if n in pivot.columns]

        if len(existing_neighbors) == 0:
            continue

        neighbor_features = pivot[existing_neighbors].copy()

        rename_map = {
            n: f"sca_lag_{lag}_station{n}"
            for n in existing_neighbors
        }

        neighbor_features = neighbor_features.rename(columns=rename_map)
        neighbor_features = neighbor_features.reset_index()

        station_df = station_df.merge(
            neighbor_features,
            on=TIMESTAMP_COL,
            how="left"
        )

    # -----------------------------
    # FEATURE SELECTION
    # -----------------------------
    drop_feature_cols = [
        TIMESTAMP_COL,
        STATION_COL,
        RIDERSHIP_COL,
        TARGET_COL,
        "dataset_split",

        # remove year features
        "is_year_2023",
        "is_year_2024",
        "is_year_2025",

        # remove holiday
        "isHoliday",
    ]

    feature_cols = [
        c for c in station_df.columns
        if c not in drop_feature_cols
    ]

    # Keep only numeric features
    feature_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(station_df[c])
    ]

    # Remove duplicate columns
    feature_cols = list(dict.fromkeys(feature_cols))

    station_df = station_df.dropna(subset=[TARGET_COL]).copy()

    station_train = station_df[station_df["dataset_split"] == "train"].copy()
    station_test = station_df[station_df["dataset_split"] == "test"].copy()

    if len(station_train) == 0 or len(station_test) == 0:
        print(f"Skipping station {station_id}: empty train/test.")
        continue

    # Fill missing feature values using training-set mode

    X_train = station_train[feature_cols]
    y_train = station_train[TARGET_COL]

    X_test = station_test[feature_cols]
    y_test = station_test[TARGET_COL]

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} | Features: {len(feature_cols)}")
    print("Feature sample:", feature_cols[:15])

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

results_df = results_df.sort_values("Test R^2", ascending=False)

results_df.to_csv(OUT_PATH, index=False)

print("\nSaved results to:")
print(OUT_PATH)

print("\nPreview:")
print(results_df.head(20).to_string(index=False))