import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================
# CONFIG
# ============================================================
TRAIN_PATH = "train_data_featured_clean.csv"
TEST_PATH = "test_data_featured_clean.csv"
ADJ_PATH = "SC_adjacency_matrix.csv"

OUT_PATH = "all_station_ridge_model_results.csv"

RIDERSHIP_COL = "total_complex_ridership"
STATION_COL = "station_complex_id"
TIMESTAMP_COL = "transit_timestamp"
TARGET_COL = "target_3h_ahead"

SCA_LAGS = [1, 3, 6, 12, 24]

# ============================================================
# METRICS
# ============================================================
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    mask = y_true != 0
    mape = np.nan if mask.sum() == 0 else np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

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
# LOAD ADJ MATRIX
# ============================================================
adj = pd.read_csv(ADJ_PATH, index_col=0)
adj.index = adj.index.astype(int)
adj.columns = adj.columns.astype(int)

# ============================================================
# PRECOMPUTE SCA PIVOTS
# ============================================================
lag_pivots = {}

for lag in SCA_LAGS:
    lag_col = f"lag_{lag}"
    lag_pivots[lag] = full_df.pivot_table(
        index=TIMESTAMP_COL,
        columns=STATION_COL,
        values=lag_col
    )

print("Lag pivots ready.")

# ============================================================
# STATION LIST
# ============================================================
stations = sorted(
    set(train_df[STATION_COL]) &
    set(test_df[STATION_COL]) &
    set(adj.index)
)

print(f"Total stations: {len(stations)}")

# ============================================================
# MODEL LOOP
# ============================================================
results = []

for i, station_id in enumerate(stations, 1):
    print(f"\n{'='*80}")
    print(f"[{i}/{len(stations)}] Ridge model for station {station_id}")
    print(f"{'='*80}")

    station_df = full_df[full_df[STATION_COL] == station_id].copy()
    neighbors = adj.columns[adj.loc[station_id] == 1].tolist()

    # Add SCA features
    for lag in SCA_LAGS:
        pivot = lag_pivots[lag]
        valid_neighbors = [n for n in neighbors if n in pivot.columns]

        if len(valid_neighbors) == 0:
            continue

        temp = pivot[valid_neighbors].copy()
        temp = temp.rename(columns={
            n: f"sca_lag_{lag}_station{n}" for n in valid_neighbors
        }).reset_index()

        station_df = station_df.merge(temp, on=TIMESTAMP_COL, how="left")

    # ========================================================
    # FEATURES (same as before, no year features)
    # ========================================================
    # --------------------------------------------------------
    # FEATURE SELECTION: use all valid numeric features
    # --------------------------------------------------------
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

    train = station_df[station_df["dataset_split"] == "train"].copy()
    test = station_df[station_df["dataset_split"] == "test"].copy()

    if len(train) == 0 or len(test) == 0:
        continue

    # Fill missing

    X_train = train[feature_cols]
    y_train = train[TARGET_COL]

    X_test = test[feature_cols]
    y_test = test[TARGET_COL]

    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feature_cols)}")

    # ========================================================
    # RIDGE MODEL
    # ========================================================
    model = Ridge(alpha=10.0)   # you can tune this later
    model.fit(X_train, y_train)

    train_preds = np.clip(model.predict(X_train), 0, None)
    test_preds = np.clip(model.predict(X_test), 0, None)

    train_rmse, train_r2, train_mae, train_mape = evaluate(y_train.values, train_preds)
    test_rmse, test_r2, test_mae, test_mape = evaluate(y_test.values, test_preds)

    print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")

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
# SAVE
# ============================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Test R^2", ascending=False)

results_df.to_csv(OUT_PATH, index=False)

print("\nSaved Ridge results to:")
print(OUT_PATH)

print(results_df.head(20).to_string(index=False))