import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# CONFIG
TRAIN_PATH = "train_data_featured_clean.csv"
TEST_PATH = "test_data_featured_clean.csv"
OUT_PATH = "all_station_lasso_results.csv"
ADJ_PATH = "SC_adjacency_matrix.csv"

RIDERSHIP_COL = "total_complex_ridership"
STATION_COL = "station_complex_id"
TIMESTAMP_COL = "transit_timestamp"
TARGET_COL = "target_3h_ahead"

SCA_LAGS = [1, 3, 6, 12, 24]

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.nan if mask.sum() == 0 else np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100
    return rmse, r2, mae, mape

# LOAD DATA
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df[TIMESTAMP_COL] = pd.to_datetime(train_df[TIMESTAMP_COL])
test_df[TIMESTAMP_COL] = pd.to_datetime(test_df[TIMESTAMP_COL])

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

# adjacency
adj = pd.read_csv(ADJ_PATH, index_col=0)
adj.index = adj.index.astype(int)
adj.columns = adj.columns.astype(int)

# precompute lag pivots
lag_pivots = {
    lag: full_df.pivot_table(index=TIMESTAMP_COL, columns=STATION_COL, values=f"lag_{lag}")
    for lag in SCA_LAGS
}

stations = sorted(set(train_df[STATION_COL]) & set(test_df[STATION_COL]) & set(adj.index))

results = []

for i, station_id in enumerate(stations, 1):
    print(f"\n[{i}/{len(stations)}] Lasso station {station_id}")

    station_df = full_df[full_df[STATION_COL] == station_id].copy()

    neighbors = adj.columns[adj.loc[station_id] == 1].tolist()
    print(f"Neighbors: {neighbors}")

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

    # FEATURES
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

    station_train = station_df[station_df["dataset_split"] == "train"].copy()
    station_test = station_df[station_df["dataset_split"] == "test"].copy()

    if len(station_train) == 0 or len(station_test) == 0:
        print(f"Skipping station {station_id}: empty train/test.")
        continue

    X_train = station_train[feature_cols]
    y_train = station_train[TARGET_COL]

    X_test = station_test[feature_cols]
    y_test = station_test[TARGET_COL]

    model = Lasso(alpha=0.1, max_iter=10000)
    model.fit(X_train, y_train)

    train_pred = np.clip(model.predict(X_train),0,None)
    test_pred = np.clip(model.predict(X_test),0,None)

    tr = evaluate(y_train, train_pred)
    te = evaluate(y_test, test_pred)

    print(f"Test R²: {te[1]:.4f}")

    results.append({
        "station_complex_id": station_id,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),

        "Train RMSE": tr[0],
        "Train R^2": tr[1],
        "Train MAE": tr[2],
        "Train MAPE": tr[3],

        "Test RMSE": te[0],
        "Test R^2": te[1],
        "Test MAE": te[2],
        "Test MAPE": te[3],
    })

pd.DataFrame(results).to_csv(OUT_PATH, index=False)