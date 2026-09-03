# NOTE: Code written by Junwen (Jerry) Zeng
# libraries used for Data Preprocessing
import pandas as pd
import numpy as np

# CSV is from https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-Beginning-2025/5wq4-mkjj/about_data, which should be updated, however, our group used the August 26, 2026 version

# Call train and test datasets
TRAIN_PATH = "train_data_with_nulls_df.csv"
TEST_PATH = "test_data_with_nulls_df.csv"

TRAIN_OUT = "train_data_featured_clean.csv"
TEST_OUT = "test_data_featured_clean.csv"

# Call specific columns/attributes
RIDERSHIP_COL = "total_complex_ridership"
STATION_COL = "station_complex_id"
TIMESTAMP_COL = "transit_timestamp"

# Time/lag to help prediction of ridership from 1 hour, 2, and 3 multiplied by 2
LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
# To predict time/lag 3 hours further multipled by 2
ROLL_WINDOWS = [3, 6, 12, 24, 48, 168]

DROP_FIRST_N = 504   # first 3 weeks


# Functions/Helpers
def first_mode(series):
    modes = series.mode(dropna=True)
    if len(modes) > 0:
        return modes.iloc[0]
    return np.nan

# Find CSV, errors, and sort by month by timestamp
def load_and_prepare(path):
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL])
    df = df.sort_values([STATION_COL, TIMESTAMP_COL]).reset_index(drop=True)

    # month helper from timestamp
    df["month_num"] = df[TIMESTAMP_COL].dt.month

    return df

# Any null values in stations column (there were a few)
def find_all_null_stations(df):
    null_status = df.groupby(STATION_COL)[RIDERSHIP_COL].apply(lambda x: x.isna().all())
    return set(null_status[null_status].index)

# Missing ridership values location, check whether it has any null values and fill in NA/Null values
def fill_missing_ridership(df, label):
    print(f"\nFilling missing ridership for {label}...")

    before_nulls = df[RIDERSHIP_COL].isna().sum()
    print(f"Missing ridership before fill: {before_nulls}")

    # dataset-wide mode fallback
    dataset_mode = first_mode(df[RIDERSHIP_COL])

    if pd.isna(dataset_mode):
        raise ValueError(f"{label} has no non-null ridership values at all.")

    # station-month mode
    station_month_modes = (
        df.groupby([STATION_COL, "month_num"])[RIDERSHIP_COL]
        .transform(first_mode)
    )

    # first fill with station-month mode
    df[RIDERSHIP_COL] = df[RIDERSHIP_COL].fillna(station_month_modes)

    # if a station-month is fully null, fill with dataset-wide mode
    df[RIDERSHIP_COL] = df[RIDERSHIP_COL].fillna(dataset_mode)

    after_nulls = df[RIDERSHIP_COL].isna().sum()
    print(f"Missing ridership after fill: {after_nulls}")

    return df

# Function to predict ridership by looking at past times
def add_lag_rolling_diff_features(df, label):
    print(f"\nCreating lag / rolling / diff features for {label}...")

    df = df.sort_values([STATION_COL, TIMESTAMP_COL]).reset_index(drop=True)

    grouped = df.groupby(STATION_COL)[RIDERSHIP_COL]

    # lag features
    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # rolling features using past values only
    shifted = grouped.shift(1)

    for w in ROLL_WINDOWS:
        df[f"rolling_mean_{w}"] = (
            shifted.groupby(df[STATION_COL])
            .rolling(window=w, min_periods=w)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"rolling_std_{w}"] = (
            shifted.groupby(df[STATION_COL])
            .rolling(window=w, min_periods=w)
            .std()
            .reset_index(level=0, drop=True)
        )

    # diff features
    df["diff_1"] = df["lag_1"] - df["lag_2"]
    df["diff_24"] = df["lag_1"] - df["lag_24"]

    return df

# Rows removed, how many rows left, and comparsion of how many leftover rows
def drop_warmup(df, label):
    print(f"\nDropping first {DROP_FIRST_N} rows per station for {label}...")

    before = len(df)

    def trim_station(group):
        if len(group) <= DROP_FIRST_N:
            return group.iloc[0:0]
        return group.iloc[DROP_FIRST_N:]

    df = (
        df.groupby(STATION_COL, group_keys=False)
        .apply(trim_station)
        .reset_index(drop=True)
    )

    after = len(df)

    print(f"Rows before trim: {before}")
    print(f"Rows after trim:  {after}")
    print(f"Rows removed:     {before - after}")

    return df

# Check if fthere are any null values left (just in case)
def final_null_check(df, label):
    print(f"\n===== NULL CHECK FOR {label} =====")

    null_counts = df.isna().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)

    if len(null_counts) == 0:
        print("No null values left.")
    else:
        print("Columns with null values:")
        print(null_counts)


# Clean and prepare initial CSV files (2020 to 2025 of NYC Subway Ridership)
train_df = load_and_prepare(TRAIN_PATH)
test_df = load_and_prepare(TEST_PATH)

print("Initial train shape:", train_df.shape)
print("Initial test shape:", test_df.shape)

# Remove invalid/null stations
invalid_train_stations = find_all_null_stations(train_df)
invalid_test_stations = find_all_null_stations(test_df)

invalid_stations = invalid_train_stations | invalid_test_stations

print("\nInvalid stations to drop:")
print(sorted(invalid_stations))

train_df = train_df[~train_df[STATION_COL].isin(invalid_stations)].copy()
test_df = test_df[~test_df[STATION_COL].isin(invalid_stations)].copy()

print("Train shape after dropping invalid stations:", train_df.shape)
print("Test shape after dropping invalid stations:", test_df.shape)

# Fill null values in train and test datasets
train_df = fill_missing_ridership(train_df, "TRAIN")
test_df = fill_missing_ridership(test_df, "TEST")

# Add lag, rolling, and diff(erence by past times) columns
train_df = add_lag_rolling_diff_features(train_df, "TRAIN")
test_df = add_lag_rolling_diff_features(test_df, "TEST")

# Remove first 3 weeks to make sure datasets are consistent (this decision was made to remove possible volatility)
train_df = drop_warmup(train_df, "TRAIN")
test_df = drop_warmup(test_df, "TEST")

# Remove Helper column
train_df = train_df.drop(columns=["month_num"], errors="ignore")
test_df = test_df.drop(columns=["month_num"], errors="ignore")

# Make sure null values are gone
final_null_check(train_df, "TRAIN")
final_null_check(test_df, "TEST")

# Quick check of train and test dataset
print("\nTRAIN preview:")
print(train_df.head())

print("\nTEST preview:")
print(test_df.head())

# Save and confirm output
train_df.to_csv(TRAIN_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)

print("\nSaved:")
print(TRAIN_OUT)
print(TEST_OUT)
