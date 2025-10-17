# calculate_load_per_minute.py

import pandas as pd


def calculate_load_per_minute(csv_path="data/soccer_data_cleaned.csv"):
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # Map intensity from action values
    action_to_intensity = {"Train_Low": 0.3, "Train_Med": 0.6, "Train_High": 0.9}
    df["intensity"] = df["action"].map(action_to_intensity)

    # Filter to only rows with known training intensities
    train_df = df.dropna(subset=["intensity", "load", "total_duration"])

    # Aggregate total load and duration per intensity value
    stats = (
        train_df
        .groupby("intensity")
        .agg(total_load=("load", "sum"), total_duration=("total_duration", "sum"))
        .reset_index()
    )

    # Create dictionary mapping intensity -> load per minute
    load_per_minute = {
        round(row.intensity, 2): row.total_load / row.total_duration
        for _, row in stats.iterrows()
    }

    return load_per_minute


if __name__ == "__main__":
    load_map = calculate_load_per_minute()
    for intensity, rate in load_map.items():
        print(f"Intensity {intensity}: Load/min = {rate:.4f}")
