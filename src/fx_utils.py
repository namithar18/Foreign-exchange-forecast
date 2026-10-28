from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PAIRS = ["GBP_USD", "USD_EUR", "USD_JPY"]

def load_forex_data(path=None):
    path = Path(path) if path else DATA_DIR / "daily_forex_rates.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def build_pair_rates(forex_df):
    # The source data expresses each currency as units per EUR.
    target = forex_df[
        forex_df["base_currency"].eq("EUR")
        & forex_df["currency"].isin(["GBP", "USD", "JPY"])
    ].copy()
    if target.duplicated(["date", "currency"]).any():
        raise ValueError("Duplicate date/currency records found; inspect the source data.")
    wide = target.pivot(index="date", columns="currency", values="exchange_rate").sort_index()
    complete = wide.dropna(subset=["GBP", "USD", "JPY"]).copy()
    complete["GBP_USD"] = complete["USD"] / complete["GBP"]
    complete["USD_EUR"] = 1 / complete["USD"]
    complete["USD_JPY"] = complete["JPY"] / complete["USD"]
    return complete[PAIRS].sort_index()

def create_fx_features(data, column):
    result = pd.DataFrame(index=data.index)
    price = data[column]
    log_return = np.log(price / price.shift(1))
    log_price = np.log(price)

    result[f"{column}_log_return"] = log_return
    result[f"{column}_prev_log_return"] = log_return.shift(1)
    for window in [5, 10, 20, 60, 120]:
        result[f"{column}_ret_ma{window}"] = log_return.rolling(window).mean()

    result[f"{column}_log"] = log_price
    result[f"{column}_log_sq"] = log_price ** 2
    result[f"{column}_log_cube"] = log_price ** 3
    for window in [5, 10, 20, 60, 220]:
        result[f"{column}_log_ma{window}"] = log_price.rolling(window).mean()

    # Stationary replacements for the price-level features: they describe the price
    # relative to its own recent history, so their range does not drift over time.
    for window in [5, 10, 20, 60, 220]:
        result[f"{column}_dev_ma{window}"] = log_price - result[f"{column}_log_ma{window}"]
    result[f"{column}_vol20"] = log_return.rolling(20).std()
    result[f"{column}_vol60"] = log_return.rolling(60).std()
    rolling_220 = log_price.rolling(220)
    result[f"{column}_z220"] = (log_price - rolling_220.mean()) / rolling_220.std()
    return result

def build_features_and_targets(pair_df):
    feature_frames = [create_fx_features(pair_df, pair) for pair in PAIRS]
    features_df = pd.concat(feature_frames, axis=1)
    target_frames = {}
    for pair in PAIRS:
        log_return = np.log(pair_df[pair] / pair_df[pair].shift(1))
        target_frames[pair] = pd.DataFrame(index=pair_df.index)
        target_frames[pair][f"{pair}_target_return"] = log_return.shift(-1)
        target_frames[pair][f"{pair}_target_direction"] = (
            log_return.shift(-1) > 0
        ).astype("Int64")
    targets_df = pd.concat(target_frames.values(), axis=1)

    model_datasets = {}
    for pair in PAIRS:
        feature_cols = [c for c in features_df.columns if c.startswith(f"{pair}_")]
        target_cols = [f"{pair}_target_return", f"{pair}_target_direction"]
        model_df = pd.concat([features_df[feature_cols], targets_df[target_cols]], axis=1).dropna().copy()
        model_df[f"{pair}_target_direction"] = model_df[f"{pair}_target_direction"].astype(int)
        model_datasets[pair] = model_df
    return features_df, targets_df, model_datasets

def chronological_split(data):
    n = len(data)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return {
        "train": data.iloc[:train_end].copy(),
        "val": data.iloc[train_end:val_end].copy(),
        "test": data.iloc[val_end:].copy()
    }

def get_7_features(pair):
    return [
        f"{pair}_log_return", f"{pair}_prev_log_return",
        f"{pair}_ret_ma5", f"{pair}_ret_ma10", f"{pair}_ret_ma20",
        f"{pair}_ret_ma60", f"{pair}_ret_ma120"
    ]

def get_15_paper_features(pair):
    # The paper's 15 features: the 7 plus raw log-price level terms.
    return get_7_features(pair) + [
        f"{pair}_log", f"{pair}_log_sq", f"{pair}_log_cube",
        f"{pair}_log_ma5", f"{pair}_log_ma10", f"{pair}_log_ma20",
        f"{pair}_log_ma60", f"{pair}_log_ma220"
    ]

def get_15_stationary_features(pair):
    # Same size as the paper's set, with each level term replaced by a relative one.
    return get_7_features(pair) + [
        f"{pair}_dev_ma5", f"{pair}_dev_ma10", f"{pair}_dev_ma20",
        f"{pair}_dev_ma60", f"{pair}_dev_ma220",
        f"{pair}_vol20", f"{pair}_vol60", f"{pair}_z220"
    ]

def get_feature_sets(pair):
    return {
        "7": get_7_features(pair),
        "15-paper": get_15_paper_features(pair),
        "15-stationary": get_15_stationary_features(pair)
    }
