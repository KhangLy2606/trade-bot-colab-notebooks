# Multi-Timeframe Return Training — Trade Bot
#
# Reusable training script for 5m, 15m, and 1h candle experiments.
# Targets: next-bar return direction (classification) and magnitude (regression).
# Validation: timeframe-aware walk-forward CV with random-walk/persistence baselines.
#
# Usage:
#   python train_regime.py                      # run all 3 timeframes
#   python train_regime.py --timeframe 5m       # run single timeframe
#   python train_regime.py --symbols SPY QQQ    # custom symbol list

import json
import pickle
import warnings
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import s3fs

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, mean_squared_error,
    classification_report,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings('ignore', category=UserWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'MSFT', 'NVDA', 'AMZN',
                   'META', 'GOOGL', 'TSLA', 'AMD']
DEFAULT_TIMEFRAMES = ['5m', '15m', '1h']
DEFAULT_START_DATE = '2024-06-01'
DEFAULT_END_DATE = '2025-12-31'
SAVE_DIR = './trained-models'

# Timeframe config: bars_per_minute, annualization factor, and wall-clock conversions
TIMEFRAME_CONFIG = {
    '1m': {'minutes': 1, 'bars_per_day': 390, 'bars_per_hour': 60},
    '5m': {'minutes': 5, 'bars_per_day': 78, 'bars_per_hour': 12},
    '15m': {'minutes': 15, 'bars_per_day': 26, 'bars_per_hour': 4},
    '1h': {'minutes': 60, 'bars_per_day': 7, 'bars_per_hour': 1},  # ~6.5h/day ≈ 7
}

# Wall-clock durations for CV (applied uniformly across timeframes)
CV_PURGE_HOURS = 1        # 1 hour purge gap between train/test
CV_TEST_DAYS = 5          # 1 trading week test window
CV_N_SPLITS = 8

# ── Load credentials ───────────────────────────────────────────────────────────
def load_credentials():
    access_key = os.getenv('MASSIVE_S3_ACCESS_KEY')
    secret_key = os.getenv('MASSIVE_S3_SECRET_KEY')

    if not access_key:
        try:
            from dotenv import load_dotenv
            for d in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
                env_file = d / '.env'
                if env_file.exists():
                    load_dotenv(env_file)
                    access_key = os.getenv('MASSIVE_S3_ACCESS_KEY')
                    secret_key = os.getenv('MASSIVE_S3_SECRET_KEY')
                    if access_key:
                        break
        except ImportError:
            pass

    if not access_key:
        access_key = 'YOUR_ACCESS_KEY_HERE'
        secret_key = 'YOUR_SECRET_KEY_HERE'
        print('WARNING: Using placeholder credentials')

    return {
        'endpoint': os.getenv('MASSIVE_S3_ENDPOINT', 'https://files.massive.com'),
        'bucket': os.getenv('MASSIVE_S3_BUCKET', 'flatfiles'),
        'access_key': access_key,
        'secret_key': secret_key,
    }


# ── Data loader ────────────────────────────────────────────────────────────────
KEY_FORMATS = [
    'us_stocks_sip/minute_aggs_v1/{year}/{month:02d}/{date}.csv.gz',
    'us_stocks_sip/minute-aggregates/{year}/{month:02d}/{date}.csv.gz',
]

_fs_cache = {}


def _get_fs(cfg: dict) -> s3fs.S3FileSystem:
    key = cfg['endpoint']
    if key not in _fs_cache:
        _fs_cache[key] = s3fs.S3FileSystem(
            key=cfg['access_key'],
            secret=cfg['secret_key'],
            client_kwargs={'endpoint_url': cfg['endpoint']},
        )
    return _fs_cache[key]


def _load_single_file(args):
    """Load a single date's file from S3 for a given symbol."""
    symbol, date_str, cfg = args
    fs = _get_fs(cfg)
    symbol = symbol.upper()
    for fmt in KEY_FORMATS:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        path = f"{cfg['bucket']}/{fmt.format(year=dt.year, month=dt.month, date=date_str)}"
        try:
            with fs.open(path, 'rb') as f:
                df = pd.read_csv(f, compression='gzip')
                df.columns = [c.lower() for c in df.columns]
                sym_col = 'ticker' if 'ticker' in df.columns else 'symbol'
                filtered = df[df[sym_col].str.upper() == symbol].copy()
                if not filtered.empty:
                    return 'success', date_str, filtered
        except FileNotFoundError:
            continue
        except PermissionError:
            return 'forbidden', date_str, None
        except Exception:
            return 'error', date_str, None
    return 'notfound', date_str, None


def load_minute_bars(symbol: str, start_date: str, end_date: str, cfg: dict,
                     max_workers: int = 15) -> pd.DataFrame:
    """Load 1-minute bars from S3 for a single symbol."""
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    trading_days = []
    while current <= end:
        if current.weekday() < 5:
            trading_days.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)

    print(f'  Loading {symbol} ({len(trading_days)} trading days)...')
    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_load_single_file, (symbol, d, cfg)): d
            for d in trading_days
        }
        for future in as_completed(futures):
            status, _, df = future.result()
            if status == 'success' and df is not None:
                dfs.append(df)

    if not dfs:
        print(f'  WARNING: No data for {symbol}')
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    print(f'  {symbol}: {len(result):,} 1m bars')
    return result


def load_multi_symbol_minute_bars(symbols: list[str], start_date: str,
                                  end_date: str, cfg: dict) -> pd.DataFrame:
    """Load 1-minute bars for multiple symbols, tagged with symbol column."""
    all_dfs = []
    for sym in symbols:
        df = load_minute_bars(sym, start_date, end_date, cfg)
        if not df.empty:
            sym_col = 'ticker' if 'ticker' in df.columns else 'symbol'
            df['_symbol'] = df[sym_col].str.upper()
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f'No data found for any symbol in {symbols}')

    result = pd.concat(all_dfs, ignore_index=True)
    print(f'Total: {len(result):,} 1m bars across {len(all_dfs)} symbols')
    return result


# ── Resampling ────────────────────────────────────────────────────────────────
def resample_to_timeframe(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1-minute bars to a higher timeframe.

    Parameters
    ----------
    df_1m : DataFrame with columns: open, high, low, close, volume, _symbol, window_start
    timeframe : one of '1m', '5m', '15m', '1h'

    Returns
    -------
    DataFrame with OHLCV columns resampled, plus _symbol and _timeframe metadata.
    """
    if timeframe == '1m':
        out = df_1m.copy()
        out['_timeframe'] = '1m'
        return out

    cfg = TIMEFRAME_CONFIG[timeframe]
    minutes = cfg['minutes']

    # Parse timestamp
    if 'window_start' in df_1m.columns:
        # Massive uses nanosecond epoch
        ts = pd.to_datetime(df_1m['window_start'], unit='ns')
    elif 'timestamp' in df_1m.columns:
        ts = pd.to_datetime(df_1m['timestamp'])
    else:
        raise ValueError('No timestamp column found (expected window_start or timestamp)')

    df = df_1m.copy()
    df['_ts'] = ts
    df = df.sort_values(['_symbol', '_ts']).reset_index(drop=True)

    resampled_parts = []
    freq = f'{minutes}min'

    for symbol, group in df.groupby('_symbol'):
        g = group.set_index('_ts').sort_index()
        # origin='start' anchors bins to the first timestamp in the series
        # (typically 09:30 ET) rather than midnight, so hourly/multi-minute
        # candles align to market open instead of midnight-aligned bins.
        ohlcv = g[['open', 'high', 'low', 'close', 'volume']].resample(
            freq, origin='start'
        ).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna(subset=['open'])
        ohlcv['_symbol'] = symbol
        ohlcv['_timeframe'] = timeframe
        ohlcv = ohlcv.reset_index().rename(columns={'_ts': 'datetime'})
        resampled_parts.append(ohlcv)

    result = pd.concat(resampled_parts, ignore_index=True)
    result = result.sort_values(['_symbol', 'datetime']).reset_index(drop=True)
    print(f'  Resampled to {timeframe}: {len(result):,} bars')
    return result


# ── Timeframe-aware features ──────────────────────────────────────────────────
def _distinct_windows(
    wall_clock_minutes: list,
    minutes_per_bar: int,
    min_bars: int = 2,
) -> tuple:
    """Convert wall-clock durations to distinct bar counts with honest labels.

    Guarantees:
    - Each window is at least min_bars (need ≥2 to compute rolling std).
    - All bar counts are distinct (deduplicates by incrementing).
    - Labels reflect the *actual* wall-clock span after rounding, not the
      requested span — so 25min for a 60min bar reports "120min", not "25min".

    Returns
    -------
    (windows_bars, label_strings) : tuple of lists
    """
    bars_list = []
    labels_list = []
    seen: set = set()
    for wm in wall_clock_minutes:
        b = max(min_bars, round(wm / minutes_per_bar))
        while b in seen:
            b += 1
        seen.add(b)
        actual_wm = b * minutes_per_bar
        bars_list.append(b)
        labels_list.append(f"{actual_wm}min")
    return bars_list, labels_list


def _hurst_rs(series: np.ndarray) -> float:
    """Hurst exponent via R/S analysis — pure numpy."""
    n = len(series)
    if n < 20:
        return 0.5
    mean = series.mean()
    deviation = np.cumsum(series - mean)
    r = deviation.max() - deviation.min()
    s = series.std(ddof=1)
    if s == 0:
        return 0.5
    return float(np.clip(np.log(r / s) / np.log(n), 0.0, 1.0))


def compute_features_timeframe_aware(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Compute features with window sizes scaled to wall-clock time.

    Window semantics:
    - 'short':  ~25 min wall clock
    - 'medium': ~100 min wall clock (~1.5 hours)
    - 'long':   ~300 min wall clock (~5 hours)

    These translate to different bar counts depending on timeframe.
    """
    cfg = TIMEFRAME_CONFIG[timeframe]
    minutes_per_bar = cfg['minutes']

    # Convert wall-clock targets to distinct bar counts with honest labels.
    # For 1h bars: 25min→2bars="120min", 100min→3bars="180min", 300min→5bars="300min"
    # For 5m bars: 25min→5bars="25min",  100min→20bars="100min", 300min→60bars="300min"
    windows_minutes = [25, 100, 300]
    windows, w_labels = _distinct_windows(windows_minutes, minutes_per_bar)

    feat = pd.DataFrame(index=df.index)
    log_ret = np.log(df['close'] / df['close'].shift(1))

    for w, label in zip(windows, w_labels):
        feat[f'realized_vol_{label}'] = (
            log_ret.rolling(w, min_periods=w).std() * np.sqrt(252 * cfg['bars_per_day'])
        )
        feat[f'log_return_{label}'] = log_ret.rolling(w, min_periods=1).sum()

    # RSI(14 bars) — period stays fixed in bars for RSI convention
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=14).mean()
    rsi = 100.0 - (100.0 / (1.0 + gain / loss.replace(0, np.nan)))

    feat['rsi_14'] = rsi
    # Rolling z-score window: ~5 hours of bars
    zscore_window = max(10, round(300 / minutes_per_bar))
    rsi_mean = rsi.rolling(zscore_window, min_periods=max(5, zscore_window // 3)).mean()
    rsi_std = rsi.rolling(zscore_window, min_periods=max(5, zscore_window // 3)).std()
    feat['rsi_14_zscore'] = (rsi - rsi_mean) / rsi_std.replace(0, np.nan)
    feat['rsi_14_dev50'] = rsi - 50.0

    # Volume z-score
    vol_window = max(10, round(300 / minutes_per_bar))
    vm = df['volume'].rolling(vol_window, min_periods=max(5, vol_window // 3)).mean()
    vs = df['volume'].rolling(vol_window, min_periods=max(5, vol_window // 3)).std()
    feat['volume_zscore'] = (df['volume'] - vm) / vs.replace(0, np.nan)

    # Price range normalized
    range_window = max(5, round(100 / minutes_per_bar))
    bar_range = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
    feat['price_range_norm'] = bar_range.rolling(range_window, min_periods=max(3, range_window // 4)).mean()

    # Hurst exponent: ~100 min of bars
    hurst_window = max(20, round(100 / minutes_per_bar))
    log_price = np.log(df['close'].replace(0, np.nan).ffill().values)
    hurst_vals = np.full(len(df), np.nan)
    for i in range(hurst_window, len(log_price)):
        seg = log_price[i - hurst_window: i]
        if not np.isnan(seg).any():
            try:
                hurst_vals[i] = _hurst_rs(seg)
            except Exception:
                pass
    feat['hurst'] = hurst_vals

    return feat


# ── Next-return targets ──────────────────────────────────────────────────────
def compute_return_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute next-bar return targets. Label is always at t+1.

    Returns DataFrame with:
    - next_log_return: float (regression target)
    - next_direction:  int 1 if next return > 0, else 0 (classification target)
    """
    log_ret = np.log(df['close'] / df['close'].shift(1))
    next_ret = log_ret.shift(-1)  # shift(-1) = next bar's return

    targets = pd.DataFrame(index=df.index)
    targets['next_log_return'] = next_ret
    targets['next_direction'] = (next_ret > 0).astype(int)
    return targets


# ── Timeframe-aware walk-forward splitter ────────────────────────────────────
def make_timeframe_cv(timeframe: str, n_splits: int = CV_N_SPLITS) -> TimeSeriesSplit:
    """Create a TimeSeriesSplit with gap and test_size scaled to wall-clock time.

    Gap = CV_PURGE_HOURS hours (prevents feature leakage)
    Test = CV_TEST_DAYS trading days
    """
    cfg = TIMEFRAME_CONFIG[timeframe]
    gap_bars = CV_PURGE_HOURS * cfg['bars_per_hour']
    test_bars = CV_TEST_DAYS * cfg['bars_per_day']
    print(f'  CV for {timeframe}: gap={gap_bars} bars ({CV_PURGE_HOURS}h), '
          f'test_size={test_bars} bars ({CV_TEST_DAYS}d), {n_splits} splits')
    return TimeSeriesSplit(n_splits=n_splits, gap=gap_bars, test_size=test_bars)


# ── Baselines ────────────────────────────────────────────────────────────────
def evaluate_baselines(y_true_direction: np.ndarray, y_true_return: np.ndarray,
                       last_return: np.ndarray) -> dict:
    """Evaluate naive baselines.

    Persistence baseline:
    - Direction: predict sign(last return)
    - Magnitude: predict last return value

    Majority-class baseline (labelled accurately — not a stochastic random walk):
    - Direction: always predict the majority class (constant predictor)
    - Magnitude: predict zero return
    """
    # Persistence: direction = sign of last return
    persistence_dir = (last_return > 0).astype(int)
    persistence_acc = accuracy_score(y_true_direction, persistence_dir)
    persistence_f1 = f1_score(y_true_direction, persistence_dir, average='weighted')

    # Persistence: magnitude = last return
    persistence_mse = mean_squared_error(y_true_return, last_return)

    # Majority-class constant predictor (not a true random walk)
    majority_class = int(y_true_direction.mean() > 0.5)
    majority_dir = np.full_like(y_true_direction, majority_class)
    majority_acc = accuracy_score(y_true_direction, majority_dir)
    majority_f1 = f1_score(y_true_direction, majority_dir, average='weighted')

    # Zero-return predictor for magnitude
    zero_mse = mean_squared_error(y_true_return, np.zeros_like(y_true_return))

    return {
        'persistence_accuracy': persistence_acc,
        'persistence_f1': persistence_f1,
        'persistence_mse': persistence_mse,
        'majority_class_accuracy': majority_acc,
        'majority_class_f1': majority_f1,
        'zero_return_mse': zero_mse,
        # Keep legacy keys to avoid breaking notebook result table
        'random_walk_accuracy': majority_acc,
        'random_walk_f1': majority_f1,
        'random_walk_mse': zero_mse,
    }


# ── Per-symbol feature + target builder ──────────────────────────────────────
def build_dataset_for_timeframe(df_1m: pd.DataFrame, timeframe: str,
                                min_rows: int = 100_000) -> dict:
    """Build features, targets, and metadata for a single timeframe.

    Parameters
    ----------
    df_1m : DataFrame with 1m bars, must have _symbol column
    timeframe : '5m', '15m', or '1h'
    min_rows : target minimum row count (informational warning if not met)

    Returns
    -------
    dict with keys: X, y_direction, y_return, last_return, symbols, timeframe, n_rows
    """
    print(f'\nBuilding {timeframe} dataset...')
    resampled = resample_to_timeframe(df_1m, timeframe)

    all_X, all_y_dir, all_y_ret, all_last_ret = [], [], [], []
    symbols_used = []

    for symbol, group in resampled.groupby('_symbol'):
        group = group.sort_values('datetime').reset_index(drop=True)
        if len(group) < 200:
            print(f'  Skipping {symbol}: only {len(group)} bars')
            continue

        feat = compute_features_timeframe_aware(group, timeframe)
        targets = compute_return_targets(group)

        # last return for persistence baseline
        log_ret = np.log(group['close'] / group['close'].shift(1))

        # Combine and drop NaN
        combined = pd.concat([feat, targets, log_ret.rename('last_return')], axis=1)
        valid = combined.notna().all(axis=1)
        combined = combined[valid]

        if len(combined) < 50:
            continue

        feature_cols = [c for c in feat.columns]
        all_X.append(combined[feature_cols])
        all_y_dir.append(combined['next_direction'])
        all_y_ret.append(combined['next_log_return'])
        all_last_ret.append(combined['last_return'])
        symbols_used.append(symbol)

    if not all_X:
        raise ValueError(f'No valid data for timeframe {timeframe}')

    X = pd.concat(all_X, ignore_index=True)
    y_dir = pd.concat(all_y_dir, ignore_index=True)
    y_ret = pd.concat(all_y_ret, ignore_index=True)
    last_ret = pd.concat(all_last_ret, ignore_index=True)

    n_rows = len(X)
    if n_rows < min_rows:
        print(f'  WARNING: {timeframe} has {n_rows:,} rows (target: {min_rows:,})')
    else:
        print(f'  {timeframe}: {n_rows:,} rows from {len(symbols_used)} symbols')

    return {
        'X': X,
        'y_direction': y_dir,
        'y_return': y_ret,
        'last_return': last_ret,
        'symbols': symbols_used,
        'timeframe': timeframe,
        'n_rows': n_rows,
        'feature_columns': list(X.columns),
    }


# ── Training ─────────────────────────────────────────────────────────────────
def train_direction_model(dataset: dict) -> dict:
    """Train a next-bar direction classifier for a single timeframe.

    Returns dict with model, CV metrics, baseline comparison, and metadata.
    """
    X = dataset['X']
    y = dataset['y_direction']
    timeframe = dataset['timeframe']

    print(f'\n{"="*60}')
    print(f'Training DIRECTION model — {timeframe} ({len(X):,} rows)')
    print(f'{"="*60}')

    tscv = make_timeframe_cv(timeframe)

    pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('clf', LGBMClassifier(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.1,
            min_child_samples=20,
            class_weight='balanced',
            verbose=-1,
        )),
    ])

    # Walk-forward CV
    print('  Running walk-forward CV...')
    cv_results = cross_validate(
        pipe, X, y, cv=tscv,
        scoring=['accuracy', 'f1_weighted', 'roc_auc'],
        return_estimator=True,
    )

    mean_acc = cv_results['test_accuracy'].mean()
    mean_f1 = cv_results['test_f1_weighted'].mean()
    mean_auc = cv_results['test_roc_auc'].mean()

    print(f'  CV Accuracy:  {mean_acc:.4f}')
    print(f'  CV F1:        {mean_f1:.4f}')
    print(f'  CV ROC-AUC:   {mean_auc:.4f}')

    # Baseline evaluation: compute persistence on each CV fold and average so
    # lift is apples-to-apples with the mean CV model metrics.
    fold_persistence_accs = []
    fold_persistence_f1s = []
    for _, test_idx in tscv.split(X):
        y_fold = y.iloc[test_idx].values
        last_ret_fold = dataset['last_return'].iloc[test_idx].values
        pers_dir = (last_ret_fold > 0).astype(int)
        fold_persistence_accs.append(accuracy_score(y_fold, pers_dir))
        fold_persistence_f1s.append(f1_score(y_fold, pers_dir, average='weighted'))

    mean_persistence_acc = float(np.mean(fold_persistence_accs))
    mean_persistence_f1 = float(np.mean(fold_persistence_f1s))

    # Use last fold for magnitude metrics (need y_return; these are not CV-averaged)
    last_fold_test_idx = list(tscv.split(X))[-1][1]
    y_test = y.iloc[last_fold_test_idx].values
    ret_test = dataset['y_return'].iloc[last_fold_test_idx].values
    last_ret_test = dataset['last_return'].iloc[last_fold_test_idx].values

    baselines = evaluate_baselines(y_test, ret_test, last_ret_test)
    # Replace direction metrics with per-fold averages for honest lift reporting
    baselines['persistence_accuracy'] = mean_persistence_acc
    baselines['persistence_f1'] = mean_persistence_f1

    print(f'\n  Baselines (per-fold mean):')
    print(f'    Persistence acc: {mean_persistence_acc:.4f}  f1: {mean_persistence_f1:.4f}')
    print(f'    Majority-class acc: {baselines["random_walk_accuracy"]:.4f}  f1: {baselines["random_walk_f1"]:.4f}')

    lift_acc = mean_acc - mean_persistence_acc
    lift_f1 = mean_f1 - mean_persistence_f1
    print(f'    Model lift (vs persistence, per-fold mean): acc={lift_acc:+.4f}  f1={lift_f1:+.4f}')

    # Final fit
    print('  Fitting final model...')
    pipe.fit(X, y)

    return {
        'model': pipe,
        'timeframe': timeframe,
        'target': 'direction',
        'cv_accuracy': mean_acc,
        'cv_f1': mean_f1,
        'cv_roc_auc': mean_auc,
        'cv_accuracy_per_fold': cv_results['test_accuracy'].tolist(),
        'cv_f1_per_fold': cv_results['test_f1_weighted'].tolist(),
        'cv_auc_per_fold': cv_results['test_roc_auc'].tolist(),
        'baselines': baselines,
        'lift_accuracy': lift_acc,
        'lift_f1': lift_f1,
        'n_rows': dataset['n_rows'],
        'symbols': dataset['symbols'],
        'feature_columns': dataset['feature_columns'],
    }


def train_magnitude_model(dataset: dict) -> dict:
    """Train a next-bar return magnitude regressor for a single timeframe."""
    X = dataset['X']
    y = dataset['y_return']
    timeframe = dataset['timeframe']

    print(f'\n{"="*60}')
    print(f'Training MAGNITUDE model — {timeframe} ({len(X):,} rows)')
    print(f'{"="*60}')

    tscv = make_timeframe_cv(timeframe)

    pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('reg', LGBMRegressor(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.1,
            min_child_samples=20,
            verbose=-1,
        )),
    ])

    print('  Running walk-forward CV...')
    cv_results = cross_validate(
        pipe, X, y, cv=tscv,
        scoring=['neg_mean_squared_error', 'r2'],
    )

    mean_mse = -cv_results['test_neg_mean_squared_error'].mean()
    mean_r2 = cv_results['test_r2'].mean()

    print(f'  CV MSE: {mean_mse:.8f}')
    print(f'  CV R2:  {mean_r2:.4f}')

    # Baseline on last fold
    last_fold_test_idx = list(tscv.split(X))[-1][1]
    ret_test = y.iloc[last_fold_test_idx].values
    last_ret_test = dataset['last_return'].iloc[last_fold_test_idx].values

    persistence_mse = mean_squared_error(ret_test, last_ret_test)
    rw_mse = mean_squared_error(ret_test, np.zeros_like(ret_test))

    print(f'\n  Baselines (last fold):')
    print(f'    Persistence MSE: {persistence_mse:.8f}')
    print(f'    Random-walk MSE: {rw_mse:.8f}')
    print(f'    Model MSE:       {mean_mse:.8f}')

    # Final fit
    print('  Fitting final model...')
    pipe.fit(X, y)

    return {
        'model': pipe,
        'timeframe': timeframe,
        'target': 'magnitude',
        'cv_mse': mean_mse,
        'cv_r2': mean_r2,
        'cv_mse_per_fold': (-cv_results['test_neg_mean_squared_error']).tolist(),
        'cv_r2_per_fold': cv_results['test_r2'].tolist(),
        'baseline_persistence_mse': persistence_mse,
        'baseline_rw_mse': rw_mse,
        'n_rows': dataset['n_rows'],
        'symbols': dataset['symbols'],
        'feature_columns': dataset['feature_columns'],
    }


# ── Experiment runner ─────────────────────────────────────────────────────────
def run_experiment_matrix(df_1m: pd.DataFrame, timeframes: list[str],
                          save_dir: str = SAVE_DIR) -> dict:
    """Run the full experiment matrix: direction + magnitude for each timeframe.

    Returns a dict keyed by timeframe with direction and magnitude results.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    for tf in timeframes:
        dataset = build_dataset_for_timeframe(df_1m, tf)
        dir_result = train_direction_model(dataset)
        mag_result = train_magnitude_model(dataset)
        results[tf] = {'direction': dir_result, 'magnitude': mag_result}

        # Save models
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for target_name, result in [('direction', dir_result), ('magnitude', mag_result)]:
            model_path = Path(save_dir) / f'return_{target_name}_{tf}_{timestamp}.pkl'
            meta_path = model_path.with_suffix('.json')

            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)

            meta = {
                'timeframe': tf,
                'target': target_name,
                'trained_at': timestamp,
                'n_rows': result['n_rows'],
                'symbols': result['symbols'],
                'feature_columns': result['feature_columns'],
                'feature_schema_version': '3.0.0',
            }
            if target_name == 'direction':
                meta.update({
                    'cv_accuracy': round(result['cv_accuracy'], 4),
                    'cv_f1': round(result['cv_f1'], 4),
                    'cv_roc_auc': round(result['cv_roc_auc'], 4),
                    'baselines': result['baselines'],
                    'lift_accuracy': round(result['lift_accuracy'], 4),
                    'lift_f1': round(result['lift_f1'], 4),
                })
            else:
                meta.update({
                    'cv_mse': float(result['cv_mse']),
                    'cv_r2': round(result['cv_r2'], 4),
                    'baseline_persistence_mse': float(result['baseline_persistence_mse']),
                    'baseline_rw_mse': float(result['baseline_rw_mse']),
                })

            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            print(f'  Saved: {model_path.name}')

    # Print comparison table
    print(f'\n{"="*80}')
    print('EXPERIMENT MATRIX RESULTS')
    print(f'{"="*80}')
    print(f'\n{"Timeframe":<12} {"Accuracy":<12} {"F1":<12} {"AUC":<12} '
          f'{"Pers.Acc":<12} {"Lift(Acc)":<12}')
    print('-' * 72)
    for tf in timeframes:
        d = results[tf]['direction']
        print(f'{tf:<12} {d["cv_accuracy"]:<12.4f} {d["cv_f1"]:<12.4f} '
              f'{d["cv_roc_auc"]:<12.4f} '
              f'{d["baselines"]["persistence_accuracy"]:<12.4f} '
              f'{d["lift_accuracy"]:<+12.4f}')

    print(f'\n{"Timeframe":<12} {"MSE":<16} {"R2":<12} {"Pers.MSE":<16} {"RW MSE":<16}')
    print('-' * 72)
    for tf in timeframes:
        m = results[tf]['magnitude']
        print(f'{tf:<12} {m["cv_mse"]:<16.8f} {m["cv_r2"]:<12.4f} '
              f'{m["baseline_persistence_mse"]:<16.8f} '
              f'{m["baseline_rw_mse"]:<16.8f}')

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Multi-timeframe return training')
    parser.add_argument('--symbols', nargs='+', default=DEFAULT_SYMBOLS,
                        help='Symbols to pool')
    parser.add_argument('--timeframes', nargs='+', default=DEFAULT_TIMEFRAMES,
                        help='Timeframes to train')
    parser.add_argument('--start-date', default=DEFAULT_START_DATE)
    parser.add_argument('--end-date', default=DEFAULT_END_DATE)
    parser.add_argument('--save-dir', default=SAVE_DIR)
    args = parser.parse_args()

    cfg = load_credentials()
    print(f'Loading 1m bars for {len(args.symbols)} symbols...')
    df_1m = load_multi_symbol_minute_bars(args.symbols, args.start_date, args.end_date, cfg)

    results = run_experiment_matrix(df_1m, args.timeframes, args.save_dir)
    print('\nDone!')
    return results


if __name__ == '__main__':
    main()
