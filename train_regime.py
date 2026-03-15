# Regime Classifier Training — Trade Bot (Converted from Notebook)
import json
import pickle
import warnings
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report,
    roc_auc_score, f1_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore', category=UserWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
SYMBOL     = 'SPY'
START_DATE = '2025-01-01'
END_DATE   = '2025-03-01'
SAVE_DIR   = './trained-models'
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load credentials ───────────────────────────────────────────────────────────
MASSIVE_ACCESS_KEY = os.getenv('MASSIVE_S3_ACCESS_KEY')
MASSIVE_SECRET_KEY = os.getenv('MASSIVE_S3_SECRET_KEY')

if not MASSIVE_ACCESS_KEY:
    # Hardcoded fallback from notebook
    MASSIVE_ACCESS_KEY = 'bd65025c-6f4b-4ca9-ad0a-bbafceb8a161'
    MASSIVE_SECRET_KEY = 'HLw8C29nc07bW1fTGtNzq1AOteWzzcZX'
    print('Using hardcoded fallback credentials')

S3_CONFIG = {
    'endpoint':   'https://files.massive.com',
    'bucket':     'flatfiles',
    'access_key': MASSIVE_ACCESS_KEY,
    'secret_key': MASSIVE_SECRET_KEY,
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

def load_minute_bars(symbol: str, start_date: str, end_date: str, cfg: dict) -> pd.DataFrame:
    fs = _get_fs(cfg)
    symbol = symbol.upper()
    dfs = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    loaded = 0

    while current <= end:
        if current.weekday() < 5:
            date_str = current.strftime('%Y-%m-%d')
            for fmt in KEY_FORMATS:
                path = f"{cfg['bucket']}/{fmt.format(year=current.year, month=current.month, date=date_str)}"
                try:
                    with fs.open(path, 'rb') as f:
                        df = pd.read_csv(f, compression='gzip')
                        df.columns = [c.lower() for c in df.columns]
                        sym_col = 'ticker' if 'ticker' in df.columns else 'symbol'
                        filtered = df[df[sym_col].str.upper() == symbol].copy()
                        if not filtered.empty:
                            dfs.append(filtered)
                            loaded += 1
                            break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f'  Warning {date_str}: {e}')
        current += timedelta(days=1)

    if not dfs:
        raise ValueError(f'No data found for {symbol} between {start_date} and {end_date}')

    result = pd.concat(dfs, ignore_index=True)
    print(f'Loaded {len(result):,} bars across {loaded} trading days for {symbol}')
    return result

# ── Feature engineering ────────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    log_ret = np.log(df['close'] / df['close'].shift(1))

    for w in [5, 20, 60]:
        feat[f'realized_vol_{w}'] = (
            log_ret.rolling(w, min_periods=w).std() * np.sqrt(252 * 390)
        )
        feat[f'log_return_{w}'] = log_ret.rolling(w, min_periods=1).sum()

    try:
        import pandas_ta as ta
        rsi = ta.rsi(df['close'], length=14)
    except Exception:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=14).mean()
        rsi = 100.0 - (100.0 / (1.0 + gain / loss.replace(0, np.nan)))

    feat['rsi_14'] = rsi
    rsi_mean = rsi.rolling(60, min_periods=20).mean()
    rsi_std  = rsi.rolling(60, min_periods=20).std()
    feat['rsi_14_zscore'] = (rsi - rsi_mean) / rsi_std.replace(0, np.nan)
    feat['rsi_14_dev50']  = rsi - 50.0

    vm = df['volume'].rolling(60, min_periods=20).mean()
    vs = df['volume'].rolling(60, min_periods=20).std()
    feat['volume_zscore_60'] = (df['volume'] - vm) / vs.replace(0, np.nan)

    bar_range = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
    feat['price_range_norm'] = bar_range.rolling(20, min_periods=5).mean()

    print('Computing Hurst exponent...')
    try:
        import nolds
        log_price  = np.log(df['close'].replace(0, np.nan))
        hurst_vals = np.full(len(df), np.nan)
        window     = 100
        total      = len(log_price) - window
        
        # We'll use a larger step or sample for speed in this run if needed, 
        # but the request is to "run all code blocks". 
        # For full verification we should run it all.
        for i in range(window, len(log_price)):
            seg = log_price.iloc[i - window : i].dropna()
            if len(seg) >= 80:
                try:
                    hurst_vals[i] = float(np.clip(nolds.hurst_rs(seg.values, fit='poly'), 0.0, 1.0))
                except Exception:
                    pass
            if (i - window) % 1000 == 0:
                print(f'  Hurst: {((i-window)/total)*100:.1f}% complete', end='\r')

        feat['hurst'] = hurst_vals
        print('\nHurst computation complete')
    except ImportError:
        print('nolds not available — defaulting Hurst to 0.5')
        feat['hurst'] = 0.5

    return feat

# ── Compute labels ─────────────────────────────────────────────────────────────
LABEL_MAP = {0: 'mean_reverting', 1: 'choppy', 2: 'trending', 3: 'crisis'}

def compute_labels(df: pd.DataFrame, feat: pd.DataFrame) -> pd.Series:
    hurst = feat['hurst'].fillna(0.5)
    vol   = feat['realized_vol_20']

    labels = pd.Series(1, index=df.index, dtype=int)
    labels[hurst < 0.45] = 0
    labels[hurst > 0.55] = 2

    vm = vol.rolling(60, min_periods=20).mean()
    vs = vol.rolling(60, min_periods=20).std()
    labels[vol > (vm + 2.0 * vs)] = 3
    return labels

# ── Main Execution ───────────────────────────────────────────────────────────
print(f'Starting training pipeline for {SYMBOL}...')
df_raw = load_minute_bars(SYMBOL, START_DATE, END_DATE, S3_CONFIG)

print('Computing features...')
features = compute_features(df_raw)

print('Computing labels...')
labels = compute_labels(df_raw, features)

valid = features.notna().all(axis=1)
X = features[valid]
y = labels[valid]

print(f'Training set: {len(X):,} bars')

# ── Walk-forward CV ────────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=8, gap=60, test_size=2016)

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

calibrated = CalibratedClassifierCV(pipe, method='isotonic', cv=5)

print('Running 8-fold walk-forward CV...')
cv_results = cross_validate(
    calibrated, X, y, cv=tscv,
    scoring=['roc_auc_ovr_weighted', 'f1_weighted'],
    verbose=1,
)

mean_auc = cv_results['test_roc_auc_ovr_weighted'].mean()
mean_f1  = cv_results['test_f1_weighted'].mean()

print(f'\n═══ CV Results ═══')
print(f'  ROC-AUC: {mean_auc:.4f}')
print(f'  F1:      {mean_f1:.4f}')

# ── Final Fit ───────────────────────────────────────────────────────────
print('Fitting final model...')
calibrated.fit(X, y)

# ── Save ────────────────────────────────────────────────────────────
timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = Path(SAVE_DIR) / f'{SYMBOL}_regime_classifier_{timestamp}.pkl'
meta_path  = model_path.with_suffix('.json')

with open(model_path, 'wb') as f:
    pickle.dump(calibrated, f)

meta = {
    'symbol':                 SYMBOL,
    'trained_at':             timestamp,
    'train_start':            START_DATE,
    'train_end':              END_DATE,
    'n_samples':              len(X),
    'feature_schema_version': '1.0.0',
    'feature_columns':        list(X.columns),
    'cv_mean_roc_auc':        round(float(mean_auc), 4),
    'cv_mean_f1':             round(float(mean_f1), 4),
}
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

print(f'Model saved to {model_path}')
print('Done!')
