"""
Reusable, transparent cleaning pipeline for multivariate OHLCV time series
(5-minute crypto candles). Uses pandas, numpy, and the standard library only.

Report-friendly: clear comments and structured log dictionary for auditing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class CleanConfig:
    """Configuration for the OHLCV cleaning pipeline."""

    freq: str = "5min"
    tz: str = "UTC"
    volume_fill_value: float = 0.0
    return_outlier_z: float = 5.0
    normalize_close: bool = True


# -----------------------------------------------------------------------------
# Symbol and loading
# -----------------------------------------------------------------------------


def infer_symbol_from_filename(path: str | Path) -> str:
    """
    Infer a symbol (e.g. BTCUSD) from a filename like 'bybit_BTCUSD_ohlcv_5min.csv'.

    Rule: return the segment between the first underscore and '_ohlcv'.
    Uses regex ^[^_]+_([^_]+)_ohlcv; fallback to stem.upper() if not matched.
    """
    path = Path(path)
    stem = path.stem
    match = re.match(r"^[^_]+_([^_]+)_ohlcv", stem)
    if match:
        return match.group(1).upper()
    return stem.upper()


def load_ohlcv_csv(path: str | Path, config: CleanConfig) -> pd.DataFrame:
    """
    Load a single OHLCV CSV with timestamps parsed as UTC.
    Drops duplicate timestamps and sorts by time.
    """
    path = Path(path)
    df = pd.read_csv(path)

    # Normalize column names to lowercase for consistent handling
    df.columns = df.columns.str.strip().str.lower()

    # Parse timestamp as datetime, ensure UTC
    ts_col = "timestamp"
    if ts_col not in df.columns:
        raise ValueError(f"Expected column '{ts_col}' in {path.name}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    if df[ts_col].dt.tz is None:
        df[ts_col] = df[ts_col].dt.tz_localize(config.tz, ambiguous="infer")

    # Drop duplicate timestamps (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset=[ts_col], keep="first")
    n_dupes = before - len(df)

    # Sort by time
    df = df.sort_values(ts_col).reset_index(drop=True)

    return df


def prefix_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Prefix OHLCV columns with the symbol (e.g. open -> BTCUSD_open).
    Leaves 'timestamp' unchanged so it can be used as merge key.
    """
    out = df.copy()
    skip = {"timestamp"}
    rename = {
        c: f"{symbol}_{c}"
        for c in out.columns
        if c not in skip
    }
    return out.rename(columns=rename)


# -----------------------------------------------------------------------------
# Merge and grid
# -----------------------------------------------------------------------------


def merge_multivariate_time_series(
    paths: list[str | Path],
    config: CleanConfig,
) -> pd.DataFrame:
    """
    Load multiple OHLCV CSVs, prefix columns by symbol, and outer-merge on
    timestamp. Then reindex to a full 5-minute grid (no gaps).
    """
    if not paths:
        return pd.DataFrame()

    dfs = []
    for p in paths:
        symbol = infer_symbol_from_filename(p)
        raw = load_ohlcv_csv(p, config)
        prefixed = prefix_columns(raw, symbol)
        dfs.append(prefixed)

    # Outer merge on timestamp
    merged = dfs[0]
    for other in dfs[1:]:
        merged = pd.merge(
            merged,
            other,
            on="timestamp",
            how="outer",
            suffixes=("", "_dup"),
        )
    # Drop any duplicate columns from merge (e.g. timestamp_dup)
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]

    # Sort by time
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Build full 5-minute grid from min to max timestamp
    t_min = merged["timestamp"].min()
    t_max = merged["timestamp"].max()
    full_index = pd.date_range(
        start=t_min,
        end=t_max,
        freq=config.freq,
        tz=config.tz,
        name="timestamp",
    )

    # Reindex to full grid (introduces NaN where bars are missing)
    merged = merged.set_index("timestamp")
    merged = merged.reindex(full_index).reset_index()
    if "index" in merged.columns and "timestamp" not in merged.columns:
        merged = merged.rename(columns={"index": "timestamp"})

    return merged


# -----------------------------------------------------------------------------
# Missingness and filling
# -----------------------------------------------------------------------------

# OHLCV column suffixes: only these get _was_missing flags and missingness stats
OHLCV_SUFFIXES = ("_open", "_high", "_low", "_close", "_volume")


def _is_ohlcv_column(name: str) -> bool:
    """True if column name ends with an OHLCV suffix."""
    return name.endswith(OHLCV_SUFFIXES)


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean columns indicating whether each OHLCV value was missing
    (before fill). Only targets columns ending in _open, _high, _low,
    _close, _volume. Adds {col}_was_missing only for those columns.
    """
    out = df.copy()
    for c in out.columns:
        if not _is_ohlcv_column(c):
            continue
        if out[c].isna().any():
            out[f"{c}_was_missing"] = out[c].isna()
    return out


def fill_missing_ohlcv(df: pd.DataFrame, config: CleanConfig) -> pd.DataFrame:
    """
    Forward-fill then back-fill OHLC and close; fill volume NaN with
    config.volume_fill_value (default 0).
    Skips auxiliary columns (e.g. *_was_missing).
    """
    out = df.copy()
    for c in out.columns:
        if c == "timestamp" or "_was_missing" in c or "_outlier" in c:
            continue
        if "volume" in c.lower():
            out[c] = out[c].fillna(config.volume_fill_value)
        elif any(x in c.lower() for x in ("open", "high", "low", "close")):
            # OHLC: forward then backward fill
            out[c] = out[c].ffill().bfill()
    return out


# -----------------------------------------------------------------------------
# Returns and outliers
# -----------------------------------------------------------------------------


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from all *_close columns.
    New columns: {symbol}_log_return (log(close_t / close_{t-1})).
    """
    out = df.copy()
    close_cols = [c for c in df.columns if c.endswith("_close")]
    for col in close_cols:
        base = col.replace("_close", "")
        out[f"{base}_log_return"] = np.log(out[col] / out[col].shift(1))
    return out


def flag_return_outliers(
    returns: pd.DataFrame,
    z_thresh: float,
) -> pd.DataFrame:
    """
    Add outlier flags for each return column using z-score.
    New columns: {col}_outlier (True where |z| > z_thresh).
    """
    out = returns.copy()
    return_cols = [c for c in returns.columns if "log_return" in c]
    for col in return_cols:
        r = out[col]
        mean_r = r.mean()
        std_r = r.std()
        if std_r == 0 or np.isnan(std_r):
            z = np.zeros_like(r)
        else:
            z = (r - mean_r) / std_r
        out[f"{col}_outlier"] = np.abs(z) > z_thresh
    return out


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------


def normalize_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each *_close series to start at 1.0 (divide by first valid close).
    New columns: {symbol}_close_normalized.
    """
    out = df.copy()
    close_cols = [c for c in df.columns if c.endswith("_close") and "normalized" not in c]
    for col in close_cols:
        base = col.replace("_close", "")
        first = out[col].dropna().iloc[0] if out[col].notna().any() else 1.0
        if first == 0:
            first = 1.0
        out[f"{base}_close_normalized"] = out[col] / first
    return out


# -----------------------------------------------------------------------------
# Full pipeline and log
# -----------------------------------------------------------------------------


def auto_clean_multivariate(
    paths: list[str | Path],
    config: CleanConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Run the full cleaning pipeline on multiple OHLCV CSVs.

    Steps:
      1. Merge multivariate series (outer join, full 5-min grid).
      2. Add missing flags, then fill OHLC (ffill/bfill) and volume (0).
      3. Compute log returns from *_close.
      4. Flag return outliers by z-score.
      5. Optionally normalize *_close to start at 1.

    Returns:
      cleaned_df: Filled OHLCV + timestamp + *_was_missing; no NaNs in OHLC/volume.
      normalized_close_df: Same index as cleaned_df, with *_close_normalized (and returns/outliers if present).
      returns_flagged_df: cleaned_df plus *_log_return and *_outlier columns.
      log: Dictionary with config, paths, symbols, row counts, fill counts, and n_return_outliers per series.
    """
    if config is None:
        config = CleanConfig()

    log: dict[str, Any] = {
        "config": {
            "freq": config.freq,
            "tz": config.tz,
            "volume_fill_value": config.volume_fill_value,
            "return_outlier_z": config.return_outlier_z,
            "normalize_close": config.normalize_close,
        },
        "paths": [str(p) for p in paths],
        "symbols": [infer_symbol_from_filename(p) for p in paths],
        "n_rows_raw_merged": None,
        "n_rows_after_grid": None,
        "n_filled_ohlc": None,
        "n_filled_volume": None,
        "n_return_outliers": {},
    }

    # Merge and reindex to full grid
    df = merge_multivariate_time_series(paths, config)
    log["n_rows_raw_merged"] = len(df)

    # Add missing flags (only for OHLCV columns)
    df = add_missing_flags(df)

    # Missingness summary: only columns ending in _open, _high, _low, _close, _volume
    ohlcv_cols = [c for c in df.columns if _is_ohlcv_column(c)]
    missing_matrix = df[ohlcv_cols].isna() if ohlcv_cols else pd.DataFrame()
    if ohlcv_cols:
        df["any_missing_before_fill"] = missing_matrix.any(axis=1)
        log["missing"] = {
            "missing_rows_count": int(df["any_missing_before_fill"].sum()),
            "missing_cells_count": int(missing_matrix.sum().sum()),
            "missing_cells_by_column": missing_matrix.sum().to_dict(),
        }
    else:
        df["any_missing_before_fill"] = False
        log["missing"] = {
            "missing_rows_count": 0,
            "missing_cells_count": 0,
            "missing_cells_by_column": {},
        }

    # Fill OHLC (ffill/bfill) and volume (0)
    df = fill_missing_ohlcv(df, config)
    if ohlcv_cols:
        log["missing"]["missing_cells_after_fill"] = int(df[ohlcv_cols].isna().sum().sum())

    # Report fill counts (OHLC vs volume) for backward compatibility
    ohlc_cols = [c for c in df.columns if c.endswith(("_open", "_high", "_low", "_close"))]
    volume_cols = [c for c in df.columns if c.endswith("_volume")]
    if ohlcv_cols:
        n_missing_ohlc = int(missing_matrix[ohlc_cols].sum().sum()) if ohlc_cols else 0
        n_missing_vol = int(missing_matrix[volume_cols].sum().sum()) if volume_cols else 0
    else:
        n_missing_ohlc = n_missing_vol = 0
    log["n_filled_ohlc"] = n_missing_ohlc
    log["n_filled_volume"] = n_missing_vol

    cleaned_df = df.copy()

    # Log returns
    df = compute_log_returns(df)
    df = flag_return_outliers(df, config.return_outlier_z)

    # Outlier counts per symbol
    return_cols = [c for c in df.columns if "log_return" in c and "outlier" not in c]
    for col in return_cols:
        flag_col = f"{col}_outlier"
        if flag_col in df.columns:
            log["n_return_outliers"][col] = int(df[flag_col].sum())

    returns_flagged_df = df.copy()

    # Normalized close (optional)
    if config.normalize_close:
        df = normalize_close(df)
    normalized_close_df = df.copy()

    log["n_rows_after_grid"] = len(cleaned_df)

    return cleaned_df, normalized_close_df, returns_flagged_df, log
"""
EDA for merged multivariate OHLCV dataset from auto_wrangle.auto_clean_multivariate.
Uses pandas, numpy, matplotlib only. Saves figures to outputs/.
"""

from __future__ import annotations

import json
import glob
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.auto_wrangle import CleanConfig, auto_clean_multivariate


# Default symbols for close trend plots (3–4 most important)
DEFAULT_TREND_SYMBOLS = ("BTCUSD", "ETHUSD", "SOLUSDT", "XRPUSD")

# Symbol order for correlation heatmap
HEATMAP_SYMBOL_ORDER = (
    "ADAUSD", "BNBUSDT", "BTCUSD", "DOGEUSDT", "ETHUSD", "SOLUSDT", "XRPUSD",
)


def _ensure_out_dir(out_dir: str | Path) -> Path:
    """Create outputs directory if it does not exist."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def summarize_dataset(
    cleaned_df: pd.DataFrame,
    returns_flagged_df: pd.DataFrame,
    log: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a summary dict: rows/cols, date range, missing stats from log,
    per-symbol close stats (*_close), per-symbol return stats (*_log_return),
    and outlier counts from log['n_return_outliers'].
    """
    ts = cleaned_df["timestamp"] if "timestamp" in cleaned_df.columns else cleaned_df.index
    missing = log.get("missing", {})

    summary: dict[str, Any] = {
        "n_rows": int(len(cleaned_df)),
        "n_columns": int(len(cleaned_df.columns)),
        "date_range": {
            "min": pd.Timestamp(ts.min()).isoformat() if len(ts) else None,
            "max": pd.Timestamp(ts.max()).isoformat() if len(ts) else None,
        },
        "missing": {
            "missing_rows_count": int(missing.get("missing_rows_count", 0)),
            "missing_cells_count": int(missing.get("missing_cells_count", 0)),
        },
        "close_stats": {},
        "return_stats": {},
        "n_return_outliers": dict(log.get("n_return_outliers", {})),
    }

    # Per-symbol close stats (columns ending with _close, excluding _close_normalized)
    close_cols = [
        c for c in cleaned_df.columns
        if c.endswith("_close") and "normalized" not in c
    ]
    for col in close_cols:
        s = cleaned_df[col].dropna()
        if len(s) == 0:
            continue
        symbol = col.replace("_close", "")
        summary["close_stats"][symbol] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    # Per-symbol return stats (*_log_return)
    return_cols = [c for c in returns_flagged_df.columns if c.endswith("_log_return")]
    for col in return_cols:
        s = returns_flagged_df[col].dropna()
        if len(s) == 0:
            continue
        summary["return_stats"][col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    return summary


def plot_close_trends(
    df: pd.DataFrame,
    out_dir: str | Path,
    use_normalized: bool = False,
    symbols: tuple[str, ...] = DEFAULT_TREND_SYMBOLS,
) -> None:
    """
    Plot close (or normalized close) time series for selected symbols.
    use_normalized=False: plot columns ending with _close.
    use_normalized=True: plot columns ending with _close_normalized.
    Saves to outputs/close_trends.png or outputs/close_trends_normalized.png.
    """
    out_dir = _ensure_out_dir(out_dir)
    suffix = "_close_normalized" if use_normalized else "_close"
    cols = [c for c in df.columns if c.endswith(suffix)]
    # Restrict to requested symbols that exist
    plot_cols = [c for c in cols if c.replace(suffix, "") in symbols]
    if not plot_cols:
        plot_cols = cols[:4]  # fallback: first 4

    ts = df["timestamp"] if "timestamp" in df.columns else df.index

    fig, ax = plt.subplots(figsize=(12, 5))
    for col in plot_cols:
        ax.plot(ts, df[col], label=col.replace(suffix, ""), alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Close (normalized)" if use_normalized else "Close")
    ax.set_title("Close trends" + (" (normalized)" if use_normalized else ""))
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "close_trends_normalized.png" if use_normalized else "close_trends.png"
    fig.savefig(out_dir / fname, dpi=120)
    plt.close(fig)


def plot_return_distributions(returns_flagged_df: pd.DataFrame, out_dir: str | Path) -> None:
    """
    For each *_log_return column: save histogram and boxplot.
    Files: outputs/hist_<SYMBOL>_returns.png, outputs/box_<SYMBOL>_returns.png.
    """
    out_dir = _ensure_out_dir(out_dir)
    return_cols = [c for c in returns_flagged_df.columns if c.endswith("_log_return")]
    for col in return_cols:
        symbol = col.replace("_log_return", "")
        data = returns_flagged_df[col].dropna()

        # Histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Log return")
        ax.set_ylabel("Count")
        ax.set_title(f"Log return distribution: {symbol}")
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{symbol}_returns.png", dpi=120)
        plt.close(fig)

        # Boxplot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data, vert=True)
        ax.set_ylabel("Log return")
        ax.set_title(f"Log return boxplot: {symbol}")
        fig.tight_layout()
        fig.savefig(out_dir / f"box_{symbol}_returns.png", dpi=120)
        plt.close(fig)


def plot_correlation_heatmap(returns_flagged_df: pd.DataFrame, out_dir: str | Path) -> None:
    """
    Build returns-only dataframe in symbol order, compute correlation matrix,
    plot heatmap with imshow + colorbar + tick labels.
    Saves to outputs/returns_corr_heatmap.png.
    """
    out_dir = _ensure_out_dir(out_dir)
    # Select *_log_return columns in HEATMAP_SYMBOL_ORDER (only those present)
    return_cols = [c for c in returns_flagged_df.columns if c.endswith("_log_return")]
    ordered = []
    for sym in HEATMAP_SYMBOL_ORDER:
        c = f"{sym}_log_return"
        if c in return_cols:
            ordered.append(c)
    # Add any remaining return columns not in the fixed order
    for c in return_cols:
        if c not in ordered:
            ordered.append(c)

    if not ordered:
        return
    returns_sub = returns_flagged_df[ordered].dropna(how="all")
    corr = returns_sub.corr()
    labels = [c.replace("_log_return", "") for c in ordered]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Return correlation heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "returns_corr_heatmap.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """
    Load data/*.csv, run auto_clean_multivariate, create outputs/,
    generate all EDA plots and write outputs/summary.json.
    """
    out_dir = Path("outputs")
    _ensure_out_dir(out_dir)

    paths = sorted(glob.glob("data/*.csv"))
    if not paths:
        raise SystemExit("No data/*.csv files found.")
    print("Running EDA...")
    print(f"CSV files found: {len(paths)}")

    config = CleanConfig()
    cleaned_df, normalized_close_df, returns_flagged_df, log = auto_clean_multivariate(
        paths, config
    )

    # Summary dict and JSON
    summary = summarize_dataset(cleaned_df, returns_flagged_df, log)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    plot_close_trends(cleaned_df, out_dir, use_normalized=False)
    plot_close_trends(normalized_close_df, out_dir, use_normalized=True)
    plot_return_distributions(returns_flagged_df, out_dir)
    plot_correlation_heatmap(returns_flagged_df, out_dir)

    n_files = len(list(out_dir.iterdir()))
    print("EDA complete.")
    print(f"Outputs: {n_files} files in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
