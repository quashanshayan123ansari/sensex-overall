"""
Quantitative Modeling Engine: BSE SENSEX vs Geometric Brownian Motion (GBM)
Features:
- Historical Data Retrieval & Local CSV Caching
- Dynamic & Expanding Window GBM Monte Carlo Simulation
- Empirical Distribution vs Gaussian Normal vs Scaled Student-t
- GARCH(1,1) Volatility Clustering & Conditional Variance
- High-order Moment & Outlier Analytics (Skewness, Kurtosis, 3s/4s/5s violations)
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Tuple, Dict, Any, Optional
import yfinance as yf
from arch import arch_model

CACHE_FILE = os.path.join(os.path.dirname(__file__), "sensex_data.csv")


def fetch_index_data(ticker: str = "^BSESN", start_date: str = "1997-07-01", force_refresh: bool = False) -> pd.DataFrame:
    """
    Retrieves historical index data. Uses local CSV cache if available.
    Computes log returns and normalized base-100 series.
    """
    df = None
    
    # Try reading from cache first if ticker is default Sensex
    if not force_refresh and ticker == "^BSESN" and os.path.exists(CACHE_FILE):
        try:
            df = pd.read_csv(CACHE_FILE)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
                df.set_index("Date", inplace=True)
            elif df.index.name == "Date":
                df.index = pd.to_datetime(df.index, utc=True).dt.tz_localize(None)
        except Exception:
            df = None

    if df is None or df.empty or "Close" not in df.columns:
        t = yf.Ticker(ticker)
        df = t.history(period="max")
        if df.empty:
            df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        
        if df.empty or "Close" not in df.columns:
            raise ValueError(f"Failed to fetch data for ticker: {ticker}")
        
        df.index = pd.to_datetime(df.index, utc=True).dt.tz_localize(None)
        if ticker == "^BSESN":
            df.to_csv(CACHE_FILE)

    clean_df = pd.DataFrame()
    clean_df["Price"] = df["Close"].astype(float).dropna()
    clean_df["Log_Return"] = np.log(clean_df["Price"] / clean_df["Price"].shift(1))
    clean_df["Pct_Return"] = clean_df["Price"].pct_change()
    clean_df.dropna(inplace=True)
    
    # Normalize price base 100
    clean_df["Normalized_Price"] = 100.0 * (clean_df["Price"] / clean_df["Price"].iloc[0])
    return clean_df


def calculate_summary_stats(returns: pd.Series, price_series: pd.Series) -> Dict[str, Any]:
    """
    Computes summary metrics, distribution parameters, and extreme tail counts.
    """
    n_days = len(returns)
    mean_daily = returns.mean()
    std_daily = returns.std(ddof=1)
    
    ann_factor = 252.0
    ann_vol = std_daily * np.sqrt(ann_factor)
    
    total_days = (price_series.index[-1] - price_series.index[0]).days
    total_years = max(total_days / 365.25, 0.01)
    ann_geo_return = (price_series.iloc[-1] / price_series.iloc[0]) ** (1.0 / total_years) - 1.0
    
    worst_day = float(returns.min())
    best_day = float(returns.max())
    skewness = float(stats.skew(returns, bias=False))
    kurtosis_excess = float(stats.kurtosis(returns, fisher=True, bias=False))
    
    # Standardized deviations
    z_scores = (returns - mean_daily) / std_daily
    
    # Outliers
    outliers_3s = int(np.sum(np.abs(z_scores) >= 3.0))
    outliers_4s = int(np.sum(np.abs(z_scores) >= 4.0))
    outliers_5s = int(np.sum(np.abs(z_scores) >= 5.0))
    
    # Theoretical normal expectations
    p_3s = 2.0 * (1.0 - stats.norm.cdf(3.0))
    p_4s = 2.0 * (1.0 - stats.norm.cdf(4.0))
    p_5s = 2.0 * (1.0 - stats.norm.cdf(5.0))
    
    expected_3s = n_days * p_3s
    expected_4s = n_days * p_4s
    expected_5s = n_days * p_5s
    
    # Student-t MLE fit
    t_df, t_loc, t_scale = stats.t.fit(returns)
    
    return {
        "n_days": n_days,
        "ann_geo_return": ann_geo_return,
        "ann_vol": ann_vol,
        "worst_day": worst_day,
        "best_day": best_day,
        "skewness": skewness,
        "excess_kurtosis": kurtosis_excess,
        "outliers_3s": outliers_3s,
        "expected_3s": expected_3s,
        "outliers_4s": outliers_4s,
        "expected_4s": expected_4s,
        "outliers_5s": outliers_5s,
        "expected_5s": expected_5s,
        "student_t_df": float(t_df),
        "student_t_loc": float(t_loc),
        "student_t_scale": float(t_scale),
        "daily_mean": float(mean_daily),
        "daily_std": float(std_daily)
    }


def simulate_gbm_ensemble(
    s0: float,
    drift: float,
    vol: float,
    n_steps: int,
    n_paths: int = 200,
    dt: float = 1.0 / 252.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates Geometric Brownian Motion paths with fixed drift and vol.
    Returns:
    - paths: ndarray (n_paths, n_steps + 1)
    - p5: 5th percentile path
    - p50: Median path
    - p95: 95th percentile path
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    
    drift_step = (drift - 0.5 * (vol ** 2)) * dt
    diffusion_step = vol * np.sqrt(dt) * z
    log_increments = drift_step + diffusion_step
    
    log_paths = np.zeros((n_paths, n_steps + 1))
    log_paths[:, 1:] = np.cumsum(log_increments, axis=1)
    paths = s0 * np.exp(log_paths)
    
    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    
    return paths, p5, p50, p95


def fit_garch_model(returns: pd.Series) -> Tuple[pd.Series, pd.Series, float]:
    """
    Fits GARCH(1,1) model and returns conditional volatility, standardized residuals,
    and standardized excess kurtosis.
    """
    scaled_ret = returns * 100.0
    am = arch_model(scaled_ret, vol="Garch", p=1, o=0, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    
    cond_vol = (res.conditional_volatility / 100.0).copy()
    std_resid = (returns - returns.mean()) / cond_vol
    std_kurt = float(stats.kurtosis(std_resid.dropna(), fisher=True, bias=False))
    
    return cond_vol, std_resid, std_kurt
