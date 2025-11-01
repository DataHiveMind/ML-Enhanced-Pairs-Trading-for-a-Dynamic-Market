"""
Utility functions for the pairs trading project.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        prices: DataFrame with price data
        
    Returns:
        DataFrame with returns
    """
    return prices.pct_change().dropna()


def calculate_rolling_statistics(series: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate rolling mean and standard deviation.
    
    Args:
        series: Time series data
        window: Rolling window size
        
    Returns:
        Tuple of (rolling_mean, rolling_std)
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return rolling_mean, rolling_std


def normalize_series(series: pd.Series) -> pd.Series:
    """
    Normalize a time series to have mean 0 and std 1.
    
    Args:
        series: Time series data
        
    Returns:
        Normalized series
    """
    return (series - series.mean()) / series.std()
