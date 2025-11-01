"""
Feature engineering module for creating ML features from price data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def create_technical_features(data: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
    """
    Create technical indicators as features.
    
    Args:
        data: DataFrame with price data
        windows: List of window sizes for rolling calculations
        
    Returns:
        DataFrame with technical features
    """
    features = pd.DataFrame(index=data.index)
    
    for window in windows:
        # Simple Moving Average
        features[f'sma_{window}'] = data.rolling(window=window).mean()
        
        # Exponential Moving Average
        features[f'ema_{window}'] = data.ewm(span=window, adjust=False).mean()
        
        # Volatility
        features[f'volatility_{window}'] = data.rolling(window=window).std()
        
        # Momentum
        features[f'momentum_{window}'] = data - data.shift(window)
    
    return features


def create_spread_features(spread: pd.Series, windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
    """
    Create features from the spread between pairs.
    
    Args:
        spread: Spread series between two stocks
        windows: List of window sizes for rolling calculations
        
    Returns:
        DataFrame with spread features
    """
    features = pd.DataFrame(index=spread.index)
    
    for window in windows:
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        
        # Z-score of spread
        features[f'z_score_{window}'] = (spread - rolling_mean) / rolling_std
        
        # Distance from mean
        features[f'distance_from_mean_{window}'] = spread - rolling_mean
        
        # Normalized spread
        features[f'normalized_spread_{window}'] = spread / rolling_mean
    
    return features


def create_lag_features(data: pd.Series, lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
    """
    Create lagged features.
    
    Args:
        data: Time series data
        lags: List of lag periods
        
    Returns:
        DataFrame with lagged features
    """
    features = pd.DataFrame(index=data.index)
    
    for lag in lags:
        features[f'lag_{lag}'] = data.shift(lag)
    
    return features
