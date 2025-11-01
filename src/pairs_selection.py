"""
Pairs selection module for identifying cointegrated pairs of assets.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from typing import List, Tuple, Dict


def test_cointegration(stock1: pd.Series, stock2: pd.Series, significance_level: float = 0.05) -> Tuple[bool, float]:
    """
    Test if two stocks are cointegrated using the Engle-Granger test.
    
    Args:
        stock1: Price series for first stock
        stock2: Price series for second stock
        significance_level: Significance level for the test (default: 0.05)
        
    Returns:
        Tuple of (is_cointegrated, p_value)
    """
    score, p_value, _ = coint(stock1, stock2)
    is_cointegrated = p_value < significance_level
    return is_cointegrated, p_value


def find_cointegrated_pairs(data: pd.DataFrame, significance_level: float = 0.05) -> List[Tuple[str, str, float]]:
    """
    Find all cointegrated pairs in a dataset.
    
    Args:
        data: DataFrame with price data for multiple stocks (columns are tickers)
        significance_level: Significance level for the cointegration test
        
    Returns:
        List of tuples (stock1, stock2, p_value) for cointegrated pairs
    """
    n = data.shape[1]
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            stock1 = data.columns[i]
            stock2 = data.columns[j]
            
            is_cointegrated, p_value = test_cointegration(
                data[stock1], data[stock2], significance_level
            )
            
            if is_cointegrated:
                pairs.append((stock1, stock2, p_value))
    
    # Sort by p-value (lower is better)
    pairs.sort(key=lambda x: x[2])
    return pairs


def calculate_spread(stock1: pd.Series, stock2: pd.Series) -> pd.Series:
    """
    Calculate the spread between two stocks.
    
    Args:
        stock1: Price series for first stock
        stock2: Price series for second stock
        
    Returns:
        Spread series
    """
    # Simple spread calculation (can be enhanced with hedge ratio)
    return stock1 - stock2
