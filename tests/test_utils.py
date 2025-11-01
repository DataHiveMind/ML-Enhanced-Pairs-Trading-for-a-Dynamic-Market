"""
Unit tests for utility functions.
"""

import pytest
import pandas as pd
import numpy as np
from src.utils import calculate_returns, calculate_rolling_statistics, normalize_series


def test_calculate_returns():
    """Test returns calculation."""
    prices = pd.Series([100, 105, 103, 110], index=pd.date_range('2023-01-01', periods=4))
    returns = calculate_returns(pd.DataFrame({'stock': prices}))
    
    assert len(returns) == 3  # One less than original due to pct_change
    assert returns.iloc[0]['stock'] == pytest.approx(0.05, rel=1e-5)


def test_calculate_rolling_statistics():
    """Test rolling statistics calculation."""
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    rolling_mean, rolling_std = calculate_rolling_statistics(series, window=3)
    
    assert len(rolling_mean) == len(series)
    assert rolling_mean.iloc[2] == pytest.approx(2.0, rel=1e-5)
    assert rolling_std.iloc[2] == pytest.approx(1.0, rel=1e-5)


def test_normalize_series():
    """Test series normalization."""
    series = pd.Series([1, 2, 3, 4, 5])
    normalized = normalize_series(series)
    
    assert normalized.mean() == pytest.approx(0.0, abs=1e-10)
    assert normalized.std() == pytest.approx(1.0, rel=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
