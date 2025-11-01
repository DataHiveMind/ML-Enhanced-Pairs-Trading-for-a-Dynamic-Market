"""
Backtesting module for evaluating trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def calculate_returns(signals: pd.DataFrame, stock1: pd.Series, stock2: pd.Series) -> pd.Series:
    """
    Calculate returns based on trading signals.
    
    Args:
        signals: DataFrame with trading signals
        stock1: Price series for first stock
        stock2: Price series for second stock
        
    Returns:
        Returns series
    """
    stock1_returns = stock1.pct_change()
    stock2_returns = stock2.pct_change()
    
    # Strategy returns: long stock1, short stock2 when signal is 1
    strategy_returns = signals['position'] * (stock1_returns - stock2_returns)
    
    return strategy_returns


def calculate_performance_metrics(returns: pd.Series, signals: pd.DataFrame = None) -> Dict[str, float]:
    """
    Calculate performance metrics for a strategy.
    
    Args:
        returns: Series of strategy returns
        signals: Optional DataFrame with trading signals to calculate num_trades
        
    Returns:
        Dictionary with performance metrics
    """
    total_return = (1 + returns).cumprod().iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    # Calculate number of trades from signals if provided
    num_trades = 0
    if signals is not None and 'position' in signals.columns:
        num_trades = (signals['position'].diff() != 0).sum()
    
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': num_trades
    }
    
    return metrics


def backtest_strategy(signals: pd.DataFrame, stock1: pd.Series, stock2: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Backtest a pairs trading strategy.
    
    Args:
        signals: DataFrame with trading signals
        stock1: Price series for first stock
        stock2: Price series for second stock
        
    Returns:
        Tuple of (returns_series, performance_metrics)
    """
    returns = calculate_returns(signals, stock1, stock2)
    metrics = calculate_performance_metrics(returns, signals)
    
    return returns, metrics
