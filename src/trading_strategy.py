"""
Trading strategy module for generating trading signals.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class PairsTradingStrategy:
    """
    Basic pairs trading strategy based on spread z-score.
    """
    
    def __init__(self, entry_threshold: float = 2.0, exit_threshold: float = 0.5):
        """
        Initialize the strategy.
        
        Args:
            entry_threshold: Z-score threshold for entering a trade
            exit_threshold: Z-score threshold for exiting a trade
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def calculate_spread(self, stock1: pd.Series, stock2: pd.Series, hedge_ratio: float = 1.0) -> pd.Series:
        """
        Calculate the spread between two stocks.
        
        Args:
            stock1: Price series for first stock
            stock2: Price series for second stock
            hedge_ratio: Hedge ratio for the pair
            
        Returns:
            Spread series
        """
        return stock1 - hedge_ratio * stock2
    
    def calculate_zscore(self, spread: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate z-score of the spread.
        
        Args:
            spread: Spread series
            window: Rolling window for mean and std calculation
            
        Returns:
            Z-score series
        """
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        return (spread - rolling_mean) / rolling_std
    
    def generate_signals(self, spread: pd.Series, window: int = 20) -> pd.DataFrame:
        """
        Generate trading signals based on z-score.
        
        Args:
            spread: Spread series
            window: Rolling window for z-score calculation
            
        Returns:
            DataFrame with signals (1: long, -1: short, 0: no position)
        """
        zscore = self.calculate_zscore(spread, window)
        
        signals = pd.DataFrame(index=spread.index)
        signals['zscore'] = zscore
        signals['signal'] = 0
        
        # Generate signals
        signals.loc[zscore > self.entry_threshold, 'signal'] = -1  # Short spread
        signals.loc[zscore < -self.entry_threshold, 'signal'] = 1  # Long spread
        signals.loc[abs(zscore) < self.exit_threshold, 'signal'] = 0  # Exit position
        
        # Forward fill to maintain positions
        signals['position'] = signals['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        return signals
