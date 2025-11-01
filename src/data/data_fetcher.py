"""
Data fetching and preprocessing utilities for pairs trading
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime, timedelta


class DataFetcher:
    """Fetch and preprocess stock data for pairs trading"""
    
    def __init__(self, start_date: str = None, end_date: str = None):
        """
        Initialize DataFetcher
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        self.start_date = start_date
        self.end_date = end_date
    
    def fetch_stock_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch historical stock data for given tickers
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            DataFrame with adjusted close prices
        """
        data = yf.download(tickers, start=self.start_date, end=self.end_date, progress=False)
        
        if len(tickers) == 1:
            prices = data['Adj Close'].to_frame()
            prices.columns = tickers
        else:
            prices = data['Adj Close']
        
        # Forward fill then backward fill to handle missing data
        prices = prices.ffill().bfill()
        
        return prices
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns from prices
        
        Args:
            prices: DataFrame of stock prices
            
        Returns:
            DataFrame of log returns
        """
        return np.log(prices / prices.shift(1)).dropna()
    
    def normalize_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize prices to start at 100
        
        Args:
            prices: DataFrame of stock prices
            
        Returns:
            Normalized prices
        """
        return prices / prices.iloc[0] * 100
    
    def create_feature_matrix(self, prices: pd.DataFrame, 
                             lookback_windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create technical features for ML models
        
        Args:
            prices: DataFrame of stock prices
            lookback_windows: List of window sizes for rolling statistics
            
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=prices.index)
        
        for col in prices.columns:
            # Returns
            features[f'{col}_return'] = prices[col].pct_change()
            
            # Rolling statistics
            for window in lookback_windows:
                features[f'{col}_ma_{window}'] = prices[col].rolling(window).mean()
                features[f'{col}_std_{window}'] = prices[col].rolling(window).std()
                features[f'{col}_rsi_{window}'] = self._calculate_rsi(prices[col], window)
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def prepare_pairs_data(self, ticker1: str, ticker2: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fetch and prepare data for a pair of stocks
        
        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            
        Returns:
            Tuple of (DataFrame with both prices, Series of spread)
        """
        prices = self.fetch_stock_data([ticker1, ticker2])
        
        # Calculate spread (difference in log prices)
        spread = np.log(prices[ticker1]) - np.log(prices[ticker2])
        
        return prices, spread
