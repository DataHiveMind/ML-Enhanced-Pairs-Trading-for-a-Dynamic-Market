"""
Cointegration testing and pair selection utilities
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from typing import List, Tuple, Dict
from itertools import combinations


class PairSelector:
    """Select and validate cointegrated pairs"""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize PairSelector
        
        Args:
            significance_level: P-value threshold for cointegration test
        """
        self.significance_level = significance_level
    
    def test_cointegration(self, price1: pd.Series, price2: pd.Series) -> Tuple[float, float, bool]:
        """
        Test cointegration between two price series using Engle-Granger test
        
        Args:
            price1: First price series
            price2: Second price series
            
        Returns:
            Tuple of (test statistic, p-value, is_cointegrated)
        """
        # Engle-Granger cointegration test
        score, pvalue, _ = coint(price1, price2)
        
        is_cointegrated = pvalue < self.significance_level
        
        return score, pvalue, is_cointegrated
    
    def test_stationarity(self, series: pd.Series) -> Tuple[float, float, bool]:
        """
        Test if series is stationary using Augmented Dickey-Fuller test
        
        Args:
            series: Time series to test
            
        Returns:
            Tuple of (test statistic, p-value, is_stationary)
        """
        result = adfuller(series.dropna())
        
        adf_stat = result[0]
        pvalue = result[1]
        is_stationary = pvalue < self.significance_level
        
        return adf_stat, pvalue, is_stationary
    
    def calculate_hedge_ratio(self, price1: pd.Series, price2: pd.Series) -> float:
        """
        Calculate optimal hedge ratio using OLS regression
        
        Args:
            price1: Dependent variable (price series 1)
            price2: Independent variable (price series 2)
            
        Returns:
            Hedge ratio (beta coefficient)
        """
        model = OLS(price1, price2).fit()
        hedge_ratio = model.params.iloc[0] if hasattr(model.params, 'iloc') else model.params[0]
        
        return hedge_ratio
    
    def find_cointegrated_pairs(self, prices: pd.DataFrame) -> List[Dict]:
        """
        Find all cointegrated pairs from a set of stocks
        
        Args:
            prices: DataFrame with stock prices (columns are tickers)
            
        Returns:
            List of dictionaries with pair information
        """
        tickers = prices.columns.tolist()
        pairs = []
        
        # Test all combinations
        for ticker1, ticker2 in combinations(tickers, 2):
            score, pvalue, is_coint = self.test_cointegration(
                prices[ticker1], prices[ticker2]
            )
            
            if is_coint:
                hedge_ratio = self.calculate_hedge_ratio(
                    prices[ticker1], prices[ticker2]
                )
                
                # Calculate spread
                spread = prices[ticker1] - hedge_ratio * prices[ticker2]
                
                # Test spread stationarity
                adf_stat, adf_pval, is_stationary = self.test_stationarity(spread)
                
                pairs.append({
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'coint_score': score,
                    'coint_pvalue': pvalue,
                    'hedge_ratio': hedge_ratio,
                    'spread_adf_stat': adf_stat,
                    'spread_adf_pval': adf_pval,
                    'is_spread_stationary': is_stationary
                })
        
        # Sort by cointegration p-value (lower is better)
        pairs.sort(key=lambda x: x['coint_pvalue'])
        
        return pairs
    
    def calculate_spread(self, price1: pd.Series, price2: pd.Series, 
                        hedge_ratio: float = None) -> pd.Series:
        """
        Calculate spread between two price series
        
        Args:
            price1: First price series
            price2: Second price series
            hedge_ratio: Hedge ratio (if None, will be calculated)
            
        Returns:
            Spread series
        """
        if hedge_ratio is None:
            hedge_ratio = self.calculate_hedge_ratio(price1, price2)
        
        spread = price1 - hedge_ratio * price2
        
        return spread
    
    def calculate_zscore(self, spread: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate rolling z-score of spread
        
        Args:
            spread: Spread series
            window: Rolling window size
            
        Returns:
            Z-score series
        """
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        
        zscore = (spread - rolling_mean) / rolling_std
        
        return zscore
    
    def get_pair_metrics(self, price1: pd.Series, price2: pd.Series) -> Dict:
        """
        Calculate comprehensive metrics for a pair
        
        Args:
            price1: First price series
            price2: Second price series
            
        Returns:
            Dictionary with pair metrics
        """
        # Cointegration test
        coint_score, coint_pval, is_coint = self.test_cointegration(price1, price2)
        
        # Hedge ratio
        hedge_ratio = self.calculate_hedge_ratio(price1, price2)
        
        # Spread
        spread = self.calculate_spread(price1, price2, hedge_ratio)
        
        # Stationarity test on spread
        adf_stat, adf_pval, is_stationary = self.test_stationarity(spread)
        
        # Z-score
        zscore = self.calculate_zscore(spread)
        
        return {
            'cointegration_score': coint_score,
            'cointegration_pvalue': coint_pval,
            'is_cointegrated': is_coint,
            'hedge_ratio': hedge_ratio,
            'spread': spread,
            'spread_adf_stat': adf_stat,
            'spread_adf_pval': adf_pval,
            'is_spread_stationary': is_stationary,
            'zscore': zscore
        }
