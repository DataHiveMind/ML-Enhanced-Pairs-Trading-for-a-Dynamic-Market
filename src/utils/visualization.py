"""
Visualization utilities for pairs trading analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List


def plot_cointegration_test(prices1: pd.Series, prices2: pd.Series, 
                            spread: pd.Series, zscore: pd.Series,
                            ticker1: str, ticker2: str,
                            save_path: Optional[str] = None):
    """
    Plot cointegration analysis results
    
    Args:
        prices1: First stock prices
        prices2: Second stock prices
        spread: Spread series
        zscore: Z-score series
        ticker1: First ticker symbol
        ticker2: Second ticker symbol
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Normalized prices
    norm_prices1 = prices1 / prices1.iloc[0] * 100
    norm_prices2 = prices2 / prices2.iloc[0] * 100
    
    axes[0].plot(norm_prices1.index, norm_prices1, label=ticker1, linewidth=2)
    axes[0].plot(norm_prices2.index, norm_prices2, label=ticker2, linewidth=2)
    axes[0].set_title(f'Normalized Prices: {ticker1} vs {ticker2}')
    axes[0].set_ylabel('Normalized Price (Base=100)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Spread
    axes[1].plot(spread.index, spread, color='blue', linewidth=2)
    axes[1].axhline(y=spread.mean(), color='red', linestyle='--', label='Mean')
    axes[1].fill_between(spread.index, 
                         spread.mean() - spread.std(), 
                         spread.mean() + spread.std(), 
                         alpha=0.2, color='gray', label='±1 Std')
    axes[1].set_title('Spread')
    axes[1].set_ylabel('Spread')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Z-score
    axes[2].plot(zscore.index, zscore, color='purple', linewidth=2)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[2].axhline(y=2, color='red', linestyle='--', label='±2σ')
    axes[2].axhline(y=-2, color='red', linestyle='--')
    axes[2].axhline(y=3, color='orange', linestyle='--', label='±3σ')
    axes[2].axhline(y=-3, color='orange', linestyle='--')
    axes[2].set_title('Z-Score')
    axes[2].set_ylabel('Z-Score')
    axes[2].set_xlabel('Date')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_pairs_correlation_matrix(prices: pd.DataFrame, 
                                  title: str = "Stock Correlation Matrix",
                                  save_path: Optional[str] = None):
    """
    Plot correlation matrix of stock returns
    
    Args:
        prices: DataFrame with stock prices
        title: Plot title
        save_path: Path to save plot
    """
    returns = prices.pct_change().dropna()
    corr_matrix = returns.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_regime_analysis(regime_labels: pd.Series, spread: pd.Series, 
                        zscore: pd.Series, save_path: Optional[str] = None):
    """
    Plot regime classification results
    
    Args:
        regime_labels: Series with regime classifications
        spread: Spread series
        zscore: Z-score series
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Define regime colors
    regime_colors = {0: 'green', 1: 'orange', 2: 'red'}
    regime_names = {0: 'Low Vol Mean-Reverting', 1: 'High Vol Mean-Reverting', 2: 'Trending'}
    
    # Spread with regime coloring
    for regime in [0, 1, 2]:
        mask = regime_labels == regime
        if mask.any():
            regime_data = spread[mask]
            axes[0].scatter(regime_data.index, regime_data.values, 
                          c=regime_colors[regime], label=regime_names[regime], 
                          alpha=0.6, s=20)
    
    axes[0].set_title('Spread by Market Regime')
    axes[0].set_ylabel('Spread')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Z-score with regime coloring
    for regime in [0, 1, 2]:
        mask = regime_labels == regime
        if mask.any():
            regime_data = zscore[mask]
            axes[1].scatter(regime_data.index, regime_data.values, 
                          c=regime_colors[regime], label=regime_names[regime], 
                          alpha=0.6, s=20)
    
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].axhline(y=2, color='gray', linestyle='--', alpha=0.5)
    axes[1].axhline(y=-2, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title('Z-Score by Market Regime')
    axes[1].set_ylabel('Z-Score')
    axes[1].set_xlabel('Date')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(feature_importance: dict, 
                           top_n: int = 15,
                           title: str = "Feature Importance",
                           save_path: Optional[str] = None):
    """
    Plot feature importance from ML models
    
    Args:
        feature_importance: Dictionary of feature names and importances
        top_n: Number of top features to display
        title: Plot title
        save_path: Path to save plot
    """
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    features, importances = zip(*top_features)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), importances, color='steelblue')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
