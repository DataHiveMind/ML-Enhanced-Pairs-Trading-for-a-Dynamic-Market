"""
ML-Enhanced Pairs Trading for Dynamic Markets

This package provides tools for pairs trading using machine learning techniques.
"""

__version__ = '0.1.0'

from . import utils
from . import pairs_selection
from . import feature_engineering
from . import trading_strategy
from . import backtesting

__all__ = [
    'utils',
    'pairs_selection',
    'feature_engineering',
    'trading_strategy',
    'backtesting'
]
