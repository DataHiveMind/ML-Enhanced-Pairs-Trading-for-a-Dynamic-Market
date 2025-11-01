# ML-Enhanced Pairs Trading for Dynamic Markets

A machine learning-enhanced pairs trading system designed to identify and exploit mean-reversion opportunities in financial markets.

## Overview

This project implements a sophisticated pairs trading strategy that combines traditional statistical arbitrage techniques with modern machine learning algorithms to improve trading performance in dynamic market conditions.

### Key Features

- **Statistical Cointegration Analysis**: Identify pairs of assets with strong long-term relationships
- **ML-Enhanced Signal Generation**: Use machine learning models to predict optimal entry/exit points
- **Comprehensive Backtesting**: Evaluate strategy performance with detailed metrics
- **Feature Engineering**: Create technical and spread-based features for ML models
- **Flexible Configuration**: Easy-to-use YAML configuration for strategy parameters

## Project Structure

```
ML-Enhanced-Pairs-Trading-for-a-Dynamic-Market/
│
├── README.md                # Project overview and documentation
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── .gitignore              # Git ignore rules
│
├── data/                    # Data directory
│   ├── raw/                # Raw market data
│   ├── processed/          # Cleaned and processed data
│   └── external/           # External datasets
│
├── notebooks/              # Jupyter notebooks
│   ├── exploratory/       # Data exploration and analysis
│   └── modeling/          # Model development and experiments
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── utils.py           # Utility functions
│   ├── pairs_selection.py  # Cointegration and pair selection
│   ├── feature_engineering.py  # Feature creation
│   ├── trading_strategy.py     # Trading signal generation
│   └── backtesting.py          # Strategy backtesting
│
├── models/                 # Saved ML models
│
├── configs/                # Configuration files
│   └── config.yaml        # Main configuration
│
├── tests/                  # Unit tests
│   └── test_utils.py
│
├── docs/                   # Documentation
│   └── README.md
│
└── reports/                # Generated reports and visualizations
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DataHiveMind/ML-Enhanced-Pairs-Trading-for-a-Dynamic-Market.git
cd ML-Enhanced-Pairs-Trading-for-a-Dynamic-Market
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Configure Your Strategy

Edit `configs/config.yaml` to set your trading parameters:
- Stock tickers to analyze
- Cointegration significance level
- Entry/exit thresholds
- Model hyperparameters

### 2. Identify Cointegrated Pairs

```python
from src.pairs_selection import find_cointegrated_pairs
import pandas as pd

# Load your price data
data = pd.read_csv('data/raw/prices.csv', index_col='Date', parse_dates=True)

# Find cointegrated pairs
pairs = find_cointegrated_pairs(data, significance_level=0.05)
print(f"Found {len(pairs)} cointegrated pairs")
```

### 3. Generate Trading Signals

```python
from src.trading_strategy import PairsTradingStrategy

strategy = PairsTradingStrategy(entry_threshold=2.0, exit_threshold=0.5)
spread = strategy.calculate_spread(data['STOCK1'], data['STOCK2'])
signals = strategy.generate_signals(spread, window=20)
```

### 4. Backtest the Strategy

```python
from src.backtesting import backtest_strategy

returns, metrics = backtest_strategy(signals, data['STOCK1'], data['STOCK2'])
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Documentation

Detailed documentation is available in the `docs/` directory:
- [Methodology](docs/README.md)
- API Reference
- Tutorials

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading in financial markets involves risk, and you should not trade with money you cannot afford to lose. Past performance is not indicative of future results.

## Acknowledgments

- Statistical methods based on established pairs trading research
- Machine learning techniques adapted for financial time series
- Inspired by quantitative trading literature and academic research