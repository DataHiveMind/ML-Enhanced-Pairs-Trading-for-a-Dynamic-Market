# Quick Start Guide

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/DataHiveMind/ML-Enhanced-Pairs-Trading-for-a-Dynamic-Market.git
cd ML-Enhanced-Pairs-Trading-for-a-Dynamic-Market
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Demo

Start with the demo to see the strategy in action with synthetic data:

```bash
python demo.py
```

This will:
- Generate synthetic cointegrated stock data
- Test cointegration
- Train LSTM and Random Forest models
- Run a backtest simulation
- Display performance metrics

Expected output:
```
ML-ENHANCED PAIRS TRADING STRATEGY - DEMO
...
✓ Successfully demonstrated cointegration testing
✓ Trained LSTM model for spread prediction
✓ Trained Random Forest for regime classification
✓ Executed ML-enhanced pairs trading strategy
```

## Running with Real Data

Execute the full pipeline with real stock data:

```bash
python main.py
```

This will:
1. Download historical data for GLD/GDX pair (or customize tickers in main.py)
2. Test cointegration and calculate spread
3. Visualize the pair relationship
4. Train ML models (LSTM, Random Forest)
5. Run backtest with ML-enhanced signals
6. Display comprehensive performance metrics and plots

## Customizing the Strategy

### Change Stock Pair

Edit `main.py`:
```python
TICKERS = ['STOCK1', 'STOCK2']  # Replace with your tickers
```

### Adjust Strategy Parameters

```python
strategy = MLEnhancedPairsStrategy(
    entry_zscore=2.5,      # Entry threshold
    exit_zscore=0.3,       # Exit threshold
    stop_loss_zscore=4.0,  # Stop loss
    use_lstm=True,         # Enable LSTM
    use_regime=True,       # Enable regime classification
    use_rl=False           # Enable RL (optional)
)
```

### Tune ML Models

**LSTM:**
```python
lstm = LSTMSpreadPredictor(
    lookback_window=30,    # History length
    lstm_units=128,        # Model complexity
    dropout_rate=0.3,      # Regularization
    learning_rate=0.0005   # Learning rate
)
```

**Random Forest:**
```python
regime = RegimeClassifier(
    n_estimators=200,      # Number of trees
    max_depth=15,          # Tree depth
    random_state=42
)
```

## Using the Jupyter Notebook

Launch Jupyter:
```bash
jupyter notebook notebooks/pairs_trading_example.ipynb
```

The notebook provides an interactive environment for:
- Step-by-step execution
- Data visualization
- Model training
- Performance analysis
- Strategy comparison

## Testing Components

Run the component tests:
```bash
python test_components.py
```

This validates that all modules are working correctly.

## Example Pairs to Try

### Highly Correlated Pairs:
- **GLD/GDX** - Gold ETF vs Gold Miners ETF
- **PEP/KO** - Pepsi vs Coca-Cola
- **XLE/XOM** - Energy Sector ETF vs Exxon Mobil
- **USO/XLE** - Oil ETF vs Energy Sector ETF

### Sector Pairs:
- **JPM/BAC** - Major banks
- **AAPL/MSFT** - Big tech
- **WMT/TGT** - Retail

## Troubleshooting

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Data Download Issues
```bash
# Check internet connection
# Verify ticker symbols are correct
# Try different date ranges
```

### CUDA Warnings
These are informational and can be ignored. The code will run on CPU.

### Memory Issues
- Reduce lookback_window in LSTM
- Decrease n_estimators in Random Forest
- Use smaller date ranges

## Project Structure

```
ML-Enhanced-Pairs-Trading/
├── src/
│   ├── data/              # Data fetching and pair selection
│   ├── models/            # ML models (LSTM, RF, DQN)
│   ├── strategies/        # Trading strategy implementation
│   ├── backtesting/       # Backtesting framework
│   └── utils/             # Visualization utilities
├── notebooks/             # Jupyter notebooks
├── main.py               # Main pipeline
├── demo.py               # Demo with synthetic data
├── test_components.py    # Component tests
└── requirements.txt      # Dependencies
```

## Next Steps

1. **Experiment with different pairs** - Some pairs are more stable than others
2. **Optimize parameters** - Use grid search or Bayesian optimization
3. **Add more features** - Incorporate volume, sentiment, or macro data
4. **Implement portfolio** - Trade multiple pairs simultaneously
5. **Paper trade** - Test with live data before using real capital

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the README for detailed documentation
- Review the example notebook for guidance

## Disclaimer

⚠️ **Important:** This software is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always use proper risk management and never trade with money you cannot afford to lose.
