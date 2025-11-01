# ML-Enhanced Pairs Trading for Dynamic Markets

A sophisticated pairs trading strategy that combines classical statistical arbitrage techniques with modern machine learning methods to adapt to dynamic market regimes. This project demonstrates how traditional cointegration-based trading can be enhanced with LSTMs, Random Forests, and Reinforcement Learning for improved robustness and adaptability.

## 🎯 Overview

This project implements a **machine learning–enhanced pairs trading strategy** that:

- Uses **cointegration testing** to identify statistically related stock pairs
- Applies **z-score thresholds** for classical mean-reversion trading signals
- Leverages **LSTM neural networks** to predict spread mean reversion
- Employs **Random Forest classifiers** to identify market regimes (low volatility, high volatility, trending)
- Optionally uses **Deep Q-Learning (DQN)** for adaptive decision-making
- Includes a comprehensive **backtesting framework** with performance metrics and visualizations

## 🏗️ Architecture

```
ML-Enhanced-Pairs-Trading/
├── src/
│   ├── data/
│   │   ├── data_fetcher.py      # Data downloading and preprocessing
│   │   └── pair_selector.py     # Cointegration testing and pair selection
│   ├── models/
│   │   ├── lstm_model.py        # LSTM for spread prediction
│   │   ├── regime_classifier.py # Random Forest for regime detection
│   │   └── dqn_agent.py         # Deep Q-Learning agent
│   ├── strategies/
│   │   └── ml_pairs_strategy.py # ML-enhanced trading strategy
│   ├── backtesting/
│   │   └── backtester.py        # Backtesting framework
│   └── utils/
│       └── visualization.py     # Plotting utilities
├── notebooks/                    # Jupyter notebooks for analysis
├── main.py                       # Main execution pipeline
└── requirements.txt              # Python dependencies
```

## 🚀 Features

### Classical Statistical Arbitrage
- **Cointegration Testing**: Engle-Granger test to identify mean-reverting pairs
- **Hedge Ratio Calculation**: OLS regression for optimal pair weighting
- **Z-Score Trading**: Entry/exit signals based on spread deviation from mean
- **Stationarity Testing**: Augmented Dickey-Fuller test for spread validation

### Machine Learning Enhancements

#### 1. LSTM Spread Predictor
- Predicts future spread values to anticipate mean reversion
- Uses sequence-to-sequence architecture with dropout regularization
- Features: spread history, z-scores, price returns, rolling statistics
- Prevents trades when reversion is unlikely

#### 2. Regime Classifier (Random Forest)
- Identifies market regimes:
  - **Low volatility mean-reverting** (ideal for pairs trading)
  - **High volatility mean-reverting** (reduce position size)
  - **Trending/diverging** (avoid trading)
- Features: volatility, correlation, momentum, spread characteristics
- Adjusts trading behavior based on current regime

#### 3. DQN Reinforcement Learning Agent (Optional)
- Learns optimal trading actions through experience
- Actions: hold, long spread, short spread
- State representation includes spread, z-score, position, and ML predictions
- Experience replay and target networks for stable training

### Backtesting Framework
- Train/test split for realistic ML model evaluation
- Transaction cost modeling
- Comprehensive performance metrics:
  - Total and annualized returns
  - Sharpe ratio
  - Maximum drawdown
  - Win rate, profit factor
- Rich visualizations:
  - Portfolio value over time
  - Spread and position markers
  - Z-score analysis
  - Cumulative P&L

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/DataHiveMind/ML-Enhanced-Pairs-Trading-for-a-Dynamic-Market.git
cd ML-Enhanced-Pairs-Trading-for-a-Dynamic-Market

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Usage

### Quick Start

Run the main pipeline with default settings:

```bash
python main.py
```

This will:
1. Download historical data for a sample pair (GLD/GDX)
2. Test cointegration and calculate spread metrics
3. Train LSTM and Random Forest models
4. Run backtest with ML-enhanced signals
5. Display performance metrics and visualizations

### Custom Pair Analysis

```python
from src.data.data_fetcher import DataFetcher
from src.data.pair_selector import PairSelector
from src.strategies.ml_pairs_strategy import MLEnhancedPairsStrategy
from src.backtesting.backtester import PairsBacktester

# Fetch data
fetcher = DataFetcher(start_date='2020-01-01', end_date='2023-12-31')
prices = fetcher.fetch_stock_data(['STOCK1', 'STOCK2'])

# Test cointegration
selector = PairSelector()
metrics = selector.get_pair_metrics(prices['STOCK1'], prices['STOCK2'])

# Initialize strategy
strategy = MLEnhancedPairsStrategy(
    entry_zscore=2.0,
    exit_zscore=0.5,
    use_lstm=True,
    use_regime=True,
    use_rl=False
)

# Run backtest
backtester = PairsBacktester(initial_capital=100000)
results = backtester.run_backtest(
    strategy=strategy,
    prices1=prices['STOCK1'],
    prices2=prices['STOCK2'],
    spread=metrics['spread'],
    zscore=metrics['zscore']
)

# View results
backtester.print_summary()
backtester.plot_results()
```

## 📊 Example Results

The strategy has been tested on various cointegrated pairs including:
- **GLD/GDX** (Gold ETF / Gold Miners ETF)
- **PEP/KO** (Pepsi / Coca-Cola)
- **XLE/XOM** (Energy Sector ETF / Exxon Mobil)

Performance metrics typically show:
- Improved Sharpe ratios compared to classical z-score strategies
- Better drawdown control through regime-aware trading
- Higher win rates due to LSTM-based trade filtering

## 🔬 Methodology

### 1. Pair Selection
- Calculate correlation and cointegration for stock pairs
- Use Engle-Granger test (p-value < 0.05 for significance)
- Verify spread stationarity with ADF test

### 2. Feature Engineering
- **Spread features**: value, change, moving averages, volatility
- **Price features**: returns, momentum, RSI
- **Regime features**: volatility ratios, correlation, trend indicators

### 3. Model Training
- **LSTM**: Trained on 70% of data to predict next-day spread
- **Random Forest**: Trained on labeled regimes based on volatility and momentum
- **DQN**: Trained through simulation with reward = profit/loss

### 4. Signal Generation
- Classical signal: Trade when |z-score| > entry_threshold
- ML enhancement: Filter trades using LSTM predictions and regime
- Position sizing: Adjust based on signal strength and regime

### 5. Risk Management
- Stop-loss at z-score > 3σ
- Position limits based on capital
- Regime-based exposure reduction

## 📈 Performance Metrics

The backtesting framework calculates:

- **Return Metrics**: Total return, annualized return
- **Risk Metrics**: Sharpe ratio, maximum drawdown, volatility
- **Trading Metrics**: Number of trades, win rate, avg win/loss, profit factor

## 🛠️ Customization

### Adjust Strategy Parameters

```python
strategy = MLEnhancedPairsStrategy(
    entry_zscore=2.5,      # Higher = fewer but stronger signals
    exit_zscore=0.3,       # Lower = earlier exits
    stop_loss_zscore=4.0,  # Wider stop loss
    use_lstm=True,         # Enable/disable LSTM
    use_regime=True,       # Enable/disable regime classification
    use_rl=False           # Enable/disable RL agent
)
```

### Tune ML Models

```python
# LSTM configuration
lstm = LSTMSpreadPredictor(
    lookback_window=30,    # Longer history
    lstm_units=128,        # More complex model
    dropout_rate=0.3,      # More regularization
    learning_rate=0.0005   # Slower learning
)

# Random Forest configuration
regime = RegimeClassifier(
    n_estimators=200,      # More trees
    max_depth=15,          # Deeper trees
    random_state=42
)
```

## 📚 Dependencies

- **Data**: yfinance, pandas, numpy
- **ML/DL**: tensorflow, keras, torch, scikit-learn
- **Statistics**: statsmodels, scipy
- **RL**: gym, stable-baselines3
- **Visualization**: matplotlib, seaborn
- **Utils**: joblib, tqdm

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. 

- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- This code is not financial advice
- Always perform your own due diligence before trading
- Test thoroughly with paper trading before using real capital
- Consider transaction costs, slippage, and market impact in live trading

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional ML models (Transformers, GRU, etc.)
- More sophisticated feature engineering
- Multi-pair portfolio optimization
- Real-time data integration
- Enhanced risk management techniques
- Performance attribution analysis

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

## 🙏 Acknowledgments

This project combines concepts from:
- Classical pairs trading literature (Gatev et al., 1999)
- Modern machine learning for finance
- Reinforcement learning for trading (Deep Q-Learning)

---

**Built with ❤️ for quantitative traders and ML enthusiasts**