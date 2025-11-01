"""
Main pipeline for ML-Enhanced Pairs Trading Strategy
"""
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import custom modules
from src.data.data_fetcher import DataFetcher
from src.data.pair_selector import PairSelector
from src.models.lstm_model import LSTMSpreadPredictor
from src.models.regime_classifier import RegimeClassifier
from src.models.dqn_agent import DQNAgent
from src.strategies.ml_pairs_strategy import MLEnhancedPairsStrategy
from src.backtesting.backtester import PairsBacktester
from src.utils.visualization import (
    plot_cointegration_test,
    plot_pairs_correlation_matrix,
    plot_regime_analysis,
    plot_feature_importance
)


def main():
    """
    Main execution pipeline for ML-Enhanced Pairs Trading
    """
    print("=" * 80)
    print("ML-ENHANCED PAIRS TRADING STRATEGY")
    print("=" * 80)
    print("\nThis strategy combines classical statistical arbitrage with modern ML methods:")
    print("  • Cointegration testing for pair selection")
    print("  • Z-score based spread trading")
    print("  • LSTM for spread prediction")
    print("  • Random Forest for regime classification")
    print("  • Deep Q-Learning for adaptive decision making")
    print("=" * 80)
    
    # Configuration
    TICKERS = ['GLD', 'GDX']  # Example: Gold ETF and Gold Miners ETF (typically cointegrated)
    START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    INITIAL_CAPITAL = 100000
    
    print(f"\nConfiguration:")
    print(f"  Pair: {TICKERS[0]} / {TICKERS[1]}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,}")
    
    # Step 1: Fetch data
    print("\n" + "-" * 80)
    print("Step 1: Fetching historical data...")
    print("-" * 80)
    
    data_fetcher = DataFetcher(start_date=START_DATE, end_date=END_DATE)
    prices = data_fetcher.fetch_stock_data(TICKERS)
    
    print(f"  ✓ Downloaded {len(prices)} days of data")
    print(f"  ✓ Data range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # Step 2: Test cointegration
    print("\n" + "-" * 80)
    print("Step 2: Testing cointegration...")
    print("-" * 80)
    
    pair_selector = PairSelector()
    metrics = pair_selector.get_pair_metrics(prices[TICKERS[0]], prices[TICKERS[1]])
    
    print(f"  Cointegration p-value: {metrics['cointegration_pvalue']:.4f}")
    print(f"  Is cointegrated: {metrics['is_cointegrated']}")
    print(f"  Hedge ratio: {metrics['hedge_ratio']:.4f}")
    print(f"  Spread ADF p-value: {metrics['spread_adf_pval']:.4f}")
    print(f"  Is spread stationary: {metrics['is_spread_stationary']}")
    
    spread = metrics['spread']
    zscore = metrics['zscore']
    
    if not metrics['is_cointegrated']:
        print("\n  ⚠ WARNING: Pair is not cointegrated. Results may not be reliable.")
    else:
        print("\n  ✓ Pair is cointegrated - good for pairs trading!")
    
    # Step 3: Visualize pair relationship
    print("\n" + "-" * 80)
    print("Step 3: Visualizing pair relationship...")
    print("-" * 80)
    
    try:
        plot_cointegration_test(
            prices[TICKERS[0]], 
            prices[TICKERS[1]], 
            spread, 
            zscore,
            TICKERS[0], 
            TICKERS[1]
        )
        print("  ✓ Cointegration plots displayed")
    except Exception as e:
        print(f"  ⚠ Could not display plots: {e}")
    
    # Step 4: Initialize and train ML models
    print("\n" + "-" * 80)
    print("Step 4: Initializing ML-Enhanced Strategy...")
    print("-" * 80)
    
    strategy = MLEnhancedPairsStrategy(
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=3.0,
        use_lstm=True,
        use_regime=True,
        use_rl=False  # Can enable RL for more advanced scenarios
    )
    
    print("  ✓ Strategy initialized with:")
    print(f"    - Entry Z-score: ±{strategy.entry_zscore}")
    print(f"    - Exit Z-score: ±{strategy.exit_zscore}")
    print(f"    - Stop loss Z-score: ±{strategy.stop_loss_zscore}")
    print(f"    - LSTM enabled: {strategy.use_lstm}")
    print(f"    - Regime classifier enabled: {strategy.use_regime}")
    print(f"    - RL agent enabled: {strategy.use_rl}")
    
    # Step 5: Run backtest
    print("\n" + "-" * 80)
    print("Step 5: Running backtest...")
    print("-" * 80)
    
    backtester = PairsBacktester(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=0.001  # 0.1% transaction cost
    )
    
    print("  Training ML models and executing backtest...")
    results = backtester.run_backtest(
        strategy=strategy,
        prices1=prices[TICKERS[0]],
        prices2=prices[TICKERS[1]],
        spread=spread,
        zscore=zscore,
        train_split=0.7
    )
    
    print(f"  ✓ Backtest completed - {len(results)} periods simulated")
    
    # Step 6: Display results
    print("\n" + "-" * 80)
    print("Step 6: Performance Analysis")
    print("-" * 80)
    
    backtester.print_summary()
    
    # Step 7: Plot results
    print("\n" + "-" * 80)
    print("Step 7: Generating performance plots...")
    print("-" * 80)
    
    try:
        backtester.plot_results()
        print("  ✓ Performance plots displayed")
    except Exception as e:
        print(f"  ⚠ Could not display plots: {e}")
    
    # Step 8: Additional analysis
    print("\n" + "-" * 80)
    print("Step 8: Additional ML Model Analysis")
    print("-" * 80)
    
    if strategy.use_regime:
        try:
            feature_importance = strategy.regime_classifier.get_feature_importance()
            if feature_importance:
                print("\n  Top 10 Most Important Features for Regime Classification:")
                sorted_features = sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
                for i, (feature, importance) in enumerate(sorted_features, 1):
                    print(f"    {i}. {feature}: {importance:.4f}")
                
                plot_feature_importance(
                    feature_importance,
                    title="Regime Classifier Feature Importance"
                )
        except Exception as e:
            print(f"  ⚠ Could not analyze feature importance: {e}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Adjust strategy parameters for optimization")
    print("  2. Test on different stock pairs")
    print("  3. Implement portfolio of multiple pairs")
    print("  4. Deploy for live trading (with proper risk management)")
    print("=" * 80)


if __name__ == "__main__":
    main()
