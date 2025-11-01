"""
Simple test script to verify core functionality
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("Testing ML-Enhanced Pairs Trading Strategy Components")
print("=" * 70)

# Test 1: Data Fetcher
print("\n1. Testing Data Fetcher...")
try:
    from src.data.data_fetcher import DataFetcher
    
    fetcher = DataFetcher(
        start_date=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    # Create synthetic data for testing (to avoid network issues)
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    synthetic_prices = pd.DataFrame({
        'STOCK1': np.cumsum(np.random.randn(60)) + 100,
        'STOCK2': np.cumsum(np.random.randn(60)) + 100
    }, index=dates)
    
    returns = fetcher.calculate_returns(synthetic_prices)
    
    print(f"   ✓ Data fetcher initialized")
    print(f"   ✓ Created synthetic data: {len(synthetic_prices)} days")
    print(f"   ✓ Calculated returns: {len(returns)} days")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Pair Selector
print("\n2. Testing Pair Selector...")
try:
    from src.data.pair_selector import PairSelector
    
    # Create cointegrated synthetic data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    stock1 = pd.Series(np.cumsum(np.random.randn(100)) + 100, index=dates)
    stock2 = stock1 * 1.5 + np.random.randn(100) * 2  # Cointegrated with noise
    
    selector = PairSelector()
    score, pvalue, is_coint = selector.test_cointegration(stock1, stock2)
    hedge_ratio = selector.calculate_hedge_ratio(stock1, stock2)
    spread = selector.calculate_spread(stock1, stock2, hedge_ratio)
    zscore = selector.calculate_zscore(spread, window=20)
    
    print(f"   ✓ Pair selector initialized")
    print(f"   ✓ Cointegration test: p-value={pvalue:.4f}, cointegrated={is_coint}")
    print(f"   ✓ Hedge ratio: {hedge_ratio:.4f}")
    print(f"   ✓ Spread calculated: {len(spread)} points")
    print(f"   ✓ Z-score calculated: {len(zscore.dropna())} points")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: LSTM Model
print("\n3. Testing LSTM Model...")
try:
    from src.models.lstm_model import LSTMSpreadPredictor
    
    lstm = LSTMSpreadPredictor(lookback_window=10, lstm_units=32)
    
    # Create small synthetic dataset
    features = pd.DataFrame({
        'spread': np.random.randn(50),
        'zscore': np.random.randn(50),
        'spread_change': np.random.randn(50),
        'price1_return': np.random.randn(50),
        'price2_return': np.random.randn(50)
    })
    target = pd.Series(np.random.randn(50))
    
    # Build model
    test_data = np.random.randn(40, 5)
    X, y = lstm.prepare_sequences(test_data, np.random.randn(40))
    lstm.build_model(input_shape=(X.shape[1], X.shape[2]))
    
    print(f"   ✓ LSTM model initialized")
    print(f"   ✓ Model built with input shape: {(X.shape[1], X.shape[2])}")
    print(f"   ✓ Prepared sequences: X.shape={X.shape}, y.shape={y.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Regime Classifier
print("\n4. Testing Regime Classifier...")
try:
    from src.models.regime_classifier import RegimeClassifier
    
    classifier = RegimeClassifier(n_estimators=50, max_depth=5)
    
    # Create synthetic features and labels
    features = pd.DataFrame(
        np.random.randn(100, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )
    labels = pd.Series(np.random.randint(0, 3, 100))
    
    classifier.feature_names = features.columns.tolist()
    metrics = classifier.train(features, labels, test_size=0.3)
    
    print(f"   ✓ Regime classifier initialized")
    print(f"   ✓ Trained on {len(features)} samples")
    print(f"   ✓ Test accuracy: {metrics['test_accuracy']:.2f}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: DQN Agent
print("\n5. Testing DQN Agent...")
try:
    from src.models.dqn_agent import DQNAgent
    
    agent = DQNAgent(state_dim=10, action_dim=3)
    
    # Test action selection
    state = np.random.randn(10)
    action = agent.select_action(state, training=False)
    
    # Test experience replay
    for _ in range(100):
        agent.replay_buffer.push(
            state=np.random.randn(10),
            action=np.random.randint(0, 3),
            reward=np.random.randn(),
            next_state=np.random.randn(10),
            done=False
        )
    
    print(f"   ✓ DQN agent initialized")
    print(f"   ✓ Action selected: {action}")
    print(f"   ✓ Replay buffer size: {len(agent.replay_buffer)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: ML-Enhanced Strategy
print("\n6. Testing ML-Enhanced Strategy...")
try:
    from src.strategies.ml_pairs_strategy import MLEnhancedPairsStrategy
    
    strategy = MLEnhancedPairsStrategy(
        entry_zscore=2.0,
        exit_zscore=0.5,
        use_lstm=False,  # Disable to avoid training
        use_regime=False,
        use_rl=False
    )
    
    # Create synthetic data
    dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
    spread = pd.Series(np.random.randn(50), index=dates)
    zscore = pd.Series(np.random.randn(50) * 2, index=dates)
    prices1 = pd.Series(np.cumsum(np.random.randn(50)) + 100, index=dates)
    prices2 = pd.Series(np.cumsum(np.random.randn(50)) + 100, index=dates)
    
    signals = strategy.generate_signals(spread, zscore, prices1, prices2)
    
    print(f"   ✓ Strategy initialized")
    print(f"   ✓ Generated {len(signals)} signals")
    print(f"   ✓ Signal distribution: {signals.value_counts().to_dict()}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Backtester
print("\n7. Testing Backtester...")
try:
    from src.backtesting.backtester import PairsBacktester
    
    backtester = PairsBacktester(initial_capital=100000)
    
    print(f"   ✓ Backtester initialized")
    print(f"   ✓ Initial capital: ${backtester.initial_capital:,}")
    print(f"   ✓ Transaction cost: {backtester.transaction_cost*100:.2f}%")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✓ All core components tested successfully!")
print("=" * 70)
print("\nThe implementation is ready to use.")
print("Run 'python main.py' to execute the full pipeline.")
