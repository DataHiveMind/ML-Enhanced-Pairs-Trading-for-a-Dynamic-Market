#!/usr/bin/env python3
"""
Example script demonstrating basic pairs trading workflow.

This script shows how to:
1. Load stock price data
2. Find cointegrated pairs
3. Generate trading signals
4. Backtest a strategy
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from src.pairs_selection import find_cointegrated_pairs, test_cointegration
from src.trading_strategy import PairsTradingStrategy
from src.backtesting import backtest_strategy


def main():
    """Main execution function."""
    
    print("=" * 80)
    print("ML-Enhanced Pairs Trading - Example Workflow")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n[1/4] Loading stock data...")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        print(f"   ✓ Downloaded {len(data)} days of data for {len(tickers)} stocks")
    except Exception as e:
        print(f"   ✗ Error downloading data: {e}")
        return
    
    # Step 2: Find cointegrated pairs
    print("\n[2/4] Finding cointegrated pairs...")
    pairs = find_cointegrated_pairs(data, significance_level=0.05)
    
    if len(pairs) == 0:
        print("   ✗ No cointegrated pairs found. Try different stocks or time period.")
        return
    
    print(f"   ✓ Found {len(pairs)} cointegrated pair(s):")
    for i, (stock1, stock2, p_value) in enumerate(pairs[:3], 1):  # Show top 3
        print(f"      {i}. {stock1} - {stock2} (p-value: {p_value:.4f})")
    
    # Step 3: Generate trading signals for best pair
    print("\n[3/4] Generating trading signals...")
    stock1, stock2, p_value = pairs[0]  # Use best pair (lowest p-value)
    
    strategy = PairsTradingStrategy(entry_threshold=2.0, exit_threshold=0.5)
    spread = strategy.calculate_spread(data[stock1], data[stock2])
    signals = strategy.generate_signals(spread, window=20)
    
    num_trades = (signals['position'].diff() != 0).sum()
    print(f"   ✓ Generated signals for {stock1}-{stock2} pair")
    print(f"      Number of trade entries: {num_trades}")
    
    # Step 4: Backtest the strategy
    print("\n[4/4] Backtesting strategy...")
    returns, metrics = backtest_strategy(signals, data[stock1], data[stock2])
    
    print(f"   ✓ Backtest complete!")
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print(f"Total Return:     {metrics['total_return']:>10.2%}")
    print(f"Annual Return:    {metrics['annual_return']:>10.2%}")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:     {metrics['max_drawdown']:>10.2%}")
    print(f"Win Rate:         {metrics['win_rate']:>10.2%}")
    print("=" * 80)
    
    # Additional analysis
    print("\n📊 Additional Statistics:")
    print(f"   Average Daily Return:  {returns.mean():.4%}")
    print(f"   Daily Volatility:      {returns.std():.4%}")
    print(f"   Best Day:              {returns.max():.2%}")
    print(f"   Worst Day:             {returns.min():.2%}")
    
    print("\n" + "=" * 80)
    print("Example workflow completed successfully!")
    print("\nNext steps:")
    print("  - Explore notebooks/ for detailed analysis")
    print("  - Modify configs/config.yaml for different parameters")
    print("  - Implement ML models for signal enhancement")
    print("=" * 80)


if __name__ == '__main__':
    main()
