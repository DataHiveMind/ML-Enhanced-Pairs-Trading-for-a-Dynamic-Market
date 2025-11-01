"""
Backtesting framework for pairs trading strategies
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from ..strategies.ml_pairs_strategy import MLEnhancedPairsStrategy


class PairsBacktester:
    """Backtest pairs trading strategies"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction of trade value
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = None
    
    def run_backtest(self,
                    strategy: MLEnhancedPairsStrategy,
                    prices1: pd.Series,
                    prices2: pd.Series,
                    spread: pd.Series,
                    zscore: pd.Series,
                    train_split: float = 0.7) -> pd.DataFrame:
        """
        Run backtest on historical data
        
        Args:
            strategy: Trading strategy instance
            prices1: First stock prices
            prices2: Second stock prices
            spread: Spread series
            zscore: Z-score series
            train_split: Fraction of data for training ML models
            
        Returns:
            DataFrame with backtest results
        """
        # Split data into train and test
        split_idx = int(len(spread) * train_split)
        
        # Train ML models if enabled
        if strategy.use_lstm:
            train_features = strategy.lstm_predictor.prepare_features(
                spread[:split_idx],
                zscore[:split_idx],
                prices1[:split_idx],
                prices2[:split_idx]
            )
            train_target = spread[:split_idx].shift(-1)  # Predict next spread
            strategy.lstm_predictor.train(train_features, train_target, verbose=0)
        
        if strategy.use_regime:
            train_regime_features = strategy.regime_classifier.create_regime_features(
                prices1[:split_idx],
                prices2[:split_idx],
                spread[:split_idx]
            )
            train_regime_labels = strategy.regime_classifier.create_regime_labels(
                spread[:split_idx],
                zscore[:split_idx]
            )
            strategy.regime_classifier.train(train_regime_features, train_regime_labels)
        
        # Initialize backtest results
        results = []
        capital = self.initial_capital
        
        # Run through test period
        test_start_idx = split_idx
        
        for i in range(test_start_idx, len(spread)):
            timestamp = spread.index[i]
            current_spread = spread.iloc[i]
            current_zscore = zscore.iloc[i]
            current_price1 = prices1.iloc[i]
            current_price2 = prices2.iloc[i]
            
            # Prepare features for ML models
            lstm_features = None
            if strategy.use_lstm:
                try:
                    window_start = max(0, i - 30)
                    lstm_features = strategy.lstm_predictor.prepare_features(
                        spread[window_start:i+1],
                        zscore[window_start:i+1],
                        prices1[window_start:i+1],
                        prices2[window_start:i+1]
                    )
                except Exception:
                    pass
            
            regime_features = None
            regime = None
            if strategy.use_regime:
                try:
                    window_start = max(0, i - 30)
                    regime_features = strategy.regime_classifier.create_regime_features(
                        prices1[window_start:i+1],
                        prices2[window_start:i+1],
                        spread[window_start:i+1]
                    )
                    if len(regime_features) > 0:
                        regime = strategy.regime_classifier.predict(regime_features)[-1]
                except Exception:
                    pass
            
            # Generate signals
            signal_window = zscore[max(0, i-1):i+1]
            signals = strategy.generate_signals(
                spread[max(0, i-1):i+1],
                signal_window,
                prices1[max(0, i-1):i+1],
                prices2[max(0, i-1):i+1],
                lstm_features=lstm_features,
                regime_features=regime_features
            )
            
            signal = signals.iloc[-1] if len(signals) > 0 else 0
            
            # Execute trade
            trade_info = strategy.execute_trade(
                signal,
                current_spread,
                current_price1,
                current_price2,
                timestamp,
                capital
            )
            
            # Update capital
            if trade_info['pnl'] != 0:
                # Apply transaction costs
                trade_cost = abs(trade_info['pnl']) * self.transaction_cost
                capital += trade_info['pnl'] - trade_cost
            
            # Record results
            results.append({
                'timestamp': timestamp,
                'capital': capital,
                'position': strategy.position,
                'spread': current_spread,
                'zscore': current_zscore,
                'signal': signal,
                'regime': regime,
                'pnl': trade_info['pnl'],
                'action': trade_info['action']
            })
        
        self.results = pd.DataFrame(results)
        self.results.set_index('timestamp', inplace=True)
        
        return self.results
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics from backtest results"""
        if self.results is None:
            return {}
        
        capital_series = self.results['capital']
        returns = capital_series.pct_change().dropna()
        
        # Basic metrics
        total_return = (capital_series.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized metrics
        days = (self.results.index[-1] - self.results.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(capital_series)
        
        # Trade metrics
        trades = self.results[self.results['action'] != 'none']
        num_trades = len(trades)
        
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': capital_series.iloc[-1],
            'num_trades': num_trades,
            'win_rate': len(winning_trades) / num_trades if num_trades > 0 else 0,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum())
                           if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        excess_returns = returns.mean() - (risk_free_rate / 252)  # Daily risk-free rate
        sharpe = excess_returns / returns.std() * np.sqrt(252)  # Annualized
        
        return sharpe
    
    def _calculate_max_drawdown(self, capital_series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cummax = capital_series.cummax()
        drawdown = (capital_series - cummax) / cummax
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot backtest results
        
        Args:
            save_path: Path to save plot (optional)
        """
        if self.results is None:
            print("No results to plot. Run backtest first.")
            return
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # Capital curve
        axes[0].plot(self.results.index, self.results['capital'], label='Portfolio Value')
        axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        axes[0].set_title('Portfolio Value Over Time')
        axes[0].set_ylabel('Capital ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Spread and positions
        axes[1].plot(self.results.index, self.results['spread'], label='Spread', color='blue')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Mark positions
        long_positions = self.results[self.results['position'] == 1]
        short_positions = self.results[self.results['position'] == -1]
        
        axes[1].scatter(long_positions.index, long_positions['spread'], 
                       color='green', marker='^', s=50, label='Long', alpha=0.6)
        axes[1].scatter(short_positions.index, short_positions['spread'], 
                       color='red', marker='v', s=50, label='Short', alpha=0.6)
        
        axes[1].set_title('Spread and Positions')
        axes[1].set_ylabel('Spread')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Z-score
        axes[2].plot(self.results.index, self.results['zscore'], label='Z-Score', color='purple')
        axes[2].axhline(y=2, color='r', linestyle='--', label='Entry Threshold', alpha=0.5)
        axes[2].axhline(y=-2, color='r', linestyle='--', alpha=0.5)
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2].set_title('Z-Score Over Time')
        axes[2].set_ylabel('Z-Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Cumulative PnL
        cumulative_pnl = self.results['pnl'].cumsum()
        axes[3].plot(self.results.index, cumulative_pnl, label='Cumulative PnL', color='orange')
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[3].set_title('Cumulative Profit/Loss')
        axes[3].set_ylabel('PnL ($)')
        axes[3].set_xlabel('Date')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_summary(self):
        """Print backtest summary"""
        if self.results is None:
            print("No results to display. Run backtest first.")
            return
        
        metrics = self.calculate_metrics()
        
        print("=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        print(f"\nPerformance Metrics:")
        print(f"  Total Return:        {metrics['total_return']*100:.2f}%")
        print(f"  Annualized Return:   {metrics['annualized_return']*100:.2f}%")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']*100:.2f}%")
        print(f"  Final Capital:       ${metrics['final_capital']:,.2f}")
        print(f"\nTrading Metrics:")
        print(f"  Number of Trades:    {metrics['num_trades']}")
        print(f"  Win Rate:            {metrics['win_rate']*100:.2f}%")
        print(f"  Average Win:         ${metrics['avg_win']:.2f}")
        print(f"  Average Loss:        ${metrics['avg_loss']:.2f}")
        print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
        print("=" * 60)
