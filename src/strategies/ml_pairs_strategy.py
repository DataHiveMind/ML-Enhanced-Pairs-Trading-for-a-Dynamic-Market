"""
ML-Enhanced Pairs Trading Strategy combining classical and ML approaches
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from ..data.pair_selector import PairSelector
from ..models.lstm_model import LSTMSpreadPredictor
from ..models.regime_classifier import RegimeClassifier
from ..models.dqn_agent import DQNAgent


class MLEnhancedPairsStrategy:
    """
    ML-Enhanced Pairs Trading Strategy
    
    Combines:
    - Classical cointegration and z-score based trading
    - LSTM for spread prediction
    - Random Forest for regime classification
    - DQN for adaptive decision making
    """
    
    def __init__(self, 
                 entry_zscore: float = 2.0,
                 exit_zscore: float = 0.5,
                 stop_loss_zscore: float = 3.0,
                 use_lstm: bool = True,
                 use_regime: bool = True,
                 use_rl: bool = True):
        """
        Initialize ML-Enhanced strategy
        
        Args:
            entry_zscore: Z-score threshold for entry
            exit_zscore: Z-score threshold for exit
            stop_loss_zscore: Z-score threshold for stop loss
            use_lstm: Whether to use LSTM predictions
            use_regime: Whether to use regime classification
            use_rl: Whether to use RL agent
        """
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.use_lstm = use_lstm
        self.use_regime = use_regime
        self.use_rl = use_rl
        
        # Initialize components
        self.pair_selector = PairSelector()
        self.lstm_predictor = LSTMSpreadPredictor() if use_lstm else None
        self.regime_classifier = RegimeClassifier() if use_regime else None
        self.dqn_agent = None  # Will be initialized when state_dim is known
        
        # Trading state
        self.position = 0  # -1: short spread, 0: no position, 1: long spread
        self.entry_spread = None
        self.trades = []
    
    def generate_signals(self, 
                        spread: pd.Series,
                        zscore: pd.Series,
                        prices1: pd.Series,
                        prices2: pd.Series,
                        lstm_features: Optional[pd.DataFrame] = None,
                        regime_features: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate trading signals combining classical and ML approaches
        
        Args:
            spread: Spread series
            zscore: Z-score series
            prices1: First stock prices
            prices2: Second stock prices
            lstm_features: Features for LSTM (if using LSTM)
            regime_features: Features for regime classifier (if using regime)
            
        Returns:
            Series of trading signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=zscore.index)
        
        # Classical z-score signals
        classical_signals = self._generate_classical_signals(zscore)
        
        # LSTM predictions (if enabled)
        lstm_signals = None
        if self.use_lstm and lstm_features is not None:
            lstm_signals = self._generate_lstm_signals(lstm_features, spread)
        
        # Regime classification (if enabled)
        regime_signals = None
        if self.use_regime and regime_features is not None:
            regime_signals = self._generate_regime_signals(regime_features)
        
        # Combine signals
        for i in range(len(signals)):
            signal = classical_signals.iloc[i]
            
            # Modify signal based on LSTM prediction
            if lstm_signals is not None and i < len(lstm_signals):
                # If LSTM predicts no reversion, reduce signal strength or skip
                if lstm_signals[i] == 0:
                    signal = 0
            
            # Modify signal based on regime
            if regime_signals is not None and i < len(regime_signals):
                regime = regime_signals[i]
                # In trending regime (2), avoid or reduce trading
                if regime == 2:
                    signal = 0
                # In high volatility regime (1), reduce position size (handled in execution)
            
            signals.iloc[i] = signal
        
        return signals
    
    def _generate_classical_signals(self, zscore: pd.Series) -> pd.Series:
        """Generate signals based on classical z-score thresholds"""
        signals = pd.Series(0, index=zscore.index)
        
        # Entry signals
        signals[zscore > self.entry_zscore] = -1  # Short spread
        signals[zscore < -self.entry_zscore] = 1   # Long spread
        
        # Exit signals
        signals[abs(zscore) < self.exit_zscore] = 0
        
        return signals
    
    def _generate_lstm_signals(self, features: pd.DataFrame, spread: pd.Series) -> np.ndarray:
        """Generate signals based on LSTM predictions"""
        try:
            predictions = self.lstm_predictor.predict(features)
            
            # Align predictions with spread
            aligned_spread = spread.iloc[-len(predictions):].values
            
            # If LSTM predicts spread moving away from mean, signal = 0 (no trade)
            # If predicts mean reversion, signal = 1 (trade)
            signals = np.zeros(len(predictions))
            for i in range(len(predictions)):
                if abs(predictions[i]) < abs(aligned_spread[i]):
                    signals[i] = 1  # Reversion expected
                else:
                    signals[i] = 0  # Continuation expected
            
            return signals
        except Exception:
            return np.zeros(len(features))
    
    def _generate_regime_signals(self, features: pd.DataFrame) -> np.ndarray:
        """Generate signals based on regime classification"""
        try:
            regimes = self.regime_classifier.predict(features)
            return regimes
        except Exception:
            return np.zeros(len(features))
    
    def generate_rl_action(self, state: np.ndarray, training: bool = False) -> int:
        """
        Generate action using DQN agent
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Action (0: hold, 1: long, 2: short)
        """
        if self.dqn_agent is None:
            # Initialize DQN agent with state dimension
            self.dqn_agent = DQNAgent(state_dim=len(state))
        
        return self.dqn_agent.select_action(state, training=training)
    
    def calculate_position_size(self, 
                               zscore: float,
                               regime: Optional[int] = None,
                               capital: float = 100000) -> float:
        """
        Calculate position size based on signal strength and regime
        
        Args:
            zscore: Current z-score
            regime: Current market regime
            capital: Available capital
            
        Returns:
            Position size in dollars
        """
        # Base position size (e.g., 10% of capital)
        base_size = capital * 0.1
        
        # Adjust based on z-score magnitude (stronger signal = larger position)
        zscore_multiplier = min(abs(zscore) / self.entry_zscore, 2.0)
        
        # Adjust based on regime
        regime_multiplier = 1.0
        if regime == 1:  # High volatility
            regime_multiplier = 0.5  # Reduce size
        elif regime == 2:  # Trending
            regime_multiplier = 0.0  # No trade
        
        position_size = base_size * zscore_multiplier * regime_multiplier
        
        return position_size
    
    def execute_trade(self, 
                     signal: int,
                     spread: float,
                     prices1: float,
                     prices2: float,
                     timestamp: pd.Timestamp,
                     capital: float = 100000) -> Dict:
        """
        Execute trade based on signal
        
        Args:
            signal: Trading signal (-1, 0, 1)
            spread: Current spread
            prices1: Current price of stock 1
            prices2: Current price of stock 2
            timestamp: Trade timestamp
            capital: Available capital
            
        Returns:
            Trade information dictionary
        """
        trade_info = {
            'timestamp': timestamp,
            'signal': signal,
            'spread': spread,
            'prices1': prices1,
            'prices2': prices2,
            'action': 'none',
            'position': self.position,
            'pnl': 0
        }
        
        # Check for stop loss
        if self.position != 0 and self.entry_spread is not None:
            spread_change = spread - self.entry_spread
            if self.position == 1 and spread_change < -self.stop_loss_zscore:
                # Stop loss for long position
                signal = 0  # Force exit
            elif self.position == -1 and spread_change > self.stop_loss_zscore:
                # Stop loss for short position
                signal = 0  # Force exit
        
        # Execute based on current position and signal
        if self.position == 0:
            # No position - enter if signal
            if signal != 0:
                self.position = signal
                self.entry_spread = spread
                trade_info['action'] = 'enter_long' if signal == 1 else 'enter_short'
        
        elif self.position != 0:
            # Have position - check for exit or reversal
            if signal == 0:
                # Exit position
                pnl = self._calculate_pnl(spread, self.entry_spread, self.position)
                trade_info['pnl'] = pnl
                trade_info['action'] = 'exit'
                self.position = 0
                self.entry_spread = None
            
            elif signal == -self.position:
                # Reverse position
                pnl = self._calculate_pnl(spread, self.entry_spread, self.position)
                trade_info['pnl'] = pnl
                trade_info['action'] = 'reverse'
                self.position = signal
                self.entry_spread = spread
        
        # Record trade
        if trade_info['action'] != 'none':
            self.trades.append(trade_info)
        
        return trade_info
    
    def _calculate_pnl(self, exit_spread: float, entry_spread: float, position: int) -> float:
        """Calculate profit/loss for a trade"""
        spread_change = exit_spread - entry_spread
        
        # Long spread profits when spread increases
        # Short spread profits when spread decreases
        pnl = position * spread_change
        
        return pnl
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics from trades"""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        closed_trades = trades_df[trades_df['pnl'] != 0]
        
        if len(closed_trades) == 0:
            return {}
        
        total_pnl = closed_trades['pnl'].sum()
        num_trades = len(closed_trades)
        winning_trades = closed_trades[closed_trades['pnl'] > 0]
        losing_trades = closed_trades[closed_trades['pnl'] < 0]
        
        metrics = {
            'total_pnl': total_pnl,
            'num_trades': num_trades,
            'win_rate': len(winning_trades) / num_trades if num_trades > 0 else 0,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) 
                           if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
        }
        
        return metrics
