"""
LSTM model for spread prediction
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import joblib


class LSTMSpreadPredictor:
    """LSTM model to predict spread mean reversion"""
    
    def __init__(self, lookback_window: int = 20, lstm_units: int = 64, 
                 dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        Initialize LSTM predictor
        
        Args:
            lookback_window: Number of time steps to look back
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.lookback_window = lookback_window
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: (timesteps, features)
        """
        self.model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    def prepare_sequences(self, data: np.ndarray, 
                         target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare sequences for LSTM training/prediction
        
        Args:
            data: Input data array
            target: Target values (for training)
            
        Returns:
            Tuple of (X sequences, y targets or None)
        """
        X = []
        y = [] if target is not None else None
        
        for i in range(len(data) - self.lookback_window):
            X.append(data[i:i + self.lookback_window])
            if target is not None:
                y.append(target[i + self.lookback_window])
        
        X = np.array(X)
        y = np.array(y) if y is not None else None
        
        return X, y
    
    def prepare_features(self, spread: pd.Series, zscore: pd.Series, 
                        prices1: pd.Series, prices2: pd.Series) -> pd.DataFrame:
        """
        Create feature set for LSTM
        
        Args:
            spread: Spread series
            zscore: Z-score of spread
            prices1: First stock prices
            prices2: Second stock prices
            
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=spread.index)
        
        features['spread'] = spread
        features['zscore'] = zscore
        features['spread_change'] = spread.diff()
        features['price1_return'] = prices1.pct_change()
        features['price2_return'] = prices2.pct_change()
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'spread_ma_{window}'] = spread.rolling(window).mean()
            features[f'spread_std_{window}'] = spread.rolling(window).std()
        
        features = features.dropna()
        
        return features
    
    def train(self, features: pd.DataFrame, target: pd.Series, 
              epochs: int = 50, batch_size: int = 32, 
              validation_split: float = 0.2, verbose: int = 0) -> None:
        """
        Train LSTM model
        
        Args:
            features: Feature DataFrame
            target: Target values (future spread or spread change)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation split ratio
            verbose: Verbosity level
        """
        # Align features and target
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features.values)
        
        # Prepare sequences
        X, y = self.prepare_sequences(features_scaled, target.values)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Train model
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with trained model
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale features
        features_scaled = self.scaler.transform(features.values)
        
        # Prepare sequences
        X, _ = self.prepare_sequences(features_scaled)
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        
        return predictions.flatten()
    
    def predict_reversion_probability(self, features: pd.DataFrame, 
                                     threshold: float = 0.0) -> np.ndarray:
        """
        Predict probability of mean reversion
        
        Args:
            features: Feature DataFrame
            threshold: Threshold for reversion (spread moving towards 0)
            
        Returns:
            Binary predictions (1 for reversion, 0 for continuation)
        """
        predictions = self.predict(features)
        
        # If prediction is close to threshold, consider it mean reversion
        reversion_probs = (np.abs(predictions) < np.abs(threshold)).astype(int)
        
        return reversion_probs
    
    def save_model(self, model_path: str, scaler_path: str) -> None:
        """Save model and scaler"""
        if self.model is not None:
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
    
    def load_model(self, model_path: str, scaler_path: str) -> None:
        """Load model and scaler"""
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
