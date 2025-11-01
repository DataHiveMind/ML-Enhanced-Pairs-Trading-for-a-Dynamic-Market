"""
Random Forest model for market regime classification
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import joblib


class RegimeClassifier:
    """Random Forest classifier for identifying market regimes"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, 
                 random_state: int = 42):
        """
        Initialize Random Forest regime classifier
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def create_regime_features(self, prices1: pd.Series, prices2: pd.Series, 
                              spread: pd.Series, market_index: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Create features for regime classification
        
        Args:
            prices1: First stock prices
            prices2: Second stock prices
            spread: Spread series
            market_index: Market index prices (optional)
            
        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=spread.index)
        
        # Volatility features
        for window in [5, 10, 20]:
            features[f'volatility_stock1_{window}'] = prices1.pct_change().rolling(window).std()
            features[f'volatility_stock2_{window}'] = prices2.pct_change().rolling(window).std()
            features[f'spread_volatility_{window}'] = spread.rolling(window).std()
        
        # Trend features (moving average slopes)
        for window in [10, 20]:
            features[f'trend_stock1_{window}'] = prices1.rolling(window).mean().diff()
            features[f'trend_stock2_{window}'] = prices2.rolling(window).mean().diff()
            features[f'spread_trend_{window}'] = spread.rolling(window).mean().diff()
        
        # Correlation features
        for window in [10, 20, 30]:
            returns1 = prices1.pct_change()
            returns2 = prices2.pct_change()
            features[f'correlation_{window}'] = returns1.rolling(window).corr(returns2)
        
        # Volume indicators (if available, using price changes as proxy)
        features['price1_momentum'] = prices1.pct_change(5)
        features['price2_momentum'] = prices2.pct_change(5)
        
        # Spread characteristics
        features['spread_mean_reversion_speed'] = spread.diff().abs()
        features['spread_autocorr'] = spread.rolling(20).apply(
            lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
        )
        
        # Market features (if market index provided)
        if market_index is not None:
            market_returns = market_index.pct_change()
            features['market_volatility'] = market_returns.rolling(20).std()
            features['market_trend'] = market_index.rolling(20).mean().diff()
            features['beta_stock1'] = returns1.rolling(30).cov(market_returns) / market_returns.rolling(30).var()
            features['beta_stock2'] = returns2.rolling(30).cov(market_returns) / market_returns.rolling(30).var()
        
        features = features.dropna()
        self.feature_names = features.columns.tolist()
        
        return features
    
    def create_regime_labels(self, spread: pd.Series, zscore: pd.Series, 
                           volatility_threshold: float = 1.5) -> pd.Series:
        """
        Create regime labels based on spread characteristics
        
        Regimes:
        0 - Low volatility mean-reverting
        1 - High volatility mean-reverting
        2 - Trending/diverging
        
        Args:
            spread: Spread series
            zscore: Z-score of spread
            volatility_threshold: Threshold for high volatility regime
            
        Returns:
            Series with regime labels
        """
        labels = pd.Series(index=spread.index, dtype=int)
        
        # Calculate spread volatility
        spread_vol = spread.rolling(20).std()
        mean_vol = spread_vol.mean()
        
        # Calculate spread momentum
        spread_momentum = spread.rolling(5).mean().diff()
        
        for i in range(len(spread)):
            if pd.isna(spread_vol.iloc[i]) or pd.isna(spread_momentum.iloc[i]):
                labels.iloc[i] = 0
                continue
            
            # High volatility regime
            if spread_vol.iloc[i] > volatility_threshold * mean_vol:
                labels.iloc[i] = 1
            # Trending/diverging regime
            elif abs(spread_momentum.iloc[i]) > spread_vol.iloc[i]:
                labels.iloc[i] = 2
            # Low volatility mean-reverting regime
            else:
                labels.iloc[i] = 0
        
        return labels
    
    def train(self, features: pd.DataFrame, labels: pd.Series, 
              test_size: float = 0.2) -> Dict[str, float]:
        """
        Train Random Forest classifier
        
        Args:
            features: Feature DataFrame
            labels: Regime labels
            test_size: Test set size for evaluation
            
        Returns:
            Dictionary with training metrics
        """
        # Align features and labels
        common_index = features.index.intersection(labels.index)
        features = features.loc[common_index]
        labels = labels.loc[common_index]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features.values, labels.values, 
            test_size=test_size, 
            random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict regime for new data
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Predicted regime labels
        """
        features_scaled = self.scaler.transform(features.values)
        predictions = self.model.predict(features_scaled)
        
        return predictions
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Probability matrix (n_samples x n_regimes)
        """
        features_scaled = self.scaler.transform(features.values)
        probabilities = self.model.predict_proba(features_scaled)
        
        return probabilities
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.feature_names is None:
            return {}
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save_model(self, model_path: str, scaler_path: str) -> None:
        """Save model and scaler"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
    
    def load_model(self, model_path: str, scaler_path: str) -> None:
        """Load model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
