"""
Feature Engineering Module for Volatility Regime Prediction

This module implements feature engineering specifically for predicting volatility regimes.
It includes target variable creation, feature calculation, and data preparation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RegimeFeatureEngineer:
    """
    Feature engineering class for volatility regime prediction.
    
    Parameters
    ----------
    data_path : str
        Path to the processed data file
    output_path : str
        Path to save the engineered features
    """
    
    def __init__(self, data_path: str, output_path: str):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.data = None
        self.features = None
        self.target = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load the processed data."""
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            logger.info(f"Successfully loaded {len(self.data)} rows of data")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_target_variable(self) -> None:
        """
        Create the target variable (volatility regime).
        High volatility regime is defined as volatility > mean + std.
        """
        # Calculate volatility threshold
        vol_mean = self.data['realized_volatility'].mean()
        vol_std = self.data['realized_volatility'].std()
        vol_threshold = vol_mean + vol_std
        
        # Create target variable (1 for high volatility, 0 for normal)
        self.target = (self.data['realized_volatility'] > vol_threshold).astype(int)
        
        # Log regime statistics
        high_vol_count = self.target.sum()
        total_count = len(self.target)
        logger.info(f"High volatility regime: {high_vol_count}/{total_count} days ({high_vol_count/total_count*100:.2f}%)")
    
    def create_price_features(self) -> pd.DataFrame:
        """Create price-based features."""
        features = pd.DataFrame(index=self.data.index)
        
        # Price changes at different horizons
        for horizon in [1, 3, 5, 10]:
            features[f'price_change_{horizon}d'] = self.data['Close'].pct_change(horizon)
        
        # Price volatility at different horizons
        for window in [5, 10, 20]:
            features[f'price_volatility_{window}d'] = self.data['Close'].pct_change().rolling(window).std()
        
        # Price momentum
        for window in [5, 10, 20]:
            features[f'price_momentum_{window}d'] = self.data['Close'].pct_change(window)
        
        # Price trend strength
        for window in [5, 10, 20]:
            ma = self.data['Close'].rolling(window).mean()
            features[f'trend_strength_{window}d'] = (self.data['Close'] - ma) / ma
        
        return features
    
    def create_volume_features(self) -> pd.DataFrame:
        """Create volume-based features."""
        features = pd.DataFrame(index=self.data.index)
        
        # Volume changes
        for horizon in [1, 3, 5, 10]:
            features[f'volume_change_{horizon}d'] = self.data['Volume'].pct_change(horizon)
        
        # Volume moving averages
        for window in [5, 10, 20]:
            features[f'volume_ma_{window}d'] = self.data['Volume'].rolling(window).mean()
            features[f'volume_std_{window}d'] = self.data['Volume'].rolling(window).std()
        
        # Volume momentum
        for window in [5, 10, 20]:
            features[f'volume_momentum_{window}d'] = self.data['Volume'].pct_change(window)
        
        # Volume trend strength
        for window in [5, 10, 20]:
            ma = self.data['Volume'].rolling(window).mean()
            features[f'volume_trend_strength_{window}d'] = (self.data['Volume'] - ma) / ma
        
        return features
    
    def create_volatility_features(self) -> pd.DataFrame:
        """Create volatility-based features."""
        features = pd.DataFrame(index=self.data.index)
        
        # Volatility changes
        for horizon in [1, 3, 5, 10]:
            features[f'volatility_change_{horizon}d'] = self.data['realized_volatility'].pct_change(horizon)
        
        # Volatility moving averages
        for window in [5, 10, 20]:
            features[f'volatility_ma_{window}d'] = self.data['realized_volatility'].rolling(window).mean()
            features[f'volatility_std_{window}d'] = self.data['realized_volatility'].rolling(window).std()
        
        # Volatility momentum
        for window in [5, 10, 20]:
            features[f'volatility_momentum_{window}d'] = self.data['realized_volatility'].pct_change(window)
        
        # Volatility regime persistence
        for window in [5, 10, 20]:
            features[f'regime_persistence_{window}d'] = self.target.rolling(window).mean()
        
        return features
    
    def create_time_features(self) -> pd.DataFrame:
        """Create time-based features."""
        features = pd.DataFrame(index=self.data.index)
        
        # Day of week features
        features['day_of_week'] = self.data['Date'].dt.dayofweek
        features['is_weekend'] = self.data['Date'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Month features
        features['month'] = self.data['Date'].dt.month
        features['is_month_end'] = self.data['Date'].dt.is_month_end.astype(int)
        features['is_month_start'] = self.data['Date'].dt.is_month_start.astype(int)
        
        # Quarter features
        features['quarter'] = self.data['Date'].dt.quarter
        
        # Cyclical encoding for day of week and month
        features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        return features
    
    def create_interaction_features(self) -> pd.DataFrame:
        """Create interaction features between different metrics."""
        features = pd.DataFrame(index=self.data.index)
        
        # Price-Volume interactions
        features['price_volume_correlation_5d'] = (
            self.data['Close'].pct_change().rolling(5)
            .corr(self.data['Volume'].pct_change())
        )
        
        # Volatility-Volume interactions
        features['volatility_volume_correlation_5d'] = (
            self.data['realized_volatility'].rolling(5)
            .corr(self.data['Volume'])
        )
        
        # Price-Volatility interactions
        features['price_volatility_correlation_5d'] = (
            self.data['Close'].pct_change().rolling(5)
            .corr(self.data['realized_volatility'])
        )
        
        return features
    
    def prepare_features(self) -> None:
        """Prepare all features for regime prediction."""
        logger.info("Starting feature preparation")
        
        # Create target variable
        self.create_target_variable()
        
        # Create feature sets
        price_features = self.create_price_features()
        volume_features = self.create_volume_features()
        volatility_features = self.create_volatility_features()
        time_features = self.create_time_features()
        interaction_features = self.create_interaction_features()
        
        # Combine all features
        self.features = pd.concat([
            price_features,
            volume_features,
            volatility_features,
            time_features,
            interaction_features
        ], axis=1)
        
        # Handle missing values
        self.features = self.features.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any infinite values
        self.features = self.features.replace([np.inf, -np.inf], np.nan)
        self.features = self.features.fillna(method='ffill').fillna(method='bfill')
        
        # Log feature information
        logger.info(f"Created {len(self.features.columns)} features")
        logger.info("Feature preparation completed")
    
    def save_features(self) -> None:
        """Save the prepared features and target variable."""
        try:
            # Combine features and target
            data_to_save = pd.concat([
                self.data[['Date']],
                self.features,
                pd.Series(self.target, name='volatility_regime')
            ], axis=1)
            
            # Save to CSV
            data_to_save.to_csv(self.output_path, index=False)
            logger.info(f"Features saved to {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise
    
    def run_pipeline(self) -> None:
        """Run the complete feature engineering pipeline."""
        try:
            self.prepare_features()
            self.save_features()
            logger.info("Feature engineering pipeline completed successfully")
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise 