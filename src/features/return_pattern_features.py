import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReturnPatternFeatures:
    """
    A class to create features based on return patterns and characteristics.
    This helps us understand and predict return behavior and momentum.
    """
    
    def __init__(self):
        """Initialize the feature creator."""
        pass
    
    def calculate_return_streaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate return streaks (consecutive positive/negative returns).
        This helps us understand return momentum and persistence.
        
        Args:
            df (pd.DataFrame): DataFrame containing return data
            
        Returns:
            pd.DataFrame: DataFrame with added streak features
        """
        # Calculate return sign (1 for positive, -1 for negative)
        df['return_sign'] = np.sign(df['log_returns'])
        
        # Create streak groups (changes when return sign changes)
        df['streak_group'] = (df['return_sign'] != df['return_sign'].shift()).cumsum()
        
        # Calculate streak length
        df['streak_length'] = df.groupby('streak_group').cumcount()
        
        # Calculate streak type (positive/negative)
        df['streak_type'] = df['return_sign'].map({1: 'positive', -1: 'negative'})
        
        return df
    
    def calculate_return_momentum(self, df: pd.DataFrame, windows: list = [5, 10, 20]) -> pd.DataFrame:
        """
        Calculate return momentum over different windows.
        This helps us understand return trends and acceleration.
        
        Args:
            df (pd.DataFrame): DataFrame containing return data
            windows (list): List of window sizes for momentum calculation
            
        Returns:
            pd.DataFrame: DataFrame with added momentum features
        """
        for window in windows:
            # Calculate rolling mean of returns
            df[f'return_momentum_{window}d'] = df['log_returns'].rolling(window=window).mean()
            
            # Calculate rolling standard deviation
            df[f'return_volatility_{window}d'] = df['log_returns'].rolling(window=window).std()
            
            # Calculate return acceleration (change in momentum)
            df[f'return_acceleration_{window}d'] = df[f'return_momentum_{window}d'].diff()
        
        return df
    
    def calculate_return_asymmetry(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate return asymmetry metrics.
        This helps us understand if returns are more positive or negative.
        
        Args:
            df (pd.DataFrame): DataFrame containing return data
            window (int): Window size for asymmetry calculation
            
        Returns:
            pd.DataFrame: DataFrame with added asymmetry features
        """
        # Calculate positive and negative returns separately
        positive_returns = df['log_returns'].clip(lower=0)
        negative_returns = df['log_returns'].clip(upper=0)
        
        # Calculate rolling means
        df['positive_return_mean'] = positive_returns.rolling(window=window).mean()
        df['negative_return_mean'] = negative_returns.rolling(window=window).mean()
        
        # Calculate asymmetry ratio
        df['return_asymmetry'] = df['positive_return_mean'] / abs(df['negative_return_mean'])
        
        # Calculate skewness
        df['return_skewness'] = df['log_returns'].rolling(window=window).skew()
        
        return df
    
    def calculate_return_autocorrelation(self, df: pd.DataFrame, lags: list = [1, 5, 10]) -> pd.DataFrame:
        """
        Calculate return autocorrelation at different lags.
        This helps us understand return persistence and mean reversion.
        
        Args:
            df (pd.DataFrame): DataFrame containing return data
            lags (list): List of lag periods for autocorrelation calculation
            
        Returns:
            pd.DataFrame: DataFrame with added autocorrelation features
        """
        for lag in lags:
            # Calculate autocorrelation
            df[f'return_autocorr_{lag}d'] = df['log_returns'].rolling(window=20).apply(
                lambda x: x.autocorr(lag=lag)
            )
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all return pattern features.
        This is the main method that combines all the feature creation steps.
        
        Args:
            df (pd.DataFrame): DataFrame with return data
            
        Returns:
            pd.DataFrame: DataFrame with all return pattern features
        """
        logger.info("Creating return pattern features...")
        
        # Create a copy to avoid modifying the original DataFrame
        df_features = df.copy()
        
        # Apply all feature creation methods
        df_features = self.calculate_return_streaks(df_features)
        df_features = self.calculate_return_momentum(df_features)
        df_features = self.calculate_return_asymmetry(df_features)
        df_features = self.calculate_return_autocorrelation(df_features)
        
        logger.info("Return pattern features created successfully")
        return df_features

def main():
    """
    Example usage of the ReturnPatternFeatures class.
    """
    # Example data (you would typically load your actual data here)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    example_data = pd.DataFrame({
        'log_returns': np.random.normal(0.001, 0.02, len(dates))
    }, index=dates)
    
    # Create features
    feature_creator = ReturnPatternFeatures()
    df_with_features = feature_creator.create_features(example_data)
    
    # Display results
    print("\nFeature Summary:")
    print(df_with_features.head())
    print("\nStreak Statistics:")
    print(df_with_features.groupby('streak_type')['streak_length'].describe())

if __name__ == "__main__":
    main() 