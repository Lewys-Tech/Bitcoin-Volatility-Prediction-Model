import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolatilityRegimeFeatures:
    """
    A class to create features based on volatility regimes.
    This helps us understand and predict how volatility patterns change over time.
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize the feature creator.
        
        Args:
            n_regimes (int): Number of volatility regimes to identify (default: 3 for Low/Medium/High)
        """
        self.n_regimes = n_regimes
        self.regime_labels = ['Low', 'Medium', 'High'] if n_regimes == 3 else [f'Regime_{i+1}' for i in range(n_regimes)]
    
    def identify_regimes(self, df: pd.DataFrame, volatility_col: str = 'realized_volatility') -> pd.DataFrame:
        """
        Identify different volatility regimes using quantiles.
        This splits the volatility data into different levels (Low/Medium/High).
        
        Args:
            df (pd.DataFrame): DataFrame containing volatility data
            volatility_col (str): Name of the column containing volatility values
            
        Returns:
            pd.DataFrame: DataFrame with added regime labels
        """
        # Create regime labels based on volatility quantiles
        df['vol_regime'] = pd.qcut(df[volatility_col], 
                                 q=self.n_regimes, 
                                 labels=self.regime_labels)
        return df
    
    def calculate_regime_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate how long each volatility regime has been active.
        This helps us understand regime persistence.
        
        Args:
            df (pd.DataFrame): DataFrame with regime labels
            
        Returns:
            pd.DataFrame: DataFrame with added regime duration
        """
        # Create a group that changes when regime changes
        df['regime_group'] = (df['vol_regime'] != df['vol_regime'].shift()).cumsum()
        
        # Calculate duration of each regime
        regime_durations = df.groupby('regime_group').cumcount()
        df['regime_duration'] = regime_durations
        
        return df
    
    def calculate_regime_transitions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate probabilities of transitioning between different volatility regimes.
        This helps us understand how likely it is to move from one regime to another.
        
        Args:
            df (pd.DataFrame): DataFrame with regime labels
            
        Returns:
            pd.DataFrame: DataFrame with added transition probabilities
        """
        # Create transition matrix
        transitions = pd.crosstab(df['vol_regime'].shift(), df['vol_regime'])
        transition_probs = transitions.div(transitions.sum(axis=1), axis=0)
        
        # Add transition probabilities to DataFrame
        for regime in self.regime_labels:
            df[f'transition_prob_{regime}'] = df['vol_regime'].map(
                transition_probs[regime]
            )
        
        return df
    
    def calculate_regime_boundary_distance(self, df: pd.DataFrame, 
                                        volatility_col: str = 'realized_volatility') -> pd.DataFrame:
        """
        Calculate how close the current volatility is to regime boundaries.
        This helps us identify when we might be about to switch regimes.
        
        Args:
            df (pd.DataFrame): DataFrame with volatility data
            volatility_col (str): Name of the column containing volatility values
            
        Returns:
            pd.DataFrame: DataFrame with added boundary distances
        """
        # Calculate quantile boundaries
        quantiles = df[volatility_col].quantile([i/self.n_regimes for i in range(self.n_regimes + 1)])
        
        # Calculate distance to lower and upper boundaries
        df['distance_to_lower_boundary'] = df[volatility_col] - quantiles.iloc[0]
        df['distance_to_upper_boundary'] = quantiles.iloc[-1] - df[volatility_col]
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all volatility regime features.
        This is the main method that combines all the feature creation steps.
        
        Args:
            df (pd.DataFrame): DataFrame with volatility data
            
        Returns:
            pd.DataFrame: DataFrame with all volatility regime features
        """
        logger.info("Creating volatility regime features...")
        
        # Create a copy to avoid modifying the original DataFrame
        df_features = df.copy()
        
        # Apply all feature creation methods
        df_features = self.identify_regimes(df_features)
        df_features = self.calculate_regime_duration(df_features)
        df_features = self.calculate_regime_transitions(df_features)
        df_features = self.calculate_regime_boundary_distance(df_features)
        
        logger.info("Volatility regime features created successfully")
        return df_features

def main():
    """
    Example usage of the VolatilityRegimeFeatures class.
    """
    # Example data (you would typically load your actual data here)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    example_data = pd.DataFrame({
        'realized_volatility': np.random.normal(0.02, 0.01, len(dates))
    }, index=dates)
    
    # Create features
    feature_creator = VolatilityRegimeFeatures()
    df_with_features = feature_creator.create_features(example_data)
    
    # Display results
    print("\nFeature Summary:")
    print(df_with_features.head())
    print("\nRegime Distribution:")
    print(df_with_features['vol_regime'].value_counts())
    print("\nTransition Probabilities:")
    print(pd.crosstab(df_with_features['vol_regime'].shift(), 
                      df_with_features['vol_regime']).round(4))

if __name__ == "__main__":
    main() 