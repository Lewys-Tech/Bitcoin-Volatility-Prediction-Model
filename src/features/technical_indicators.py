import pandas as pd
import numpy as np
import ta
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    A class to create technical indicators for cryptocurrency price analysis.
    These indicators help identify trends, momentum, and potential reversal points.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the technical indicators creator.
        
        Args:
            config (Dict): Configuration dictionary with indicator parameters
        """
        self.config = config or {
            'rsi_period': 14,
            'bb_period': 20,
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'atr_period': 14,
            'stoch_period': 14,
            'adx_period': 14
        }
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-following indicators.
        These help identify the overall direction of the market.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added trend indicators
        """
        # Moving Averages
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(
            close=df['Close'],
            window_slow=self.config['macd']['slow'],
            window_fast=self.config['macd']['fast'],
            window_sign=self.config['macd']['signal']
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(
            close=df['Close'],
            window=self.config['bb_period']
        )
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators.
        These help identify the strength of price movements.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added momentum indicators
        """
        # RSI (Relative Strength Index)
        df['rsi'] = ta.momentum.rsi(
            df['Close'],
            window=self.config['rsi_period']
        )
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=self.config['stoch_period']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=self.config['adx_period']
        )
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators.
        These help understand trading activity and volume trends.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added volume indicators
        """
        # On-Balance Volume (OBV)
        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # Volume Weighted Average Price (VWAP)
        df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Money Flow Index
        df['mfi'] = ta.volume.money_flow_index(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume'],
            window=14
        )
        
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators.
        These help measure price volatility and potential breakouts.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added volatility indicators
        """
        # Average True Range (ATR)
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=self.config['atr_period']
        )
        df['atr'] = atr_indicator.average_true_range()
        
        # Bollinger Band Width (as a volatility measure)
        bb = ta.volatility.BollingerBands(
            close=df['Close'],
            window=self.config['bb_period']
        )
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        
        # Normalized ATR (Volatility Index)
        df['volatility_index'] = df['atr'] / df['Close']
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical indicators.
        This is the main method that combines all the indicator creation steps.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with all technical indicators
        """
        logger.info("Creating technical indicators...")
        
        # Create a copy to avoid modifying the original DataFrame
        df_features = df.copy()
        
        # Apply all indicator creation methods
        df_features = self.add_trend_indicators(df_features)
        df_features = self.add_momentum_indicators(df_features)
        df_features = self.add_volume_indicators(df_features)
        df_features = self.add_volatility_indicators(df_features)
        
        logger.info("Technical indicators created successfully")
        return df_features

def main():
    """
    Example usage of the TechnicalIndicators class.
    """
    # Example data (you would typically load your actual data here)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    example_data = pd.DataFrame({
        'Open': np.random.normal(50000, 1000, len(dates)),
        'High': np.random.normal(51000, 1000, len(dates)),
        'Low': np.random.normal(49000, 1000, len(dates)),
        'Close': np.random.normal(50000, 1000, len(dates)),
        'Volume': np.random.normal(1000000, 100000, len(dates))
    }, index=dates)
    
    # Create indicators
    indicator_creator = TechnicalIndicators()
    df_with_indicators = indicator_creator.create_features(example_data)
    
    # Display results
    print("\nFeature Summary:")
    print(df_with_indicators.head())
    print("\nAvailable Indicators:")
    print(df_with_indicators.columns.tolist())

if __name__ == "__main__":
    main() 