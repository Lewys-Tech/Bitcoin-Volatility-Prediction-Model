import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import logging
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import PATHS
from technical_indicators import TechnicalIndicators

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load the raw cryptocurrency data."""
    file_path = Path(PATHS['raw_data']) / 'BTC_USD_raw.csv'
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    
    # Ensure required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Sort index to ensure chronological order
    df = df.sort_index()
    
    # Remove any duplicate indices
    df = df[~df.index.duplicated(keep='first')]
    
    logger.info(f"Loaded data shape: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Available columns: {df.columns.tolist()}")
    
    return df

def plot_technical_indicators(df: pd.DataFrame):
    """Create visualizations of the technical indicators analysis."""
    try:
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bitcoin Technical Indicators Analysis', fontsize=16)
        
        # Plot 1: Price and Moving Averages
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['Close'], label='Price', alpha=0.7)
        ax1.plot(df.index, df['sma_20'], label='SMA 20', alpha=0.7)
        ax1.plot(df.index, df['sma_50'], label='SMA 50', alpha=0.7)
        ax1.plot(df.index, df['bb_high'], '--', label='BB Upper', alpha=0.5)
        ax1.plot(df.index, df['bb_low'], '--', label='BB Lower', alpha=0.5)
        ax1.set_title('Price and Moving Averages')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        
        # Plot 2: Momentum Indicators
        ax2 = axes[0, 1]
        ax2.plot(df.index, df['rsi'], label='RSI', alpha=0.7)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.set_title('RSI and Overbought/Oversold Levels')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('RSI')
        ax2.legend()
        
        # Plot 3: Volume Indicators
        ax3 = axes[1, 0]
        ax3.plot(df.index, df['obv'], label='OBV', alpha=0.7)
        ax3.plot(df.index, df['vwap'], label='VWAP', alpha=0.7)
        ax3.set_title('Volume Indicators')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Value')
        ax3.legend()
        
        # Plot 4: Volatility Indicators
        ax4 = axes[1, 1]
        ax4.plot(df.index, df['atr'], label='ATR', alpha=0.7)
        ax4.plot(df.index, df['volatility_index'], label='Volatility Index', alpha=0.7)
        ax4.set_title('Volatility Indicators')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Value')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(Path(PATHS['results']) / 'technical_indicators_analysis.png')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting indicators: {str(e)}")
        raise

def analyze_technical_indicators(df: pd.DataFrame):
    """Analyze and print statistics about the technical indicators."""
    try:
        print("\n=== Technical Indicators Analysis ===")
        
        # Print Moving Averages Statistics
        print("\nMoving Averages Statistics:")
        ma_stats = df[['sma_20', 'sma_50', 'ema_20']].describe()
        print(ma_stats)
        
        # Print RSI Statistics
        print("\nRSI Statistics:")
        rsi_stats = df['rsi'].describe()
        print(rsi_stats)
        
        # Print Bollinger Bands Statistics
        print("\nBollinger Bands Statistics:")
        bb_stats = df[['bb_high', 'bb_low', 'bb_width']].describe()
        print(bb_stats)
        
        # Print Volume Indicators Statistics
        print("\nVolume Indicators Statistics:")
        volume_stats = df[['obv', 'mfi']].describe()
        print(volume_stats)
        
        # Print Volatility Indicators Statistics
        print("\nVolatility Indicators Statistics:")
        vol_stats = df[['atr', 'volatility_index']].describe()
        print(vol_stats)
        
        # Calculate correlation with realized volatility
        if 'realized_volatility' in df.columns:
            print("\nCorrelation with Realized Volatility:")
            vol_corr = df[['realized_volatility', 'atr', 'volatility_index', 'bb_width']].corr()
            print(vol_corr['realized_volatility'])
        
    except Exception as e:
        logger.error(f"Error analyzing indicators: {str(e)}")
        raise

def main():
    try:
        # Create results directory if it doesn't exist
        Path(PATHS['results']).mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info("Loading Bitcoin data...")
        df = load_data()
        
        # Create technical indicators
        logger.info("Creating technical indicators...")
        indicator_creator = TechnicalIndicators()
        df_with_indicators = indicator_creator.create_features(df)
        
        # Check for NaN values
        nan_cols = df_with_indicators.columns[df_with_indicators.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"NaN values found in columns: {nan_cols}")
            logger.info("Filling NaN values with forward fill method...")
            df_with_indicators = df_with_indicators.fillna(method='ffill').fillna(method='bfill')
        
        # Analyze the indicators
        analyze_technical_indicators(df_with_indicators)
        
        # Create visualizations
        logger.info("Generating visualizations...")
        plot_technical_indicators(df_with_indicators)
        logger.info(f"Visualizations saved to {PATHS['results']}/technical_indicators_analysis.png")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 