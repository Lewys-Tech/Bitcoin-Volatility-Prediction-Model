import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import PATHS

def load_data():
    """Load the raw cryptocurrency data."""
    file_path = Path(PATHS['raw_data']) / 'BTC_USD_raw.csv'
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return df

def analyze_basic_stats(df):
    """Analyze basic statistical characteristics of the data."""
    print("\n=== Basic Statistical Analysis ===")
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nData Types:")
    print(df.dtypes)

def analyze_volatility_patterns(df):
    """Analyze volatility patterns and characteristics."""
    print("\n=== Volatility Analysis ===")
    
    # Calculate volatility statistics
    vol_stats = df['realized_volatility'].describe()
    print("\nVolatility Statistics:")
    print(vol_stats)
    
    # Calculate volatility clustering
    vol_autocorr = df['realized_volatility'].autocorr()
    print(f"\nVolatility Autocorrelation (1-day lag): {vol_autocorr:.4f}")
    
    # Calculate volatility persistence
    vol_persistence = df['realized_volatility'].rolling(window=5).mean().autocorr()
    print(f"Volatility Persistence (5-day rolling mean): {vol_persistence:.4f}")

def analyze_return_characteristics(df):
    """Analyze return characteristics and distribution."""
    print("\n=== Return Analysis ===")
    
    # Calculate return statistics
    return_stats = df['log_returns'].describe()
    print("\nReturn Statistics:")
    print(return_stats)
    
    # Calculate return skewness and kurtosis
    skewness = df['log_returns'].skew()
    kurtosis = df['log_returns'].kurtosis()
    print(f"\nReturn Skewness: {skewness:.4f}")
    print(f"Return Kurtosis: {kurtosis:.4f}")
    
    # Calculate return autocorrelation
    return_autocorr = df['log_returns'].autocorr()
    print(f"Return Autocorrelation (1-day lag): {return_autocorr:.4f}")

def plot_characteristics(df):
    """Create visualizations of key characteristics."""
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bitcoin Price and Volatility Characteristics', fontsize=16)
    
    # Price and Volume
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['Close'], label='Price')
    ax1.set_title('Bitcoin Price Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    
    # Volatility
    ax2 = axes[0, 1]
    ax2.plot(df.index, df['realized_volatility'], label='Volatility')
    ax2.set_title('Realized Volatility Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volatility')
    ax2.legend()
    
    # Return Distribution
    ax3 = axes[1, 0]
    sns.histplot(data=df, x='log_returns', bins=50, ax=ax3)
    ax3.set_title('Return Distribution')
    ax3.set_xlabel('Log Returns')
    ax3.set_ylabel('Count')
    
    # Volatility vs Returns
    ax4 = axes[1, 1]
    ax4.scatter(df['log_returns'], df['realized_volatility'], alpha=0.5)
    ax4.set_title('Volatility vs Returns')
    ax4.set_xlabel('Log Returns')
    ax4.set_ylabel('Volatility')
    
    plt.tight_layout()
    plt.savefig(Path(PATHS['results']) / 'data_characteristics.png')
    plt.close()

def main():
    # Create results directory if it doesn't exist
    Path(PATHS['results']).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Perform analyses
    analyze_basic_stats(df)
    analyze_volatility_patterns(df)
    analyze_return_characteristics(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_characteristics(df)
    print(f"\nVisualizations saved to {PATHS['results']}/data_characteristics.png")

if __name__ == "__main__":
    main() 