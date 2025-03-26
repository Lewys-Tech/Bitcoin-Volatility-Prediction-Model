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

def analyze_extreme_events(df):
    """Analyze extreme price movements and volatility events."""
    print("\n=== Extreme Events Analysis ===")
    
    # Find largest price movements
    df['price_change'] = df['Close'].pct_change()
    largest_gains = df.nlargest(5, 'price_change')
    largest_losses = df.nsmallest(5, 'price_change')
    
    print("\nLargest Daily Gains:")
    for idx, row in largest_gains.iterrows():
        print(f"{idx.date()}: {row['price_change']*100:.2f}% (${row['Close']:,.2f})")
    
    print("\nLargest Daily Losses:")
    for idx, row in largest_losses.iterrows():
        print(f"{idx.date()}: {row['price_change']*100:.2f}% (${row['Close']:,.2f})")
    
    # Find highest volatility periods
    highest_vol = df.nlargest(5, 'realized_volatility')
    print("\nHighest Volatility Periods:")
    for idx, row in highest_vol.iterrows():
        print(f"{idx.date()}: {row['realized_volatility']*100:.2f}% (Price: ${row['Close']:,.2f})")

def analyze_volatility_regimes(df):
    """Analyze different volatility regimes and their characteristics."""
    print("\n=== Volatility Regime Analysis ===")
    
    # Define volatility regimes
    df['vol_regime'] = pd.qcut(df['realized_volatility'], q=3, labels=['Low', 'Medium', 'High'])
    
    # Analyze each regime
    regime_stats = df.groupby('vol_regime').agg({
        'log_returns': ['mean', 'std', 'skew'],
        'Close': 'mean',
        'Volume': 'mean'
    }).round(4)
    
    # Calculate kurtosis separately
    kurtosis = df.groupby('vol_regime')['log_returns'].apply(lambda x: x.kurtosis()).round(4)
    regime_stats[('log_returns', 'kurtosis')] = kurtosis
    
    print("\nCharacteristics of Different Volatility Regimes:")
    print(regime_stats)
    
    # Calculate regime transition probabilities
    regime_transitions = pd.crosstab(df['vol_regime'].shift(), df['vol_regime'])
    regime_transitions = regime_transitions.div(regime_transitions.sum(axis=1), axis=0)
    
    print("\nVolatility Regime Transition Probabilities:")
    print(regime_transitions.round(4))

def analyze_return_patterns(df):
    """Analyze return patterns and their relationship with volatility."""
    print("\n=== Return Pattern Analysis ===")
    
    # Calculate consecutive positive/negative returns
    df['return_sign'] = np.sign(df['log_returns'])
    df['consecutive'] = (df['return_sign'] != df['return_sign'].shift()).cumsum()
    
    # Calculate streak lengths for positive and negative returns
    positive_streaks = df[df['return_sign'] > 0].groupby('consecutive').size()
    negative_streaks = df[df['return_sign'] < 0].groupby('consecutive').size()
    
    print("\nLongest Streaks:")
    print("Positive Returns:", positive_streaks.max(), "days")
    print("Negative Returns:", negative_streaks.max(), "days")
    
    # Analyze return patterns by volatility regime
    df['vol_regime'] = pd.qcut(df['realized_volatility'], q=3, labels=['Low', 'Medium', 'High'])
    regime_returns = df.groupby('vol_regime')['log_returns'].agg(['mean', 'std', 'skew'])
    regime_returns['kurtosis'] = df.groupby('vol_regime')['log_returns'].apply(lambda x: x.kurtosis())
    
    print("\nReturn Characteristics by Volatility Regime:")
    print(regime_returns.round(4))

def plot_patterns(df):
    """Create visualizations of key patterns."""
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bitcoin Market Patterns Analysis', fontsize=16)
    
    # Price and Volatility Regimes
    ax1 = axes[0, 0]
    df['vol_regime'] = pd.qcut(df['realized_volatility'], q=3, labels=['Low', 'Medium', 'High'])
    colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
    for regime in ['Low', 'Medium', 'High']:
        mask = df['vol_regime'] == regime
        ax1.scatter(df.index[mask], df['Close'][mask], 
                   label=regime, color=colors[regime], alpha=0.5)
    ax1.set_title('Price by Volatility Regime')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    
    # Return Distribution by Regime
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x='vol_regime', y='log_returns', ax=ax2)
    ax2.set_title('Return Distribution by Volatility Regime')
    ax2.set_xlabel('Volatility Regime')
    ax2.set_ylabel('Log Returns')
    
    # Rolling Volatility
    ax3 = axes[1, 0]
    ax3.plot(df.index, df['realized_volatility'], label='7-day Volatility')
    ax3.plot(df.index, df['realized_volatility'].rolling(window=30).mean(), 
             label='30-day Average', alpha=0.7)
    ax3.set_title('Volatility Over Time')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Volatility')
    ax3.legend()
    
    # Return vs Volatility
    ax4 = axes[1, 1]
    sns.scatterplot(data=df, x='log_returns', y='realized_volatility', 
                   hue='vol_regime', alpha=0.5, ax=ax4)
    ax4.set_title('Returns vs Volatility')
    ax4.set_xlabel('Log Returns')
    ax4.set_ylabel('Volatility')
    
    plt.tight_layout()
    plt.savefig(Path(PATHS['results']) / 'market_patterns.png')
    plt.close()

def main():
    # Create results directory if it doesn't exist
    Path(PATHS['results']).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Perform detailed analyses
    analyze_extreme_events(df)
    analyze_volatility_regimes(df)
    analyze_return_patterns(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_patterns(df)
    print(f"\nVisualizations saved to {PATHS['results']}/market_patterns.png")

if __name__ == "__main__":
    main() 