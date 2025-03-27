import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import PATHS
from return_pattern_features import ReturnPatternFeatures

def load_data():
    """Load the raw cryptocurrency data."""
    file_path = Path(PATHS['raw_data']) / 'BTC_USD_raw.csv'
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return df

def plot_return_patterns(df: pd.DataFrame):
    """Create visualizations of the return pattern analysis."""
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bitcoin Return Pattern Analysis', fontsize=16)
    
    # Plot 1: Return Streaks
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='streak_type', y='streak_length', ax=ax1)
    ax1.set_title('Return Streak Length Distribution')
    ax1.set_xlabel('Streak Type')
    ax1.set_ylabel('Length (days)')
    
    # Plot 2: Return Momentum
    ax2 = axes[0, 1]
    for window in [5, 10, 20]:
        ax2.plot(df.index, df[f'return_momentum_{window}d'], 
                label=f'{window}d Momentum', alpha=0.7)
    ax2.set_title('Return Momentum Over Different Windows')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Momentum')
    ax2.legend()
    
    # Plot 3: Return Asymmetry
    ax3 = axes[1, 0]
    ax3.plot(df.index, df['return_asymmetry'], label='Asymmetry Ratio')
    ax3.plot(df.index, df['return_skewness'], label='Skewness', alpha=0.7)
    ax3.set_title('Return Asymmetry Metrics')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Value')
    ax3.legend()
    
    # Plot 4: Return Autocorrelation
    ax4 = axes[1, 1]
    for lag in [1, 5, 10]:
        ax4.plot(df.index, df[f'return_autocorr_{lag}d'], 
                label=f'{lag}d Lag', alpha=0.7)
    ax4.set_title('Return Autocorrelation at Different Lags')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Autocorrelation')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(Path(PATHS['results']) / 'return_pattern_analysis.png')
    plt.close()

def analyze_return_features(df: pd.DataFrame):
    """Analyze and print statistics about the return patterns."""
    print("\n=== Return Pattern Analysis ===")
    
    # Print streak statistics
    print("\nStreak Statistics:")
    streak_stats = df.groupby('streak_type')['streak_length'].describe()
    print(streak_stats)
    
    # Print momentum statistics
    print("\nMomentum Statistics (20-day window):")
    momentum_stats = df['return_momentum_20d'].describe()
    print(momentum_stats)
    
    # Print asymmetry statistics
    print("\nAsymmetry Statistics:")
    asymmetry_stats = df[['return_asymmetry', 'return_skewness']].describe()
    print(asymmetry_stats)
    
    # Print autocorrelation statistics
    print("\nAutocorrelation Statistics:")
    autocorr_stats = df[[f'return_autocorr_{lag}d' for lag in [1, 5, 10]]].describe()
    print(autocorr_stats)

def main():
    # Create results directory if it doesn't exist
    Path(PATHS['results']).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading Bitcoin data...")
    df = load_data()
    
    # Create return pattern features
    print("\nCreating return pattern features...")
    feature_creator = ReturnPatternFeatures()
    df_with_features = feature_creator.create_features(df)
    
    # Analyze the features
    analyze_return_features(df_with_features)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_return_patterns(df_with_features)
    print(f"\nVisualizations saved to {PATHS['results']}/return_pattern_analysis.png")

if __name__ == "__main__":
    main() 