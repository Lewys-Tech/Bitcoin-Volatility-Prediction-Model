import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import PATHS
from volatility_regime_features import VolatilityRegimeFeatures

def load_data():
    """Load the raw cryptocurrency data."""
    file_path = Path(PATHS['raw_data']) / 'BTC_USD_raw.csv'
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return df

def plot_regime_analysis(df: pd.DataFrame):
    """Create visualizations of the volatility regime analysis."""
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bitcoin Volatility Regime Analysis', fontsize=16)
    
    # Plot 1: Price by Volatility Regime
    ax1 = axes[0, 0]
    colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
    for regime in ['Low', 'Medium', 'High']:
        mask = df['vol_regime'] == regime
        ax1.scatter(df.index[mask], df['Close'][mask], 
                   label=regime, color=colors[regime], alpha=0.5)
    ax1.set_title('Price by Volatility Regime')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    
    # Plot 2: Regime Duration Distribution
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x='vol_regime', y='regime_duration', ax=ax2)
    ax2.set_title('Regime Duration Distribution')
    ax2.set_xlabel('Volatility Regime')
    ax2.set_ylabel('Duration (days)')
    
    # Plot 3: Transition Probabilities Heatmap
    ax3 = axes[1, 0]
    transitions = pd.crosstab(df['vol_regime'].shift(), df['vol_regime'])
    transition_probs = transitions.div(transitions.sum(axis=1), axis=0)
    sns.heatmap(transition_probs, annot=True, fmt='.2%', cmap='YlOrRd', ax=ax3)
    ax3.set_title('Regime Transition Probabilities')
    
    # Plot 4: Boundary Distances
    ax4 = axes[1, 1]
    sns.boxplot(data=df, x='vol_regime', y='distance_to_upper_boundary', ax=ax4)
    ax4.set_title('Distance to Upper Boundary by Regime')
    ax4.set_xlabel('Volatility Regime')
    ax4.set_ylabel('Distance to Upper Boundary')
    
    plt.tight_layout()
    plt.savefig(Path(PATHS['results']) / 'volatility_regime_analysis.png')
    plt.close()

def analyze_regime_features(df: pd.DataFrame):
    """Analyze and print statistics about the volatility regimes."""
    print("\n=== Volatility Regime Analysis ===")
    
    # Print regime distribution
    print("\nRegime Distribution:")
    regime_dist = df['vol_regime'].value_counts()
    print(regime_dist)
    
    # Print average duration in each regime
    print("\nAverage Duration in Each Regime (days):")
    avg_duration = df.groupby('vol_regime')['regime_duration'].mean()
    print(avg_duration)
    
    # Print transition probabilities
    print("\nRegime Transition Probabilities:")
    transitions = pd.crosstab(df['vol_regime'].shift(), df['vol_regime'])
    transition_probs = transitions.div(transitions.sum(axis=1), axis=0)
    print(transition_probs.round(4))
    
    # Print boundary distance statistics
    print("\nBoundary Distance Statistics:")
    boundary_stats = df.groupby('vol_regime')[['distance_to_lower_boundary', 'distance_to_upper_boundary']].describe()
    print(boundary_stats.round(4))

def main():
    # Create results directory if it doesn't exist
    Path(PATHS['results']).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading Bitcoin data...")
    df = load_data()
    
    # Create volatility regime features
    print("\nCreating volatility regime features...")
    feature_creator = VolatilityRegimeFeatures()
    df_with_features = feature_creator.create_features(df)
    
    # Analyze the features
    analyze_regime_features(df_with_features)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_regime_analysis(df_with_features)
    print(f"\nVisualizations saved to {PATHS['results']}/volatility_regime_analysis.png")

if __name__ == "__main__":
    main() 