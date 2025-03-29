"""
Data Analysis Visualization Module

This module provides comprehensive visualization tools for analyzing the processed cryptocurrency data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import List, Optional, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """
    A class for analyzing and visualizing cryptocurrency data.
    
    Parameters
    ----------
    data_path : str
        Path to the processed data file
    output_dir : str
        Directory to save the generated plots
    """
    
    def __init__(self, data_path: str, output_dir: str):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = None
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
    
    def plot_price_and_volume(self) -> None:
        """Plot price and volume over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot price
        ax1.plot(self.data['Date'], self.data['Close'], label='Close Price')
        ax1.set_title('Bitcoin Price Over Time')
        ax1.set_ylabel('Price (USD)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot volume
        ax2.plot(self.data['Date'], self.data['Volume'], label='Volume', color='orange')
        ax2.set_title('Trading Volume Over Time')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'price_and_volume.png')
        plt.close()
        logger.info("Price and volume plot saved")
    
    def plot_volatility_analysis(self) -> None:
        """Plot volatility metrics and their relationships."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Realized volatility over time
        ax1.plot(self.data['Date'], self.data['realized_volatility'], label='Realized Volatility')
        ax1.set_title('Realized Volatility Over Time')
        ax1.set_ylabel('Volatility')
        ax1.grid(True)
        ax1.legend()
        
        # Volatility distribution
        sns.histplot(data=self.data, x='realized_volatility', bins=50, ax=ax2)
        ax2.set_title('Distribution of Realized Volatility')
        ax2.set_xlabel('Volatility')
        
        # Volatility vs Price Change
        ax3.scatter(self.data['Price_Change'], self.data['realized_volatility'], alpha=0.5)
        ax3.set_title('Volatility vs Price Change')
        ax3.set_xlabel('Price Change')
        ax3.set_ylabel('Realized Volatility')
        
        # Volatility moving averages
        ax4.plot(self.data['Date'], self.data['realized_volatility'], label='Realized Volatility', alpha=0.5)
        ax4.plot(self.data['Date'], self.data['Volatility_MA5'], label='5-day MA')
        ax4.plot(self.data['Date'], self.data['Volatility_MA20'], label='20-day MA')
        ax4.set_title('Volatility Moving Averages')
        ax4.set_ylabel('Volatility')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'volatility_analysis.png')
        plt.close()
        logger.info("Volatility analysis plots saved")
    
    def plot_feature_correlations(self) -> None:
        """Plot correlation heatmap of features."""
        # Select features for correlation analysis
        features = [
            'realized_volatility', 'Price_Change', 'Daily_Range', 'Volume',
            'Volume_Ratio', 'Volatility_Ratio', 'Price_Trend', 'Volume_Trend',
            'Return_Momentum', 'Volatility_Momentum'
        ]
        
        # Calculate correlation matrix
        corr_matrix = self.data[features].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_correlations.png')
        plt.close()
        logger.info("Feature correlation heatmap saved")
    
    def plot_time_based_patterns(self) -> None:
        """Plot patterns based on time features."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Volatility by day of week
        sns.boxplot(data=self.data, x='Day_of_Week', y='realized_volatility', ax=ax1)
        ax1.set_title('Volatility by Day of Week')
        ax1.set_xlabel('Day of Week (0=Monday)')
        ax1.set_ylabel('Realized Volatility')
        
        # Volume by day of week
        sns.boxplot(data=self.data, x='Day_of_Week', y='Volume', ax=ax2)
        ax2.set_title('Volume by Day of Week')
        ax2.set_xlabel('Day of Week (0=Monday)')
        ax2.set_ylabel('Volume')
        
        # Volatility by month
        sns.boxplot(data=self.data, x='Month', y='realized_volatility', ax=ax3)
        ax3.set_title('Volatility by Month')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Realized Volatility')
        
        # Weekend vs Weekday comparison
        weekend_vol = self.data[self.data['Is_Weekend'] == 1]['realized_volatility']
        weekday_vol = self.data[self.data['Is_Weekend'] == 0]['realized_volatility']
        
        ax4.boxplot([weekday_vol, weekend_vol], labels=['Weekday', 'Weekend'])
        ax4.set_title('Volatility: Weekend vs Weekday')
        ax4.set_ylabel('Realized Volatility')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_patterns.png')
        plt.close()
        logger.info("Time-based pattern plots saved")
    
    def plot_market_regime_analysis(self) -> None:
        """Plot market regime analysis based on volatility and returns."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Volatility regimes
        vol_threshold = self.data['realized_volatility'].mean() + self.data['realized_volatility'].std()
        high_vol_mask = self.data['realized_volatility'] > vol_threshold
        
        ax1.scatter(self.data[~high_vol_mask]['Date'], 
                   self.data[~high_vol_mask]['realized_volatility'],
                   label='Normal Volatility', alpha=0.5)
        ax1.scatter(self.data[high_vol_mask]['Date'],
                   self.data[high_vol_mask]['realized_volatility'],
                   label='High Volatility', color='red', alpha=0.5)
        ax1.set_title('Volatility Regimes Over Time')
        ax1.set_ylabel('Realized Volatility')
        ax1.grid(True)
        ax1.legend()
        
        # Return regimes
        return_threshold = self.data['log_returns'].std()
        high_return_mask = abs(self.data['log_returns']) > return_threshold
        
        ax2.scatter(self.data[~high_return_mask]['Date'],
                   self.data[~high_return_mask]['log_returns'],
                   label='Normal Returns', alpha=0.5)
        ax2.scatter(self.data[high_return_mask]['Date'],
                   self.data[high_return_mask]['log_returns'],
                   label='High Returns', color='red', alpha=0.5)
        ax2.set_title('Return Regimes Over Time')
        ax2.set_ylabel('Log Returns')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'market_regimes.png')
        plt.close()
        logger.info("Market regime analysis plots saved")
    
    def generate_all_plots(self) -> None:
        """Generate all analysis plots."""
        logger.info("Starting to generate all analysis plots")
        
        self.plot_price_and_volume()
        self.plot_volatility_analysis()
        self.plot_feature_correlations()
        self.plot_time_based_patterns()
        self.plot_market_regime_analysis()
        
        logger.info("All analysis plots generated successfully") 