"""
On-Chain Metrics Visualization Module

This module provides visualization utilities for on-chain metrics data.
It includes various plotting functions to help analyze network activity,
exchange flows, and whale activity patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class OnChainVisualizer:
    """
    A class to create visualizations for on-chain metrics.
    
    Parameters
    ----------
    style : str, optional (default='seaborn')
        Matplotlib style to use for plots
    figsize : Tuple[float, float], optional (default=(12, 8))
        Default figure size for plots
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[float, float] = (12, 8)):
        plt.style.use(style)
        self.figsize = figsize
    
    def plot_network_activity(self,
                            transaction_volume: List[float],
                            active_addresses: List[int],
                            hash_rate: List[float],
                            difficulty: List[float],
                            timestamps: Optional[List[datetime]] = None,
                            save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive visualization of network activity metrics.
        
        Parameters
        ----------
        transaction_volume : List[float]
            List of transaction volumes
        active_addresses : List[int]
            List of active addresses
        hash_rate : List[float]
            List of network hash rates
        difficulty : List[float]
            List of mining difficulties
        timestamps : List[datetime], optional
            List of timestamps for x-axis
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Network Activity Analysis', fontsize=16)
        
        # Create x-axis if timestamps not provided
        if timestamps is None:
            x = range(len(transaction_volume))
        else:
            x = timestamps
        
        # Transaction Volume
        axes[0, 0].plot(x, transaction_volume, label='Volume')
        axes[0, 0].set_title('Transaction Volume')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Volume')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Active Addresses
        axes[0, 1].plot(x, active_addresses, label='Addresses', color='green')
        axes[0, 1].set_title('Active Addresses')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Hash Rate
        axes[1, 0].plot(x, hash_rate, label='Hash Rate', color='orange')
        axes[1, 0].set_title('Network Hash Rate')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Hash Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Mining Difficulty
        axes[1, 1].plot(x, difficulty, label='Difficulty', color='red')
        axes[1, 1].set_title('Mining Difficulty')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Difficulty')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_exchange_flow(self,
                          exchange_inflow: List[float],
                          exchange_outflow: List[float],
                          exchange_balance: List[float],
                          timestamps: Optional[List[datetime]] = None,
                          save_path: Optional[str] = None) -> None:
        """
        Create a visualization of exchange flow metrics.
        
        Parameters
        ----------
        exchange_inflow : List[float]
            List of exchange inflows
        exchange_outflow : List[float]
            List of exchange outflows
        exchange_balance : List[float]
            List of exchange balances
        timestamps : List[datetime], optional
            List of timestamps for x-axis
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        fig.suptitle('Exchange Flow Analysis', fontsize=16)
        
        # Create x-axis if timestamps not provided
        if timestamps is None:
            x = range(len(exchange_inflow))
        else:
            x = timestamps
        
        # Inflow vs Outflow
        axes[0].plot(x, exchange_inflow, label='Inflow', color='green')
        axes[0].plot(x, exchange_outflow, label='Outflow', color='red')
        axes[0].set_title('Exchange Flow')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Volume')
        axes[0].legend()
        axes[0].grid(True)
        
        # Balance
        axes[1].plot(x, exchange_balance, label='Balance', color='blue')
        axes[1].set_title('Exchange Balance')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Balance')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_whale_activity(self,
                           large_transactions: List[float],
                           whale_wallets: List[int],
                           timestamps: Optional[List[datetime]] = None,
                           save_path: Optional[str] = None) -> None:
        """
        Create a visualization of whale activity metrics.
        
        Parameters
        ----------
        large_transactions : List[float]
            List of large transaction volumes
        whale_wallets : List[int]
            List of active whale wallets
        timestamps : List[datetime], optional
            List of timestamps for x-axis
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        fig.suptitle('Whale Activity Analysis', fontsize=16)
        
        # Create x-axis if timestamps not provided
        if timestamps is None:
            x = range(len(large_transactions))
        else:
            x = timestamps
        
        # Large Transactions
        axes[0].plot(x, large_transactions, label='Large Transactions', color='purple')
        axes[0].set_title('Large Transaction Volume')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Volume')
        axes[0].legend()
        axes[0].grid(True)
        
        # Whale Wallets
        axes[1].plot(x, whale_wallets, label='Whale Wallets', color='orange')
        axes[1].set_title('Active Whale Wallets')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Count')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_correlation_heatmap(self,
                               network_data: Dict[str, List[float]],
                               exchange_data: Dict[str, List[float]],
                               whale_data: Dict[str, List[float]],
                               save_path: Optional[str] = None) -> None:
        """
        Create a correlation heatmap of all metrics.
        
        Parameters
        ----------
        network_data : Dict[str, List[float]]
            Dictionary containing network activity data
        exchange_data : Dict[str, List[float]]
            Dictionary containing exchange flow data
        whale_data : Dict[str, List[float]]
            Dictionary containing whale activity data
        save_path : str, optional
            Path to save the plot
        """
        # Combine all data into a DataFrame
        data = {}
        data.update(network_data)
        data.update(exchange_data)
        data.update(whale_data)
        
        df = pd.DataFrame(data)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        plt.figure(figsize=self.figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap of On-Chain Metrics')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_rolling_statistics(self,
                              data: List[float],
                              window_size: int,
                              title: str,
                              timestamps: Optional[List[datetime]] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Create a visualization of rolling statistics for a metric.
        
        Parameters
        ----------
        data : List[float]
            List of values for the metric
        window_size : int
            Size of the rolling window
        title : str
            Title for the plot
        timestamps : List[datetime], optional
            List of timestamps for x-axis
        save_path : str, optional
            Path to save the plot
        """
        # Convert to pandas Series for rolling calculations
        series = pd.Series(data)
        
        # Calculate rolling statistics
        rolling_mean = series.rolling(window=window_size).mean()
        rolling_std = series.rolling(window=window_size).std()
        
        # Create x-axis if timestamps not provided
        if timestamps is None:
            x = range(len(data))
        else:
            x = timestamps
        
        # Create plot
        plt.figure(figsize=self.figsize)
        plt.plot(x, data, label='Original', alpha=0.5)
        plt.plot(x, rolling_mean, label=f'{window_size}-period Mean', color='red')
        plt.fill_between(x,
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        alpha=0.2,
                        color='red',
                        label=f'{window_size}-period Std Dev')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show() 