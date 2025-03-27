"""
On-Chain Metrics Features Module

This module implements various on-chain metrics that help understand the fundamental
health and activity of the cryptocurrency network. These metrics are particularly
useful for volatility prediction as they provide insights into network usage and
mining activity.

Key Features:
1. Network Activity Metrics
   - Transaction volume
   - Active addresses
   - Network hash rate
   - Mining difficulty
2. Exchange Flow Metrics
   - Exchange inflow/outflow
   - Exchange balance
3. Whale Activity Metrics
   - Large transaction count
   - Whale wallet movements
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class OnChainMetrics:
    """
    A class to calculate various on-chain metrics from blockchain data.
    
    Parameters
    ----------
    window_size : int, optional (default=24)
        Number of hours to consider for rolling calculations
    whale_threshold : float, optional (default=1000.0)
        Threshold in base currency for considering a transaction as whale activity
    """
    
    def __init__(self, window_size: int = 24, whale_threshold: float = 1000.0):
        self.window_size = window_size
        self.whale_threshold = whale_threshold
    
    def calculate_network_activity(self, 
                                 transaction_volume: List[float],
                                 active_addresses: List[int],
                                 hash_rate: List[float],
                                 difficulty: List[float]) -> Dict[str, float]:
        """
        Calculate network activity metrics.
        
        Parameters
        ----------
        transaction_volume : List[float]
            List of hourly transaction volumes
        active_addresses : List[int]
            List of hourly active addresses
        hash_rate : List[float]
            List of hourly network hash rates
        difficulty : List[float]
            List of hourly mining difficulties
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing network activity metrics
        """
        # Take only the specified window size
        window_data = {
            'transaction_volume': transaction_volume[-self.window_size:],
            'active_addresses': active_addresses[-self.window_size:],
            'hash_rate': hash_rate[-self.window_size:],
            'difficulty': difficulty[-self.window_size:]
        }
        
        metrics = {}
        
        # Calculate basic statistics
        for metric, data in window_data.items():
            if data:  # Check if data is not empty
                metrics[f'{metric}_mean'] = np.mean(data)
                metrics[f'{metric}_std'] = np.std(data)
                metrics[f'{metric}_max'] = np.max(data)
                metrics[f'{metric}_min'] = np.min(data)
                metrics[f'{metric}_trend'] = (data[-1] - data[0]) / data[0] if data[0] != 0 else 0
        
        # Calculate derived metrics
        if window_data['transaction_volume'] and window_data['active_addresses']:
            # Average transaction size
            metrics['avg_transaction_size'] = (
                np.mean(window_data['transaction_volume']) / 
                np.mean(window_data['active_addresses']) 
                if np.mean(window_data['active_addresses']) != 0 else 0
            )
            
            # Network utilization
            metrics['network_utilization'] = (
                np.mean(window_data['transaction_volume']) / 
                np.max(window_data['transaction_volume'])
                if np.max(window_data['transaction_volume']) != 0 else 0
            )
        
        # Mining metrics
        if window_data['hash_rate'] and window_data['difficulty']:
            # Mining efficiency
            metrics['mining_efficiency'] = (
                np.mean(window_data['hash_rate']) / 
                np.mean(window_data['difficulty'])
                if np.mean(window_data['difficulty']) != 0 else 0
            )
            
            # Difficulty adjustment
            metrics['difficulty_adjustment'] = (
                (window_data['difficulty'][-1] - window_data['difficulty'][0]) / 
                window_data['difficulty'][0]
                if window_data['difficulty'][0] != 0 else 0
            )
        
        return metrics
    
    def calculate_exchange_flow(self,
                              exchange_inflow: List[float],
                              exchange_outflow: List[float],
                              exchange_balance: List[float]) -> Dict[str, float]:
        """
        Calculate exchange flow metrics.
        
        Parameters
        ----------
        exchange_inflow : List[float]
            List of hourly exchange inflows
        exchange_outflow : List[float]
            List of hourly exchange outflows
        exchange_balance : List[float]
            List of hourly exchange balances
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing exchange flow metrics
        """
        # Take only the specified window size
        window_data = {
            'inflow': exchange_inflow[-self.window_size:],
            'outflow': exchange_outflow[-self.window_size:],
            'balance': exchange_balance[-self.window_size:]
        }
        
        metrics = {}
        
        # Calculate basic flow metrics
        if window_data['inflow'] and window_data['outflow']:
            metrics['net_flow'] = np.mean(window_data['inflow']) - np.mean(window_data['outflow'])
            metrics['flow_ratio'] = (
                np.mean(window_data['inflow']) / 
                np.mean(window_data['outflow'])
                if np.mean(window_data['outflow']) != 0 else float('inf')
            )
            
            # Flow volatility
            metrics['flow_volatility'] = np.std([
                i - o for i, o in zip(window_data['inflow'], window_data['outflow'])
            ])
        
        # Calculate balance metrics
        if window_data['balance']:
            metrics['balance_mean'] = np.mean(window_data['balance'])
            metrics['balance_std'] = np.std(window_data['balance'])
            metrics['balance_trend'] = (
                (window_data['balance'][-1] - window_data['balance'][0]) / 
                window_data['balance'][0]
                if window_data['balance'][0] != 0 else 0
            )
        
        return metrics
    
    def calculate_whale_activity(self,
                               large_transactions: List[float],
                               whale_wallets: List[int]) -> Dict[str, float]:
        """
        Calculate whale activity metrics.
        
        Parameters
        ----------
        large_transactions : List[float]
            List of hourly large transaction volumes
        whale_wallets : List[int]
            List of hourly active whale wallets
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing whale activity metrics
        """
        # Take only the specified window size
        window_data = {
            'large_transactions': large_transactions[-self.window_size:],
            'whale_wallets': whale_wallets[-self.window_size:]
        }
        
        metrics = {}
        
        # Calculate whale transaction metrics
        if window_data['large_transactions']:
            metrics['whale_transaction_volume'] = np.sum(window_data['large_transactions'])
            metrics['whale_transaction_count'] = len([
                t for t in window_data['large_transactions'] 
                if t >= self.whale_threshold
            ])
            metrics['whale_transaction_ratio'] = (
                metrics['whale_transaction_volume'] / 
                np.sum(window_data['large_transactions'])
                if np.sum(window_data['large_transactions']) != 0 else 0
            )
        
        # Calculate whale wallet metrics
        if window_data['whale_wallets']:
            metrics['whale_wallet_count'] = np.mean(window_data['whale_wallets'])
            metrics['whale_wallet_trend'] = (
                (window_data['whale_wallets'][-1] - window_data['whale_wallets'][0]) / 
                window_data['whale_wallets'][0]
                if window_data['whale_wallets'][0] != 0 else 0
            )
        
        return metrics
    
    def calculate_all_features(self,
                             network_data: Dict[str, List[float]],
                             exchange_data: Dict[str, List[float]],
                             whale_data: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculate all on-chain metrics.
        
        Parameters
        ----------
        network_data : Dict[str, List[float]]
            Dictionary containing network activity data
        exchange_data : Dict[str, List[float]]
            Dictionary containing exchange flow data
        whale_data : Dict[str, List[float]]
            Dictionary containing whale activity data
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing all calculated features
        """
        features = {}
        
        # Calculate network activity features
        network_features = self.calculate_network_activity(
            network_data.get('transaction_volume', []),
            network_data.get('active_addresses', []),
            network_data.get('hash_rate', []),
            network_data.get('difficulty', [])
        )
        features.update(network_features)
        
        # Calculate exchange flow features
        exchange_features = self.calculate_exchange_flow(
            exchange_data.get('inflow', []),
            exchange_data.get('outflow', []),
            exchange_data.get('balance', [])
        )
        features.update(exchange_features)
        
        # Calculate whale activity features
        whale_features = self.calculate_whale_activity(
            whale_data.get('large_transactions', []),
            whale_data.get('whale_wallets', [])
        )
        features.update(whale_features)
        
        return features