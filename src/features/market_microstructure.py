"""
Market Microstructure Features Module

This module implements various market microstructure features that help understand
the underlying market dynamics and liquidity conditions. These features are particularly
useful for short-term volatility prediction as they provide insights into market
pressure and liquidity conditions.

Key Features:
1. Order Book Imbalance
2. Bid-Ask Spread
3. Trading Volume Profile
4. Market Depth
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class MarketMicrostructureFeatures:
    """
    A class to calculate various market microstructure features from order book and trade data.
    
    Parameters
    ----------
    price_levels : int, optional (default=10)
        Number of price levels to consider for order book features
    volume_bins : int, optional (default=20)
        Number of bins for volume profile calculation
    """
    
    def __init__(self, price_levels: int = 10, volume_bins: int = 20):
        self.price_levels = price_levels
        self.volume_bins = volume_bins
    
    def calculate_order_book_imbalance(self, 
                                     bids: List[Tuple[float, float]], 
                                     asks: List[Tuple[float, float]]) -> float:
        """
        Calculate the order book imbalance between buy and sell orders.
        
        Parameters
        ----------
        bids : List[Tuple[float, float]]
            List of (price, volume) tuples for bid orders
        asks : List[Tuple[float, float]]
            List of (price, volume) tuples for ask orders
            
        Returns
        -------
        float
            Order book imbalance value between -1 and 1
            Positive values indicate more buy pressure
            Negative values indicate more sell pressure
        """
        # Take only the specified number of price levels
        bids = bids[:self.price_levels]
        asks = asks[:self.price_levels]
        
        # Calculate total volumes
        bid_volume = sum(vol for _, vol in bids)
        ask_volume = sum(vol for _, vol in asks)
        
        # Calculate imbalance
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
            
        return (bid_volume - ask_volume) / total_volume
    
    def calculate_bid_ask_spread(self, 
                               best_bid: float, 
                               best_ask: float) -> Tuple[float, float]:
        """
        Calculate the bid-ask spread and its percentage.
        
        Parameters
        ----------
        best_bid : float
            Best bid price
        best_ask : float
            Best ask price
            
        Returns
        -------
        Tuple[float, float]
            (Absolute spread, Percentage spread)
        """
        if best_bid <= 0 or best_ask <= 0:
            return 0.0, 0.0
            
        absolute_spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        percentage_spread = absolute_spread / mid_price
        
        return absolute_spread, percentage_spread
    
    def calculate_volume_profile(self, 
                               trades: List[Tuple[float, float]], 
                               price_range: Optional[Tuple[float, float]] = None) -> Dict[float, float]:
        """
        Calculate the volume profile across price levels.
        
        Parameters
        ----------
        trades : List[Tuple[float, float]]
            List of (price, volume) tuples for trades
        price_range : Tuple[float, float], optional
            (min_price, max_price) for binning. If None, uses min/max from trades
            
        Returns
        -------
        Dict[float, float]
            Dictionary mapping price levels to their volume
        """
        if not trades:
            return {}
            
        # Get price range if not provided
        if price_range is None:
            prices = [price for price, _ in trades]
            price_range = (min(prices), max(prices))
            
        min_price, max_price = price_range
        price_step = (max_price - min_price) / self.volume_bins
        
        # Initialize volume profile
        volume_profile = {}
        
        # Calculate volume for each price bin
        for price, volume in trades:
            bin_index = int((price - min_price) / price_step)
            bin_price = min_price + (bin_index * price_step)
            volume_profile[bin_price] = volume_profile.get(bin_price, 0) + volume
            
        return volume_profile
    
    def calculate_market_depth(self, 
                             bids: List[Tuple[float, float]], 
                             asks: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Calculate market depth metrics.
        
        Parameters
        ----------
        bids : List[Tuple[float, float]]
            List of (price, volume) tuples for bid orders
        asks : List[Tuple[float, float]]
            List of (price, volume) tuples for ask orders
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing various depth metrics:
            - bid_depth: Total volume on bid side
            - ask_depth: Total volume on ask side
            - depth_ratio: Ratio of bid to ask depth
            - total_depth: Total volume on both sides
        """
        # Calculate depths
        bid_depth = sum(vol for _, vol in bids[:self.price_levels])
        ask_depth = sum(vol for _, vol in asks[:self.price_levels])
        
        # Calculate depth ratio
        depth_ratio = bid_depth / ask_depth if ask_depth > 0 else float('inf')
        
        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'depth_ratio': depth_ratio,
            'total_depth': bid_depth + ask_depth
        }
    
    def calculate_all_features(self, 
                             order_book: Dict[str, List[Tuple[float, float]]],
                             trades: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Calculate all market microstructure features.
        
        Parameters
        ----------
        order_book : Dict[str, List[Tuple[float, float]]]
            Dictionary containing 'bids' and 'asks' lists
        trades : List[Tuple[float, float]]
            List of recent trades
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing all calculated features
        """
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        # Get best bid and ask
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        
        # Calculate all features
        features = {
            'order_book_imbalance': self.calculate_order_book_imbalance(bids, asks),
            'bid_ask_spread': self.calculate_bid_ask_spread(best_bid, best_ask)[0],
            'bid_ask_spread_pct': self.calculate_bid_ask_spread(best_bid, best_ask)[1],
        }
        
        # Add market depth features
        depth_features = self.calculate_market_depth(bids, asks)
        features.update(depth_features)
        
        # Add volume profile features
        volume_profile = self.calculate_volume_profile(trades)
        features['volume_profile_std'] = np.std(list(volume_profile.values())) if volume_profile else 0
        
        return features 