"""
Demo script for on-chain metrics visualization.
This script demonstrates the usage of OnChainVisualizer with sample data.
"""

import numpy as np
from datetime import datetime, timedelta
from features.onchain_visualization import OnChainVisualizer

def generate_sample_data(n_points: int = 100) -> tuple:
    """
    Generate sample data for visualization.
    
    Parameters
    ----------
    n_points : int
        Number of data points to generate
        
    Returns
    -------
    tuple
        (network_data, exchange_data, whale_data, timestamps)
    """
    # Generate timestamps
    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(hours=i) for i in range(n_points)]
    
    # Generate network data
    network_data = {
        'transaction_volume': np.random.normal(1000, 100, n_points).cumsum().tolist(),
        'active_addresses': np.random.normal(5000, 500, n_points).cumsum().tolist(),
        'hash_rate': np.random.normal(100, 10, n_points).cumsum().tolist(),
        'difficulty': np.random.normal(1, 0.1, n_points).cumsum().tolist()
    }
    
    # Generate exchange data
    exchange_data = {
        'inflow': np.random.normal(500, 50, n_points).cumsum().tolist(),
        'outflow': np.random.normal(450, 45, n_points).cumsum().tolist(),
        'balance': np.random.normal(10000, 1000, n_points).cumsum().tolist()
    }
    
    # Generate whale data
    whale_data = {
        'large_transactions': np.random.normal(2000, 200, n_points).cumsum().tolist(),
        'whale_wallets': np.random.randint(10, 20, n_points).tolist()
    }
    
    return network_data, exchange_data, whale_data, timestamps

def main():
    """Main function to demonstrate visualization utilities."""
    # Create visualizer instance
    visualizer = OnChainVisualizer(style='seaborn-v0_8', figsize=(12, 8))
    
    # Generate sample data
    network_data, exchange_data, whale_data, timestamps = generate_sample_data()
    
    # Create output directory for saved plots
    import os
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot Network Activity
    print("Generating Network Activity plot...")
    visualizer.plot_network_activity(
        network_data['transaction_volume'],
        network_data['active_addresses'],
        network_data['hash_rate'],
        network_data['difficulty'],
        timestamps=timestamps,
        save_path=f'{output_dir}/network_activity.png'
    )
    
    # 2. Plot Exchange Flow
    print("Generating Exchange Flow plot...")
    visualizer.plot_exchange_flow(
        exchange_data['inflow'],
        exchange_data['outflow'],
        exchange_data['balance'],
        timestamps=timestamps,
        save_path=f'{output_dir}/exchange_flow.png'
    )
    
    # 3. Plot Whale Activity
    print("Generating Whale Activity plot...")
    visualizer.plot_whale_activity(
        whale_data['large_transactions'],
        whale_data['whale_wallets'],
        timestamps=timestamps,
        save_path=f'{output_dir}/whale_activity.png'
    )
    
    # 4. Plot Correlation Heatmap
    print("Generating Correlation Heatmap...")
    visualizer.plot_correlation_heatmap(
        network_data,
        exchange_data,
        whale_data,
        save_path=f'{output_dir}/correlation_heatmap.png'
    )
    
    # 5. Plot Rolling Statistics for Transaction Volume
    print("Generating Rolling Statistics plot...")
    visualizer.plot_rolling_statistics(
        network_data['transaction_volume'],
        window_size=24,
        title='24-Hour Rolling Statistics for Transaction Volume',
        timestamps=timestamps,
        save_path=f'{output_dir}/rolling_stats.png'
    )
    
    print(f"\nAll plots have been saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main() 