import os
import sys
from pathlib import Path
import pandas as pd
from data_collector import CryptoDataCollector
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import DATA_CONFIG, PATHS

def main():
    # Create data directories if they don't exist
    for path in [PATHS['raw_data'], PATHS['processed_data']]:
        os.makedirs(path, exist_ok=True)

    # Initialize data collector
    collector = CryptoDataCollector(
        symbol=DATA_CONFIG['symbol'],
        interval=DATA_CONFIG['interval']
    )

    try:
        # Fetch and process data
        print(f"Fetching data for {DATA_CONFIG['symbol']}...")
        data = collector.process_data()
        
        # Save raw data
        raw_file_path = os.path.join(PATHS['raw_data'], f"{DATA_CONFIG['symbol'].replace('-', '_')}_raw.csv")
        data.to_csv(raw_file_path)
        print(f"Raw data saved to: {raw_file_path}")
        
        # Display data summary
        print("\nData Summary:")
        print(f"Time period: {data.index[0]} to {data.index[-1]}")
        print(f"Number of records: {len(data)}")
        print(f"Columns: {data.columns.tolist()}")
        print("\nFirst few rows:")
        print(data.head())
        
        return data

    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

if __name__ == "__main__":
    data = main() 