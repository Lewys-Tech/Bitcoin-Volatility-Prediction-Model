"""
Script to run the feature engineering pipeline for volatility regime prediction.
"""

import os
from pathlib import Path
from features.regime_features import RegimeFeatureEngineer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the feature engineering pipeline."""
    # Define paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "processed" / "BTC_USD_processed.csv"
    output_path = base_dir / "data" / "features" / "regime_features.csv"
    
    # Create features directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run feature engineering
    engineer = RegimeFeatureEngineer(str(data_path), str(output_path))
    
    logger.info("Starting feature engineering pipeline")
    engineer.run_pipeline()
    logger.info("Feature engineering pipeline completed")

if __name__ == "__main__":
    main() 