"""
Script to run data analysis and generate visualization plots.
"""

import os
from pathlib import Path
from visualization.data_analysis import DataAnalyzer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run data analysis and generate plots."""
    # Define paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "processed" / "BTC_USD_processed.csv"
    output_dir = base_dir / "visualization_output"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run analyzer
    analyzer = DataAnalyzer(str(data_path), str(output_dir))
    
    logger.info("Starting data analysis and visualization")
    analyzer.generate_all_plots()
    logger.info("Data analysis and visualization completed")

if __name__ == "__main__":
    main() 