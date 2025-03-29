"""
Script to run the data quality pipeline on cryptocurrency data.
"""

import os
from pathlib import Path
from data.data_quality_pipeline import DataQualityPipeline
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the data quality pipeline."""
    # Define paths
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "data" / "raw" / "BTC_USD_raw.csv"
    output_path = base_dir / "data" / "processed" / "BTC_USD_processed.csv"
    
    # Create processed directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run pipeline
    pipeline = DataQualityPipeline(str(input_path), str(output_path))
    
    logger.info("Starting data quality pipeline")
    success = pipeline.run_pipeline()
    
    if success:
        logger.info("Data quality pipeline completed successfully")
        logger.info(f"Processed data saved to: {output_path}")
    else:
        logger.error("Data quality pipeline failed")
        if pipeline.validation_errors:
            logger.error("Validation errors found:")
            for error in pipeline.validation_errors:
                logger.error(f"- {error}")

if __name__ == "__main__":
    main() 