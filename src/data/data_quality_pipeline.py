"""
Data Quality Improvement Pipeline

This module implements a comprehensive pipeline for improving the quality of our cryptocurrency data.
It includes data cleaning, validation, and enhancement steps.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataQualityPipeline:
    """
    A pipeline for improving cryptocurrency data quality.
    
    Parameters
    ----------
    input_path : str
        Path to the input data file
    output_path : str
        Path to save the processed data
    """
    
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.data = None
        self.validation_errors = []
        
    def load_data(self) -> None:
        """Load the raw data from CSV file."""
        try:
            logger.info(f"Loading data from {self.input_path}")
            self.data = pd.read_csv(self.input_path)
            logger.info(f"Successfully loaded {len(self.data)} rows of data")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data_structure(self) -> bool:
        """
        Validate the basic structure of the data.
        
        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        required_columns = {
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
            'log_returns', 'realized_volatility'
        }
        
        # Check for required columns
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            self.validation_errors.append(f"Missing required columns: {missing_columns}")
            return False
        
        # Convert data types
        try:
            # Convert Date to datetime
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
            # Convert numeric columns to float64
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                             'log_returns', 'realized_volatility']
            for col in numeric_columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            logger.info("Data type conversion completed successfully")
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Error converting data types: {str(e)}")
            return False
    
    def clean_data(self) -> None:
        """Clean the data by removing unnecessary columns and handling missing values."""
        logger.info("Starting data cleaning process")
        
        # Remove unnecessary columns
        columns_to_drop = ['Dividends', 'Stock Splits']
        self.data = self.data.drop(columns=columns_to_drop, errors='ignore')
        
        # Sort by date
        self.data = self.data.sort_values('Date')
        
        # Reset index
        self.data = self.data.reset_index(drop=True)
        
        # Handle missing values
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove duplicates
        self.data = self.data.drop_duplicates(subset=['Date'])
        
        logger.info("Data cleaning completed")
    
    def validate_price_data(self) -> bool:
        """
        Validate price data for logical consistency.
        
        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if (self.data[col] < 0).any():
                self.validation_errors.append(f"Negative values found in {col}")
        
        # Check OHLC consistency
        invalid_ohlc = (
            (self.data['High'] < self.data['Low']) |
            (self.data['Open'] > self.data['High']) |
            (self.data['Open'] < self.data['Low']) |
            (self.data['Close'] > self.data['High']) |
            (self.data['Close'] < self.data['Low'])
        )
        
        if invalid_ohlc.any():
            self.validation_errors.append("Invalid OHLC relationships found")
        
        return len(self.validation_errors) == 0
    
    def validate_volume_data(self) -> bool:
        """
        Validate volume data for logical consistency.
        
        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        # Check for negative volume
        if (self.data['Volume'] < 0).any():
            self.validation_errors.append("Negative volume values found")
        
        # Check for zero volume
        if (self.data['Volume'] == 0).any():
            self.validation_errors.append("Zero volume values found")
        
        return len(self.validation_errors) == 0
    
    def validate_volatility_data(self) -> bool:
        """
        Validate volatility and return data for logical consistency.
        
        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        # Log volatility statistics
        logger.info("Volatility statistics:")
        logger.info(f"Mean: {self.data['realized_volatility'].mean():.4f}")
        logger.info(f"Std: {self.data['realized_volatility'].std():.4f}")
        logger.info(f"Min: {self.data['realized_volatility'].min():.4f}")
        logger.info(f"Max: {self.data['realized_volatility'].max():.4f}")
        
        # Check for negative volatility
        if (self.data['realized_volatility'] < 0).any():
            self.validation_errors.append("Negative volatility values found")
        
        # Check for extreme volatility values using z-score
        z_scores = np.abs((self.data['realized_volatility'] - self.data['realized_volatility'].mean()) / 
                         self.data['realized_volatility'].std())
        extreme_values = z_scores > 3  # Values more than 3 standard deviations from mean
        
        if extreme_values.any():
            logger.warning(f"Found {extreme_values.sum()} extreme volatility values")
            # Cap extreme values at 3 standard deviations
            self.data.loc[extreme_values, 'realized_volatility'] = (
                self.data['realized_volatility'].mean() + 
                3 * self.data['realized_volatility'].std() * 
                np.sign(self.data.loc[extreme_values, 'realized_volatility'] - 
                       self.data['realized_volatility'].mean())
            )
            logger.info("Extreme volatility values have been capped")
        
        return True
    
    def enhance_data(self) -> None:
        """Enhance the data with additional features and transformations."""
        logger.info("Starting data enhancement process")
        
        # Price-based features
        self.data['Daily_Range'] = (self.data['High'] - self.data['Low']) / self.data['Close']
        self.data['Price_Change'] = self.data['Close'].pct_change()
        self.data['Price_Volatility'] = self.data['Price_Change'].rolling(5).std()
        
        # Volume-based features
        self.data['Volume_MA5'] = self.data['Volume'].rolling(5).mean()
        self.data['Volume_MA20'] = self.data['Volume'].rolling(20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA5']
        
        # Volatility-based features
        self.data['Volatility_MA5'] = self.data['realized_volatility'].rolling(5).mean()
        self.data['Volatility_MA20'] = self.data['realized_volatility'].rolling(20).mean()
        self.data['Volatility_Ratio'] = self.data['realized_volatility'] / self.data['Volatility_MA5']
        
        # Time-based features
        self.data['Day_of_Week'] = self.data['Date'].dt.dayofweek
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Is_Weekend'] = self.data['Day_of_Week'].isin([5, 6]).astype(int)
        
        # Trend features
        self.data['Price_Trend'] = self.data['Close'].rolling(5).mean().pct_change()
        self.data['Volume_Trend'] = self.data['Volume'].rolling(5).mean().pct_change()
        
        # Momentum features
        self.data['Return_Momentum'] = self.data['log_returns'].rolling(5).mean()
        self.data['Volatility_Momentum'] = self.data['realized_volatility'].rolling(5).mean()
        
        logger.info("Data enhancement completed")
    
    def normalize_data(self) -> None:
        """Normalize numerical features."""
        logger.info("Starting data normalization")
        
        # Features to normalize
        features_to_normalize = [
            'Volume', 'Daily_Range', 'Price_Change', 'Price_Volatility',
            'Volume_Ratio', 'Volatility_Ratio', 'Price_Trend', 'Volume_Trend',
            'Return_Momentum', 'Volatility_Momentum'
        ]
        
        # Z-score normalization
        for feature in features_to_normalize:
            if feature in self.data.columns:
                self.data[f'{feature}_normalized'] = (
                    (self.data[feature] - self.data[feature].mean()) / 
                    self.data[feature].std()
                )
        
        logger.info("Data normalization completed")
    
    def save_data(self) -> None:
        """Save the processed data to CSV file."""
        try:
            logger.info(f"Saving processed data to {self.output_path}")
            self.data.to_csv(self.output_path, index=False)
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def run_pipeline(self) -> bool:
        """
        Run the complete data quality improvement pipeline.
        
        Returns
        -------
        bool
            True if pipeline completes successfully, False otherwise
        """
        try:
            # Load data
            self.load_data()
            
            # Validate data structure
            if not self.validate_data_structure():
                logger.error("Data structure validation failed")
                return False
            
            # Clean data
            self.clean_data()
            
            # Validate price data
            if not self.validate_price_data():
                logger.error("Price data validation failed")
                return False
            
            # Validate volume data
            if not self.validate_volume_data():
                logger.error("Volume data validation failed")
                return False
            
            # Validate volatility data
            if not self.validate_volatility_data():
                logger.error("Volatility data validation failed")
                return False
            
            # Enhance data
            self.enhance_data()
            
            # Normalize data
            self.normalize_data()
            
            # Save processed data
            self.save_data()
            
            # Log validation errors if any
            if self.validation_errors:
                logger.warning("Validation errors found:")
                for error in self.validation_errors:
                    logger.warning(f"- {error}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return False 