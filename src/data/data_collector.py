import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataCollector:
    """
    A class to collect and process cryptocurrency data.
    """
    def __init__(self, symbol: str = "BTC-USD", 
                 start_date: str = None,
                 end_date: str = None,
                 interval: str = "1d"):
        """
        Initialize the data collector.
        
        Args:
            symbol (str): The cryptocurrency symbol (e.g., 'BTC-USD')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval ('1d' for daily, '1h' for hourly)
        """
        self.symbol = symbol
        self.start_date = start_date or (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.interval = interval
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch cryptocurrency data from Yahoo Finance.
        
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            logger.info(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}")
            crypto = yf.Ticker(self.symbol)
            df = crypto.history(start=self.start_date, 
                              end=self.end_date, 
                              interval=self.interval)
            
            if df.empty:
                raise ValueError(f"No data found for {self.symbol}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
            
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns from price data.
        
        Args:
            df (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: DataFrame with added returns column
        """
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        return df
        
    def calculate_realized_volatility(self, df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
        """
        Calculate realized volatility over a rolling window.
        
        Args:
            df (pd.DataFrame): Price data with returns
            window (int): Rolling window size in days
            
        Returns:
            pd.DataFrame: DataFrame with added volatility column
        """
        # Annualization factor (sqrt of trading days in a year)
        annualization = np.sqrt(252 if self.interval == '1d' else 252 * 24)
        
        # Calculate realized volatility
        df['realized_volatility'] = df['log_returns'].rolling(
            window=window
        ).std() * annualization
        
        return df
        
    def process_data(self) -> pd.DataFrame:
        """
        Complete data processing pipeline.
        
        Returns:
            pd.DataFrame: Processed data with all features
        """
        # Fetch raw data
        df = self.fetch_data()
        
        # Calculate returns and volatility
        df = self.calculate_returns(df)
        df = self.calculate_realized_volatility(df)
        
        # Drop NaN values created by rolling calculations
        df = df.dropna()
        
        return df

if __name__ == "__main__":
    # Example usage
    collector = CryptoDataCollector(symbol="BTC-USD", interval="1d")
    data = collector.process_data()
    print(data.head())
    print("\nData shape:", data.shape)
    print("\nColumns:", data.columns.tolist()) 