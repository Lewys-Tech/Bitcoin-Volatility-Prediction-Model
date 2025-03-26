"""
Configuration parameters for the cryptocurrency volatility prediction project.
"""

# Data Collection Parameters
DATA_CONFIG = {
    'symbol': 'BTC-USD',
    'interval': '1d',  # '1d' for daily, '1h' for hourly
    'lookback_days': 730,  # 2 years of historical data
    'prediction_horizon': 7,  # 7-day prediction window
}

# Feature Engineering Parameters
FEATURE_CONFIG = {
    'volatility_windows': [7, 14, 30],  # Multiple volatility calculation windows
    'technical_indicators': {
        'rsi_period': 14,
        'bb_period': 20,
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
    },
    'target_window': 7,  # Window for calculating target volatility
}

# Model Parameters
MODEL_CONFIG = {
    'train_test_split': 0.8,
    'validation_size': 0.1,
    'random_state': 42,
    'lstm_params': {
        'units': [64, 32],
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'epochs': 100,
        'batch_size': 32,
    },
    'garch_params': {
        'p': 1,  # GARCH lag order
        'q': 1,  # ARCH lag order
    },
}

# Training Parameters
TRAINING_CONFIG = {
    'early_stopping_patience': 10,
    'learning_rate': 0.001,
    'min_delta': 0.0001,
}

# Evaluation Metrics
EVALUATION_CONFIG = {
    'metrics': [
        'rmse',
        'mae',
        'mape',
        'directional_accuracy',
    ],
    'rolling_window_size': 30,  # For rolling evaluation
}

# Paths
PATHS = {
    'raw_data': 'data/raw',
    'processed_data': 'data/processed',
    'models': 'models',
    'results': 'results',
    'logs': 'logs',
} 