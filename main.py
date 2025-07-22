# main.py
import os
import yaml
import pandas as pd
import logging
from pathlib import Path
from m5_data_preparation import M5DataPreparator, get_model_info
from m5_training_pipeline import train_models_for_level

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Downcast numeric columns to save memory
def downcast_dtypes(df):
    start_mem = df.memory_usage().sum() / 1024**2
    cols_float = df.select_dtypes('float').columns
    cols_int = df.select_dtypes('integer').columns
    df[cols_float] = df[cols_float].apply(pd.to_numeric, downcast='float')
    df[cols_int] = df[cols_int].apply(pd.to_numeric, downcast='integer')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    end_mem = df.memory_usage().sum() / 1024**2
    print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def load_m5_data(data_dir: str) -> tuple:
    """Load M5 dataset files."""
    logger.info(f"Loading data from {data_dir}")

    # Load calendar data
    calendar_df = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
    logger.info(f"Calendar shape: {calendar_df.shape}")

    # Load sales data
    sales_df = pd.read_csv(os.path.join(data_dir, "sales_train_validation.csv"))
    logger.info(f"Sales shape: {sales_df.shape}")

    # Load prices (if needed for your model)
    if os.path.exists(os.path.join(data_dir, "sell_prices.csv")):
        prices_df = pd.read_csv(os.path.join(data_dir, "sell_prices.csv"))
        logger.info(f"Prices shape: {prices_df.shape}")
    else:
        prices_df = None

    calendar_df = downcast_dtypes(calendar_df)
    sales_df = downcast_dtypes(sales_df)
    prices_df = downcast_dtypes(prices_df)

    return sales_df, calendar_df, prices_df


def main():
    """Main pipeline execution."""
    # Load configuration
    config = load_config()

    # Get model level info
    model_info = get_model_info(config['model_level'])
    logger.info(f"Running pipeline for: {model_info['description']}")

    # Create output directories
    output_dir = Path(config.get('output_dir', 'output'))
    models_dir = output_dir / 'models' / f"level_{config['model_level']}"
    results_dir = output_dir / 'results' / f"level_{config['model_level']}"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    sales_df, calendar_df, prices_df = load_m5_data(config['data_path'])

    # Train models for the specified level
    trained_groups = train_models_for_level(sales_df, calendar_df, config)

    # Save model metadata
    import json
    metadata = {
        'model_level': config['model_level'],
        'description': model_info['description'],
        'n_models': len(trained_groups),
        'models': [group['id'] for group in trained_groups],
        'config': config
    }

    with open(results_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Pipeline completed! Trained {len(trained_groups)} models.")
    logger.info(f"Models saved to: {models_dir}")
    logger.info(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()

# config.yaml
"""
# M5 Sales Forecasting Configuration

# Model level (1-12) - determines aggregation strategy
# 1: All products, all stores (1 model)
# 2: All products, by state (3 models)
# 3: All products, by store (10 models)
# 4: All products, by category (3 models)
# 5: All products, by department (7 models)
# 6: All products, by state and category (9 models)
# 7: All products, by state and department (21 models)
# 8: All products, by store and category (30 models)
# 9: All products, by store and department (70 models)
# 10: By item, all stores (139 models)
# 11: By item and state (417 models)
# 12: By item and store (1390 models)
model_level: 10

# Data configuration
data_dir: "data"
output_dir: "output"

# Model parameters
n_training: 28  # Number of days to use as input
n_forecast: 28  # Number of days to forecast

# Training configuration
parallel: true  # Enable parallel training
n_workers: 4    # Number of parallel workers

# Model hyperparameters
learning_rate: 0.001
batch_size: 100
epochs: 50
patience: 20
"""