import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class M5DataPreparator:
    """
    A configurable data preparation class for M5 dataset that handles
    different model levels (1-12) as specified in the challenge.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model_level = config.get("model_level", 12)
        self.n_training = config.get("n_training", 28)
        self.n_forecast = config.get("n_forecast", 28)

    def get_aggregation_key(self) -> List[str]:
        """
        Returns the aggregation key columns based on model level.
        """
        level_keys = {
            1: [],  # All products, all stores
            2: ['state_id'],  # By state
            3: ['store_id'],  # By store
            4: ['cat_id'],  # By category
            5: ['dept_id'],  # By department
            6: ['state_id', 'cat_id'],  # By state and category
            7: ['state_id', 'dept_id'],  # By state and department
            8: ['store_id', 'cat_id'],  # By store and category
            9: ['store_id', 'dept_id'],  # By store and department
            10: ['item_id'],  # By item (across all stores)
            11: ['item_id', 'state_id'],  # By item and state
            12: ['item_id', 'store_id']  # By item and store (original)
        }
        return level_keys.get(self.model_level, [])

    def aggregate_sales(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates sales data based on the model level.
        """
        agg_keys = self.get_aggregation_key()

        # Get date columns
        date_cols = [col for col in sales_df.columns if col.startswith('d_')]

        # Get metadata columns
        meta_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

        if not agg_keys:  # Level 1: aggregate all
            # Sum all sales across all dimensions
            agg_df = pd.DataFrame({
                'id': ['total'],
                **{col: [sales_df[col].sum()] for col in date_cols}
            })
        elif self.model_level == 10:
            # Special handling for level 10: keep all item-store rows but group by item
            agg_df = sales_df.copy()
            agg_df['id'] = agg_df['item_id'].astype(str)
            # We keep all rows but will group them when creating models
        elif self.model_level == 11:
            # Level 11: aggregate by item and state
            agg_df = sales_df.groupby(['item_id', 'state_id'])[date_cols].sum().reset_index()
            agg_df['id'] = agg_df['item_id'].astype(str) + '_' + agg_df['state_id'].astype(str)
        elif self.model_level == 12:
            # Level 12: no aggregation, each item-store is separate
            agg_df = sales_df.copy()
            agg_df['id'] = agg_df['item_id'].astype(str) + '_' + agg_df['store_id'].astype(str)
        else:
            # For other levels, group by specified keys and sum
            agg_df = sales_df.groupby(agg_keys)[date_cols].sum().reset_index()

            # Create unique ID for each aggregation
            if len(agg_keys) == 1:
                agg_df['id'] = agg_df[agg_keys[0]].astype(str)
            else:
                agg_df['id'] = agg_df[agg_keys].apply(lambda x: '_'.join(x.astype(str)), axis=1)

        return agg_df

    def prepare_exogenous_features(self, calendar_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares exogenous features from calendar data.
        """
        calendar_df = calendar_df.copy()

        # Create event flags
        calendar_df['event_1_flag'] = calendar_df['event_name_1'].notna().astype(int)
        calendar_df['event_2_flag'] = calendar_df['event_name_2'].notna().astype(int)

        # For levels that aggregate across states, we need to handle SNAP features differently
        agg_keys = self.get_aggregation_key()

        if 'state_id' not in agg_keys and self.model_level not in [11, 12]:
            # Average SNAP features across states for aggregated models
            calendar_df['snap_avg'] = calendar_df[['snap_CA', 'snap_TX', 'snap_WI']].mean(axis=1)
            exo_features = ['snap_avg', 'event_1_flag', 'event_2_flag']
        else:
            # Keep state-specific SNAP features
            exo_features = ['snap_CA', 'snap_TX', 'snap_WI', 'event_1_flag', 'event_2_flag']

        return calendar_df[['d'] + exo_features]

    def prepare_data(self, sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> Tuple:
        """
        Main method to prepare data for training based on model level.
        """
        logger.info(f"Preparing data for model level {self.model_level}")

        # Aggregate sales based on model level
        aggregated_sales = self.aggregate_sales(sales_df)
        logger.info(f"Aggregated sales shape: {aggregated_sales.shape}")

        # Prepare train/validation split
        date_cols = [col for col in aggregated_sales.columns if col.startswith('d_')]
        non_date_cols = [col for col in aggregated_sales.columns if not col.startswith('d_')]

        train_df = aggregated_sales[non_date_cols + date_cols[:-self.n_forecast]].copy()
        valid_df = aggregated_sales[non_date_cols + date_cols[-self.n_forecast:]].copy()

        # Transpose sales data for time series format
        sales_t_df = aggregated_sales[date_cols].T.reset_index().rename(columns={'index': 'd'})
        n_series = len(aggregated_sales)  # Number of time series at this level

        # Prepare exogenous features
        calendar_exo = self.prepare_exogenous_features(calendar_df)

        # Merge with exogenous features
        sales_t_exo_df = pd.merge(sales_t_df, calendar_exo, on='d')
        sales_t_df = sales_t_df.drop(columns=['d'])
        sales_t_exo_df = sales_t_exo_df.drop(columns=['d'])

        # Get feature dimensions
        n_products_stores = sales_t_df.shape[1]
        n_products_stores_exo = sales_t_exo_df.shape[1]

        # Split into train and validation
        train_sales_t_df = sales_t_df.iloc[:-self.n_forecast, :]
        valid_sales_t_df = sales_t_df.iloc[-self.n_forecast:, :]

        train_sales_t_exo_df = sales_t_exo_df.iloc[:-self.n_forecast, :]
        valid_sales_t_exo_df = sales_t_exo_df.iloc[-self.n_forecast:, :]

        # Feature scaling
        sc = MinMaxScaler(feature_range=(0, 1))
        train_sales_t_df.columns = train_sales_t_df.columns.astype(str)
        valid_sales_t_df.columns = valid_sales_t_df.columns.astype(str)
        train_sales_t_df_scaled = sc.fit_transform(train_sales_t_df)
        valid_sales_t_df_scaled = sc.transform(valid_sales_t_df)

        sc_exo = MinMaxScaler(feature_range=(0, 1))
        train_sales_t_exo_df.columns = train_sales_t_exo_df.columns.astype(str)
        valid_sales_t_exo_df.columns = valid_sales_t_exo_df.columns.astype(str)
        train_sales_t_exo_df_scaled = sc_exo.fit_transform(train_sales_t_exo_df)
        valid_sales_t_exo_df_scaled = sc_exo.transform(valid_sales_t_exo_df)

        # Log information about the prepared data
        logger.info(f"Number of time series: {n_series}")
        logger.info(f"Train shape (with exo): {train_sales_t_exo_df_scaled.shape}")
        logger.info(f"Valid shape (with exo): {valid_sales_t_exo_df_scaled.shape}")

        return (n_products_stores, n_products_stores_exo,
                train_sales_t_exo_df_scaled, valid_sales_t_exo_df_scaled,
                train_df, valid_df, non_date_cols, sc, aggregated_sales)


def prepare_data(sales_df: pd.DataFrame, calendar_df: pd.DataFrame, config: Dict) -> Tuple:
    """
    Backward compatible function that uses the M5DataPreparator class.
    """
    preparator = M5DataPreparator(config)
    return preparator.prepare_data(sales_df, calendar_df)


# Example usage for different model levels
def get_model_info(model_level: int) -> Dict:
    """
    Returns information about what each model level represents.
    """
    model_descriptions = {
        1: "All products, all stores (1 model, 1390 series)",
        2: "All products, by state (3 models)",
        3: "All products, by store (10 models)",
        4: "All products, by category (3 models)",
        5: "All products, by department (7 models)",
        6: "All products, by state and category (9 models)",
        7: "All products, by state and department (21 models)",
        8: "All products, by store and category (30 models)",
        9: "All products, by store and department (70 models)",
        10: "By item, all stores (139 models)",
        11: "By item and state (417 models)",
        12: "By item and store (1390 models)"
    }
    return {
        "level": model_level,
        "description": model_descriptions.get(model_level, "Unknown level")
    }