import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, BatchNormalization, Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from multiprocessing import Pool, cpu_count
import warnings
import pickle
import json

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class M5ModelTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.model_level = config.get("model_level", 12)
        self.n_training = config.get("n_training", 28)
        self.n_forecast = config.get("n_forecast", 28)
        self.output_dir = config.get("output_dir", "output")
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 100)
        self.learning_rate = config.get("learning_rate", 0.001)

    def build_model(self, n_features: int, n_outputs: int) -> Sequential:
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=7, strides=1, padding="causal",
                         activation="relu", input_shape=(self.n_training, n_features)))
        model.add(MaxPooling1D())
        model.add(Conv1D(filters=32, kernel_size=7, strides=1, padding="causal", activation="relu"))
        model.add(MaxPooling1D())
        model.add(LSTM(units=256, return_sequences=True))
        model.add(BatchNormalization())
        model.add(LSTM(units=128, return_sequences=True))
        model.add(BatchNormalization())
        model.add(LSTM(units=64))
        model.add(BatchNormalization())
        model.add(Dense(units=n_outputs))

        opt = Adam(learning_rate=self.learning_rate, clipvalue=0.5)
        model.compile(loss="mse", optimizer=opt, metrics=["RootMeanSquaredError"])
        return model

    def create_dataset(self, data_array: np.ndarray, series_indices: List[int]) -> tuple:
        X_train, y_train = [], []
        X_valid, y_valid = [], []

        total_length = data_array.shape[0]
        cutoff = total_length - self.n_training

        for i in range(cutoff):
            X_train.append(data_array[i:i + self.n_training])
            y_train.append(data_array[i + self.n_training, series_indices])

        #for i in range(cutoff, total_length - self.n_training):
        #    X_valid.append(data_array[i:i + self.n_training])
        #    y_valid.append(data_array[i + self.n_training, series_indices])
        print(np.array(X_train).shape, np.array(y_train).shape)
        return np.array(X_train), np.array(y_train)

    def train_model(self,
                    train_data: np.ndarray,
                    valid_data: np.ndarray,
                    aggregated_sales: pd.DataFrame,
                    n_features: int):
        logger.info(f"Training shared model for model_level={self.model_level}")

        series_indices = list(range(aggregated_sales.shape[0]))
        id_list = aggregated_sales['id'].tolist()

        # Prepare training data (sliding windows)
        X_train, y_train = self.create_dataset(train_data, series_indices)
        X_valid, y_valid = self.create_dataset(valid_data, series_indices)

        logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        if X_valid is not None:
            logger.info(f"X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")

        # Build and compile model
        model = self.build_model(n_features, len(series_indices))

        # TensorBoard + EarlyStopping
        model_id = f"level_{self.model_level}_shared"
        log_dir = os.path.join(self.output_dir, "results", f"level_{self.model_level}", model_id, "tb_logs")
        os.makedirs(log_dir, exist_ok=True)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=20, mode="min", verbose=1),
            TensorBoard(log_dir=log_dir)
        ]

        # Train model
        model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid) if X_valid is not None else None,
            epochs=self.epochs,
            batch_size=min(self.batch_size, max(8, X_train.shape[0] // 10)),
            callbacks=callbacks,
            verbose=1
        )

        # Save model
        model_dir = os.path.join(self.output_dir, "models", f"level_{self.model_level}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_id}_model.keras")
        model.save(model_path)
        logger.info(f"Saved model to: {model_path}")

        # Save validation features for streamlit
        np.save(f"{self.output_dir}/results/level_{self.model_level}/X_valid.npy", X_valid)

        # Predict for validation period
        y_pred_scaled = model.predict(X_valid, verbose=0)
        y_pred_full = y_pred_scaled

        # Load scaler
        with open(f"{self.output_dir}/results/level_{self.model_level}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        #import pdb;pdb.set_trace()
        inv_preds = scaler.inverse_transform(y_pred_full)

        # Save predictions in same format as valid_df
        d_cols = [f"d_{i}" for i in range(1886, 1886 + self.n_forecast)]
        pred_df = pd.DataFrame(inv_preds.T, columns=d_cols)
        pred_df.insert(0, "id", id_list)
        pred_df.to_csv(f"{self.output_dir}/results/level_{self.model_level}/pred_df.csv", index=False)

        # Save id list and metadata
        with open(f"{self.output_dir}/results/level_{self.model_level}/id_list.json", "w") as f:
            json.dump(id_list, f)

        with open(f"{self.output_dir}/results/level_{self.model_level}/metadata.json", "w") as f:
            json.dump({
                "n_models": 1,
                "models": [f"level_{self.model_level}_shared"],
                "description": f"Shared model for level {self.model_level}",
                "series_count": len(id_list)
            }, f)


def train_models_for_level(sales_df: pd.DataFrame, calendar_df: pd.DataFrame, config: Dict):
    from m5_data_preparation import M5DataPreparator

    # Prepare data
    preparator = M5DataPreparator(config)
    (n_products_stores, n_products_stores_exo,
     train_sales_t_exo_df_scaled, valid_sales_t_exo_df_scaled,
     train_df, valid_df, non_date_cols, sc, aggregated_sales) = preparator.prepare_data(sales_df, calendar_df)
    model_level = config.get("model_level", 12)

    train_df.to_csv(f"output/results/level_{model_level}/train_df.csv", index=False)
    valid_df.to_csv(f"output/results/level_{model_level}/valid_df.csv", index=False)

    # Save scaler
    output_dir = config.get("output_dir", "output")
    with open(f"{output_dir}/results/level_{model_level}/scaler.pkl", "wb") as f:
        pickle.dump(sc, f)

    # Train shared model
    trainer = M5ModelTrainer(config)
    trainer.train_model(train_sales_t_exo_df_scaled, valid_sales_t_exo_df_scaled, aggregated_sales, n_products_stores_exo)

    return [{
        'id': f'level_{config["model_level"]}_shared',
        'n_series': len(aggregated_sales)
    }]
