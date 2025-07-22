# streamlit_app.py (Final version: supports all levels and fallback to product-store forecasts)

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import pickle
from pathlib import Path
import plotly.graph_objects as go
import yaml

st.set_page_config(page_title="M5 Sales Forecasting Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

@st.cache_data
def load_metadata(model_level):
    metadata_path = Path(f"output/results/level_{model_level}/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_saved_predictions(model_level):
    pred_path = Path(f"output/results/level_{model_level}/pred_df.csv")
    if not pred_path.exists():
        return None
    return pd.read_csv(pred_path)

@st.cache_data
def load_id_list(model_level):
    with open(f"output/results/level_{model_level}/id_list.json", "r") as f:
        return json.load(f)

def get_aggregation_key(level):
    level_keys = {
        1: [],
        2: ['state_id'],
        3: ['store_id'],
        4: ['cat_id'],
        5: ['dept_id'],
        6: ['state_id', 'cat_id'],
        7: ['state_id', 'dept_id'],
        8: ['store_id', 'cat_id'],
        9: ['store_id', 'dept_id'],
        10: ['item_id'],
        11: ['item_id', 'state_id'],
        12: ['item_id', 'store_id']
    }
    return level_keys.get(level, [])

def extract_key_values_from_id(series_id, level):
    parts = series_id.split('_')
    try:
        if level == 1:
            return 'total'
        elif level == 2:
            return parts[-1]  # state_id
        elif level == 3:
            return '_'.join(parts[-2:])  # store_id
        elif level == 4:
            return parts[0]  # cat_id
        elif level == 5:
            return '_'.join(parts[:2])  # dept_id
        elif level == 6:
            return f"{parts[0]}_{parts[-1]}"  # state_id + cat_id
        elif level == 7:
            return f"{parts[-1]}_{'_'.join(parts[:2])}"  # state_id + dept_id
        elif level == 8:
            return f"{'_'.join(parts[-2:])}_{parts[0]}"  # store_id + cat_id
        elif level == 9:
            return f"{'_'.join(parts[-2:])}_{'_'.join(parts[:2])}"  # store_id + dept_id
        elif level == 10:
            return '_'.join(parts[:3])  # item_id
        elif level == 11:
            return f"{'_'.join(parts[:3])}_{parts[3]}"  # item_id + state
        elif level == 12:
            return f"{'_'.join(parts[:3])}_{parts[3]}"  # item_id + store
        else:
            return series_id
    except IndexError:
        return series_id

def group_ids_by_model(id_list, level):
    group_dict = {}
    for i in id_list:
        group_key = extract_key_values_from_id(i, level)
        if group_key not in group_dict:
            group_dict[group_key] = []
        group_dict[group_key].append(i)
    return group_dict

def plot_forecast(historical_sales, predictions, dates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[:-28], y=historical_sales[:-28], mode='lines', name='Historical', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=dates[-28:], y=historical_sales[-28:], mode='lines', name='Validation', line=dict(color='green', dash='dash')))
    #pred_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=len(predictions))
    pred_dates = dates[-28:]
    fig.add_trace(go.Scatter(x=pred_dates, y=predictions, mode='lines+markers', name='Predictions', line=dict(color='red')))
    fig.update_layout(title='Sales Forecast', xaxis_title='Date', yaxis_title='Sales', hovermode='x unified', height=500)
    return fig

def main():
    st.title("👚 M5 Sales Forecasting Dashboard")
    config = load_config()

    st.sidebar.header("Model Level")
    available_levels = [i for i in range(1, 13) if Path(f"output/models/level_{i}").exists()]
    selected_level = st.sidebar.selectbox("Select Level", available_levels)

    metadata = load_metadata(selected_level)
    if not metadata:
        st.error("Metadata missing.")
        return

    pred_df = load_saved_predictions(selected_level)
    if pred_df is None:
        st.error("Prediction file not found. Please run training first.")
        return

    id_list = load_id_list(selected_level)
    grouped_ids = group_ids_by_model(id_list, selected_level)
    group_keys = sorted(grouped_ids.keys())

    selected_model_key = st.sidebar.selectbox("Select Model", group_keys)
    series_options = grouped_ids[selected_model_key]
    selected_series_id = st.sidebar.selectbox("Select Time Series", series_options)

    st.sidebar.info(f"Level {selected_level}: {metadata['description']}")
    st.subheader("📈 Forecast Visualization")

    sales_df = pd.read_csv(os.path.join(config['data_path'], "sales_train_validation.csv"))
    sales_df['id'] = sales_df['id'].astype(str)

    if selected_level in [1,2,3,4,5,6,7,8,9,10,11]:
        if selected_level == 1:
            ts_data = sales_df.copy()
        elif selected_level == 2:
            ts_data = sales_df[sales_df['state_id'] == selected_series_id]
        elif selected_level == 3:
            ts_data = sales_df[sales_df['store_id'] == selected_series_id]
        elif selected_level == 4:
            ts_data = sales_df[sales_df['cat_id'] == selected_series_id]
        elif selected_level == 5:
            ts_data = sales_df[sales_df['dept_id'] == selected_series_id]
        elif selected_level == 6:
            parts = selected_series_id.split('_')
            cat = parts[-1]
            state = '_'.join(parts[:-1])
            ts_data = sales_df[
                (sales_df['state_id'].str.strip().str.upper() == state.strip().upper()) &
                (sales_df['cat_id'].str.strip().str.upper() == cat.strip().upper())
                ]
        elif selected_level == 7:
            parts = selected_series_id.split('_')
            state = parts[0]
            dept = '_'.join(parts[1:])
            ts_data = sales_df[(sales_df['state_id'] == state) & (sales_df['dept_id'] == dept)]
        elif selected_level == 8:
            parts = selected_series_id.split('_')
            store = '_'.join(parts[:2])
            cat = '_'.join(parts[2:])
            ts_data = sales_df[(sales_df['store_id'] == store) & (sales_df['cat_id'] == cat)]
        elif selected_level == 9:
            parts = selected_series_id.split('_')
            store = '_'.join(parts[:2])
            dept = '_'.join(parts[2:])
            ts_data = sales_df[(sales_df['store_id'] == store) & (sales_df['dept_id'] == dept)]
        elif selected_level == 10:
            ts_data = sales_df[sales_df['item_id'] == selected_series_id]
        elif selected_level == 11:
            item = '_'.join(selected_series_id.split('_')[:3])
            state = selected_series_id.split('_')[3]
            ts_data = sales_df[(sales_df['item_id'] == item) & (sales_df['state_id'] == state)]

        if ts_data.empty:
            st.error(f"No aggregated sales found for {selected_series_id}")
            return

        ts_vals = ts_data[[col for col in sales_df.columns if col.startswith('d_')]].sum(axis=0).values

    else:
        ts_data = sales_df[sales_df['id'] == selected_series_id+'_validation']
        if ts_data.empty:
            st.error(f"No sales data found for ID: {selected_series_id}")
            return
        ts_vals = ts_data[[col for col in ts_data.columns if col.startswith('d_')]].values.flatten()

    date_range = pd.date_range("2011-01-29", periods=len(ts_vals))

    pred_row = pred_df[pred_df['id'] == selected_series_id]
    if pred_row.empty:
        st.error(f"No prediction data found for ID: {selected_series_id}")
        return

    pred_values = pred_row.drop(columns=['id']).values.flatten()
    fig = plot_forecast(ts_vals, pred_values, date_range)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
