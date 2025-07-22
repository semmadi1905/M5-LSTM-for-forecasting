# 📊 M5 Forecasting Application

This repository contains code and configurations to train and visualize time series forecasting models for the M5 dataset.

## 📁 Project Structure
<img width="146" height="187" alt="image" src="https://github.com/user-attachments/assets/894435c6-063e-490e-8aff-80ea31dcb362" />

project/
│
├── data/ # Input dataset files
├── output/ # Trained models and result files
│
├── main.py # Main script to launch training
├── m5_data_preparation.py # Data preparation module
├── m5_training_pipeline.py# Model architecture and training pipeline
├── streamlit_app.py # Streamlit app to visualize results
│
└── config.yaml # Configuration file for levels, epochs, and other parameters

## 🚀 How to Run the Application

### Step 1: Create and activate the Conda environment

conda create -n m5env python=3.10 -y
conda activate m5env

Step 2: Install the dependencies
pip install -r requirements.txt


Step 3: Configure the model
Edit config.yaml to specify:
model_level: from 1 to 12
epochs: desired number of training epochs
Other tunable parameters

Step 4: Train the model
python main.py

Step 5: Launch the Streamlit dashboard
streamlit run streamlit_app.py

Step 6: Explore the Results
Use dropdown menus to select model levels
Zoom and pan to inspect forecasted vs. actual time series plots

<img width="309" height="129" alt="image" src="https://github.com/user-attachments/assets/c2933bfb-f833-406d-b462-f50136ae3909" />


📝 Notes
All training artifacts and predictions will be saved in the output/folder.
You can repeat the training process with different levels by updating config.yaml.
