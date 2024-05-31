import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from keras.models import load_model
import joblib

# Load preprocessed data
final_data = pd.read_csv('data/final_data.csv')

# Convert the 'Timestamp' column to datetime format
final_data['Timestamp'] = pd.to_datetime(final_data['Timestamp'])

# Filter data for a single day (e.g., May 1, 2024)
one_day_data = final_data[final_data['Timestamp'].dt.date == pd.Timestamp('2024-05-01').date()]

# Define features and target variable for the single day
X_one_day = one_day_data[['Max_Capacity_kWh', 'Current_Load_kWh', 'Max_Generation_Rate_kWh', 'Current_Generation_Rate_kWh', 'Load_Ratio', 'Generation_Ratio']]
y_one_day = one_day_data['Power_Demand_kWh']

# Load models
lr_model = joblib.load('models/lr_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
gb_model = joblib.load('models/gb_model.pkl')
nn_model = load_model('models/nn_model.h5')

# Make predictions for the single day
y_pred_lr_one_day = lr_model.predict(X_one_day)
y_pred_rf_one_day = rf_model.predict(X_one_day)
y_pred_gb_one_day = gb_model.predict(X_one_day)
y_pred_nn_one_day = nn_model.predict(X_one_day)

# Plot actual vs predicted demand for each model
def plot_actual_vs_predicted_one_day(y_actual, y_pred, model_name, timestamps):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, y_actual, label='Actual Demand')
    plt.plot(timestamps, y_pred, label='Predicted Demand', linestyle='-.')
    plt.title(f'Actual vs Predicted Power Demand ({model_name}) - May 1, 2024')
    plt.xlabel('Time')
    plt.ylabel('Power Demand (kWh)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig(f'results/actual_vs_predicted_{model_name}_one_day.png')


# Plotting for each model for the single day
plot_actual_vs_predicted_one_day(y_one_day, y_pred_lr_one_day, 'Linear Regression', one_day_data['Timestamp'])
plot_actual_vs_predicted_one_day(y_one_day, y_pred_rf_one_day, 'Random Forest', one_day_data['Timestamp'])
plot_actual_vs_predicted_one_day(y_one_day, y_pred_gb_one_day, 'Gradient Boosting', one_day_data['Timestamp'])
plot_actual_vs_predicted_one_day(y_one_day, y_pred_nn_one_day, 'Neural Network', one_day_data['Timestamp'])

print("Plots for a single day have been saved in the 'results/' directory.")

# Additional EDA before and after AI model for one day
def plot_eda_before_after_one_day(data, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Timestamp'], data['Power_Demand_kWh'], label='Actual Power Demand')
    plt.scatter(data['Timestamp'], y_pred, label='Predicted Power Demand', alpha=0.7)
    plt.title(f'Before and After AI Model ({model_name}) - May 1, 2024')
    plt.xlabel('Timestamp')
    plt.ylabel('Power Demand (kWh)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig(f'results/before_after_{model_name}_one_day.png')


# Plotting before and after for each model for the single day
plot_eda_before_after_one_day(one_day_data, y_pred_lr_one_day, 'Linear Regression')
plot_eda_before_after_one_day(one_day_data, y_pred_rf_one_day, 'Random Forest')
plot_eda_before_after_one_day(one_day_data, y_pred_gb_one_day, 'Gradient Boosting')
plot_eda_before_after_one_day(one_day_data, y_pred_nn_one_day, 'Neural Network')

print("Before and after plots for a single day have been saved in the 'results/' directory.")
