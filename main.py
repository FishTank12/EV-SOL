import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from keras.models import load_model
import joblib

# Load preprocessed data
final_data = pd.read_csv('data/final_data.csv')

# Define features and target variable
X = final_data[['Max_Capacity_kWh', 'Current_Load_kWh', 'Max_Generation_Rate_kWh', 'Current_Generation_Rate_kWh', 'Load_Ratio', 'Generation_Ratio']]
y = final_data['Power_Demand_kWh']

# Load models
lr_model = joblib.load('models/lr_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
gb_model = joblib.load('models/gb_model.pkl')
nn_model = load_model('models/nn_model.h5')

# Make predictions
y_pred_lr = lr_model.predict(X)
y_pred_rf = rf_model.predict(X)
y_pred_gb = gb_model.predict(X)
y_pred_nn = nn_model.predict(X)

# Plot actual vs predicted demand for each model
def plot_actual_vs_predicted(y_actual, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(y_actual, label='Actual Demand')
    plt.plot(y_pred, label='Predicted Demand', linestyle='--')
    plt.title(f'Actual vs Predicted Power Demand ({model_name})')
    plt.xlabel('Hour')
    plt.ylabel('Power Demand (kWh)')
    plt.legend()
    plt.savefig(f'results/actual_vs_predicted_{model_name}.png')
    plt.show()

# Plotting for each model
plot_actual_vs_predicted(y, y_pred_lr, 'Linear Regression')
plot_actual_vs_predicted(y, y_pred_rf, 'Random Forest')
plot_actual_vs_predicted(y, y_pred_gb, 'Gradient Boosting')
plot_actual_vs_predicted(y, y_pred_nn, 'Neural Network')

print("Plots have been saved in the 'results/' directory.")

# Additional EDA before and after AI model
def plot_eda_before_after(final_data, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(final_data['Timestamp'], final_data['Power_Demand_kWh'], label='Actual Power Demand')
    plt.scatter(final_data['Timestamp'], y_pred, label='Predicted Power Demand', alpha=0.7)
    plt.title(f'Before and After AI Model ({model_name})')
    plt.xlabel('Timestamp')
    plt.ylabel('Power Demand (kWh)')
    plt.legend()
    plt.savefig(f'results/before_after_{model_name}.png')
    plt.show()

# Plotting before and after for each model
plot_eda_before_after(final_data, y_pred_lr, 'Linear Regression')
plot_eda_before_after(final_data, y_pred_rf, 'Random Forest')
plot_eda_before_after(final_data, y_pred_gb, 'Gradient Boosting')
plot_eda_before_after(final_data, y_pred_nn, 'Neural Network')

print("Before and after plots have been saved in the 'results/' directory.")
