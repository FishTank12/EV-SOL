# main.py
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
nn_models = {
    'relu': load_model('models/nn_model_relu.h5'),
    'leaky_relu': load_model('models/nn_model_leaky_relu.h5'),
    'elu': load_model('models/nn_model_elu.h5'),
    'swish': load_model('models/nn_model_swish.h5'),
    'tanh': load_model('models/nn_model_tanh.h5')
}

# Make predictions
y_pred_lr = lr_model.predict(X)
y_pred_rf = rf_model.predict(X)
y_pred_gb = gb_model.predict(X)
y_pred_nn = {key: model.predict(X).flatten() for key, model in nn_models.items()}

# Select one day of data for comparison
selected_day = final_data[final_data['Timestamp'].str.contains('2024-05-01')]
X_day = selected_day[['Max_Capacity_kWh', 'Current_Load_kWh', 'Max_Generation_Rate_kWh', 'Current_Generation_Rate_kWh', 'Load_Ratio', 'Generation_Ratio']]
y_day = selected_day['Power_Demand_kWh']

# Make predictions for the selected day
y_pred_day_lr = lr_model.predict(X_day)
y_pred_day_rf = rf_model.predict(X_day)
y_pred_day_gb = gb_model.predict(X_day)
y_pred_day_nn = {key: model.predict(X_day).flatten() for key, model in nn_models.items()}

# Plot actual vs predicted demand for each model
def plot_actual_vs_predicted(y_actual, y_pred, model_name, day):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_actual)), y_actual, label='Actual Demand')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted Demand', linestyle='--')
    plt.title(f'Actual vs Predicted Power Demand ({model_name}) - {day}')
    plt.xlabel('Hour')
    plt.ylabel('Power Demand (kWh)')
    plt.legend()
    plt.savefig(f'results/actual_vs_predicted_{model_name}_{day}.png')
    plt.close()

# Plotting for each model for the selected day
day_str = '2024-05-01'
plot_actual_vs_predicted(y_day, y_pred_day_lr, 'Linear Regression', day_str)
plot_actual_vs_predicted(y_day, y_pred_day_rf, 'Random Forest', day_str)
plot_actual_vs_predicted(y_day, y_pred_day_gb, 'Gradient Boosting', day_str)

for activation in y_pred_day_nn.keys():
    plot_actual_vs_predicted(y_day, y_pred_day_nn[activation], f'Neural Network ({activation})', day_str)

print("Plots have been saved in the 'results/' directory.")
