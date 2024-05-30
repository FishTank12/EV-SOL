# src/training.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the final data
final_data = pd.read_csv('../data/final_synthetic_data.csv')

# Calculate distance between suppliers and distributors
final_data['Distance'] = np.sqrt((final_data['Distributor_Latitude'] - final_data['Supplier_Latitude'])**2 +
                                 (final_data['Distributor_Longitude'] - final_data['Supplier_Longitude'])**2)

# Calculate load ratio
final_data['Load_Ratio'] = final_data['Current_Load_kWh'] / final_data['Max_Capacity_kWh']

# Load the data with features
data = final_data

# Define features and target
X = data[['Max_Capacity_kWh', 'Current_Load_kWh', 'Max_Generation_Rate_kWh', 'Current_Generation_Rate_kWh', 'Distance', 'Load_Ratio']]
y = data['Power_Demand_kWh']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
